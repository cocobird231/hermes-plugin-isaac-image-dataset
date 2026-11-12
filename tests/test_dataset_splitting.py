import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.splitting import assign_groups, finalize_dataset, ratios_and_targets, verify_nearby_groups
from dataset.checkpoints import prepare_output, commit_sample


class SplitTests(unittest.TestCase):
    def rows(self, n=32):
        return [{"sample_id": f"{i:06d}", "group_id": f"g{i}", "pose_xyzw":[i*3.,0,1.5,0,0,0,1]} for i in range(n)]

    def test_default_ratio(self):
        assignment, result = assign_groups(self.rows(), [4,2,2], 42)
        self.assertEqual(result["actual_counts"], {"train":16,"validation":8,"test":8})
        self.assertEqual(len(assignment), 32)
        self.assertEqual((assignment,result), assign_groups(self.rows(), [4,2,2],42))

    def test_changed_seed_changes_assignment(self):
        self.assertNotEqual(assign_groups(self.rows(), [4,2,2],42)[0],assign_groups(self.rows(), [4,2,2],43)[0])

    def test_fractional_and_zero_ratios(self):
        self.assertEqual(ratios_and_targets([4,2,2],7)[1], [3,2,2])
        self.assertEqual(assign_groups(self.rows(),[1,0,0],42)[1]["actual_counts"],{"train":32,"validation":0,"test":0})

    def test_invalid_ratios(self):
        for value in ([0,0,0],[-1,1,1],[True,1,1],[1,float("nan"),1],[1,float("inf"),1],[1,1],"4:2:2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ratios_and_targets(value,32)

    def test_huge_ratios_do_not_overflow(self):
        self.assertEqual(ratios_and_targets([1e308]*3,3)[1],[1,1,1])

    def test_group_integrity_and_impossible(self):
        rows=self.rows(12)
        for i,row in enumerate(rows):
            row["group_id"]=str(i//4)
        assignment,result=assign_groups(rows,[4,2,2],42)
        self.assertEqual(set(result["actual_counts"].values()),{4})
        self.assertEqual(len(assignment),3)
        for row in rows:
            row["group_id"]="single"
        with self.assertRaises(ValueError):
            assign_groups(rows,[4,2,2],42)

    def test_nearby_group_guard(self):
        rows=self.rows(2)
        rows[1]["pose_xyzw"][0]=1
        with self.assertRaises(ValueError):
            verify_nearby_groups(rows,2)
        rows[1]["group_id"]=rows[0]["group_id"]
        verify_nearby_groups(rows,2)

    def fixture(self):
        from PIL import Image
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        output=root/"dataset"
        rows=self.rows(8)
        plan={"complete":True,"config":{"n":8,"seed":42,"group_distance_m":2,"camera":{"resolution":[8,4]},"split_ratios":[4,2,2]},"samples":rows,"scene":{}}
        source=root/"scene.usda"
        source.write_text("#usda 1.0\n")
        plan["scene"]={"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
        for i,row in enumerate(rows):
            row.update(source_face_id=i,region_id="road",ground_z_m=0.,angles_deg={"yaw":0.,"pitch":0.,"roll":0.})
        plan_path=root/"plan.json"
        plan_path.write_text(json.dumps(plan))
        prepare_output(plan_path, output)
        captured=[]
        for i,row in enumerate(rows):
            p=output/"images"/(row["sample_id"]+".png")
            Image.new("RGB",(8,4),(i*20,40,60)).save(p)
            captured.append({**row,"image_path":str(p.relative_to(output)),"image_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        K=[[4,0,4],[0,4,2],[0,0,1]]
        for index,row in enumerate(captured):
            x,y,z,*quaternion=row["pose_xyzw"]
            matrix=[[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]
            evidence={"sample_id":row["sample_id"],"validation":{"passed":True,"intrinsics_K":K},
                      "native_nurec_crop_unchanged":True,"camera_readback_after_same_render_call":True,"lossless_png_readback":True,
                      "T_world_camera":matrix}
            camera=output/"camera"/(row["sample_id"]+".json")
            camera.write_text(json.dumps(evidence))
            row.update(camera_evidence_path=str(camera.relative_to(output)),
                       camera_evidence_sha256=hashlib.sha256(camera.read_bytes()).hexdigest(),
                       calibration_passed=True,T_world_camera=matrix,frame_index=index,
                       requested_pose_xyzw=row["pose_xyzw"])
            row.update(dict(zip(("x","y","z","qx","qy","qz","qw"),row["pose_xyzw"])))
            commit_sample(output, hashlib.sha256(plan_path.read_bytes()).hexdigest(), index, row)
        with (output/"samples.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=("sample_id","image_path","x","y","z","qx","qy","qz","qw"),extrasaction="ignore")
            writer.writeheader()
            writer.writerows(captured)
        (output/"metadata.json").write_text(json.dumps({"complete":True,"captured_count":8,"scene":plan["scene"],
            "plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),"intrinsics_K":K}))
        (output/"result.json").write_text(json.dumps({"ok":True,"app_closed":True,"source_unchanged":True,
                                                    "captured_count":8,"requested_count":8}))
        self.write_rows(output,captured)
        return plan_path,output,captured

    def write_rows(self, output, rows):
        (output/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))

    def test_real_png_disk_readback_and_no_overwrite(self):
        p,d,rows=self.fixture()
        result=finalize_dataset(p,d)
        self.assertEqual(result["actual_counts"],{"train":4,"validation":2,"test":2})
        self.assertEqual(sum(len((d/"splits"/(n+".txt")).read_text().splitlines()) for n in ("train","validation","test")),8)
        self.assertEqual(finalize_dataset(p,d), result)

    def test_missing_manifest_does_not_allow_legacy_fallback(self):
        plan, output, _ = self.fixture()
        (output/"capture-manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "capture-manifest"):
            finalize_dataset(plan, output)
        self.assertFalse((output/"splits").exists())

    def test_sha_corruption_refused(self):
        p,d,rows=self.fixture()
        (d/rows[0]["image_path"]).write_bytes(b"bad")
        with self.assertRaises(ValueError):
            finalize_dataset(p,d)
        self.assertFalse((d/"splits").exists())

    def test_pose_mismatch_refused(self):
        p,d,rows=self.fixture()
        rows[0]["pose_xyzw"][2]+=1
        self.write_rows(d,rows)
        with self.assertRaises(ValueError):
            finalize_dataset(p,d)

    def test_association_and_duplicate_refused(self):
        for mode in ("group","duplicate","missing"):
            with self.subTest(mode=mode):
                p,d,rows=self.fixture()
                if mode=="group": rows[0]["group_id"]="bad"
                elif mode=="duplicate": rows[1]=rows[0]
                else: rows.pop()
                self.write_rows(d,rows)
                with self.assertRaises(ValueError):
                    finalize_dataset(p,d)

    def test_partial_capture_refused(self):
        p,d,rows=self.fixture()
        (d/"result.json").write_text('{"ok":false}')
        with self.assertRaises(ValueError):
            finalize_dataset(p,d)

    def test_camera_evidence_and_metadata_binding(self):
        for mode in ("missing","hash","id","metadata","csv"):
            with self.subTest(mode=mode):
                p,d,rows=self.fixture()
                camera=d/rows[0]["camera_evidence_path"]
                if mode=="missing": camera.unlink()
                elif mode=="hash": camera.write_text("{}")
                elif mode=="id":
                    value=json.loads(camera.read_text())
                    value["sample_id"]="other"
                    camera.write_text(json.dumps(value))
                    rows[0]["camera_evidence_sha256"]=hashlib.sha256(camera.read_bytes()).hexdigest()
                    self.write_rows(d,rows)
                elif mode=="metadata":
                    value=json.loads((d/"metadata.json").read_text())
                    value["plan_sha256"]="0"*64
                    (d/"metadata.json").write_text(json.dumps(value))
                else:
                    (d/"samples.csv").write_text("sample_id,image_path\n")
                with self.assertRaises((ValueError,FileNotFoundError)):
                    finalize_dataset(p,d)
                self.assertFalse((d/"splits").exists())

    def test_path_escape_refused(self):
        p,d,rows=self.fixture()
        rows[0]["image_path"]="../outside.png"
        self.write_rows(d,rows)
        with self.assertRaises(ValueError):
            finalize_dataset(p,d)


if __name__=="__main__":
    unittest.main()
