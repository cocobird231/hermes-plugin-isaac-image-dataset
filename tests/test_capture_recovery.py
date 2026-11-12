"""CPU fixtures test durable recovery; fixture PNGs are never dataset acceptance evidence."""
import json
import random
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import checkpoints as cp
from dataset import render_worker as renderer
from dataset import splitting


class RecoveryTests(unittest.TestCase):
    def fixture(self, count=8, committed=3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root/"scene.usda"
        source.write_text("#usda 1.0\n")
        samples = []
        for i in range(count):
            pose = [i*3., 0., 1.5, 0., 0., 0., 1.]
            matrix = [[1.,0.,0.,pose[0]], [0.,1.,0.,0.], [0.,0.,1.,1.5], [0.,0.,0.,1.]]
            samples.append({"sample_id": f"{i:06d}", "pose_xyzw": pose, "T_world_camera": matrix,
                            "group_id": "g"+str(i), "source_face_id": i, "region_id": "road",
                            "ground_z_m": 0., "angles_deg": {"yaw":0., "pitch":0., "roll":0.}})
        plan = {"schema_version":1, "complete":True, "samples":samples,
                "scene":{"path":str(source), "sha256":cp.sha256(source)},
                "config":{"n":count, "seed":42, "camera":{"resolution":[16,16]},
                          "split_ratios":[4,2,2], "group_distance_m":2.}}
        plan_path = root/"plan.json"
        cp.atomic_json(plan_path, plan)
        output = root/"capture"
        cp.prepare_output(plan_path, output)
        rows = []
        K = [[8.,0.,8.],[0.,8.,8.],[0.,0.,1.]]
        for index, sample in enumerate(samples[:committed]):
            sid = sample["sample_id"]
            image = output/"images"/(sid+".png")
            Image.new("RGB", (16,16), (index*20,40,60)).save(image)
            with image.open("rb") as stream:
                import os
                os.fsync(stream.fileno())
            evidence = {"sample_id":sid, "validation":{"passed":True, "intrinsics_K":K},
                        "T_world_camera":sample["T_world_camera"], "native_nurec_crop_unchanged":True,
                        "camera_readback_after_same_render_call":True, "lossless_png_readback":True}
            camera = output/"camera"/(sid+".json")
            cp.atomic_json(camera, evidence)
            row = {**sample, "image_path":"images/"+sid+".png", "image_sha256":cp.sha256(image),
                   "camera_evidence_path":"camera/"+sid+".json", "camera_evidence_sha256":cp.sha256(camera),
                   "requested_pose_xyzw":sample["pose_xyzw"], "calibration_passed":True,
                   "simulation_time_s":1., "kit_update_number":index+100, "frame_index":index}
            row.update(dict(zip(("x","y","z","qx","qy","qz","qw"), row["pose_xyzw"])))
            cp.commit_sample(output, cp.sha256(plan_path), index, row)
            rows.append(row)
        cp.publish_indexes(output, rows)
        cp.atomic_json(output/"metadata.json", {"complete":committed==count, "captured_count":committed,
                       "plan_sha256":cp.sha256(plan_path), "scene":plan["scene"], "intrinsics_K":K})
        cp.atomic_json(output/"result.json", {"ok":committed==count, "batch_complete":True,
                       "app_closed":True, "source_unchanged":True, "native_nurec_crop_unchanged":True,
                       "captured_count":committed, "requested_count":count})
        return plan_path, output, rows

    def test_restart_reuses_exact_verified_prefix(self):
        plan, output, rows = self.fixture()
        hashes = {str(path):cp.sha256(path) for path in (output/"images").iterdir()}
        cp.prepare_output(plan, output)
        self.assertEqual(cp.validate_capture(plan, output), rows)
        self.assertEqual(hashes, {str(path):cp.sha256(path) for path in (output/"images").iterdir()})

    def test_partial_indexes_rebuilt_from_commits(self):
        plan, output, rows = self.fixture()
        (output/"samples.jsonl").write_text("{interrupted")
        (output/"samples.csv").unlink()
        recovered = cp.validate_capture(plan, output)
        cp.publish_indexes(output, recovered)
        self.assertEqual([json.loads(line) for line in (output/"samples.jsonl").read_text().splitlines()], rows)
        self.assertEqual(len((output/"samples.csv").read_text().splitlines()), len(rows)+1)

    def test_uncommitted_owned_orphans_do_not_count(self):
        plan, output, rows = self.fixture()
        (output/"images/000003.png").write_bytes(b"interrupted")
        (output/"camera/000003.json").write_text("partial")
        (output/"commits/000003.json.orphan.tmp").write_text("partial")
        self.assertEqual(cp.validate_capture(plan, output), rows)

    def test_plan_change_rejected(self):
        plan, output, _ = self.fixture()
        original = json.loads(plan.read_text())
        original["config"]["seed"] = 43
        cp.atomic_json(plan, original)
        with self.assertRaisesRegex(ValueError, "plan SHA"):
            cp.prepare_output(plan, output)
        with self.assertRaisesRegex(ValueError, "plan SHA"):
            cp.validate_capture(plan, output)

    def test_existing_unbound_directory_rejected(self):
        plan, output, _ = self.fixture()
        (output/"capture-manifest.json").unlink()
        with self.assertRaises(ValueError):
            cp.prepare_output(plan, output)

    def test_committed_image_or_evidence_corruption_fails(self):
        for key in ("image_path", "camera_evidence_path"):
            with self.subTest(key=key):
                plan, output, rows = self.fixture()
                (output/rows[0][key]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    cp.validate_capture(plan, output)

    def test_hole_in_committed_prefix_fails(self):
        plan, output, _ = self.fixture()
        (output/"commits/000001.json").unlink()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            cp.validate_capture(plan, output)

    def test_deleted_tail_commit_rejected_by_independent_watermark(self):
        plan, output, _ = self.fixture()
        (output/"commits/000002.json").unlink()
        # Even a stale/replaced current receipt cannot erase the durable watermark.
        cp.atomic_json(output/"result.json", {"captured_count":0})
        cp.atomic_json(output/"metadata.json", {"captured_count":0})
        with self.assertRaisesRegex(ValueError, "high-water"):
            cp.validate_capture(plan, output)

    def test_prior_batch_without_watermark_preserves_receipt_high_water(self):
        plan, output, rows = self.fixture()
        (output/"capture-watermark.json").unlink()
        self.assertEqual(cp.validate_capture(plan, output), rows)
        (output/"commits/000002.json").unlink()
        with self.assertRaisesRegex(ValueError, "high-water"):
            cp.validate_capture(plan, output)
        with mock.patch.object(sys, "argv", ["renderer", "--plan", str(plan), "--output", str(output)]):
            with mock.patch.object(renderer, "run") as run:
                self.assertEqual(renderer.main(), 2)
                run.assert_not_called()
        self.assertEqual(json.loads((output/"result.json").read_text())["captured_count"], 3)

    def test_watermark_plan_and_commit_hash_tampering_rejected(self):
        for field in ("plan_sha256", "last_commit_sha256", "committed_count"):
            plan, output, _ = self.fixture()
            path = output/"capture-watermark.json"
            watermark = json.loads(path.read_text())
            watermark[field] = "0"*64 if field != "committed_count" else 4
            cp.atomic_json(path, watermark)
            with self.assertRaises(ValueError):
                cp.validate_capture(plan, output)

    def test_commit_survives_interruption_before_watermark_advance(self):
        plan, output, rows = self.fixture()
        # A killed process can leave a durable commit newer than its watermark.
        cp.atomic_json(output/"capture-watermark.json", {
            "schema_version":1, "plan_sha256":cp.sha256(plan), "committed_count":2,
            "last_commit_sha256":cp.sha256(output/"commits/000001.json")})
        self.assertEqual(cp.validate_capture(plan, output), rows)
        cp.publish_watermark(output, cp.sha256(plan), len(rows))
        self.assertEqual(cp.validate_capture(plan, output), rows)

    def test_commit_cannot_overwrite_durable_sample(self):
        plan, output, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            cp.commit_sample(output, cp.sha256(plan), 0, rows[0])

    def test_commit_symlink_and_path_mismatch_rejected(self):
        for mode in ("symlink", "path"):
            plan, output, rows = self.fixture()
            if mode == "symlink":
                path = output/rows[0]["image_path"]
                real = path.with_suffix(".backup")
                path.rename(real)
                path.symlink_to(real)
            else:
                path = output/"commits/000000.json"
                record = json.loads(path.read_text())
                record["row"]["image_path"] = "../outside.png"
                cp.atomic_json(path, record)
            with self.assertRaises(ValueError):
                cp.validate_capture(plan, output)

    def test_capture_subdirectory_symlinks_rejected(self):
        for name in ("images", "camera", "commits"):
            plan, output, _ = self.fixture()
            directory = output/name
            moved = output/(name+"-real")
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Unsafe capture directory"):
                cp.validate_capture(plan, output)

    def test_capture_root_symlink_rejected(self):
        plan, output, _ = self.fixture(committed=8)
        linked = output.parent/"linked-dataset"
        linked.symlink_to(output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
            cp.validate_capture(plan, linked)
        with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
            splitting.finalize_dataset(plan, linked)

    def test_commit_pose_and_group_association_checked(self):
        for key, value in (("group_id","different"), ("pose_xyzw",[10.,0.,1.5,0.,0.,0.,1.])):
            plan, output, _ = self.fixture()
            path = output/"commits/000000.json"
            record = json.loads(path.read_text())
            record["row"][key] = value
            cp.atomic_json(path, record)
            with self.assertRaises(ValueError):
                cp.validate_capture(plan, output)

    def test_final_requires_all_commits(self):
        plan, output, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cp.validate_capture(plan, output, require_complete=True)

    def test_atomic_write_failure_leaves_previous_contents(self):
        plan, output, _ = self.fixture()
        original = plan.read_bytes()
        with mock.patch.object(cp.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                cp.atomic_json(plan, {"broken":True})
        self.assertEqual(plan.read_bytes(), original)
        self.assertEqual(list(output.glob("*.tmp")), [])

    def test_split_is_idempotent_and_contents_exact(self):
        plan, output, _ = self.fixture(committed=8)
        first = splitting.finalize_dataset(plan, output)
        before = {p.name:cp.sha256(p) for p in (output/"splits").iterdir()}
        self.assertEqual(splitting.finalize_dataset(plan, output), first)
        self.assertEqual(before, {p.name:cp.sha256(p) for p in (output/"splits").iterdir()})
        groups = [set() for _ in range(3)]
        for index, name in enumerate(splitting.NAMES):
            groups[index] = {gid for gid, assigned in first["assignments"].items() if assigned == name}
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_split_resume_after_directory_published_before_receipt(self):
        plan, output, _ = self.fixture(committed=8)
        original = splitting.atomic_json
        def interrupt(path, value):
            if Path(path).name == "split-result.json":
                raise OSError("killed before receipt")
            original(path, value)
        with mock.patch.object(splitting, "atomic_json", side_effect=interrupt):
            with self.assertRaises(OSError):
                splitting.finalize_dataset(plan, output)
        self.assertTrue((output/"splits").is_dir())
        self.assertFalse((output/"split-result.json").exists())
        self.assertTrue(splitting.finalize_dataset(plan, output)["ok"])

    def test_split_interrupted_staging_never_published(self):
        plan, output, _ = self.fixture(committed=8)
        with mock.patch.object(splitting.os, "rename", side_effect=OSError("interrupt")):
            with self.assertRaises(OSError):
                splitting.finalize_dataset(plan, output)
        self.assertFalse((output/"splits").exists())
        self.assertFalse((output/"split-result.json").exists())
        self.assertTrue(splitting.finalize_dataset(plan, output)["ok"])

    def test_changed_split_ratios_or_contents_rejected(self):
        for mode in ("ratio", "contents", "receipt", "extra"):
            plan, output, _ = self.fixture(committed=8)
            splitting.finalize_dataset(plan, output)
            if mode == "contents":
                (output/"splits/train.txt").write_text("bad\n")
            elif mode == "receipt":
                cp.atomic_json(output/"split-result.json", {"ok":True})
            elif mode == "extra":
                (output/"splits/extra.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "Existing split"):
                splitting.finalize_dataset(plan, output, [1,1,1] if mode == "ratio" else None)

    def test_forged_master_index_cannot_override_commits(self):
        plan, output, rows = self.fixture(committed=8)
        rows[0]["capture_batch_id"] = "changed"
        cp.publish_indexes(output, rows)
        with self.assertRaisesRegex(ValueError, "committed"):
            splitting.finalize_dataset(plan, output)

    def test_source_change_refuses_split(self):
        plan, output, _ = self.fixture(committed=8)
        source = Path(json.loads(plan.read_text())["scene"]["path"])
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Scene changed"):
            splitting.finalize_dataset(plan, output)

    def test_spatial_hash_agrees_with_pairwise_reference(self):
        rng = random.Random(42)
        for threshold in (0., .3, 2.):
            for _ in range(25):
                rows = [{"pose_xyzw":[rng.uniform(-10,10),rng.uniform(-10,10)], "group_id":str(rng.randrange(3))} for _ in range(50)]
                import math
                invalid = any(a["group_id"]!=b["group_id"] and math.dist(a["pose_xyzw"],b["pose_xyzw"])<=threshold
                              for i,a in enumerate(rows) for b in rows[i+1:])
                if invalid:
                    with self.assertRaises(ValueError):
                        splitting.verify_nearby_groups(rows, threshold)
                else:
                    splitting.verify_nearby_groups(rows, threshold)
        rows = [{"pose_xyzw":[0.,0.], "group_id":"same"} for _ in range(10000)]
        splitting.verify_nearby_groups(rows, 2.)

    def test_cli_accepts_derived_timeout_public_runtime_boundaries(self):
        for timeout in (1, 2, 7800, 86_486_400):
            plan, output, _ = self.fixture()
            with mock.patch.object(sys, "argv", ["renderer", "--plan", str(plan), "--output", str(output),
                                                "--batch-size", "40", "--timeout-seconds", str(timeout)]):
                with mock.patch.object(renderer, "run", return_value=1) as run:
                    self.assertEqual(renderer.main(), 1)
                    self.assertEqual(run.call_args.args[-2:], (40, timeout))
        for timeout in (0, -1, 86_486_401):
            plan, output, _ = self.fixture()
            with mock.patch.object(sys, "argv", ["renderer", "--plan", str(plan), "--output", str(output),
                                                "--timeout-seconds", str(timeout)]):
                with mock.patch.object(renderer, "run") as run:
                    self.assertEqual(renderer.main(), 2)
                    run.assert_not_called()

    def test_partial_batch_exit_requires_close_source_and_counts(self):
        _, output, _ = self.fixture()
        receipt = {"ok":False, "batch_complete":True, "app_closed":True, "source_unchanged":True,
                   "native_nurec_crop_unchanged":True, "phase":"batch_complete",
                   "captured_count":20, "requested_count":60}
        path = output/"result.json"
        cp.atomic_json(path, receipt)
        self.assertEqual(renderer.validated_process_exit_code(0, path),0)
        self.assertEqual(renderer.validated_process_exit_code(139, path),139)
        for key,value in (("batch_complete",False),("app_closed",False),("source_unchanged",False),
                          ("captured_count",60),("captured_count",0),("captured_count",True),("ok",True)):
            cp.atomic_json(path, {**receipt,key:value})
            self.assertEqual(renderer.validated_process_exit_code(0,path),1,key)


if __name__ == "__main__":
    unittest.main()
