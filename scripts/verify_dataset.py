"""Independent read-only output audit; never repairs or publishes dataset artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(job):
    root = job/"dataset"
    plan = json.loads((job/"plan.json").read_text())
    meta = json.loads((root/"metadata.json").read_text())
    result = json.loads((root/"result.json").read_text())
    split = json.loads((root/"split-result.json").read_text())
    rows = [json.loads(x) for x in (root/"samples.jsonl").read_text().splitlines()]
    count = plan["config"]["n"]
    checks = 0
    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1
    record = json.loads((job/"job.json").read_text())
    receipt = json.loads((job/f"attempt-{record['attempt']}-exit.json").read_text())
    check(receipt["job_id"] == record["job_id"] == job.name and receipt["attempt"] == record["attempt"]
          and receipt["unit"] == record["unit"], "supervisor receipt identity")
    check(receipt["request_sha256"] == digest(job/Path(record["request_path"]).name), "request binding")
    check(receipt["worker_returncode"] == 0 and receipt["worker_status"]["state"] == "completed", "actual worker exit and final state")
    binding = json.loads((job/"plan-commit.json").read_text())
    config_digest = hashlib.sha256(json.dumps(json.loads((job/"config.json").read_text()), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    check(binding["plan_sha256"] == digest(job/"plan.json") and binding["configuration_sha256"] == config_digest, "immutable sampling commit")
    manifest = json.loads((root/"capture-manifest.json").read_text())
    watermark = json.loads((root/"capture-watermark.json").read_text())
    check(manifest["plan_sha256"] == watermark["plan_sha256"] == digest(job/"plan.json"), "capture plan binding")
    check(watermark["committed_count"] == count, "durable high water count")
    check(watermark["last_commit_sha256"] == digest(root/"commits"/f"{count-1:06d}.json"), "last commit SHA")
    check(plan["complete"] and meta["complete"] and result["ok"] and split["ok"], "all stages complete")
    check(result["app_closed"] and result["source_unchanged"], "renderer closed and sources preserved")
    check(len(rows) == len(plan["samples"]) == count == result["captured_count"] == split["sample_count"], "counts")
    check(len(list((root/"images").glob("*.png"))) == count, "exact PNG count")
    check(len(list((root/"camera").glob("*.json"))) == count, "exact camera evidence count")
    check(len(list((root/"commits").glob("*.json"))) == count, "exact commit count")
    check(len({r["sample_id"] for r in rows}) == count, "unique sample IDs")
    check(meta["plan_sha256"] == split["plan_sha256"] == digest(job/"plan.json"), "plan binding")
    check(meta["simulation"]["play_verified"] and meta["simulation"]["after_play_time_s"] > 0, "simulation advanced")
    check(split["samples_sha256"] == digest(root/"samples.jsonl"), "master index SHA")
    for path, expected in plan["scene"]["source_dependencies"].items():
        check(digest(Path(path)) == expected, "source dependency SHA:"+Path(path).name)
    width, height = plan["config"]["camera"]["resolution"]
    k = np.asarray(meta["intrinsics_K"])
    if not plan["config"]["camera"].get("template_prim"):
        f = width/(2*math.tan(math.radians(plan["config"]["camera"]["horizontal_fov_deg"])/2))
        check(np.allclose(k, [[f,0,width/2],[0,f,height/2],[0,0,1]], atol=1e-4, rtol=0), "FOV agrees with rendered K")
    hashes, black, errors = [], [], []
    batch_counts = {}
    for index, (row, sample) in enumerate(zip(rows, plan["samples"])):
        sid = row["sample_id"]
        check(sid == sample["sample_id"] and row["frame_index"] == index, sid+": sequence")
        image = root/row["image_path"]
        evidence_path = root/row["camera_evidence_path"]
        for path in (image, evidence_path):
            check(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), sid+": contained file")
        check(digest(image) == row["image_sha256"], sid+": image SHA")
        check(digest(evidence_path) == row["camera_evidence_sha256"], sid+": evidence SHA")
        evidence = json.loads(evidence_path.read_text())
        with Image.open(image) as im:
            im.load()
            check(im.format == "PNG" and im.mode == "RGB" and im.size == (width,height), sid+": PNG decode")
            pixels = np.asarray(im)
        check(float(pixels.std()) > 2, sid+": nonconstant RGB")
        check(abs(float(pixels.std())-evidence["rgb_std"]) < 1e-6, sid+": RGB statistic evidence")
        actual, expected = np.asarray(row["pose_xyzw"]), np.asarray(sample["pose_xyzw"])
        error = float(np.max(np.abs(actual[:3]-expected[:3])))
        check(error < 1e-4 and abs(float(np.dot(actual[3:],expected[3:]))) > 1-1e-7, sid+": measured pose")
        check(abs(float(np.linalg.norm(actual[3:]))-1) < 1e-6, sid+": normalized quaternion")
        check(evidence["validation"]["passed"] and np.allclose(evidence["validation"]["intrinsics_K"], k), sid+": calibration")
        check(np.allclose(np.asarray(evidence["camera_to_world_usd_gf_row_major"]).T@np.diag([1.,-1.,-1.,1.]), evidence["T_world_camera"], atol=1e-6), sid+": optical axes")
        check(abs(actual[2]-sample["ground_z_m"]-plan["config"]["sensor_height_m"]) < 1e-4, sid+": sensor height")
        for angle, interval in (("pitch", "pitch_deg"), ("roll", "roll_deg"), ("yaw", "yaw_deg")):
            lo, hi = plan["config"][interval]
            check(lo-1e-8 <= sample["angles_deg"][angle] <= hi+1e-8, sid+": "+angle)
        commit = json.loads((root/"commits"/f"{index:06d}.json").read_text())
        check(commit["row"] == row and commit["plan_sha256"] == digest(job/"plan.json"), sid+": committed row")
        check(row["group_id"] == sample["group_id"] and row["source_face_id"] == sample["source_face_id"], sid+": semantic association")
        hashes.append(row["image_sha256"])
        black.append(float(np.mean(np.max(pixels,axis=2)<8)))
        errors.append(error)
        batch_id = row["capture_batch_id"]
        batch_counts[batch_id] = batch_counts.get(batch_id,0)+1
    check(len(set(hashes)) == count, "unique RGB hashes")
    groups, ids = {}, set()
    actual_counts = {}
    for name in ("train", "validation", "test"):
        path = root/"splits"/(name+".csv")
        records = list(csv.DictReader(path.open()))
        actual_counts[name] = len(records)
        for row in records:
            check(row["sample_id"] not in ids, "no split ID overlap:"+row["sample_id"])
            ids.add(row["sample_id"])
            check(row["group_id"] not in groups or groups[row["group_id"]] == name, "no split group leakage")
            groups[row["group_id"]] = name
    check(ids == {r["sample_id"] for r in rows}, "all samples assigned")
    check(actual_counts == split["actual_counts"], "split counts agree")
    for name, expected in split["split_file_sha256"].items():
        check(digest(root/"splits"/name) == expected, "split file SHA:"+name)
    minimum = plan["config"]["min_distance_m"]
    group_distance = plan["config"]["group_distance_m"]
    cell_size = max(minimum, group_distance)
    cells = {}
    for sample in plan["samples"]:
        x,y = sample["pose_xyzw"][:2]
        cell = (math.floor(x/cell_size),math.floor(y/cell_size)) if cell_size else (x,y)
        nearby = [(cell[0]+dx,cell[1]+dy) for dx in (-1,0,1) for dy in (-1,0,1)] if cell_size else [cell]
        for key in nearby:
            for group,positions in cells.get(key,{}).items():
                if minimum == 0 and group == sample["group_id"]:
                    continue
                for a,b in positions:
                    distance = math.hypot(x-a,y-b)
                    check(distance+1e-9 >= minimum, "minimum sample spacing")
                    if distance <= group_distance:
                        check(group == sample["group_id"], "nearby samples share group")
        cells.setdefault(cell,{}).setdefault(sample["group_id"],[]).append((x,y))
    return {"ok":True,"checks":checks,"sample_count":count,"split_counts":actual_counts,
            "intrinsics_K":k.tolist(),"horizontal_fov_deg":meta["camera"]["horizontal_fov_deg"],
            "capture_batch_counts":batch_counts,"max_position_error_m":max(errors),
            "near_black_fraction_mean":sum(black)/len(black),"near_black_fraction_max":max(black),
            "high_black_sample_count":sum(x>.4 for x in black),"camera_mode":meta["camera"]["mode"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=verify(args.job_dir.expanduser().resolve(strict=True))
    text=json.dumps(result,indent=2)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text+"\n")
    print(text)


if __name__=="__main__":
    main()
