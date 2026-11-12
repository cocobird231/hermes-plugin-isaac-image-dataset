"""Validate captured image/pose pairs and split intact position groups."""
import csv
import hashlib
import json
import io
import os
import uuid
import math
from pathlib import Path
import random

from .checkpoints import atomic_bytes, atomic_json, json_bytes, sync_directory, validate_capture

NAMES = ("train", "validation", "test")
FIELDS = ("sample_id", "image_path", "x", "y", "z", "qx", "qy", "qz", "qw", "group_id")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ratios_and_targets(ratios, count):
    if (not isinstance(ratios, (list, tuple)) or len(ratios) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < 0 for x in ratios)):
        raise ValueError("split ratios require three finite nonnegative numbers")
    scale = max(ratios)
    if scale <= 0:
        raise ValueError("split ratio sum must be positive")
    weights = [x / scale for x in ratios]
    total = sum(weights)
    raw = [count * x / total for x in weights]
    targets = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda i: (-(raw[i] - targets[i]), i))[:count-sum(targets)]:
        targets[i] += 1
    return [x/total for x in weights], targets


def assign_groups(samples, ratios, seed):
    """Largest groups first; seeded ties; never split a position group."""
    weights, targets = ratios_and_targets(ratios, len(samples))
    groups = {}
    for row in samples:
        gid = row.get("group_id")
        if not isinstance(gid, str) or not gid:
            raise ValueError("sample group_id must be a nonempty string")
        groups.setdefault(gid, []).append(row)
    positive = [i for i, x in enumerate(weights) if x > 0]
    if len(groups) < len(positive):
        raise ValueError("Too few position groups for nonempty requested splits")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda gid: -len(groups[gid]))  # Stable seeded tie order.
    counts = [0, 0, 0]
    assignment = {}
    for k, gid in enumerate(order):
        empty = [i for i in positive if counts[i] == 0]
        choices = empty if len(order)-k == len(empty) else positive
        selected = max(choices, key=lambda i: (targets[i]-counts[i], -i))
        assignment[gid] = NAMES[selected]
        counts[selected] += len(groups[gid])
    return assignment, {"requested_ratios": list(ratios), "normalized_ratios": weights,
                        "target_counts": dict(zip(NAMES, targets)),
                        "actual_counts": dict(zip(NAMES, counts)),
                        "group_count": len(groups), "seed": seed,
                        "algorithm": "seeded largest-group-first deficit allocation; intact groups",
                        "strict_spatial_block_separation": False}


def verify_nearby_groups(samples, threshold):
    """Check cross-group close edges in neighboring XY cells, skipping same-group bins."""
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("group distance must be finite and nonnegative")
    cells = {}
    for row in samples:
        x, y = row["pose_xyzw"][:2]
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Group positions must be finite")
        cell = (math.floor(x/threshold), math.floor(y/threshold)) if threshold else (x, y)
        neighboring = [(cell[0]+dx, cell[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)] if threshold else [cell]
        for neighbor in neighboring:
            for group, positions in cells.get(neighbor, {}).items():
                if group != row["group_id"]:
                    if any(math.hypot(x-a, y-b) <= threshold for a, b in positions):
                        raise ValueError("Nearby positions belong to different groups")
        cells.setdefault(cell, {}).setdefault(row["group_id"], []).append((x, y))


def publish_splits(root, rows, assignments, summary):
    """Atomic directory is the commit point; receipt publication is safely repeatable."""
    contents = {}
    for name in NAMES:
        chosen = [row for row in rows if assignments[row["group_id"]] == name]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)
        contents[name+".csv"] = stream.getvalue().encode()
        contents[name+".txt"] = "".join(row["image_path"]+"\n" for row in chosen).encode()
    summary = {**summary, "split_file_sha256": {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
    contents["manifest.json"] = json_bytes(summary)
    split_root = root/"splits"
    receipt = root/"split-result.json"
    if receipt.is_symlink() or split_root.is_symlink():
        raise ValueError("Unsafe split publication path")
    if receipt.exists() and (not split_root.is_dir() or json.loads(receipt.read_text()) != summary):
        raise ValueError("Existing split receipt differs from expected split")
    if split_root.exists():
        if not split_root.is_dir() or {item.name for item in split_root.iterdir()} != set(contents):
            raise ValueError("Existing split contents differ from expected split")
        for name, expected in contents.items():
            candidate = split_root/name
            if candidate.is_symlink() or not candidate.is_file() or candidate.read_bytes() != expected:
                raise ValueError("Existing split file differs from expected: " + name)
    else:
        temporary = root/(".splits."+uuid.uuid4().hex+".tmp")
        temporary.mkdir()
        try:
            for name, content in contents.items():
                atomic_bytes(temporary/name, content)
            sync_directory(temporary)
            os.rename(temporary, split_root)
            sync_directory(root)
        finally:
            if temporary.exists():
                for item in temporary.iterdir():
                    item.unlink()
                temporary.rmdir()
    atomic_json(receipt, summary)
    return summary


def finalize_dataset(plan_path, dataset_dir, ratios=None):
    """Publish verified splits atomically; identical calls validate and reuse publication."""
    from PIL import Image
    plan_path = Path(plan_path).resolve(strict=True)
    raw_root = Path(dataset_dir)
    if raw_root.is_symlink():
        raise ValueError("Dataset root must not be a symlink")
    root = raw_root.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    result = json.loads((root/"result.json").read_text())
    if plan.get("complete") is not True or result.get("ok") is not True:
        raise ValueError("Only complete plans and successful captures can be split")
    expected = plan["samples"]
    metadata = json.loads((root/"metadata.json").read_text())
    if (result.get("app_closed") is not True or result.get("source_unchanged") is not True
            or result.get("captured_count") != len(expected)
            or result.get("requested_count") != len(expected)
            or metadata.get("complete") is not True
            or metadata.get("plan_sha256") != sha256(plan_path)
            or metadata.get("scene") != plan["scene"]
            or metadata.get("captured_count") != len(expected)):
        raise ValueError("Capture completion/metadata/plan binding is invalid")
    for path, digest in plan["scene"].get("source_dependencies", {}).items():
        if sha256(path) != (digest["sha256"] if isinstance(digest, dict) else digest):
            raise ValueError("Scene dependency changed after capture")
    if sha256(plan["scene"]["path"]) != plan["scene"]["sha256"]:
        raise ValueError("Scene changed after capture")
    if len(expected) != plan["config"]["n"]:
        raise ValueError("Plan count differs from requested N")
    planned = {x["sample_id"]: x for x in expected}
    if len(planned) != len(expected):
        raise ValueError("Duplicate plan sample IDs")
    rows = [json.loads(line) for line in (root/"samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != len(expected) or len({x["sample_id"] for x in rows}) != len(rows):
        raise ValueError("Captured count or IDs are invalid")
    if {x["sample_id"] for x in rows} != set(planned):
        raise ValueError("Captured IDs differ from plan")
    committed = validate_capture(plan_path, root, require_complete=True)
    if committed != rows:
        raise ValueError("Master JSONL differs from committed captures")
    verify_nearby_groups(expected, plan["config"].get("group_distance_m", 2.0))
    width, height = plan["config"]["camera"]["resolution"]
    image_paths = set()
    image_hashes = set()
    for row in rows:
        item = planned[row["sample_id"]]
        relative = Path(row["image_path"])
        image_path = root/relative
        if relative.is_absolute() or image_path.is_symlink() or not image_path.resolve().is_relative_to(root):
            raise ValueError("Image path escapes dataset")
        if relative in image_paths:
            raise ValueError("Different samples reference same image")
        image_paths.add(relative)
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError("Image SHA mismatch")
        image_hashes.add(row["image_sha256"])
        with Image.open(image_path) as image:
            image.load()
            if image.format != "PNG" or image.size != (width, height) or image.mode != "RGB":
                raise ValueError("Image format/resolution differs")
        camera_relative = Path(row["camera_evidence_path"])
        camera_path = root/camera_relative
        if (camera_relative.is_absolute() or camera_path.is_symlink()
                or not camera_path.resolve().is_relative_to(root)):
            raise ValueError("Camera evidence path escapes dataset")
        if sha256(camera_path) != row["camera_evidence_sha256"]:
            raise ValueError("Camera evidence SHA mismatch")
        camera_evidence = json.loads(camera_path.read_text())
        if (row.get("calibration_passed") is not True
                or camera_evidence.get("sample_id") != row["sample_id"]
                or camera_evidence.get("validation", {}).get("passed") is not True
                or camera_evidence.get("native_nurec_crop_unchanged") is not True
                or camera_evidence.get("camera_readback_after_same_render_call") is not True
                or camera_evidence.get("T_world_camera") != row.get("T_world_camera")
                or camera_evidence["validation"].get("intrinsics_K") != metadata.get("intrinsics_K")):
            raise ValueError("Camera calibration evidence association is invalid")
        pose = row["pose_xyzw"]
        request = item["pose_xyzw"]
        if len(pose) != 7 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose):
            raise ValueError("Actual pose must contain seven finite numbers")
        if abs(sum(x*x for x in pose[3:])-1) > 1e-6:
            raise ValueError("Actual quaternion is not normalized")
        if max(abs(a-b) for a,b in zip(pose[:3], request[:3])) > 1e-4:
            raise ValueError("Actual camera position differs from request")
        if abs(sum(a*b for a,b in zip(pose[3:], request[3:]))) < 1-1e-7:
            raise ValueError("Actual camera orientation differs from request")
        for key in ("group_id", "source_face_id", "region_id"):
            if row[key] != item[key]:
                raise ValueError("Captured sample association differs: "+key)
        for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose):
            if key in row and abs(row[key]-value) > 1e-10:
                raise ValueError("Flattened pose differs from pose_xyzw")
            row[key] = value
    with (root/"samples.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(rows):
        raise ValueError("Master CSV count differs")
    by_id = {row["sample_id"]: row for row in rows}
    if len({row["sample_id"] for row in csv_rows}) != len(rows):
        raise ValueError("Master CSV duplicate IDs")
    for item in csv_rows:
        linked = by_id.get(item["sample_id"])
        if linked is None or item["image_path"] != linked["image_path"]:
            raise ValueError("Master CSV image association differs")
        if any(abs(float(item[key])-linked[key]) > 1e-10 for key in FIELDS[2:9]):
            raise ValueError("Master CSV actual pose differs")
    ratio = plan["config"].get("split_ratios", [4, 2, 2]) if ratios is None else ratios
    assignments, summary = assign_groups(rows, ratio, plan["config"]["seed"])
    summary.update(ok=True, sample_count=len(rows), image_count=len(image_paths),
                   unique_image_hash_count=len(image_hashes),
                   plan_sha256=sha256(plan_path), samples_sha256=sha256(root/"samples.jsonl"),
                   assignments=assignments,
                   all_images_decoded=True, pose_readback_checked=True,
                   group_distance_m=plan["config"].get("group_distance_m", 2.0))
    return publish_splits(root, rows, assignments, summary)
