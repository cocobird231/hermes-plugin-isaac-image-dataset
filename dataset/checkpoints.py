"""Durable capture checkpoints. Committed corruption fails; only uncommitted owned names may be replaced."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid

FIELDS = ["sample_id", "image_path", "image_sha256", "x", "y", "z", "qx", "qy", "qz", "qw",
          "group_id", "source_face_id", "region_id", "ground_z_m", "simulation_time_s",
          "frame_index", "kit_update_number", "camera_evidence_path"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, content):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refuse symlink output: " + str(path))
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(data):
    def serial(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=serial) + "\n").encode()


def atomic_json(path, data):
    atomic_bytes(path, json_bytes(data))


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Required regular file missing or unsafe: " + str(path))
    return path


def prepare_output(plan_path, output):
    """Bind a new directory atomically; an existing directory must have this exact plan."""
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    manifest = {"schema_version": 1, "plan_sha256": sha256(plan_path),
                "requested_count": len(plan["samples"]),
                "recovery_policy": "verify contiguous committed prefix; fail on committed corruption; replace uncommitted owned names"}
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.mkdir()
        try:
            for directory in ("images", "camera", "commits"):
                (temporary / directory).mkdir()
            atomic_json(temporary / "capture-manifest.json", manifest)
            publish_watermark(temporary, manifest["plan_sha256"], 0)
            sync_directory(temporary)
            os.rename(temporary, output)
            sync_directory(output.parent)
        finally:
            if temporary.exists():
                # A failed initialization contains no captured data.
                for item in temporary.iterdir():
                    item.rmdir() if item.is_dir() else item.unlink()
                temporary.rmdir()
    actual = json.loads(regular(output / "capture-manifest.json").read_text())
    if actual != manifest:
        raise ValueError("Existing output immutable plan SHA does not match")
    for directory in ("images", "camera", "commits"):
        child = output / directory
        if child.is_symlink() or not child.is_dir():
            raise ValueError("Unsafe capture directory: " + str(child))
    return manifest



def read_watermark(root, plan_sha256, requested_count):
    path = Path(root)/"capture-watermark.json"
    if not path.exists() and not path.is_symlink():
        return None  # Earlier renderer batches are guarded by their durable receipts below.
    value = json.loads(regular(path).read_text())
    count = value.get("committed_count")
    if (value.get("schema_version") != 1 or value.get("plan_sha256") != plan_sha256
            or type(count) is not int or not 0 <= count <= requested_count
            or (count == 0 and value.get("last_commit_sha256") is not None)):
        raise ValueError("Capture watermark plan SHA/count is invalid")
    return value


def publish_watermark(root, plan_sha256, committed_count):
    """Advance only after the corresponding commit record has been durably published."""
    root = Path(root)
    previous = read_watermark(root, plan_sha256, committed_count)
    if previous is not None and previous["committed_count"] > committed_count:
        raise ValueError("Capture watermark cannot decrease")
    last = sha256(regular(root/"commits"/f"{committed_count-1:06d}.json")) if committed_count else None
    atomic_json(root/"capture-watermark.json", {"schema_version": 1, "plan_sha256": plan_sha256,
                "committed_count": committed_count, "last_commit_sha256": last})


def verify_high_water(root, plan_sha256, requested_count, commits):
    """Independent progress witnesses prevent deleted tail commits becoming recapturable."""
    root = Path(root)
    watermark = read_watermark(root, plan_sha256, requested_count)
    high_water = watermark["committed_count"] if watermark is not None else 0
    # Keep compatibility with a batch that started before watermarks were introduced.
    # These witnesses must be read before the renderer replaces result/progress.
    for name in ("result.json", "progress.json", "metadata.json"):
        path = root/name
        if not path.exists() and not path.is_symlink():
            continue
        value = json.loads(regular(path).read_text())
        if "captured_count" in value:
            count = value["captured_count"]
            if type(count) is not int or not 0 <= count <= requested_count:
                raise ValueError("Invalid durable captured count in " + name)
            high_water = max(high_water, count)
    if len(commits) < high_water:
        raise ValueError("Committed capture prefix regressed below durable high-water count")
    if watermark is not None and watermark["committed_count"]:
        last = commits[watermark["committed_count"]-1]
        if sha256(regular(last)) != watermark.get("last_commit_sha256"):
            raise ValueError("Capture watermark last commit SHA mismatch")

def validate_capture(plan_path, dataset_dir, require_complete=False):
    """Return verified committed rows, independent of possibly stale receipts/indexes."""
    from PIL import Image
    root = Path(dataset_dir)
    if root.is_symlink():
        raise ValueError("Capture root must not be a symlink")
    plan = json.loads(Path(plan_path).read_text())
    manifest = json.loads(regular(root / "capture-manifest.json").read_text())
    digest = sha256(plan_path)
    samples = plan["samples"]
    if manifest.get("plan_sha256") != digest or manifest.get("requested_count") != len(samples):
        raise ValueError("Capture manifest immutable plan SHA/count mismatch")
    for directory in ("images", "camera", "commits"):
        if (root / directory).is_symlink() or not (root / directory).is_dir():
            raise ValueError("Unsafe capture directory: " + directory)
    commits = sorted((root / "commits").glob("*.json"))
    expected_names = [f"{index:06d}.json" for index in range(len(commits))]
    if [item.name for item in commits] != expected_names or len(commits) > len(samples):
        raise ValueError("Committed captures must form a contiguous plan prefix")
    verify_high_water(root, digest, len(samples), commits)
    rows = []
    intrinsics = None
    for index, path in enumerate(commits):
        record = json.loads(regular(path).read_text())
        row = record.get("row", {})
        sample = samples[index]
        sid = sample["sample_id"]
        if (record.get("schema_version") != 1 or record.get("plan_sha256") != digest
                or record.get("index") != index or row.get("frame_index") != index
                or row.get("sample_id") != sid):
            raise ValueError("Committed sample/plan association mismatch at " + str(index))
        for key, directory, suffix in (("image", "images", ".png"), ("camera_evidence", "camera", ".json")):
            relative = directory + "/" + sid + suffix
            if row.get(key + "_path") != relative:
                raise ValueError("Committed path mismatch: " + sid)
            candidate = regular(root / relative)
            if sha256(candidate) != row.get(key + "_sha256"):
                raise ValueError("Committed " + key + " SHA mismatch: " + sid)
        with Image.open(root / row["image_path"]) as image:
            image.load()
            if (image.format != "PNG" or image.mode != "RGB"
                    or image.size != tuple(plan["config"]["camera"]["resolution"])):
                raise ValueError("Committed PNG format/resolution mismatch: " + sid)
        evidence = json.loads((root / row["camera_evidence_path"]).read_text())
        validation = evidence.get("validation", {})
        if (row.get("calibration_passed") is not True or evidence.get("sample_id") != sid
                or validation.get("passed") is not True
                or evidence.get("native_nurec_crop_unchanged") is not True
                or evidence.get("camera_readback_after_same_render_call") is not True
                or evidence.get("lossless_png_readback") is not True
                or evidence.get("T_world_camera") != row.get("T_world_camera")
                or row.get("requested_pose_xyzw") != sample["pose_xyzw"]):
            raise ValueError("Committed camera evidence association mismatch: " + sid)
        measured = row.get("pose_xyzw", [])
        requested = sample["pose_xyzw"]
        if (len(measured) != 7 or any(type(x) not in (int, float) or not math.isfinite(x) for x in measured)
                or abs(sum(x*x for x in measured[3:])-1) > 1e-6
                or max(abs(a-b) for a, b in zip(measured[:3], requested[:3])) > 1e-4
                or abs(sum(a*b for a, b in zip(measured[3:], requested[3:]))) < 1-1e-7):
            raise ValueError("Committed actual pose differs from plan: " + sid)
        for key in ("group_id", "source_face_id", "region_id", "ground_z_m", "angles_deg"):
            if row.get(key) != sample[key]:
                raise ValueError("Committed sample association mismatch: " + key)
        for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), measured):
            if row.get(key) != value:
                raise ValueError("Committed flattened pose mismatch: " + key)
        current = validation.get("intrinsics_K")
        if not isinstance(current, list) or len(current) != 3 or (intrinsics is not None and current != intrinsics):
            raise ValueError("Committed camera intrinsics mismatch")
        intrinsics = current
        rows.append(row)
    if require_complete and len(rows) != len(samples):
        raise ValueError("Capture commits are incomplete")
    return rows


def commit_sample(root, plan_sha256, index, row):
    """Called only after durable PNG plus durable validated camera evidence."""
    root = Path(root)
    path = root / "commits" / f"{index:06d}.json"
    if path.exists() or path.is_symlink():
        raise ValueError("Refuse overwrite of committed capture")
    atomic_json(path, {"schema_version": 1, "plan_sha256": plan_sha256, "index": index, "row": row})
    publish_watermark(root, plan_sha256, index+1)


def publish_indexes(root, rows):
    """Derived indexes may be rebuilt after interruption; commits are authoritative."""
    root = Path(root)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(root / "samples.csv", text.getvalue().encode())
    atomic_bytes(root / "samples.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode())
