#!/usr/bin/env python3
"""Capture full NuRec RGB and measured optical poses in an isolated Isaac process.

The supervisor owns the shared GPU lock and the systemd cgroup. This worker never
saves a USD layer. A result is complete only after SimulationApp.close returns.
"""
import argparse
import hashlib
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

import numpy as np

# Support package imports, direct script execution, and importlib contract tests.
if __package__:
    from . import checkpoints, quality
else:
    spec = importlib.util.spec_from_file_location("dataset_capture_checkpoints", Path(__file__).with_name("checkpoints.py"))
    checkpoints = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoints)
    spec = importlib.util.spec_from_file_location("dataset_render_quality", Path(__file__).with_name("quality.py"))
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)


AXES = np.diag([1., -1., -1., 1.])
CONVENTION = {
    "world": "right-handed Z-up metres",
    "camera": "optical: X right, Y down, Z forward",
    "pose": "camera-to-world; quaternion qx,qy,qz,qw; normalized with canonical sign",
    "matrix": "T_world_camera: row-major serialization of column-vector transform",
    "usd_camera": "X right, Y up, -Z forward; T_world_usd = T_world_camera @ diag(1,-1,-1,1)",
    "pixel": "continuous image coordinates, u right, v down, principal point width/2,height/2",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_stat(path):
    s = Path(path).stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}


def serial(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False, default=serial)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(float(np.linalg.norm(q))-1) > 1e-6:
        raise ValueError("Quaternion must be finite, normalized qx,qy,qz,qw")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quaternion(rotation):
    """Stable rotation-matrix conversion; canonical first nonzero of w,x,y,z positive."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Expected finite 3x3 rotation")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError("Rotation is not orthonormal/right-handed")
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace+1)*2
        q = np.array([(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s, s/4])
    else:
        k = int(np.argmax(np.diag(r)))
        if k == 0:
            s = math.sqrt(1+r[0, 0]-r[1, 1]-r[2, 2])*2
            q = np.array([s/4, (r[0, 1]+r[1, 0])/s, (r[0, 2]+r[2, 0])/s, (r[2, 1]-r[1, 2])/s])
        elif k == 1:
            s = math.sqrt(1+r[1, 1]-r[0, 0]-r[2, 2])*2
            q = np.array([(r[0, 1]+r[1, 0])/s, s/4, (r[1, 2]+r[2, 1])/s, (r[0, 2]-r[2, 0])/s])
        else:
            s = math.sqrt(1+r[2, 2]-r[0, 0]-r[1, 1])*2
            q = np.array([(r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s, s/4, (r[1, 0]-r[0, 1])/s])
    q /= np.linalg.norm(q)
    for v in [q[3], q[0], q[1], q[2]]:
        if abs(v) > 1e-14:
            if v < 0:
                q = -q
            break
    return q


def optical_to_usd(sample):
    pose = np.asarray(sample["pose_xyzw"], dtype=float)
    optical = np.asarray(sample["T_world_camera"], dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or optical.shape != (4, 4) or not np.isfinite(optical).all():
        raise ValueError("Sample requires finite pose7 and 4x4 T_world_camera")
    if not np.allclose(optical[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("Invalid transform homogeneous row")
    if not np.allclose(quaternion_matrix(pose[3:]), optical[:3, :3], atol=1e-6, rtol=0):
        raise ValueError("Pose quaternion and T_world_camera rotation disagree")
    if not np.allclose(pose[:3], optical[:3, 3], atol=1e-7, rtol=0):
        raise ValueError("Pose translation and T_world_camera disagree")
    usd = optical @ AXES
    return [*usd[:3, 3], *matrix_quaternion(usd[:3, :3])], usd.T


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("complete") is not True:
        raise ValueError("Only complete schema_version 1 sampling plans are accepted")
    samples = plan.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 10000:
        raise ValueError("Plan requires 1..10000 samples")
    ids = set()
    for sample in samples:
        sid = sample.get("sample_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid) or sid in ids:
            raise ValueError("Sample IDs must be unique safe filenames")
        ids.add(sid)
        optical_to_usd(sample)
        for key in ("group_id", "source_face_id", "region_id", "ground_z_m", "angles_deg"):
            if key not in sample:
                raise ValueError("Sample missing "+key)
    scene = plan.get("scene", {})
    if not isinstance(scene.get("path"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(scene.get("sha256", ""))):
        raise ValueError("Plan requires scene path and SHA256")
    quality.render_policy(plan.get("config", {}))
    count = plan.get("config", {}).get("camera_count", 1)
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("camera_count must be integer within 1..8")
    camera = plan.get("config", {}).get("camera", {})
    resolution = camera.get("resolution", [1280, 720])
    if (not isinstance(resolution, list) or len(resolution) != 2
            or any(type(x) is not int or x < 16 or x > 4096 for x in resolution)):
        raise ValueError("Resolution must contain integer width,height in 16..4096")
    fov = camera.get("horizontal_fov_deg", 90)
    if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(fov) or not 1 < fov < 179:
        raise ValueError("Pinhole horizontal FOV must be finite between 1 and 179 degrees")
    clipping = np.asarray(camera.get("clipping_range", [.1, 2000]), dtype=float)
    if clipping.shape != (2,) or not np.isfinite(clipping).all() or not 0 < clipping[0] < clipping[1]:
        raise ValueError("Positive increasing clipping range required")
    render = plan.get("config", {}).get("render", {})
    for key, default, lo, hi in (("warmup_steps", 800, 8, 4000), ("settle_updates", 60, 0, 1000),
                                ("discard_frames", 2, 1, 10), ("timeout_seconds", 1800, 30, 7200)):
        value = render.get(key, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{key} must be integer within {lo}..{hi}")
    return plan


def effective_cgroup_limits():
    entry = next((x.split("::", 1)[1] for x in Path("/proc/self/cgroup").read_text().splitlines()
                  if x.startswith("0::")), None)
    if entry is None:
        raise RuntimeError("Unified cgroup v2 is required")
    root = Path("/sys/fs/cgroup")
    current = root/entry.lstrip("/")
    rows = []
    for path in [current, *current.parents]:
        if not path.is_relative_to(root):
            break
        row = {"path": str(path)}
        for key in ("memory.max", "memory.swap.max"):
            candidate = path/key
            if candidate.is_file():
                row[key] = candidate.read_text().strip()
        rows.append(row)
    def minimum(key):
        values = [int(row[key]) for row in rows if row.get(key, "max") != "max"]
        return min(values) if values else None
    if minimum("memory.max") != 10*1024**3 or minimum("memory.swap.max") != 0:
        raise RuntimeError("Require effective whole-job MemoryMax=10GiB and MemorySwapMax=0")
    return {"memory_max": minimum("memory.max"), "memory_swap_max": minimum("memory.swap.max"),
            "path": str(current), "ancestors": rows, "gpu_lock_owned_by_supervisor": True}


def validate_readback(params, requested_usd_row, actual_usd_row, camera):
    """Compare actual annotation with USD and project known optical rays both ways."""
    def check(condition, message):
        if not condition:
            raise ValueError("Camera readback rejected: "+message)
    def arr(key, shape):
        a = np.asarray(params.get(key), dtype=float)
        check(a.size == int(np.prod(shape)) and np.isfinite(a).all(), key)
        return a.reshape(shape)
    w, h = camera["resolution"]
    focal = camera["focal_length_mm"]
    aperture = np.asarray(camera["aperture_mm"])
    check(params.get("cameraModel") == "pinhole", "not pinhole")
    check(np.array_equal(arr("renderProductResolution", (2,)), [w, h]), "resolution mismatch")
    check(abs(float(params.get("metersPerSceneUnit", 0))-1) < 1e-6, "stage scale")
    check(abs(float(params.get("cameraFocalLength", 0))-focal) < 1e-5, "focal mismatch")
    check(np.allclose(arr("cameraAperture", (2,)), aperture, atol=1e-5, rtol=0), "aperture mismatch")
    check(np.allclose(arr("cameraApertureOffset", (2,)), 0, atol=1e-6, rtol=0), "aperture offset")
    check(np.allclose(arr("cameraNearFar", (2,)), camera["clipping_range"], atol=1e-4, rtol=1e-6), "clipping mismatch")
    check(np.allclose(actual_usd_row, requested_usd_row, atol=1e-6, rtol=0), "USD pose mismatch")
    view = arr("cameraViewTransform", (4, 4))
    check(np.allclose(view, np.linalg.inv(requested_usd_row), atol=2e-4, rtol=0), "annotator stale view")
    projection = arr("cameraProjection", (4, 4))
    check(np.linalg.matrix_rank(projection) == 4, "projection rank")
    check(np.allclose([projection[0, 0], projection[1, 1]],
                     2*focal/aperture, atol=1e-6, rtol=1e-5), "projection focal scaling")
    K = np.array([[w*focal/aperture[0], 0., w/2],
                  [0., h*focal/aperture[1], h/2], [0., 0., 1.]])
    optical = actual_usd_row.T @ AXES
    rays = np.array([[0, 0, 5, 1], [1, 0, 5, 1], [0, 1, 5, 1], [-1, -1, 5, 1]], dtype=float)
    world = rays @ optical.T
    clip = world @ view @ projection
    ndc = clip[:, :2]/clip[:, 3, None]
    projected = np.column_stack(((ndc[:, 0]+1)*w/2, (1-ndc[:, 1])*h/2))
    intrinsic = rays[:, :3] @ K.T
    intrinsic = intrinsic[:, :2]/intrinsic[:, 2, None]
    error = float(np.abs(projected-intrinsic).max())
    check(error < .05, "optical/world/GPU projection disagreement")
    return {"passed": True, "intrinsics_K": K.tolist(), "intrinsics_pixel": {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": w/2, "cy": h/2},
        "known_optical_ray_projection_max_error_px": error,
        "pose_usd_tolerance": 1e-6, "annotator_view_tolerance": 2e-4,
        "projection_test": "four analytic rays projected with measured camera_params versus K"}


def configured_camera(stage, config):
    """Resolve a centered undistorted pinhole; unsupported templates fail explicitly."""
    from pxr import UsdGeom
    w, h = config.get("resolution", [1280, 720])
    focal = 18.
    horizontal = 2*focal*math.tan(math.radians(config.get("horizontal_fov_deg", 90))/2)
    values = {"mode": "configured_pinhole", "template_prim": None, "resolution": [w, h],
              "focal_length_mm": focal, "aperture_mm": [horizontal, horizontal*h/w],
              "clipping_range": config.get("clipping_range", [.1, 2000])}
    template = config.get("template_prim")
    if template:
        prim = stage.GetPrimAtPath(template)
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise ValueError("template_prim must be an existing USD Camera")
        original = UsdGeom.Camera(prim)
        if original.GetProjectionAttr().Get() != UsdGeom.Tokens.perspective:
            raise ValueError("Only perspective pinhole templates are supported")
        applied = list(prim.GetAppliedSchemas())
        if any("lensdistortion" in s.lower() or "fisheye" in s.lower() for s in applied):
            raise ValueError("Distortion schema template unsupported")
        for attr in prim.GetAttributes():
            name = attr.GetName().lower()
            if any(key in name for key in ("fisheye", "distortion", "ftheta")) and attr.HasAuthoredValue():
                raise ValueError("Authored camera distortion template unsupported: "+attr.GetName())
            if name.endswith("cameraprojectiontype") and attr.HasAuthoredValue():
                if str(attr.Get()).lower() not in ("pinhole", "perspective"):
                    raise ValueError("Non-pinhole projection template unsupported")
        if abs(original.GetHorizontalApertureOffsetAttr().Get()) > 1e-7 or abs(original.GetVerticalApertureOffsetAttr().Get()) > 1e-7:
            raise ValueError("This renderer requires centered principal point; offset template unsupported")
        values.update(mode="existing_pinhole_template", template_prim=template,
                      focal_length_mm=float(original.GetFocalLengthAttr().Get()),
                      aperture_mm=[float(original.GetHorizontalApertureAttr().Get()),
                                   float(original.GetVerticalApertureAttr().Get())])
        if "clipping_range" not in config:
            values["clipping_range"] = list(original.GetClippingRangeAttr().Get())
    clipping = np.asarray(values["clipping_range"], dtype=float)
    if clipping.shape != (2,) or not np.isfinite(clipping).all() or not 0 < clipping[0] < clipping[1]:
        raise ValueError("Camera clipping range must be positive finite and increasing")
    scalars = [values["focal_length_mm"], *values["aperture_mm"]]
    if not np.isfinite(scalars).all() or min(scalars) <= 0:
        raise ValueError("Camera focal/aperture must be positive finite")
    if not np.isclose(values["aperture_mm"][0]/values["aperture_mm"][1], w/h, rtol=1e-5):
        raise ValueError("Template aperture aspect must match requested render resolution")
    values["horizontal_fov_deg"] = math.degrees(2*math.atan(values["aperture_mm"][0]/(2*values["focal_length_mm"])))
    return values


def run(plan, plan_path, output, batch_size=20, timeout_seconds=None):
    config = plan.get("config", {})
    render = {"warmup_steps": 800, "settle_updates": 60, "discard_frames": 2,
              "timeout_seconds": 1800, **config.get("render", {})}
    if timeout_seconds is not None:
        render["timeout_seconds"] = timeout_seconds
    policy = quality.render_policy(config)
    pooled = policy["backend"] == "shared_render_products"
    if pooled:
        render = {**policy, "timeout_seconds": render["timeout_seconds"]}
    source = Path(plan["scene"]["path"]).expanduser().resolve(strict=True)
    rows = checkpoints.validate_capture(plan_path, output)
    resumed_count = len(rows)
    batch_end = min(resumed_count + batch_size, len(plan["samples"]))
    batch_id = uuid.uuid4().hex
    started = time.monotonic()
    deadline = started+render["timeout_seconds"]
    app = cap = annotator = None
    # A fully committed retry still verifies the scene and finalizes shutdown.
    pool_size = max(1, min(config.get("camera_count", 1), batch_end-resumed_count))
    phase_events = []
    capture_metrics = {"groups": [], "saves": []}
    dependencies = {}
    metadata = {"schema_version": 1, "complete": False, "coordinate_convention": CONVENTION,
                "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
                "scene": plan["scene"], "requested_count": len(plan["samples"]),
                "render": render, "capture_mode": "full native NuRec; no height cutaway",
                "sampling_config": config}
    result = {"ok": False, "error": None, "captured_count": 0, "requested_count": len(plan["samples"]),
              "pid": os.getpid(), "source_unchanged": False, "app_closed": False,
              "batch_complete": False, "batch_id": batch_id, "batch_size": batch_size,
              "resumed_count": resumed_count, "batch_target_count": batch_end,
              "batch_captured_count": 0,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "exit_policy": "explicit process exit after app.close, durable receipt and stdio flush",
              "supervisor_must_verify_process_exit": True}
    result["captured_count"] = resumed_count
    metadata["capture_manifest"] = "capture-manifest.json"
    metadata["recovery_policy"] = "verified committed prefix reused; committed corruption fails"
    if rows:
        previous = json.loads((output/"camera"/(rows[0]["sample_id"]+".json")).read_text())
        metadata["intrinsics_K"] = previous["validation"]["intrinsics_K"]
    # A killed/rejected invocation cannot leave a previous successful receipt current.
    atomic_json(output/"result.json", result)
    atomic_json(output/"progress.json", result)
    timeline = None
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Dataset renderer internal timeout")
    def progress(phase):
        check_time()
        result["phase"] = phase
        phase_events.append({"phase": phase, "elapsed_seconds": time.monotonic()-started})
        atomic_json(output/"progress.json", result)
        print("DATASET_RENDER_PHASE "+phase, flush=True)
    def source_stats_unchanged():
        return all(file_stat(path) == row["stat"] for path, row in dependencies.items())
    def source_hashes_unchanged():
        return all(file_stat(path) == row["stat"] and sha256(path) == row["sha256"]
                   for path, row in dependencies.items())
    def protect(path, expected):
        path = str(Path(path).expanduser().resolve(strict=True))
        actual = sha256(path)
        if actual != expected:
            raise ValueError("Source hash mismatch: "+path)
        dependencies[path] = {"stat": file_stat(path), "sha256": actual}
    try:
        protect(source, plan["scene"]["sha256"])
        declared = plan["scene"].get("source_dependencies", {})
        if not isinstance(declared, dict):
            raise ValueError("source_dependencies must be path-to-sha256 mapping")
        for path, digest in declared.items():
            protect(path, digest["sha256"] if isinstance(digest, dict) else digest)
        result["cgroup"] = effective_cgroup_limits()
        progress("starting_simulation_app")
        from isaacsim import SimulationApp
        package = importlib.util.find_spec("isaacsim")
        experience = next((Path(p)/"apps/isaacsim.exp.base.python.kit"
                           for p in package.submodule_search_locations or ()
                           if (Path(p)/"apps/isaacsim.exp.base.python.kit").is_file()), None)
        if experience is None:
            raise RuntimeError("Isaac base Python experience not found")
        width, height = config.get("camera", {}).get("resolution", [1280, 720])
        launch = {"headless": True, "hide_ui": True, "create_new_stage": False,
                  "disable_viewport_updates": True, "multi_gpu": False,
                  "width": width, "height": height, "renderer": "RealTimePathTracing",
                  "sync_loads": True, "fast_shutdown": False,
                  "extra_args": ["--enable", "isaacsim.replicator.nurec_utils",
                                 "--enable", "omni.rtx.spg", "--/renderer/multiGpu/enabled=false",
                                 "--/app/settings/persistent=false",
                                 "--/persistent/omni/replicator/captureOnPlay=false"]}
        metadata.update(launch_config=launch, experience=str(experience))
        app = SimulationApp(launch, experience=str(experience))
        import omni.usd
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom
        from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
        from isaacsim.replicator.nurec_utils.render import RenderTargetFactory, CameraRenderer
        metadata["isaac_version"] = (experience.parent.parent/"VERSION").read_text().strip()
        progress("opening_semantic_scene")
        context = omni.usd.get_context()
        if not context.open_stage(str(source)):
            raise RuntimeError("Isaac open_stage failed")
        stage = context.get_stage()
        if not stage or stage.GetCompositionErrors():
            raise ValueError("Source stage has composition errors")
        if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise ValueError("Only Z-up metre stages are supported")
        for layer in stage.GetUsedLayers():
            if layer.anonymous:
                continue
            path = Path(layer.realPath.split("[", 1)[0]).resolve()
            if path.is_file() and str(path) not in dependencies:
                raise ValueError("Plan omitted composed source dependency: "+str(path))
        volumes = [p for p in stage.Traverse() if p.GetTypeName() == "Volume"
                   and p.GetAttribute("omni:nurec:crop:maxBounds")]
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if len(volumes) != 1 or len(meshes) != 1:
            raise ValueError("Only a single NuRec Volume plus its one collision mesh is supported")
        volume, mesh = volumes[0], meshes[0]
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Gprim) and prim not in (mesh, volume):
                if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
                    raise ValueError("Additional visible geometry requires a mixed-scene renderer")
        for prim in (volume, mesh):
            if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
                raise ValueError("Animated source geometry unsupported")
        crop_attributes = ("omni:nurec:crop:minBounds", "omni:nurec:crop:maxBounds")
        native_crop = {key: list(volume.GetAttribute(key).Get()) for key in crop_attributes}
        def crop_unchanged():
            return all(list(volume.GetAttribute(key).Get()) == value for key, value in native_crop.items())
        camera_config = configured_camera(stage, config.get("camera", {}))
        metadata["camera"] = camera_config
        session = stage.GetSessionLayer()
        if not session.anonymous:
            raise ValueError("Anonymous session layer required")
        stage.SetEditTarget(Usd.EditTarget(session))
        mesh.SetActive(False)
        metadata["session_edits"] = {"layer": session.identifier,
                                     "deactivated_collision_mesh": str(mesh.GetPath()), "source_saved": False}
        metadata["native_nurec_crop"] = {"volume_prim": str(volume.GetPath()), "attributes": native_crop,
                                         "modified": False}
        setup = setup_for_rendering(stage)
        if not setup[0] or not setup[1] or setup[2]:
            raise ValueError("Only ordinary non-SPG NuRec camera rendering is supported")
        metadata["nurec_setup"] = setup
        rep.orchestrator.set_capture_on_play(False)
        camera_paths = ["/__ImageLocalizationCapture_"+uuid.uuid4().hex for _ in range(pool_size)]
        for camera_path in camera_paths:
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            camera.CreateFocalLengthAttr(camera_config["focal_length_mm"])
            camera.CreateHorizontalApertureAttr(camera_config["aperture_mm"][0])
            camera.CreateVerticalApertureAttr(camera_config["aperture_mm"][1])
            camera.CreateHorizontalApertureOffsetAttr(0.)
            camera.CreateVerticalApertureOffsetAttr(0.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(*camera_config["clipping_range"]))
            camera.CreateFStopAttr(0.)
        camera_path = camera_paths[0]
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        metadata["camera"].update(prim_path=camera_path, prim_paths=camera_paths,
                                  camera_count=config.get("camera_count", 1), pool_size=pool_size)
        for _ in range(3):
            check_time()
            app.update()
        while context.get_stage_loading_status()[2] > 0:
            check_time()
            app.update()
        progress("playing_simulation_then_freezing_static_scene")
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(True)
        timeline.set_current_time(0.)
        before_time = timeline.get_current_time()
        timeline.play()
        for _ in range(30):
            check_time()
            app.update()
        playing = timeline.is_playing()
        played_time = timeline.get_current_time()
        timeline.pause()
        timeline.set_auto_update(False)
        snapshot_time = timeline.get_current_time()
        if not playing or played_time <= before_time or snapshot_time <= 0:
            raise RuntimeError("Simulation timeline did not advance while playing")
        snapshot_code = snapshot_time*stage.GetTimeCodesPerSecond()
        metadata["simulation"] = {"play_verified": playing, "before_time_s": before_time,
                                  "after_play_time_s": played_time, "snapshot_time_s": snapshot_time,
                                  "snapshot_usd_time_code": snapshot_code, "paused_during_capture": True,
                                  "purpose": "static scene poses; no moving actor control"}
        progress("creating_render_products")
        if pooled:
            spec = importlib.util.spec_from_file_location("dataset_multicam", Path(__file__).with_name("multicam.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cap = module.BatchCameraRenderer(stage, app, camera_paths, (width, height), check_time,
                                             post_step_updates=policy["shared_updates"])
        else:
            target = RenderTargetFactory(False, resolution=(width, height)).create(stage, "localization", camera_path=camera_path)
            annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
            annotator.attach(target.attach_target)
            cap = CameraRenderer.open(stage, "image_localization", app, target,
                                      warmup_steps=render["warmup_steps"], force_identity_exposure=True)
        metadata["source_dependencies"] = dependencies
        atomic_json(output/"metadata.json", metadata)
        def captures():
            capture_call = 0
            if pooled:
                for group_index, start_index in enumerate(range(resumed_count, batch_end, pool_size)):
                    samples = plan["samples"][start_index:min(start_index+pool_size, batch_end)]
                    poses = [optical_to_usd(sample) for sample in samples]
                    progress("capturing_group_"+str(group_index))
                    valid_frames = None
                    for trial in range(1, 4):
                        frames, timing = cap.capture([pose for pose, _ in poses], snapshot_code)
                        timing_record = {"render_group_index": group_index, "group_size": len(samples),
                                         "trial": trial, **timing}
                        capture_metrics["groups"].append(timing_record)
                        try:
                            if len(frames) != len(samples):
                                raise ValueError("Wrong number of camera readbacks")
                            validations = []
                            for frame, (_, expected) in zip(frames, poses):
                                rgb = frame["rgb"]
                                if rgb is None or rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or float(rgb.std()) <= 2:
                                    raise ValueError("Missing, wrong size/type or featureless RGB")
                                validations.append(validate_readback(frame["params"], expected, frame["actual_row"], camera_config))
                            valid_frames = (frames, validations)
                            break
                        except (ValueError, TypeError) as exc:
                            with (output/"capture-attempts.jsonl").open("a") as attempts:
                                attempts.write(json.dumps({"sample_ids": [x["sample_id"] for x in samples],
                                                          "trial": trial, "error": str(exc)})+"\n")
                    if valid_frames is None:
                        raise RuntimeError("Camera group calibration or RGB did not converge")
                    frames, validations = valid_frames
                    # Validate every camera before publishing any image from this group.
                    for slot, (sample, frame, (_, expected), validation) in enumerate(zip(samples, frames, poses, validations)):
                        details = {"camera_slot": slot, "camera_prim_path": camera_paths[slot],
                                   "pool_size": pool_size, "render_group_index": group_index,
                                   "group_size": len(samples), "group_timing_seconds": timing,
                                   "capture_batch_id": batch_id, "render_quality": policy["render_quality"],
                                   "post_step_updates": policy["shared_updates"]}
                        yield (start_index+slot, sample, frame["rgb"], frame["params"], frame["actual_row"],
                               expected, validation, cap.render_calls, details)
                return
            for index in range(resumed_count, batch_end):
                sample = plan["samples"][index]
                sid = sample["sample_id"]
                progress("capturing_"+sid)
                usd_pose, expected_row = optical_to_usd(sample)
                check_time()
                cap.render_at_pose(usd_pose, time_code=snapshot_code)
                capture_call += 1
                for _ in range(render["settle_updates"]):
                    check_time()
                    app.update()
                for _ in range(render["discard_frames"]):
                    check_time()
                    cap.render_at_pose(usd_pose, time_code=snapshot_code)
                    capture_call += 1
                validation = None
                for trial in range(1, 4):
                    check_time()
                    rgb = cap.render_at_pose(usd_pose, time_code=snapshot_code)
                    capture_call += 1
                    params = annotator.get_data()
                    world = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode(snapshot_code))
                    actual_row = np.asarray(world, dtype=float)
                    try:
                        if rgb is None or rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or float(rgb.std()) <= 2:
                            raise ValueError("Missing, wrong size/type or featureless RGB")
                        validation = validate_readback(params, expected_row, actual_row, camera_config)
                    except (ValueError, TypeError) as exc:
                        with (output/"capture-attempts.jsonl").open("a") as attempts:
                            attempts.write(json.dumps({"sample_id": sid, "trial": trial, "error": str(exc)})+"\n")
                        validation = None
                    if validation is not None:
                        break
                if validation is None:
                    raise RuntimeError("Camera calibration or RGB did not converge for "+sid)
                yield index, sample, rgb, params, actual_row, expected_row, validation, capture_call, {}
        for index, sample, rgb, params, actual_row, expected_row, validation, capture_call, details in captures():
            sid = sample["sample_id"]
            save_started = time.monotonic()
            if not crop_unchanged() or not source_stats_unchanged():
                raise RuntimeError("Native volume crop or protected source changed")
            simulation_time = timeline.get_current_time()
            if abs(simulation_time-snapshot_time) > 1e-6:
                raise RuntimeError("Snapshot timeline changed during capture")
            actual_optical = actual_row.T @ AXES
            actual_pose = [*actual_optical[:3, 3], *matrix_quaternion(actual_optical[:3, :3])]
            image_path = output/"images"/(sid+".png")
            if image_path.is_symlink():
                raise ValueError("Unsafe uncommitted image path")
            temporary_image = image_path.with_name(image_path.name+"."+uuid.uuid4().hex+".tmp")
            try:
                with temporary_image.open("xb") as stream:
                    Image.fromarray(rgb).save(stream, format="PNG")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_image, image_path)
                checkpoints.sync_directory(image_path.parent)
            finally:
                temporary_image.unlink(missing_ok=True)
            with Image.open(image_path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (width, height) or not np.array_equal(np.asarray(image), rgb):
                    raise RuntimeError("PNG lossless readback failed")
            kit_app = omni.kit.app.get_app()
            update_number = int(kit_app.get_update_number())
            evidence_path = output/"camera"/(sid+".json")
            evidence = {"sample_id": sid, "camera_params_annotator": params,
                        "camera_to_world_usd_gf_row_major": actual_row,
                        "requested_camera_to_world_usd_gf_row_major": expected_row,
                        "T_world_camera": actual_optical, "validation": validation,
                        "pose_fixed_for_all_render_updates": True,
                        "camera_readback_after_same_render_call": True,
                        "render_call_index": capture_call, "kit_update_number": update_number,
                        "simulation_time_s": simulation_time,
                        "rgb_min": int(rgb.min()), "rgb_max": int(rgb.max()), "rgb_std": float(rgb.std()),
                        "native_nurec_crop_unchanged": True, "lossless_png_readback": True}
            evidence.update(details)
            atomic_json(evidence_path, evidence)
            row = {"sample_id": sid, "image_path": image_path.relative_to(output).as_posix(),
                   "image_sha256": sha256(image_path), "pose_xyzw": actual_pose,
                   "requested_pose_xyzw": sample["pose_xyzw"], "T_world_camera": actual_optical.tolist(),
                   "group_id": sample["group_id"], "source_face_id": sample["source_face_id"],
                   "region_id": sample["region_id"], "ground_z_m": sample["ground_z_m"],
                   "angles_deg": sample["angles_deg"], "simulation_time_s": simulation_time,
                   "usd_time_code": snapshot_code, "frame_index": index, "kit_update_number": update_number,
                   "render_call_index": capture_call, "camera_evidence_path": evidence_path.relative_to(output).as_posix(),
                   "camera_evidence_sha256": sha256(evidence_path), "calibration_passed": True,
                   "capture_batch_id": batch_id}
            row.update({k: v for k, v in details.items() if k != "group_timing_seconds"})
            row.update(dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), actual_pose)))
            # The record becomes visible only after both payloads are durable.
            checkpoints.commit_sample(output, metadata["plan_sha256"], index, row)
            rows.append(json.loads(json.dumps(row, default=serial)))
            result["captured_count"] += 1
            result["batch_captured_count"] += 1
            result["last_sample_id"] = sid
            metadata["intrinsics_K"] = validation["intrinsics_K"]
            metadata["camera"]["intrinsics_K"] = validation["intrinsics_K"]
            metadata["camera"]["intrinsics_pixel"] = validation["intrinsics_pixel"]
            metadata["camera"]["readback_verified"] = True
            atomic_json(output/"metadata.json", metadata)
            capture_metrics["saves"].append({"sample_id": sid, "seconds": time.monotonic()-save_started})
            progress("captured_"+sid)
        checkpoints.publish_indexes(output, rows)
        result["batch_complete"] = result["captured_count"] == batch_end
        result["ok"] = result["captured_count"] == len(plan["samples"])
        if rows:
            first_evidence = json.loads((output/rows[0]["camera_evidence_path"]).read_text())
            calibration = first_evidence["validation"]
            metadata["intrinsics_K"] = calibration["intrinsics_K"]
            metadata["camera"].update(intrinsics_K=calibration["intrinsics_K"],
                                      intrinsics_pixel=calibration["intrinsics_pixel"], readback_verified=True)
        result["native_nurec_crop_unchanged"] = crop_unchanged()
        result["phase"] = "capture_complete_pending_shutdown"
    except BaseException as exc:
        result["ok"] = result["batch_complete"] = False
        result["error"] = type(exc).__name__+": "+str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        result["batch_complete"] = bool(result["batch_complete"] and result.get("native_nurec_crop_unchanged"))
        result["ok"] = bool(result["ok"] and result["batch_complete"])
        for name, resource, close in (("annotator", annotator, "detach"), ("renderer", cap, "close")):
            if resource is not None:
                try:
                    getattr(resource, close)()
                except Exception as exc:
                    result.setdefault("cleanup_errors", []).append(name+": "+str(exc))
                    result["ok"] = result["batch_complete"] = False
        if timeline is not None:
            try:
                timeline.stop()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append("timeline: "+str(exc))
                result["ok"] = result["batch_complete"] = False
        # Keep the durable receipt incomplete if native shutdown aborts the interpreter.
        pending = {**result, "ok": False, "batch_complete": False, "phase": "closing_simulation_app"}
        atomic_json(output/"result.json", pending)
        if app is not None:
            try:
                app.close()
                result["app_closed"] = True
            except Exception as exc:
                result["shutdown_error"] = str(exc)
                result["ok"] = result["batch_complete"] = False
        try:
            result["source_unchanged"] = bool(dependencies) and source_hashes_unchanged()
            metadata["source_dependencies"] = dependencies
        except Exception as exc:
            result["source_check_error"] = str(exc)
        result["batch_complete"] = bool(result["batch_complete"] and result["source_unchanged"] and result["app_closed"])
        result["ok"] = bool(result["ok"] and result["batch_complete"])
        result["phase"] = "complete" if result["ok"] else "batch_complete" if result["batch_complete"] else "failed"
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["elapsed_seconds"] = time.monotonic()-started
        result["phase_events"] = phase_events
        result["capture_metrics"] = capture_metrics
        result["initialization_render_calls"] = getattr(cap, "init_render_calls", None)
        result["capture_render_calls"] = getattr(cap, "render_calls", None)
        result["render_quality"] = policy["render_quality"]
        result["render_quality_source"] = policy["render_quality_source"]
        result["post_step_updates"] = policy["shared_updates"]
        result["capture_post_step_update_calls"] = getattr(cap, "post_step_update_calls", None)
        result["camera_count"] = config.get("camera_count", 1)
        result["pool_size"] = pool_size
        result["source_hashes_after"] = {path: row["sha256"] for path, row in dependencies.items()} if result["source_unchanged"] else {}
        metadata["complete"] = result["ok"]
        metadata["captured_count"] = result["captured_count"]
        metadata["frame_index_definition"] = "zero-based accepted sample index; kit_update_number is actual Kit update count"
        atomic_json(output/"metadata.json", metadata)
        atomic_json(output/"result.json", result)
        atomic_json(output/"progress.json", result)
        print("DATASET_RENDER_RESULT "+json.dumps({key: result[key] for key in
              ("ok", "captured_count", "source_unchanged", "app_closed", "error")}), flush=True)
    return 0 if result["batch_complete"] else 1


def validated_process_exit_code(code, receipt_path):
    """Never turn a failed/incomplete render into an explicit successful exit."""
    if type(code) is not int or not 0 <= code <= 255:
        return 1
    if code != 0:
        return code
    try:
        path = Path(receipt_path)
        if path.is_symlink():
            return 1
        with path.open("r+") as stream:
            receipt = json.load(stream)
            stream.flush()
            os.fsync(stream.fileno())
        captured, requested = receipt.get("captured_count"), receipt.get("requested_count")
        final = receipt.get("ok") is True and receipt.get("phase") == "complete" and captured == requested
        partial = (receipt.get("ok") is False and receipt.get("phase") == "batch_complete"
                   and receipt.get("batch_complete") is True
                   and type(captured) is int and type(requested) is int and captured < requested)
        if (not (final or partial) or receipt.get("app_closed") is not True
                or receipt.get("source_unchanged") is not True
                or receipt.get("native_nurec_crop_unchanged") is not True
                or type(captured) is not int or type(requested) is not int
                or captured <= 0 or requested <= 0):
            return 1
    except (OSError, ValueError, TypeError, AttributeError):
        return 1
    return 0


def exit_after_finalization(code, receipt_path):
    """Exit after explicit Kit shutdown, without repeating native module teardown.

    Isaac 6.0 SimulationApp.close(fast_shutdown=False) completes application
    teardown but leaves native framework unload to interpreter exit. On this
    host that later unload segfaulted after all outputs and close completed.
    This is restricted to the executable worker: run() has returned, closed
    its files/resources, and persisted its outcome. Failure codes remain failures.
    The supervisor must still independently observe exit 0 plus a valid receipt.
    """
    code = validated_process_exit_code(code, receipt_path)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            code = code or 1
    os._exit(code)


def main(*, explicit_process_exit=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args()
    lock = None
    try:
        plan_path = Path(args.plan).expanduser().resolve(strict=True)
        plan = validate_plan(json.loads(plan_path.read_text()))
        raw_output = Path(args.output).expanduser()
        if raw_output.is_symlink():
            raise ValueError("Output must not be a symlink")
        output = raw_output.resolve()
        if not 1 <= args.batch_size <= 1000:
            raise ValueError("batch-size must be 1..1000")
        # Public runtime permits startup <= 86400 plus up to 1000 frames at <= 86400 each.
        if args.timeout_seconds is not None and not 1 <= args.timeout_seconds <= 86_486_400:
            raise ValueError("timeout-seconds must be 1..86486400")
        protected = [plan_path, Path(plan["scene"]["path"]).expanduser().resolve()]
        if any(p == output or output in p.parents for p in protected):
            raise ValueError("Output cannot contain protected inputs")
        checkpoints.prepare_output(plan_path, output)
        lock_path = output/".capture.lock"
        if lock_path.is_symlink():
            raise ValueError("Unsafe capture lock")
        lock = lock_path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Inspect all durable progress witnesses before replacing any of them.
        checkpoints.validate_capture(plan_path, output)
        atomic_json(output/"result.json", {"ok": False, "batch_complete": False,
                    "phase": "validated_resume", "app_closed": False, "source_unchanged": False})
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(type(exc).__name__+": "+str(exc), file=sys.stderr)
        if lock is not None:
            lock.close()
        return 2
    try:
        code = run(plan, plan_path, output, args.batch_size, args.timeout_seconds)
    finally:
        lock.close()
    if explicit_process_exit:
        exit_after_finalization(code, output/"result.json")
    return code


if __name__ == "__main__":
    sys.exit(main(explicit_process_exit=True))
