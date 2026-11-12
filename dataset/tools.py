"""Hermes dataset tool schemas: pure Python, no Isaac/USD imports."""
import json
import math

from . import jobs, settings

TOOLSET = "isaac-image-dataset"


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _number(lo, hi=None, *, default=None, integer=False):
    result = {"type": "integer" if integer else "number", "minimum": lo}
    if hi is not None:
        result["maximum"] = hi
    if default is not None:
        result["default"] = default
    return result


def _pair(items, default):
    return {"type": "array", "minItems": 2, "maxItems": 2,
            "items": items, "default": default}


RATIOS = {"type": "array", "minItems": 3, "maxItems": 3,
          "items": _number(0), "default": [4, 2, 2],
          "description": "Train/validation/test relative weights; positive finite sum. Spatial groups remain together."}
JOB_ID = {"type": "string", "pattern": "^[0-9a-f]{32}$",
          "description": "This plugin's UUID hex; never a path, unit or PID."}
CONFIG_PROPERTIES = {
    "scene_path": {"type": "string", "minLength": 1,
                   "description": "Absolute host path to a schema-2 semantic USD scene; defaults to runtime default_scene when configured. Only metre Z-up static triangle meshes with local dependencies are supported."},
    "scene_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "classes": {"type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "enum": ["road_surface", "drivable_area", "lane"]},
                "default": ["road_surface"],
                "description": "Eligible existing classes. road_surface does not establish individual lane boundaries."},
    "n": {**_number(1, 10000, default=32, integer=True),
          "description": "Total requested images/poses, independent of camera_count. 10000 is the schema ceiling, not guaranteed scene capacity. Geometry and spacing can exhaust max_attempts."},
    "camera_count": {**_number(1, 8, default=1, integer=True),
                     "description": "Number of concurrent reusable cameras (1 to 8), not total images. Each batch uses up to min(camera_count, remaining images) cameras. Does not change sampled poses or split groups. runtime.batch_size separately limits images per Isaac process."},
    "render_quality": {"type": "string", "enum": ["efficiency", "normal", "quality"], "default": "normal",
                       "description": "Post-step rendering strategy: efficiency=0, normal=1 (default), quality=10 extra Kit updates per group. Does not change poses; stored in the immutable plan."},
    "seed": _number(0, 2**63 - 1, default=42, integer=True),
    "sensor_height_m": _number(.05, 100, default=1.5),
    "yaw_deg": _pair(_number(-360, 360), [0, 360]),
    "pitch_deg": _pair(_number(-89, 89), [-5, 10]),
    "roll_deg": _pair(_number(-180, 180), [-5, 5]),
    "min_distance_m": _number(0, 10000, default=3),
    "group_distance_m": _number(0, 10000, default=2),
    "max_slope_deg": _number(0, 89, default=15),
    "clearance_m": _number(.001, 10, default=.15),
    "max_attempts": _number(1, 1000000, default=10000, integer=True),
    "camera": _object({
        "resolution": _pair(_number(16, 4096, integer=True), [1280, 720]),
        "horizontal_fov_deg": _number(2, 175, default=90),
        "clipping_range": {**_pair(_number(.000001), [.1, 2000]),
                           "description": "Explicit near/far overrides the template; omission inherits template clipping, or uses [0.1,2000] without a template."},
        "template_prim": {"type": "string", "pattern": "^/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*$",
                          "description": "Existing scene perspective USD Camera prim with centered undistorted pinhole projection. Template focal length/aperture override horizontal_fov_deg; requested resolution must match aperture aspect. Its pose is replaced by each sampled pose. Does not imply a physical ZED camera."},
    }),
    "split_ratios": RATIOS,
}
CONFIG = _object(CONFIG_PROPERTIES)
RUNTIME_PROPERTIES = {
    name: {"type": "string", "minLength": 1, "description": "User-provided absolute path or ~ path, persisted outside the plugin repository."}
    for name in ("isaac_python", "default_scene", "run_root", "gpu_lock_path")
}
RUNTIME_PROPERTIES["default_scene"].update(type=["string", "null"],
    description="Optional default semantic scene path; null clears it so each job supplies scene_path.")
RUNTIME_PROPERTIES.update({
    "batch_size": _number(1, 1000, integer=True),
    "startup_timeout_seconds": _number(1, 86400),
    "sampling_timeout_seconds": _number(1, 86400),
    "per_sample_timeout_seconds": _number(1, 86400),
})
DESCRIPTIONS = {
    "configure": "Persist user-provided Isaac Python entry point, optional default semantic scene, output paths and runtime limits for future calls. Read get_config first and ask only for missing paths; never guess machine paths. Merge atomically without changing active job snapshots or immutable plans. Installing this plugin does not install Isaac Sim. No simulation starts.",
    "sample": "config.render_quality chooses efficiency (0), normal (1, default), or quality (10) post-step updates. Generate N reproducible camera XYZ and XYZW poses from eligible semantic source faces, with bounded geometry/clearance checks. seed is one batch seed, not a sample count. camera_count selects 1 to 8 concurrent reusable cameras without changing N, sampled poses or groups. Creates immutable plan.json in a new owned job. Returns pending; poll until sampled. An incomplete plan fails and cannot be collected.",
    "collect": "Capture the existing sampled plan with an independent Isaac Sim 6 renderer: original full scene, the plan's configured pinhole camera pool and per-image pose evidence. Requires sampled and exclusive GPU. Returns pending; poll captured. Never opens the legacy 8768 bridge or overwrites partial outputs.",
    "split": "Verify captured RGB/pose artifacts and split by spatial groups into train/validation/test. Optional ratios override plan weights; default 4:2:2. Requires captured. Poll completed; do not confuse accepted with completed.",
    "pipeline": "config.render_quality chooses efficiency (0), normal (1, default), or quality (10) post-step updates. Run the same sample, capture and grouped split stages for a new job. config.camera_count selects 1 to 8 concurrent reusable cameras (default 1); config.n remains total images and runtime.batch_size remains the per-process image limit. Only completed after all requested samples and manifests are verified and worker exits 0. Source scenes remain unchanged; no vehicle movement or legacy bridge required. GPU must be free. Failed partial jobs are retained; resume the same immutable plan and committed images.",
    "status": "Return this job's state/artifacts with actual systemd exit evidence. Optional wait_seconds (0 to 30) bounds status waiting. sampled/captured are intermediate. completed requires worker success plus confirmed exit 0. Unknown state is not success.",
    "resume": "Continue this owned job from its immutable plan and durable image checkpoints. Existing committed images retain their SHA256; changed source/plan or corrupted checkpoints are rejected. Returns pending while remaining capture/split work runs; poll completed.",
    "get_config": "Read this standalone plugin's validated runtime configuration, optional default scene, sampling bounds and camera defaults, including camera_count concurrency versus N total images and runtime.batch_size images per process. No job is created.",
    "cancel": "Request stopping only the recorded dataset systemd unit and its child processes. Preserves partial files and other Isaac sessions. Poll cancelled; accepts only job_id.",
}
TOOLS = tuple(("isaac_dataset_" + action, action)
              for action in ("sample", "collect", "split", "status", "cancel", "pipeline", "resume", "get_config", "configure"))


def _validate(value, schema, name):
    """Small validator for exactly the scalar/array/object schemas above."""
    kind = schema["type"]
    if kind == ["string", "null"]:
        if value is None:
            return
        kind = "string"
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        if set(value) - set(schema["properties"]):
            raise ValueError(f"{name} has unsupported fields")
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{name} is missing required fields")
        for key, item in value.items():
            _validate(item, schema["properties"][key], f"{name}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000000):
            raise ValueError(f"{name} has invalid length")
        for item in value:
            _validate(item, schema["items"], name)
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError(f"{name} must contain unique items")
    elif kind in {"number", "integer"}:
        valid = (type(value) is int if kind == "integer"
                 else type(value) in {int, float})
        try:
            finite = math.isfinite(value)
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not valid or not finite:
            raise ValueError(f"{name} must be finite {kind}")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{name} is outside supported bounds")
    elif kind == "string":
        import re
        if not isinstance(value, str) or len(value) < schema.get("minLength", 0):
            raise ValueError(f"{name} must be a nonempty string")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{name} is not a supported choice")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(f"{name} has invalid format")


def parameters(action):
    if action == "get_config":
        return _object({})
    if action == "configure":
        return _object({"settings": _object(RUNTIME_PROPERTIES)}, ("settings",))
    if action in jobs.NEW_ACTIONS:
        return _object({"config": CONFIG}, ("config",))
    properties = {"job_id": JOB_ID}
    if action == "split":
        properties["ratios"] = RATIOS
    if action == "status":
        properties["wait_seconds"] = _number(0, 30, default=0)
    return _object(properties, ("job_id",))


def dispatch(action, args):
    if action not in DESCRIPTIONS:
        raise ValueError("Unsupported action")
    _validate(args, parameters(action), "arguments")
    # Detach caller-owned objects and reject all NaN/Infinity before dispatch.
    args = json.loads(json.dumps(args, allow_nan=False))
    if action == "configure":
        if not args["settings"]:
            raise ValueError("settings must contain at least one field")
        previous_root = settings.load_runtime_config()["run_root"]
        runtime = settings.configure_runtime_config(args["settings"])
        return {"saved": True, "config_path": str(settings.config_path()), "runtime": runtime,
                "previous_run_root": previous_root,
                "job_lookup_note": "Jobs are looked up under the current run_root. Changing it does not move or stop old jobs; restore their previous root to query, cancel or resume them.",
                **settings.configuration_status(runtime)}
    if action == "get_config":
        runtime = settings.load_runtime_config()
        return {"toolset": TOOLSET, "runtime": runtime, "config_path": str(settings.config_path()),
                **settings.configuration_status(runtime),
                "rendering": {"render_quality_default": "normal", "post_step_updates": {"efficiency": 0, "normal": 1, "quality": 10},
                              "legacy_plan_note": "Existing plans without render_quality retain their original strategy on resume."},
                "sampling": {"n_default": 32, "n_maximum": 10000,
                             "max_attempts_default": 10000, "max_attempts_maximum": 1000000,
                             "pitch_deg_default": [-5, 10],
                             "capacity_note": "N maximum is a request bound, not proof that scene geometry and spacing can yield N valid samples."},
                "camera": {"resolution_default": [1280, 720], "horizontal_fov_deg_default": 90,
                           "camera_count_default": 1, "camera_count_minimum": 1, "camera_count_maximum": 8,
                           "camera_count_semantics": "config.camera_count is concurrent reusable cameras; config.n is total images. Active cameras are capped by remaining images. runtime.batch_size separately limits images per Isaac process. Camera count does not alter sampled poses or spatial groups.",
                           "template_semantics": "Existing centered undistorted perspective USD Camera; template focal/aperture override FOV. Explicit clipping overrides template clipping; omission inherits it. Sampled poses replace template pose."}}
    if action == "status":
        return jobs.status(args["job_id"], wait_seconds=args.get("wait_seconds", 0))
    if action == "resume":
        return jobs.resume(args["job_id"])
    if action == "cancel":
        return jobs.cancel(args["job_id"])
    if action in jobs.NEW_ACTIONS:
        config = args["config"]
        if "scene_path" not in config:
            default_scene = settings.load_runtime_config().get("default_scene")
            if not default_scene:
                raise ValueError("scene_path is required when runtime default_scene is not configured")
            config["scene_path"] = default_scene
        if "split_ratios" in config and not 0 < sum(config["split_ratios"]) < math.inf:
            raise ValueError("split_ratios must have a positive finite sum")
        return jobs.submit(action, config)
    payload = {"ratios": args["ratios"]} if "ratios" in args else {}
    if "ratios" in payload and not 0 < sum(payload["ratios"]) < math.inf:
        raise ValueError("ratios must have a positive finite sum")
    return jobs.submit(action, payload, job_id=args["job_id"])


def handler(action):
    def handle(args, **kwargs):
        try:
            result = dispatch(action, args)
        except jobs.JobBusyError as exc:
            result = {"accepted": False, "error": "job_busy", "message": str(exc)}
        except (ValueError, TypeError, OSError) as exc:
            result = {"accepted": False, "error": "invalid_arguments", "message": str(exc)}
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
    return handle


def register(ctx):
    """Register only dataset tools; caller decides its plugin entry point."""
    for name, action in TOOLS:
        description = DESCRIPTIONS[action]
        ctx.register_tool(name=name, toolset=TOOLSET,
                          schema={"name": name, "description": description,
                                  "parameters": parameters(action)},
                          handler=handler(action), description=description)
