"""User-owned persistent runtime settings; no machine-specific defaults."""
import json
import math
import os
from pathlib import Path
import tempfile

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _xdg(name, fallback):
    value = os.environ.get(name)
    return Path(value).expanduser() if value and Path(value).expanduser().is_absolute() else Path.home() / fallback


def config_path():
    override = os.environ.get("ISAAC_DATASET_CONFIG")
    return Path(override).expanduser() if override else _xdg("XDG_CONFIG_HOME", ".config") / "isaac-image-dataset/config.json"


def validate_runtime_config(value):
    """Validate runtime separately from the immutable dataset plan.

    Unconfigured installs are readable. Execution requires an explicit Isaac
    entry point; the Hermes interpreter is never an implicit fallback.
    """
    if not isinstance(value, dict):
        raise ValueError("runtime configuration must be an object")
    defaults = {
        "isaac_python": None,
        "run_root": str(_xdg("XDG_DATA_HOME", ".local/share") / "isaac-image-dataset/runs"),
        "gpu_lock_path": str(_xdg("XDG_CACHE_HOME", ".cache") / "isaac-image-dataset/gpu.lock"),
        "batch_size": 20, "per_sample_timeout_seconds": 180,
        "startup_timeout_seconds": 600, "sampling_timeout_seconds": 600,
        "memory_max_gb": 10, "default_scene": None,
    }
    if set(value) - set(defaults):
        raise ValueError("unknown runtime settings: " + ", ".join(sorted(set(value) - set(defaults))))
    result = dict(defaults, **value)
    for key in ("isaac_python", "run_root", "gpu_lock_path", "default_scene"):
        raw = result[key]
        if key in {"default_scene", "isaac_python"} and raw is None:
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a nonempty path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{key} must be an absolute path or start with ~")
        result[key] = str(path)
    for key, low, high in (("batch_size", 1, 1000), ("per_sample_timeout_seconds", 1, 86400),
                           ("startup_timeout_seconds", 1, 86400), ("sampling_timeout_seconds", 1, 86400),
                           ("memory_max_gb", 1, 1024)):
        item = result[key]
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not low <= item <= high or not math.isfinite(item)):
            raise ValueError(f"{key} must be finite and between {low} and {high}")
    if type(result["batch_size"]) is not int:
        raise ValueError("batch_size must be an integer")
    if result["memory_max_gb"] != 10:
        raise ValueError("memory_max_gb must be 10 for the validated renderer cgroup contract")
    return result


def _read_config(path):
    if path.is_symlink():
        raise ValueError("Configuration must not be a symlink")
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read dataset runtime configuration") from exc
    if not isinstance(value, dict):
        raise ValueError("runtime configuration must be an object")
    return value


def load_runtime_config():
    value = _read_config(config_path())
    if "ISAAC_DATASET_RUN_ROOT" in os.environ:
        value["run_root"] = os.environ["ISAAC_DATASET_RUN_ROOT"]
    return validate_runtime_config(value)


def configuration_status(runtime):
    missing = [] if runtime.get("isaac_python") else ["isaac_python"]
    return {"configuration_complete": not missing, "missing_required_settings": missing,
            "default_scene_configured": bool(runtime.get("default_scene")),
            "note": "Configured paths are not a GPU/Isaac capability check. Supply scene_path per job or save default_scene."}


def configure_runtime_config(patch, path=None):
    """Merge user-provided settings atomically, without modifying active jobs.

    Environment overrides are not copied to disk. Configurations are mode 0600;
    the lock serializes concurrent merges and no existing malformed file is lost.
    """
    import fcntl
    if not isinstance(patch, dict):
        raise ValueError("settings must be an object")
    destination = Path(path).expanduser() if path is not None else config_path()
    if not destination.is_absolute():
        raise ValueError("Configuration path must be absolute")
    if destination.is_symlink():
        raise ValueError("Configuration must not be a symlink")
    # Validate field names/types before creating any configuration directory.
    validate_runtime_config(patch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = {**_read_config(destination), **patch}
        result = validate_runtime_config(merged)
        executable = result["isaac_python"]
        if executable and (not Path(executable).is_file() or not os.access(executable, os.X_OK)):
            raise ValueError("isaac_python must name an existing executable Isaac Python or python.sh entry point")
        scene = result["default_scene"]
        if scene and not Path(scene).is_file():
            raise ValueError("default_scene must name an existing local semantic USD file")
        for key in ("run_root", "gpu_lock_path"):
            target = Path(result[key])
            if target.exists() and (target.is_dir() if key == "gpu_lock_path" else not target.is_dir()):
                raise ValueError(key + " has the wrong file type")
        temp_fd, temp_path = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
        try:
            with os.fdopen(temp_fd, "w") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, destination)
            parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    return result


def batch_timeout(runtime, count):
    return math.ceil(runtime["startup_timeout_seconds"] + runtime["per_sample_timeout_seconds"] * count)


def job_timeout(runtime, count):
    batches = max(1, math.ceil(count / runtime["batch_size"]))
    return math.ceil(600 + runtime.get("sampling_timeout_seconds", 600) + batches * (runtime["startup_timeout_seconds"] + 60)
                     + runtime["per_sample_timeout_seconds"] * count)
