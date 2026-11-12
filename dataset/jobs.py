"""Owned asynchronous jobs with durable process receipts and resumable attempts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

from . import settings

PLUGIN_ROOT = settings.PLUGIN_ROOT
WORKER_PATH = Path(__file__).with_name("worker.py")
ACTIONS = frozenset({"sample", "collect", "split", "pipeline"})
NEW_ACTIONS = frozenset({"sample", "pipeline"})
SUCCESS_STATES = frozenset({"sampled", "captured", "completed"})
ACTIVE_STATES = frozenset({"pending", "running", "cancelling"})
SHOW_PROPERTIES = ("LoadState", "ActiveState", "SubState", "Result", "ExecMainCode",
                   "ExecMainStatus", "ExecMainPID")


class JobBusyError(RuntimeError):
    """A previous worker or another caller still owns the job."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _root():
    root = Path(settings.load_runtime_config()["run_root"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root.resolve(strict=True)


def job_root():
    return _root()


def resolve_job(job_id):
    path = _job_dir(job_id)
    with _locked(path):
        _record(path)
    return path


def _job_dir(job_id, *, create=False):
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
        raise ValueError("job_id must be a lowercase UUID hex string")
    root = _root()
    path = root / job_id
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("job directory escapes the configured run root")
    if create:
        path.mkdir(mode=0o700)
    elif not path.is_dir():
        raise ValueError("unknown job_id")
    return path


def _file(job_dir, name):
    path = job_dir / name
    if path.is_symlink() or path.resolve().parent != job_dir:
        raise ValueError("job metadata path must remain inside its job directory")
    return path


@contextmanager
def _locked(job_dir, name=".lock"):
    fd = os.open(_file(job_dir, name), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobBusyError("job is locked by another caller or worker") from exc
        yield fd
    finally:
        os.close(fd)


def _executing(job_dir):
    try:
        with _locked(job_dir, ".execution.lock"):
            return False
    except JobBusyError:
        return True


def _read(job_dir, name, *, optional=False):
    path = _file(job_dir, name)
    if optional and not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return result


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(job_dir, name, value, *, immutable=False):
    """Publish complete JSON atomically, including first-time immutable writes."""
    path = _file(job_dir, name)
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".metadata-", dir=job_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        _fsync_directory(job_dir)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _record(job_dir):
    record = _read(job_dir, "job.json")
    attempt = record.get("attempt")
    if (record.get("job_id") != job_dir.name or type(attempt) is not int or attempt < 1
            or record.get("action") not in ACTIONS):
        raise ValueError("invalid recorded job identity")
    if record.get("unit") != f"isaac-dataset-{job_dir.name}-{attempt}.service":
        raise ValueError("recorded unit is outside this job namespace")
    return record


def _show(unit):
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=" + ",".join(SHOW_PROPERTIES)],
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"query_error": "systemd_status_unavailable"}
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in SHOW_PROPERTIES:
            fields[key] = value
    if (result.returncode != 0 and fields.get("LoadState") != "not-found") or not fields:
        return {"query_error": "systemd_status_unavailable"}
    return fields


def _exit_receipt(job_dir, record):
    receipt = _read(job_dir, f"attempt-{record['attempt']}-exit.json", optional=True)
    if receipt is None:
        return None
    request = _file(job_dir, Path(record["request_path"]).name)
    if (receipt.get("job_id") != record["job_id"] or receipt.get("attempt") != record["attempt"]
            or receipt.get("unit") != record["unit"]
            or receipt.get("request_sha256") != _sha256(request)
            or type(receipt.get("worker_returncode")) is not int):
        raise ValueError("invalid supervisor exit receipt identity")
    return receipt


def _capture_progress(job_dir):
    """Expose renderer observations without treating them as process receipts."""
    output = _file(job_dir, "dataset")
    if not output.is_dir():
        return None
    progress = _read(output, "progress.json", optional=True)
    manifest = _read(output, "capture-manifest.json", optional=True)
    commit = _read(job_dir, "plan-commit.json", optional=True)
    if not progress or not manifest or not commit:
        return None
    if (commit.get("job_id") != job_dir.name
            or commit.get("plan_sha256") != manifest.get("plan_sha256")
            or commit.get("plan_sha256") != _sha256(_file(job_dir, "plan.json"))):
        raise ValueError("Capture progress does not match the immutable job plan")
    count, requested = progress.get("captured_count"), progress.get("requested_count")
    if (type(count) is not int or type(requested) is not int or not 0 <= count <= requested
            or requested != manifest.get("requested_count")):
        raise ValueError("Capture progress has invalid sample counts")
    fields = ("phase", "captured_count", "requested_count", "batch_id", "batch_size",
              "resumed_count", "batch_target_count", "batch_captured_count", "started_utc")
    return {key: progress[key] for key in fields if key in progress}


def _status_locked(job_dir, record=None):
    record = record or _record(job_dir)
    worker = _read(job_dir, "status.json", optional=True) or {}
    if worker.get("job_id") != record["job_id"] or worker.get("attempt") != record["attempt"]:
        worker = {}
    receipt = _exit_receipt(job_dir, record)
    if receipt and isinstance(receipt.get("worker_status"), dict):
        snapshot = receipt["worker_status"]
        if snapshot.get("job_id") == record["job_id"] and snapshot.get("attempt") == record["attempt"]:
            worker = snapshot
    unit = _show(record["unit"])
    result = dict(worker)
    result.update(job_id=record["job_id"], attempt=record["attempt"], action=record["action"],
                  unit=record["unit"], job_dir=str(job_dir), systemd=unit,
                  worker_state=worker.get("state"), queried_at=_now())
    progress = _capture_progress(job_dir)
    if progress is not None:
        result["capture_progress"] = progress
    if receipt:
        result["process_exit"] = {"worker_returncode": receipt["worker_returncode"],
                                  "receipt": str(job_dir / f"attempt-{record['attempt']}-exit.json")}
    live = (unit.get("ActiveState") in {"active", "activating", "deactivating"}
            and unit.get("SubState") not in {"exited", "dead", "failed"})
    worker_state = worker.get("state")
    if live or _executing(job_dir):
        state = "cancelling" if record.get("cancel_requested_at") else "running"
    elif receipt and receipt["worker_returncode"] == 0 and worker_state in SUCCESS_STATES:
        state = worker_state
    elif unit.get("query_error") and receipt is None:
        state = "unknown"
        result["error"] = unit["query_error"]
    elif record.get("cancel_requested_at"):
        state = "cancelled"
    elif receipt:
        state = "failed"
        result.setdefault("error", "worker_did_not_exit_successfully" if receipt["worker_returncode"]
                          else "worker_result_missing_or_incomplete")
    elif unit.get("query_error"):
        state = "unknown"
        result["error"] = unit["query_error"]
    elif record.get("dispatch_error"):
        state = "failed"
        result["error"] = record["dispatch_error"]
    else:
        normal_exit = (unit.get("LoadState") == "loaded" and unit.get("Result") == "success"
                       and unit.get("ExecMainCode") == "1" and unit.get("ExecMainStatus") == "0"
                       and unit.get("SubState") in {"exited", "dead"})
        state = worker_state if normal_exit and worker_state in SUCCESS_STATES else "failed"
        if state == "failed":
            result.setdefault("error", "worker_result_missing_or_incomplete" if normal_exit
                              else "interrupted_or_missing_process_receipt")
    result["state"] = state
    result["resumable"] = state in {"failed", "cancelled"}
    return result


def _worker_command(job_dir, record, request_path):
    request = _read(job_dir, request_path.name)
    python = Path(request["runtime"]["isaac_python"])
    if not python.is_file():
        raise RuntimeError("Isaac Python interpreter is unavailable")
    return ["/usr/bin/env", "-u", "PYTHONPATH", "-u", "PYTHONHOME", "-u", "PYTHONSTARTUP",
            "-u", "PYTHONUSERBASE", "-u", "VIRTUAL_ENV", "PYTHONNOUSERSITE=1",
            "ISAAC_DATASET_CONFIG=" + str(settings.config_path()),
            "ISAAC_DATASET_RUN_ROOT=" + str(job_dir.parent), str(python), str(WORKER_PATH),
            "--job-dir", str(job_dir), "--action", record["action"],
            "--request", str(request_path), "--supervise"]


def _launch(job_dir, record, request_path):
    request = _read(job_dir, request_path.name)
    runtime = request["runtime"]
    count = request["configuration"].get("n", 32)
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("n must be an integer between 1 and 10000")
    command = ["systemd-run", "--user", "--quiet", "--unit=" + record["unit"],
               "--property=Type=exec", f"--property=MemoryMax={runtime['memory_max_gb']}G",
               "--property=MemorySwapMax=0",
               f"--property=RuntimeMaxSec={settings.job_timeout(runtime, count)}",
               "--property=TimeoutStopSec=45", "--property=Nice=10",
               "--property=RemainAfterExit=yes", "--property=KillMode=control-group",
               "--property=WorkingDirectory=" + str(PLUGIN_ROOT)]
    try:
        command.extend(_worker_command(job_dir, record, request_path))
        launched = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except RuntimeError:
        record["dispatch_error"] = "isaac_python_unavailable"
    except subprocess.TimeoutExpired:
        record["dispatch_error"] = "systemd_dispatch_timeout"
    except OSError:
        record["dispatch_error"] = "systemd_dispatch_unavailable"
    else:
        if launched.returncode == 0:
            return True
        record["dispatch_error"] = "systemd_dispatch_failed"
    _write(job_dir, "job.json", record)
    return False


def submit(action, payload, *, job_id=None):
    """Start a stage or resume a stopped attempt without changing its plan."""
    requested_action = action
    if action not in ACTIONS | {"resume"}:
        raise ValueError("unsupported action")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    fresh = job_id is None
    if fresh:
        if action not in NEW_ACTIONS:
            raise ValueError("collect/split/resume require an existing job_id")
        job_id = uuid.uuid4().hex
    elif action in NEW_ACTIONS:
        raise ValueError("sample/pipeline always create a new job")
    elif action in {"collect", "resume"} and payload:
        raise ValueError(f"{action} accepts no configuration changes")
    elif action == "split" and set(payload) - {"ratios"}:
        raise ValueError("split accepts only ratios")
    runtime = settings.load_runtime_config()
    if not runtime.get("isaac_python"):
        raise ValueError("Isaac Python is not configured; call isaac_dataset_configure with settings.isaac_python first")
    job_dir = _job_dir(job_id, create=fresh)
    with _locked(job_dir):
        old = None if fresh else _record(job_dir)
        if old:
            previous = _status_locked(job_dir, old)
            if previous["state"] in ACTIVE_STATES | {"unknown"}:
                raise JobBusyError("previous attempt has not definitively stopped")
            if action == "resume":
                if previous["state"] in SUCCESS_STATES:
                    return dict(previous, accepted=False, already_finished=True)
                if previous["state"] not in {"failed", "cancelled"}:
                    raise ValueError("resume requires a failed or cancelled job")
                old_request = _read(job_dir, Path(old["request_path"]).name)
                action, payload = old["action"], old_request["payload"]
            else:
                required = {"collect": "sampled", "split": "captured"}[action]
                if previous["state"] != required:
                    raise ValueError(f"{action} requires state {required}; current state is {previous['state']}")
            _write(job_dir, f"attempt-{old['attempt']}-status.json", previous)
        configuration = payload if fresh else _read(job_dir, "config.json")
        attempt = 1 if fresh else old["attempt"] + 1
        stamp = _now()
        record = {"job_id": job_id, "attempt": attempt, "action": action,
                  "created_at": stamp if fresh else old["created_at"], "updated_at": stamp,
                  "unit": f"isaac-dataset-{job_id}-{attempt}.service"}
        request_path = _file(job_dir, f"request-{uuid.uuid4().hex}.json")
        request = {"job_id": job_id, "attempt": attempt, "action": action, "created_at": stamp,
                   "payload": payload, "configuration": configuration, "runtime": runtime,
                   "resuming": requested_action == "resume"}
        _write(job_dir, request_path.name, request, immutable=True)
        record["request_path"] = str(request_path)
        if fresh:
            _write(job_dir, "config.json", configuration, immutable=True)
        _write(job_dir, "job.json", record)
        _write(job_dir, "status.json", {"job_id": job_id, "attempt": attempt,
                                       "state": "pending", "updated_at": stamp})
        accepted = _launch(job_dir, record, request_path)
        return {"accepted": accepted, "job_id": job_id, "state": "pending" if accepted else "unknown",
                "unit": record["unit"], "attempt": attempt, "job_dir": str(job_dir),
                "request_path": str(request_path),
                **({"error": record["dispatch_error"]} if not accepted else {})}


def resume(job_id):
    return submit("resume", {}, job_id=job_id)


def status(job_id, wait_seconds=0):
    """Optionally wait up to 30 seconds, releasing the metadata lock between polls."""
    if (isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float))
            or not 0 <= wait_seconds <= 30):
        raise ValueError("wait_seconds must be between 0 and 30")
    job_dir = _job_dir(job_id)
    deadline = time.monotonic() + wait_seconds
    while True:
        with _locked(job_dir):
            current = _status_locked(job_dir)
        remaining = deadline - time.monotonic()
        if current["state"] not in ACTIVE_STATES or remaining <= 0:
            return current
        time.sleep(min(1, remaining))


def cancel(job_id):
    """Stop only this attempt's unit; retain all committed captures for resume."""
    job_dir = _job_dir(job_id)
    with _locked(job_dir):
        record = _record(job_dir)
        current = _status_locked(job_dir, record)
        if current["state"] not in ACTIVE_STATES | {"unknown"}:
            return dict(current, cancel_requested=False)
        record["cancel_requested_at"] = _now()
        _write(job_dir, "job.json", record)
        try:
            stopped = subprocess.run(["systemctl", "--user", "stop", "--no-block", record["unit"]],
                                     capture_output=True, text=True, timeout=5, check=False)
            accepted = stopped.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            accepted = False
        return dict(current, state="cancelling" if accepted else current["state"],
                    cancel_requested=True, cancel_accepted=accepted,
                    **({"error": "systemd_stop_not_confirmed"} if not accepted else {}))
