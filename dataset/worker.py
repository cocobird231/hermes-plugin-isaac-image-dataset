"""Bounded, restartable stage worker and its durable process-exit supervisor."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import jobs, settings

HERE = Path(__file__).resolve().parent
SAMPLING_TIMEOUT_SECONDS = 600
_PROCESS_LOCKS = set()


class JobCancelled(Exception):
    """Owned worker received a termination request."""


def _cancel_signal(signum, frame):
    raise JobCancelled(f"Worker received signal {signum}")


def available_ram():
    fields = Path("/proc/meminfo").read_text().splitlines()
    return next(int(line.split()[1]) * 1024 for line in fields if line.startswith("MemAvailable:"))


@contextmanager
def gpu_slot(runtime=None):
    """Take the configured shared lock; never terminate another GPU owner."""
    runtime = runtime or settings.load_runtime_config()
    lock = Path(runtime["gpu_lock_path"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another capture owns the GPU; resume after it finishes") from exc
        if available_ram() < 12 * 1024**3:
            raise RuntimeError("Capture requires at least 12 GiB available host RAM")
        query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15, check=True)
        if query.stdout.strip():
            raise RuntimeError("GPU compute process already active; resume after it finishes")
        _PROCESS_LOCKS.add(fd)
        yield
    finally:
        _PROCESS_LOCKS.discard(fd)
        os.close(fd)


def clean_environment(runtime=None):
    runtime = runtime or settings.load_runtime_config()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = str(Path(runtime["isaac_python"]).parent) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    return env


def _stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)


def run_child(command, log_path, timeout, *, watch_memory=False, runtime=None, check=True):
    """Wait for actual process exit and clean up only the owned process group."""
    started = time.monotonic()
    with Path(log_path).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=clean_environment(runtime),
                                 pass_fds=tuple(_PROCESS_LOCKS))
        try:
            while child.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Owned stage exceeded {timeout} seconds; see {Path(log_path).name}")
                if watch_memory and available_ram() < 1536 * 1024**2:
                    raise RuntimeError("Capture stopped: host available RAM fell below 1.5 GiB")
                time.sleep(.5)
            if check and child.returncode:
                raise RuntimeError(f"Stage process exited {child.returncode}; see {Path(log_path).name}")
            return {"returncode": child.returncode, "pid": child.pid,
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
        finally:
            _stop_child(child)


def _configuration_digest(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _plan_receipt(job, request):
    receipt = jobs._read(job, "plan-commit.json", optional=True)
    expected = _configuration_digest(request["configuration"])
    if (receipt is None or receipt.get("job_id") != request["job_id"]
            or receipt.get("configuration_sha256") != expected
            or _configuration_digest(jobs._read(job, "config.json")) != expected):
        raise ValueError("Immutable sampling plan has no matching configuration commit")
    return receipt


def validate_plan_commit(job, request):
    """Reject changed or unbound plans before reuse, capture or splitting."""
    receipt = _plan_receipt(job, request)
    path = jobs._file(job, "plan.json")
    if not path.is_file() or receipt.get("plan_sha256") != jobs._sha256(path):
        raise ValueError("Immutable sampling plan SHA256 differs from its commit")
    return jobs._read(job, "plan.json")


def commit_plan(job, request, plan):
    """Commit the expected digest first, then atomically publish the plan.

    If interrupted between these writes, deterministic resampling may recover
    only the exact committed bytes; existing unbound plans are never adopted.
    """
    path = jobs._file(job, "plan.json")
    body = (json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    receipt = {"schema_version": 1, "job_id": request["job_id"],
               "configuration_sha256": _configuration_digest(request["configuration"]),
               "plan_sha256": hashlib.sha256(body).hexdigest()}
    existing = jobs._read(job, "plan-commit.json", optional=True)
    if existing is None:
        if path.exists():
            raise ValueError("Cannot adopt an existing unbound sampling plan")
        if _configuration_digest(jobs._read(job, "config.json")) != receipt["configuration_sha256"]:
            raise ValueError("Sampling configuration changed before commit")
        jobs._write(job, "plan-commit.json", receipt, immutable=True)
    elif existing != receipt:
        raise ValueError("Recovered sampling plan differs from its immutable commit")
    if not path.exists():
        jobs._write(job, "plan.json", plan, immutable=True)
    return validate_plan_commit(job, request)


def sample_plan(job, request):
    from dataset.sampling import generate_plan
    configuration = request["configuration"]
    plan = generate_plan(configuration)
    path = jobs._file(job, "plan.json")
    existing = jobs._read(job, "plan-commit.json", optional=True)
    if existing is not None and not path.exists():
        # Only recover exact previously authenticated bytes across default-field
        # additions. Explicitly requested values and sampled geometry are never altered.
        from itertools import combinations
        receipt = _plan_receipt(job, request)
        optional = [key for key, default in (("camera_count", 1), ("render_quality", "normal"))
                    if key not in configuration and plan.get("config", {}).get(key) == default]
        def digest(candidate):
            body = (json.dumps(candidate, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
            return hashlib.sha256(body).hexdigest()
        if digest(plan) != receipt.get("plan_sha256"):
            for count in range(1, len(optional) + 1):
                matched = False
                for omitted in combinations(optional, count):
                    legacy = {**plan, "config": {key: value for key, value in plan["config"].items()
                                               if key not in omitted}}
                    if digest(legacy) == receipt.get("plan_sha256"):
                        plan, matched = legacy, True
                        break
                if matched:
                    break
    return commit_plan(job, request, plan)


def run_sampling(job, request):
    runtime = request["runtime"]
    record = jobs._record(job)
    command = [runtime["isaac_python"], str(Path(__file__).resolve()), "--job-dir", str(job),
               "--request", record["request_path"], "--action", request["action"], "--sample-only"]
    return run_child(command, job / f"attempt-{request['attempt']}-sampling.log",
                     runtime.get("sampling_timeout_seconds", SAMPLING_TIMEOUT_SECONDS), runtime=runtime)


def run_capture(plan_path, output_dir, log_path, *, runtime=None):
    """One bounded batch; a renderer receipt never substitutes for process exit."""
    runtime = runtime or settings.load_runtime_config()
    expected = len(json.loads(plan_path.read_text(encoding="utf-8"))["samples"])
    timeout = settings.batch_timeout(runtime, min(runtime["batch_size"], expected))
    command = [runtime["isaac_python"], str(HERE / "render_worker.py"),
               "--plan", str(plan_path), "--output", str(output_dir),
               "--batch-size", str(runtime["batch_size"]), "--timeout-seconds", str(timeout)]
    process = run_child(command, log_path, timeout + 30, watch_memory=True, runtime=runtime)
    receipt = output_dir / "result.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise RuntimeError("Successful capture exit has no owned result.json")
    result = json.loads(receipt.read_text(encoding="utf-8"))
    count = result.get("captured_count")
    if (result.get("batch_complete") is not True or result.get("app_closed") is not True
            or result.get("source_unchanged") is not True or type(count) is not int
            or not 0 <= count <= expected or result.get("requested_count") != expected
            or (count == expected and (result.get("ok") is not True or result.get("phase") != "complete"))
            or (count < expected and (result.get("ok") is not False or result.get("phase") != "batch_complete"))):
        raise RuntimeError("Capture receipt does not confirm batch count, close and unchanged sources")
    return dict(result, process_exit=process)


class DatasetAttempt:
    def __init__(self, job, request):
        self.job = job
        self.request = request
        self.runtime = settings.validate_runtime_config(request["runtime"])
        self.artifacts = {}
        self.plan_path = jobs._file(job, "plan.json")
        self.output_dir = jobs._file(job, "dataset")

    def state(self, state, *, step, **extra):
        result = {"job_id": self.request["job_id"], "attempt": self.request["attempt"],
                  "state": state, "step": step, "updated_at": jobs._now(),
                  "artifacts": self.artifacts, **extra}
        jobs._write(self.job, "status.json", result)
        return result

    def sample(self):
        self.state("running", step="sampling")
        if not self.plan_path.exists():
            run_sampling(self.job, self.request)
        plan = validate_plan_commit(self.job, self.request)
        self.artifacts["plan"] = str(self.plan_path)
        if plan.get("complete") is not True:
            raise RuntimeError("Sampling could not satisfy requested count; immutable plan remains for diagnosis")
        return plan

    def collect(self):
        from dataset.checkpoints import validate_capture
        if not self.plan_path.is_file():
            raise ValueError("collect requires an existing plan.json")
        plan = validate_plan_commit(self.job, self.request)
        if plan.get("complete") is not True:
            raise ValueError("An incomplete sampling plan cannot be captured")
        expected = len(plan["samples"])
        committed = len(validate_capture(self.plan_path, self.output_dir)) if self.output_dir.exists() else 0
        self.artifacts.update(plan=str(self.plan_path), dataset=str(self.output_dir))
        self.state("running", step="capturing", captured_count=committed, requested_count=expected)
        with gpu_slot(self.runtime):
            batch = 0
            while True:
                batch += 1
                prefix = f"attempt-{self.request['attempt']}-batch-{batch:04d}"
                result = run_capture(self.plan_path, self.output_dir, self.job / (prefix + ".log"),
                                     runtime=self.runtime)
                rows = validate_capture(self.plan_path, self.output_dir,
                                        require_complete=result["captured_count"] == expected)
                count = result["captured_count"]
                if len(rows) != count or (count <= committed and committed < expected):
                    raise RuntimeError("Capture batch made no verified checkpoint progress")
                if count - committed > self.runtime["batch_size"]:
                    raise RuntimeError("Capture exceeded configured batch size")
                jobs._write(self.job, prefix + "-result.json", result, immutable=True)
                committed = count
                self.state("running", step="capturing", batch=batch,
                           captured_count=count, requested_count=expected,
                           last_batch_exit=result["process_exit"])
                if count == expected:
                    break
        self.artifacts["capture_result"] = str(self.output_dir / "result.json")
        return result

    def split(self):
        from dataset.splitting import finalize_dataset
        validate_plan_commit(self.job, self.request)
        self.artifacts.update(plan=str(self.plan_path), dataset=str(self.output_dir))
        self.state("running", step="splitting")
        ratios = self.request["payload"].get("ratios") if self.request["action"] == "split" else None
        result = finalize_dataset(self.plan_path, self.output_dir, ratios=ratios)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("Split verification did not confirm success")
        self.artifacts["split_result"] = str(self.output_dir / "split-result.json")
        return result

    def execute(self):
        action = self.request["action"]
        if action in {"sample", "pipeline"}:
            self.sample()
        if action in {"collect", "pipeline"}:
            self.collect()
        if action in {"split", "pipeline"}:
            result = self.split()
            return self.state("completed", step="split", result=result)
        return self.state("sampled" if action == "sample" else "captured", step=action)


def supervise(job, request, request_path):
    """Persist the child's actual exit, independently of transient systemd state."""
    command = [request["runtime"]["isaac_python"], str(Path(__file__).resolve()),
               "--job-dir", str(job), "--request", str(request_path), "--action", request["action"]]
    process = run_child(command, job / f"attempt-{request['attempt']}-worker.log",
                        settings.job_timeout(request["runtime"], request["configuration"].get("n", 32)),
                        runtime=request["runtime"], check=False)
    record = jobs._record(job)
    snapshot = jobs._read(job, "status.json", optional=True) or {}
    receipt = {"schema_version": 1, "job_id": job.name, "attempt": request["attempt"],
               "unit": record["unit"], "request_sha256": jobs._sha256(request_path),
               "worker_returncode": process["returncode"], "worker_status": snapshot,
               "process": process, "finished_at": jobs._now()}
    jobs._write(job, f"attempt-{request['attempt']}-exit.json", receipt, immutable=True)
    return process["returncode"] if process["returncode"] >= 0 else 128 - process["returncode"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--action", choices=sorted(jobs.ACTIONS), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sample-only", action="store_true")
    modes.add_argument("--supervise", action="store_true")
    args = parser.parse_args(argv)
    job = args.job_dir.resolve(strict=True)
    if args.job_dir.is_symlink() or jobs._job_dir(job.name) != job:
        raise ValueError("Job directory must belong to the configured run root")
    request_path = args.request.resolve(strict=True)
    if args.request.is_symlink() or request_path.parent != job:
        raise ValueError("Request must belong to the owned job")
    record = jobs._record(job)
    request = jobs._read(job, request_path.name)
    if (record.get("request_path") != str(request_path) or record["action"] != args.action
            or request.get("action") != args.action or request.get("job_id") != job.name
            or record["attempt"] != request.get("attempt")):
        raise ValueError("Job/request identity mismatch")
    signal.signal(signal.SIGTERM, _cancel_signal)
    signal.signal(signal.SIGINT, _cancel_signal)
    if args.sample_only:
        if args.action not in jobs.NEW_ACTIONS:
            raise ValueError("Only sample/pipeline may create a sampling plan")
        sample_plan(job, request)
        return 0
    if args.supervise:
        return supervise(job, request, request_path)
    with jobs._locked(job, ".execution.lock") as execution_fd:
        _PROCESS_LOCKS.add(execution_fd)
        current = jobs._record(job)
        if current["attempt"] != request["attempt"]:
            raise ValueError("A newer attempt superseded this worker")
        attempt = DatasetAttempt(job, request)
        try:
            result = attempt.execute()
        except JobCancelled as exc:
            attempt.state("cancelled", step="cancelled", error=str(exc))
            return 130
        except Exception as exc:
            attempt.state("failed", step="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
            print(json.dumps({"state": "failed", "error_type": type(exc).__name__}), flush=True)
            return 1
        finally:
            _PROCESS_LOCKS.discard(execution_fd)
    print(json.dumps({"state": result["state"], "job_id": job.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
