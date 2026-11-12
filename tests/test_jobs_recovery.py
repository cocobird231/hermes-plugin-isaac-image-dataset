"""Process, restart and durable-state contracts; CPU fixtures are not Isaac renders."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import jobs, settings, worker

MISSING = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead"}
LIVE = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running"}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        runtime = settings.validate_runtime_config({"isaac_python": sys.executable,
                  "run_root": str(self.root / "runs"), "gpu_lock_path": str(self.root / "gpu.lock")})
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(runtime))
        self.environment = patch.dict(os.environ, {"ISAAC_DATASET_CONFIG": str(self.config),
                                     "ISAAC_DATASET_RUN_ROOT": runtime["run_root"]})
        self.environment.start()
        self.launch = patch.object(jobs, "_launch", return_value=True)
        self.launch_mock = self.launch.start()
        self.show = patch.object(jobs, "_show", return_value=MISSING)
        self.show_mock = self.show.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.launch.stop)
        self.addCleanup(self.show.stop)

    def create(self, action="pipeline", count=60):
        response = jobs.submit(action, {"n": count, "scene_path": "/fixture.usda"})
        root = Path(response["job_dir"])
        request = jobs._read(root, Path(response["request_path"]).name)
        return response, root, request

    def plan(self, root, plan):
        request = jobs._read(root, Path(jobs._record(root)["request_path"]).name)
        return worker.commit_plan(root, request, plan)

    def receipt(self, response, root, state="completed", code=0):
        status = {"job_id": response["job_id"], "attempt": response["attempt"], "state": state}
        jobs._write(root, "status.json", status)
        receipt = {"job_id": response["job_id"], "attempt": response["attempt"],
                   "unit": response["unit"], "worker_returncode": code, "worker_status": status,
                   "request_sha256": jobs._sha256(response["request_path"])}
        jobs._write(root, f"attempt-{response['attempt']}-exit.json", receipt, immutable=True)

    def test_completed_receipt_survives_unit_gc_and_query_failure(self):
        response, root, _ = self.create()
        self.receipt(response, root)
        for unit in (MISSING, {"query_error": "offline"}):
            self.show_mock.return_value = unit
            state = jobs.status(response["job_id"])
            self.assertEqual(state["state"], "completed")
            self.assertEqual(state["process_exit"]["worker_returncode"], 0)
            self.assertFalse(state["resumable"])

    def test_receipt_with_nonzero_exit_never_publishes_success(self):
        response, root, _ = self.create()
        self.receipt(response, root, code=-11)
        state = jobs.status(response["job_id"])
        self.assertEqual(state["state"], "failed")
        self.assertTrue(state["resumable"])

    def test_zero_exit_without_success_state_is_failed(self):
        response, root, _ = self.create()
        self.receipt(response, root, state="running")
        self.assertEqual(jobs.status(response["job_id"])["state"], "failed")

    def test_changed_request_invalidates_exit_receipt(self):
        response, root, request = self.create()
        self.receipt(response, root)
        request["configuration"]["n"] = 2
        jobs._write(root, Path(response["request_path"]).name, request)
        with self.assertRaisesRegex(ValueError, "receipt identity"):
            jobs.status(response["job_id"])

    def test_live_supervisor_blocks_even_with_final_receipt(self):
        response, root, _ = self.create()
        self.receipt(response, root)
        self.show_mock.return_value = LIVE
        self.assertEqual(jobs.status(response["job_id"])["state"], "running")
        with self.assertRaises(jobs.JobBusyError):
            jobs.resume(response["job_id"])

    def test_execution_lock_blocks_stale_or_missing_unit(self):
        response, root, _ = self.create()
        with jobs._locked(root, ".execution.lock"):
            self.assertEqual(jobs.status(response["job_id"])["state"], "running")
            with self.assertRaises(jobs.JobBusyError):
                jobs.resume(response["job_id"])

    def test_unknown_unit_without_receipt_cannot_overlap(self):
        response, _, _ = self.create()
        self.show_mock.return_value = {"query_error": "offline"}
        with self.assertRaises(jobs.JobBusyError):
            jobs.resume(response["job_id"])

    def test_reboot_interruption_resumes_identical_configuration(self):
        response, root, request = self.create()
        self.plan(root, {"complete": True, "samples": [{"sample_id": "s0"}]})
        before = (root / "plan.json").read_bytes()
        (root / "attempt-1-batch-0001.log").write_text("interrupted")
        resumed = jobs.resume(response["job_id"])
        new = jobs._read(root, Path(resumed["request_path"]).name)
        self.assertEqual(resumed["attempt"], 2)
        self.assertEqual(new["configuration"], request["configuration"])
        self.assertTrue(new["resuming"])
        self.assertEqual((root / "plan.json").read_bytes(), before)
        self.assertEqual((root / "attempt-1-batch-0001.log").read_text(), "interrupted")
        self.assertTrue((root / "attempt-1-status.json").is_file())

    def test_cancelled_attempt_resumes_without_cancellation_marker(self):
        response, root, _ = self.create()
        self.show_mock.return_value = LIVE
        with patch.object(jobs.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            jobs.cancel(response["job_id"])
        self.show_mock.return_value = MISSING
        self.assertEqual(jobs.status(response["job_id"])["state"], "cancelled")
        resumed = jobs.resume(response["job_id"])
        self.assertEqual(resumed["attempt"], 2)
        self.assertNotIn("cancel_requested_at", jobs._record(root))

    def test_resume_rejects_changed_configuration(self):
        response, _, _ = self.create()
        with self.assertRaises(ValueError):
            jobs.submit("resume", {"n": 3}, job_id=response["job_id"])

    def test_successful_resume_is_idempotent_without_launch(self):
        response, root, _ = self.create()
        self.receipt(response, root)
        self.launch_mock.reset_mock()
        same = jobs.resume(response["job_id"])
        self.assertTrue(same["already_finished"])
        self.assertEqual(same["attempt"], 1)
        self.launch_mock.assert_not_called()

    def test_split_resume_preserves_original_ratios(self):
        response, root, _ = self.create("sample")
        self.receipt(response, root, state="captured")
        split = jobs.submit("split", {"ratios": [1, 2, 3]}, job_id=response["job_id"])
        resumed = jobs.resume(split["job_id"])
        request = jobs._read(root, Path(resumed["request_path"]).name)
        self.assertEqual(request["action"], "split")
        self.assertEqual(request["payload"], {"ratios": [1, 2, 3]})

    def test_actual_supervisor_persists_actual_cpu_child_exit(self):
        response, root, _ = self.create("sample", count=1)
        self.plan(root, {"complete": True, "samples": [{}]})
        run = subprocess.run([sys.executable, str(worker.__file__), "--job-dir", str(root),
                              "--request", response["request_path"], "--action", "sample", "--supervise"],
                             capture_output=True, text=True, timeout=15, env=worker.clean_environment())
        self.assertEqual(run.returncode, 0, run.stderr)
        receipt = jobs._read(root, "attempt-1-exit.json")
        self.assertEqual(receipt["worker_returncode"], 0)
        self.assertEqual(jobs.status(response["job_id"])["state"], "sampled")

    def test_supervisor_nonzero_exit_snapshot_is_not_success(self):
        response, root, request = self.create()
        jobs._write(root, "status.json", {"job_id": root.name, "attempt": 1, "state": "completed"})
        with patch.object(worker, "run_child", return_value={"returncode": -11}):
            self.assertEqual(worker.supervise(root, request, Path(response["request_path"])), 139)
        self.assertEqual(jobs.status(response["job_id"])["state"], "failed")

    def test_sixty_samples_use_three_bounded_processes(self):
        response, root, request = self.create(count=60)
        self.plan(root, {"complete": True, "samples": [{}] * 60})
        committed = []
        calls = []
        def capture(plan, output, log, **kwargs):
            calls.append(log.name)
            committed.extend([{}] * min(20, 60 - len(committed)))
            return {"captured_count": len(committed), "process_exit": {"returncode": 0}}
        with patch("dataset.checkpoints.validate_capture", side_effect=lambda *a, **k: committed), \
                patch.object(worker, "gpu_slot", return_value=nullcontext()), \
                patch.object(worker, "run_capture", side_effect=capture):
            worker.DatasetAttempt(root, request).collect()
        self.assertEqual(calls, [f"attempt-1-batch-{i:04d}.log" for i in range(1, 4)])
        self.assertEqual(len(committed), 60)
        self.assertEqual(len(list(root.glob("attempt-1-batch-*-result.json"))), 3)

    def test_interrupted_batch_preserves_three_commits_and_resumes_next(self):
        response, root, request = self.create(count=60)
        self.plan(root, {"complete": True, "samples": [{}] * 60})
        (root / "dataset").mkdir()
        committed = [{"sample_id": str(i)} for i in range(3)]
        first = list(committed)
        calls = []
        def capture(plan, output, log, **kwargs):
            calls.append(log.name)
            committed.extend({"sample_id": str(i)} for i in range(len(committed), min(len(committed) + 20, 60)))
            return {"captured_count": len(committed), "process_exit": {"returncode": 0}}
        request["attempt"] = 2
        request["resuming"] = True
        with patch("dataset.checkpoints.validate_capture", side_effect=lambda *a, **k: committed), \
                patch.object(worker, "gpu_slot", return_value=nullcontext()), \
                patch.object(worker, "run_sampling") as sampling, \
                patch.object(worker, "run_capture", side_effect=capture):
            attempt = worker.DatasetAttempt(root, request)
            attempt.sample()
            attempt.collect()
        sampling.assert_not_called()
        self.assertEqual(committed[:3], first)
        self.assertEqual(len(committed), 60)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(name.startswith("attempt-2-") for name in calls))

    def test_zero_progress_batch_is_failure(self):
        _, root, request = self.create(count=1)
        self.plan(root, {"complete": True, "samples": [{}]})
        with patch("dataset.checkpoints.validate_capture", return_value=[]), \
                patch.object(worker, "gpu_slot", return_value=nullcontext()), \
                patch.object(worker, "run_capture", return_value={"captured_count": 0}):
            with self.assertRaisesRegex(RuntimeError, "no verified checkpoint progress"):
                worker.DatasetAttempt(root, request).collect()

    def test_worker_cannot_exceed_configured_batch_limit(self):
        _, root, request = self.create(count=60)
        self.plan(root, {"complete": True, "samples": [{}] * 60})
        with patch("dataset.checkpoints.validate_capture", return_value=[{}] * 60), \
                patch.object(worker, "gpu_slot", return_value=nullcontext()), \
                patch.object(worker, "run_capture", return_value={"captured_count": 60}):
            with self.assertRaisesRegex(RuntimeError, "exceeded configured batch size"):
                worker.DatasetAttempt(root, request).collect()

    def test_nonzero_real_child_not_replaced_by_successful_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            worker.run_child([sys.executable, "-c", "raise SystemExit(7)"], self.root / "failed.log", 5)

    def test_metadata_atomic_write_preserves_previous_on_publish_failure(self):
        _, root, _ = self.create()
        before = (root / "config.json").read_bytes()
        with patch.object(jobs.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                jobs._write(root, "config.json", {"n": 99})
        self.assertEqual((root / "config.json").read_bytes(), before)
        self.assertFalse(list(root.glob(".metadata-*")))

    def test_status_wait_does_not_allow_unbounded_or_nonfinite_wait(self):
        response, _, _ = self.create()
        for value in (-1, 31, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                jobs.status(response["job_id"], wait_seconds=value)

    def test_plan_replaced_between_sample_and_capture_is_rejected(self):
        _, root, request = self.create(count=1)
        self.plan(root, {"complete": True, "samples": [{"sample_id": "0"}]})
        jobs._write(root, "plan.json", {"complete": True, "samples": [{"sample_id": "changed"}]})
        with patch.object(worker, "run_capture") as capture:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                worker.DatasetAttempt(root, request).collect()
        capture.assert_not_called()

    def test_unbound_existing_plan_cannot_be_adopted(self):
        _, root, request = self.create(count=1)
        jobs._write(root, "plan.json", {"complete": True, "samples": [{}]})
        with self.assertRaisesRegex(ValueError, "configuration commit"):
            worker.DatasetAttempt(root, request).sample()
        with self.assertRaisesRegex(ValueError, "Cannot adopt"):
            worker.commit_plan(root, request, {"complete": True, "samples": [{}]})

    def test_plan_commit_gap_recovers_only_exact_original_plan(self):
        _, root, request = self.create(count=1)
        plan = {"complete": True, "samples": [{"sample_id": "original"}]}
        self.plan(root, plan)
        (root / "plan.json").unlink()
        with self.assertRaisesRegex(ValueError, "differs from its immutable commit"):
            worker.commit_plan(root, request, {"complete": True, "samples": [{}]})
        self.assertEqual(worker.commit_plan(root, request, plan), plan)

    def test_changed_job_configuration_invalidates_plan_commit(self):
        _, root, request = self.create(count=1)
        self.plan(root, {"complete": True, "samples": [{}]})
        jobs._write(root, "config.json", {"n": 2})
        with self.assertRaisesRegex(ValueError, "configuration commit"):
            worker.validate_plan_commit(root, request)

    def test_cancel_request_does_not_prove_unknown_unit_stopped(self):
        response, root, _ = self.create()
        record = jobs._record(root)
        record["cancel_requested_at"] = jobs._now()
        jobs._write(root, "job.json", record)
        self.show_mock.return_value = {"query_error": "offline"}
        self.assertEqual(jobs.status(response["job_id"])["state"], "unknown")
        with self.assertRaises(jobs.JobBusyError):
            jobs.resume(response["job_id"])

    def test_progress_reports_commits_without_publishing_completion(self):
        response, root, _ = self.create(count=60)
        self.plan(root, {"complete": True, "samples": [{}] * 60})
        output = root / "dataset"
        output.mkdir()
        jobs._write(output, "capture-manifest.json", {"plan_sha256": jobs._sha256(root / "plan.json"),
                                                     "requested_count": 60})
        jobs._write(output, "progress.json", {"phase": "captured_000002", "captured_count": 3,
                                             "requested_count": 60, "ok": True})
        self.show_mock.return_value = LIVE
        status = jobs.status(response["job_id"])
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["capture_progress"]["captured_count"], 3)
        self.assertNotIn("ok", status["capture_progress"])
        self.assertIn("queried_at", status)

    def test_progress_with_wrong_plan_or_symlink_is_rejected(self):
        response, root, _ = self.create(count=1)
        self.plan(root, {"complete": True, "samples": [{}]})
        output = root / "dataset"
        output.mkdir()
        jobs._write(output, "capture-manifest.json", {"plan_sha256": "wrong", "requested_count": 1})
        jobs._write(output, "progress.json", {"captured_count": 0, "requested_count": 1})
        with self.assertRaisesRegex(ValueError, "immutable job plan"):
            jobs.status(response["job_id"])
        (output / "progress.json").unlink()
        (output / "progress.json").symlink_to(root / "plan.json")
        with self.assertRaisesRegex(ValueError, "metadata path"):
            jobs.status(response["job_id"])

    def test_sampling_timeout_is_configurable_and_in_total_budget(self):
        _, root, request = self.create()
        request["runtime"]["sampling_timeout_seconds"] = 7200
        with patch.object(worker, "run_child", return_value={"returncode": 0}) as child:
            worker.run_sampling(root, request)
        self.assertEqual(child.call_args.args[2], 7200)
        old = dict(request["runtime"])
        old.pop("sampling_timeout_seconds")
        self.assertEqual(settings.job_timeout(request["runtime"], 60) - settings.job_timeout(old, 60), 6600)
        for value in (0, 86401, True, float("inf"), 10**500):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings.validate_runtime_config({"sampling_timeout_seconds": value})

    def test_descendant_keeps_execution_lock_if_worker_is_killed(self):
        _, root, _ = self.create()
        log = root / "descendant.log"
        child_code = "import os,time; print(os.getpid(), flush=True); time.sleep(20)"
        parent_code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(settings.PLUGIN_ROOT)!r}); "
            "from dataset import worker,jobs; "
            f"job=Path({str(root)!r}); "
            "lock=jobs._locked(job, '.execution.lock'); "
            "fd=lock.__enter__(); worker._PROCESS_LOCKS.add(fd); "
            f"worker.run_child([sys.executable, '-c', {child_code!r}], Path({str(log)!r}), 25)"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  env=worker.clean_environment())
        child_pid = None
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if log.exists() and log.read_text().strip():
                    child_pid = int(log.read_text().strip())
                    break
                time.sleep(.025)
            self.assertIsNotNone(child_pid, "owned CPU descendant did not start")
            self.assertTrue(jobs._executing(root))
            parent.kill()
            parent.wait(timeout=5)
            self.assertTrue(jobs._executing(root), "descendant lost the inherited execution lock")
            with self.assertRaises(jobs.JobBusyError):
                jobs.resume(root.name)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if child_pid:
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5
                while jobs._executing(root) and time.monotonic() < deadline:
                    time.sleep(.025)
        self.assertFalse(jobs._executing(root))

    def test_memory_limit_matches_renderer_contract_before_launch(self):
        for value in (1, 9, 9.5, 11, 1024):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be 10"):
                settings.validate_runtime_config({"memory_max_gb": value})
        self.assertEqual(settings.validate_runtime_config({"memory_max_gb": 10})["memory_max_gb"], 10)
        config = json.loads(self.config.read_text())
        config["memory_max_gb"] = 11
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "must be 10"):
            jobs.submit("pipeline", {"scene_path": "/fixture.usda", "n": 1})
        self.launch_mock.assert_not_called()

    def test_runtime_timeout_scales_with_sample_and_batch_count(self):
        runtime = settings.load_runtime_config()
        self.assertGreater(settings.job_timeout(runtime, 60), settings.job_timeout(runtime, 32))
        self.assertEqual(settings.batch_timeout(runtime, 20), 4200)
        for value in (True, 0, 1.5, 1001, 10**500):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings.validate_runtime_config({"batch_size": value})


if __name__ == "__main__":
    unittest.main()
