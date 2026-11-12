"""Host CPU contract checks; no simulation launches or systemd mutations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import jobs, tools, worker, settings


EXIT_OK = {"LoadState": "loaded", "ActiveState": "active", "SubState": "exited",
           "Result": "success", "ExecMainCode": "1", "ExecMainStatus": "0"}
LIVE = {**EXIT_OK, "SubState": "running", "ExecMainCode": "0"}


class JobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        runtime = {"isaac_python": sys.executable, "run_root": str(root),
                   "gpu_lock_path": str(root/"gpu.lock")}
        config = root/"runtime.json"
        config.write_text(json.dumps(runtime))
        self.env = patch.dict(os.environ, {"ISAAC_DATASET_RUN_ROOT": str(root),
                                          "ISAAC_DATASET_CONFIG": str(config)})
        self.env.start()
        self.launch = patch.object(jobs, "_launch", return_value=True)
        self.launch_mock = self.launch.start()

    def tearDown(self):
        self.launch.stop()
        self.env.stop()
        self.directory.cleanup()

    def create(self, action="sample"):
        result = jobs.submit(action, {"scene_path": "/fixture.usda", "n": 32})
        return result, Path(result["job_dir"])

    def finish(self, result, directory, state):
        jobs._write(directory, "status.json", {"job_id": result["job_id"],
                    "attempt": result["attempt"], "state": state})

    def test_new_job_immutable_request_and_no_arbitrary_action(self):
        result, directory = self.create()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["state"], "pending")
        self.assertEqual(json.loads((directory/"config.json").read_text())["n"], 32)
        with self.assertRaises(FileExistsError):
            jobs._write(directory, "config.json", {}, immutable=True)
        with self.assertRaises(ValueError):
            jobs.submit("shell", {"cmd": "ls"})

    def test_invalid_ids_and_symlink_escape(self):
        for identifier in ("../escape", "a"*31, "A"*32, None, "x.service"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                jobs.status(identifier)
        path = Path(self.directory.name)/("a"*32)
        path.symlink_to("/tmp", target_is_directory=True)
        with self.assertRaises(ValueError):
            jobs.status(path.name)

    def test_unowned_recorded_unit_is_rejected(self):
        result, directory = self.create()
        record = jobs._read(directory, "job.json")
        record["unit"] = "isaac-bridge.service"
        jobs._write(directory, "job.json", record)
        with self.assertRaises(ValueError):
            jobs.cancel(result["job_id"])

    def test_recorded_id_mismatch_rejected(self):
        result, directory = self.create()
        record = jobs._read(directory, "job.json")
        record["job_id"] = "a"*32
        jobs._write(directory, "job.json", record)
        with self.assertRaises(ValueError):
            jobs.status(result["job_id"])

    def test_metadata_symlinks_rejected(self):
        result, directory = self.create()
        (directory/"status.json").unlink()
        (directory/"status.json").symlink_to(directory/"config.json")
        with self.assertRaises(ValueError):
            jobs.status(result["job_id"])

    def test_success_requires_normal_exit(self):
        result, directory = self.create()
        self.finish(result, directory, "sampled")
        for properties, expected in ((LIVE, "running"), (EXIT_OK, "sampled"),
                ({**EXIT_OK, "ExecMainStatus": "1"}, "failed"),
                ({"query_error": "offline"}, "unknown"),
                ({**EXIT_OK, "LoadState": "not-found"}, "failed")):
            with self.subTest(properties=properties), patch.object(jobs, "_show", return_value=properties):
                self.assertEqual(jobs.status(result["job_id"])["state"], expected)

    def test_stale_worker_cannot_finish_attempt(self):
        result, directory = self.create()
        jobs._write(directory, "status.json", {"job_id": result["job_id"],
                                             "attempt": 99, "state": "completed"})
        with patch.object(jobs, "_show", return_value=EXIT_OK):
            status = jobs.status(result["job_id"])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["error"], "worker_result_missing_or_incomplete")

    def test_busy_transition_rejected(self):
        result, _ = self.create()
        with patch.object(jobs, "_show", return_value=LIVE), self.assertRaises(jobs.JobBusyError):
            jobs.submit("collect", {}, job_id=result["job_id"])

    def test_transition_sample_collect_split_and_no_restart(self):
        result, directory = self.create()
        self.finish(result, directory, "sampled")
        with patch.object(jobs, "_show", return_value=EXIT_OK):
            with self.assertRaises(ValueError):
                jobs.submit("split", {}, job_id=result["job_id"])
            second = jobs.submit("collect", {}, job_id=result["job_id"])
            self.assertEqual(second["attempt"], 2)
            self.finish(second, directory, "captured")
            third = jobs.submit("split", {"ratios": [4,2,2]}, job_id=result["job_id"])
            self.assertEqual(third["attempt"], 3)
            self.assertEqual(jobs._read(directory, "config.json")["n"], 32)
            self.finish(third, directory, "completed")
            with self.assertRaises(ValueError):
                jobs.submit("split", {}, job_id=result["job_id"])
            with self.assertRaises(ValueError):
                jobs.submit("pipeline", {}, job_id=result["job_id"])

    def test_failed_capture_cannot_overwrite(self):
        result, directory = self.create()
        self.finish(result, directory, "failed")
        with patch.object(jobs, "_show", return_value=EXIT_OK), self.assertRaises(ValueError):
            jobs.submit("collect", {}, job_id=result["job_id"])

    def test_cancel_only_owned_recorded_unit(self):
        result, _ = self.create()
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(jobs, "_show", return_value=LIVE), patch.object(jobs.subprocess, "run", return_value=completed) as run:
            cancelled = jobs.cancel(result["job_id"])
        self.assertEqual(cancelled["state"], "cancelling")
        self.assertEqual(run.call_args.args[0],
                         ["systemctl", "--user", "stop", "--no-block", result["unit"]])
        with patch.object(jobs, "_show", return_value={**EXIT_OK, "ActiveState": "inactive", "SubState": "dead"}):
            self.assertEqual(jobs.status(result["job_id"])["state"], "cancelled")

    def test_completed_job_cancel_is_noop(self):
        result, directory = self.create()
        self.finish(result, directory, "completed")
        with patch.object(jobs, "_show", return_value=EXIT_OK), patch.object(jobs.subprocess, "run") as run:
            cancelled = jobs.cancel(result["job_id"])
        self.assertFalse(cancelled["cancel_requested"])
        run.assert_not_called()

    def test_worker_command_clears_python_pollution(self):
        result, directory = self.create()
        record = jobs._record(directory)
        with patch.dict(os.environ, {"ISAAC_DATASET_RUN_ROOT": self.directory.name}):
            command = jobs._worker_command(directory, record, Path(result["request_path"]))
        self.assertEqual(command[:2], ["/usr/bin/env", "-u"])
        self.assertIn("PYTHONPATH", command)
        self.assertIn("PYTHONHOME", command)
        self.assertIn("PYTHONNOUSERSITE=1", command)
        self.assertIn("ISAAC_DATASET_RUN_ROOT="+str(directory.parent), command)
        self.assertEqual(command[-5:], ["--action", "sample", "--request", result["request_path"] , "--supervise"])

    def test_launch_is_bounded(self):
        result, directory = self.create()
        record = jobs._record(directory)
        completed = subprocess.CompletedProcess([], 0, "", "")
        self.launch.stop()
        with patch.object(jobs, "_worker_command", return_value=["/fixture/python"]), patch.object(jobs.subprocess, "run", return_value=completed) as run:
            self.assertTrue(jobs._launch(directory, record, Path(result["request_path"])))
        command = run.call_args.args[0]
        for expected in ("--property=MemoryMax=10G", "--property=MemorySwapMax=0",
                         f"--property=RuntimeMaxSec={settings.job_timeout(settings.load_runtime_config(), 32)}", "--property=KillMode=control-group"):
            self.assertIn(expected, command)

    def test_job_lock_rejects_second_caller(self):
        _, directory = self.create()
        with jobs._locked(directory), self.assertRaises(jobs.JobBusyError):
            with jobs._locked(directory):
                pass


class ToolsTests(unittest.TestCase):
    def test_register_only_unique_dataset_names(self):
        registered = []
        tools.register(types.SimpleNamespace(register_tool=lambda **kwargs: registered.append(kwargs)))
        self.assertEqual(len(registered), 9)
        self.assertEqual(len({item["name"] for item in registered}), 9)
        self.assertTrue(all(item["toolset"] == "isaac-image-dataset" for item in registered))
        self.assertTrue(all(item["name"].startswith("isaac_dataset_") for item in registered))

    def test_sample_dispatch_and_no_isaac_import(self):
        with patch.object(jobs, "submit", return_value={"accepted": True}) as submit:
            result = json.loads(tools.handler("sample")({"config": {"scene_path": "/scene.usda"}}))
        self.assertTrue(result["accepted"])
        submit.assert_called_once_with("sample", {"scene_path": "/scene.usda"})

    def test_bad_configs_rejected_before_dispatch(self):
        cases = [ {"config": {"scene_path": "/a.usda", "cmd": "ls"}},
                 {"config": {"scene_path": "/a.usda", "n": True}},
                 {"config": {"scene_path": "/a.usda", "seed": -1}},
                 {"config": {"scene_path": "/a.usda", "camera": {"pose": []}}},
                 {"config": {"scene_path": "/a.usda", "yaw_deg": [0, float("nan")]}},
                 {"config": {"scene_path": "/a.usda", "classes": ["grass"]}},
                 {"config": {"scene_path": "/a.usda"}, "job_id": "a"*32}]
        for args in cases:
            with self.subTest(args=args), patch.object(jobs, "submit") as submit:
                result = json.loads(tools.handler("sample")(args))
                self.assertEqual(result["error"], "invalid_arguments")
                submit.assert_not_called()

    def test_camera_limits_match_render_contract(self):
        for camera in ({"resolution": [8192, 720]}, {"resolution": [1280, 8192]},
                       {"horizontal_fov_deg": 1}, {"horizontal_fov_deg": 1.99}):
            with self.subTest(camera=camera), patch.object(jobs, "submit") as submit:
                result = json.loads(tools.handler("sample")({
                    "config": {"scene_path": "/scene.usda", "camera": camera}}))
                self.assertEqual(result["error"], "invalid_arguments")
                submit.assert_not_called()
        for camera in ({"resolution": [16, 4096]}, {"horizontal_fov_deg": 2},
                       {"horizontal_fov_deg": 175}):
            with self.subTest(camera=camera), patch.object(jobs, "submit", return_value={"accepted": True}) as submit:
                result = json.loads(tools.handler("sample")({
                    "config": {"scene_path": "/scene.usda", "camera": camera}}))
                self.assertTrue(result["accepted"])
                submit.assert_called_once()

    def test_continuation_rejects_configuration_and_invalid_weights(self):
        for action, args in (("collect", {"job_id": "a"*32, "config": {}}),
                             ("split", {"job_id": "a"*32, "ratios": [0,0,0]}),
                             ("status", {"job_id": "../a"}),
                             ("cancel", {"unit": "bridge.service"})):
            with self.subTest(action=action):
                self.assertEqual(json.loads(tools.handler(action)(args))["error"], "invalid_arguments")

    def test_split_has_optional_ratios(self):
        with patch.object(jobs, "submit", return_value={}) as submit:
            tools.dispatch("split", {"job_id": "a"*32, "ratios": [1,1,1]})
        submit.assert_called_once_with("split", {"ratios": [1,1,1]}, job_id="a"*32)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.job = Path(self.directory.name)
        runtime = settings.validate_runtime_config({"isaac_python": sys.executable,
                  "run_root": str(self.job/"runs"), "gpu_lock_path": str(self.job/"gpu.lock")})
        config = self.job/"runtime.json"
        config.write_text(json.dumps(runtime))
        self.env = patch.dict(os.environ, {"ISAAC_DATASET_CONFIG": str(config),
                                          "ISAAC_DATASET_RUN_ROOT": runtime["run_root"]})
        self.env.start()
        self.request = {"job_id": "a"*32, "attempt": 1, "action": "pipeline",
                        "configuration": {}, "payload": {}, "runtime": runtime}
        jobs._write(self.job, "config.json", self.request["configuration"], immutable=True)
        self.attempt = worker.DatasetAttempt(self.job, self.request)

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def test_incomplete_plan_saved_but_capture_never_called(self):
        fake = types.ModuleType("dataset.sampling")
        fake.generate_plan = lambda config: {"complete": False, "samples": [], "statistics": {}}
        with patch.dict(sys.modules, {"dataset.sampling": fake}), patch.object(worker, "run_sampling", side_effect=worker.sample_plan), patch.object(self.attempt, "collect") as collect:
            with self.assertRaises(RuntimeError):
                self.attempt.execute()
        collect.assert_not_called()
        self.assertFalse(json.loads((self.job/"plan.json").read_text())["complete"])

    def test_complete_sample_is_sampled(self):
        fake = types.ModuleType("dataset.sampling")
        fake.generate_plan = lambda config: {"complete": True, "samples": []}
        self.request["action"] = "sample"
        with patch.dict(sys.modules, {"dataset.sampling": fake}), patch.object(worker, "run_sampling", side_effect=worker.sample_plan):
            result = self.attempt.execute()
        self.assertEqual(result["state"], "sampled")

    def test_capture_rejects_incomplete_plan_and_existing_output(self):
        jobs._write(self.job, "plan.json", {"complete": False})
        with self.assertRaises(ValueError):
            self.attempt.collect()
        (self.job / "plan.json").unlink()
        worker.commit_plan(self.job, self.request, {"complete": True, "samples": []})
        (self.job/"dataset").mkdir()
        with self.assertRaises(ValueError):
            self.attempt.collect()

    def test_split_false_receipt_cannot_complete(self):
        worker.commit_plan(self.job, self.request, {"complete": True, "samples": []})
        fake = types.ModuleType("dataset.splitting")
        fake.finalize_dataset = lambda *args, **kwargs: {"ok": False}
        with patch.dict(sys.modules, {"dataset.splitting": fake}), self.assertRaises(RuntimeError):
            self.attempt.split()

    def test_pipeline_calls_same_stage_methods(self):
        with patch.object(self.attempt, "sample") as sample, patch.object(self.attempt, "collect") as collect, patch.object(self.attempt, "split", return_value={"ok": True}) as split:
            result = self.attempt.execute()
        sample.assert_called_once()
        collect.assert_called_once()
        split.assert_called_once()
        self.assertEqual(result["state"], "completed")

    def test_capture_receipt_requires_every_completion_fact(self):
        plan = self.job/"plan.json"
        plan.write_text(json.dumps({"samples": [{}, {}]}))
        output = self.job/"dataset"
        output.mkdir()
        base = {"ok": True, "captured_count": 2, "requested_count": 2,
                "app_closed": True, "source_unchanged": True, "phase": "complete", "batch_complete": True}
        with patch.object(worker, "run_child", return_value={"returncode": 0}):
            (output/"result.json").write_text(json.dumps(base))
            self.assertEqual(worker.run_capture(plan, output, self.job/"capture.log"), dict(base, process_exit={"returncode": 0}))
            for field, wrong in (("ok", False), ("captured_count", 1),
                                 ("requested_count", 3), ("app_closed", False),
                                 ("source_unchanged", False), ("phase", "rendering")):
                with self.subTest(field=field):
                    (output/"result.json").write_text(json.dumps({**base, field: wrong}))
                    with self.assertRaises(RuntimeError):
                        worker.run_capture(plan, output, self.job/"capture.log")

    def test_real_cpu_child_cleans_hermes_python_environment(self):
        polluted = {"PYTHONPATH": "/invalid/hermes", "PYTHONHOME": "/invalid/home",
                    "PYTHONSTARTUP": "/invalid/startup", "PYTHONUSERBASE": "/invalid/base",
                    "VIRTUAL_ENV": "/invalid/virtual"}
        code = ("import os,json; print(json.dumps({k:os.environ.get(k) "
                "for k in ['PYTHONPATH','PYTHONHOME','PYTHONSTARTUP',"
                "'PYTHONUSERBASE','VIRTUAL_ENV','PYTHONNOUSERSITE']}))")
        log = self.job/"clean-child.log"
        with patch.dict(os.environ, polluted):
            worker.run_child([sys.executable, "-c", code], log, 10)
        result = json.loads(log.read_text())
        self.assertEqual(result["PYTHONNOUSERSITE"], "1")
        self.assertTrue(all(result[key] is None for key in polluted))

    def test_timed_out_child_only_terminates_its_own_process_group(self):
        child = unittest.mock.Mock()
        child.pid = 424242
        child.poll.return_value = None
        with patch.object(worker.subprocess, "Popen", return_value=child), patch.object(worker.time, "monotonic", side_effect=[0, 2]), patch.object(worker.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                worker.run_child(["/fixture/python"], self.job/"timeout.log", 1)
        kill.assert_called_once_with(child.pid, worker.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=15)

    def test_existing_gpu_process_rejects_without_killing_anything(self):
        result = subprocess.CompletedProcess([], 0, "12345\\n", "")
        with patch.object(worker.settings, "load_runtime_config", return_value=self.request["runtime"]), patch.object(worker, "available_ram", return_value=20*1024**3), patch.object(worker.subprocess, "run", return_value=result), patch.object(worker.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                with worker.gpu_slot():
                    self.fail("must not enter")
        kill.assert_not_called()

    def test_shared_gpu_lock_and_low_memory_guard(self):
        self.assertEqual(self.request["runtime"]["gpu_lock_path"], str(self.job/"gpu.lock"))
        with patch.object(worker.settings, "load_runtime_config", return_value=self.request["runtime"]), patch.object(worker, "available_ram", return_value=1024):
            with self.assertRaises(RuntimeError):
                with worker.gpu_slot():
                    self.fail("must not enter")

    def test_shared_gpu_lock_rejects_concurrent_owner(self):
        with patch.object(worker.settings, "load_runtime_config", return_value=self.request["runtime"]), patch.object(worker, "available_ram", return_value=20*1024**3), patch.object(worker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with worker.gpu_slot(), self.assertRaises(RuntimeError):
                with worker.gpu_slot():
                    self.fail("must not enter")


if __name__ == "__main__":
    unittest.main()
