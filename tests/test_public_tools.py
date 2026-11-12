"""Public contract tests run without launching Isaac or a systemd worker."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import tools


class PublicToolsTests(unittest.TestCase):
    job_id = "a" * 32

    def test_standalone_registration(self):
        class Context:
            def __init__(self):
                self.tools = []
            def register_tool(self, **tool):
                self.tools.append(tool)
        context = Context()
        tools.register(context)
        self.assertEqual(len(context.tools), 9)
        self.assertEqual({t["toolset"] for t in context.tools}, {"isaac-image-dataset"})
        self.assertIn("isaac_dataset_resume", {t["name"] for t in context.tools})
        self.assertIn("isaac_dataset_get_config", {t["name"] for t in context.tools})
        for tool in context.tools:
            self.assertFalse(tool["schema"]["parameters"]["additionalProperties"])

    def test_resume_uses_existing_job(self):
        with patch.object(tools.jobs, "resume", return_value={"accepted": True}) as resume:
            self.assertTrue(tools.dispatch("resume", {"job_id": self.job_id})["accepted"])
            resume.assert_called_once_with(self.job_id)

    def test_status_wait_is_bounded(self):
        with patch.object(tools.jobs, "status", return_value={"state": "running"}) as status:
            for seconds in (0, .5, 30):
                tools.dispatch("status", {"job_id": self.job_id, "wait_seconds": seconds})
                status.assert_called_with(self.job_id, wait_seconds=seconds)
            tools.dispatch("status", {"job_id": self.job_id})
            status.assert_called_with(self.job_id, wait_seconds=0)
            for seconds in (-1, 30.1, True, float("nan"), float("inf"), 10**1000):
                with self.subTest(seconds=str(seconds)[:30]), self.assertRaises(ValueError):
                    tools.dispatch("status", {"job_id": self.job_id, "wait_seconds": seconds})

    def test_default_scene_and_caller_arguments_are_preserved(self):
        args = {"config": {"n": 60, "camera": {"horizontal_fov_deg": 120}}}
        original = copy.deepcopy(args)
        with patch.object(tools.settings, "load_runtime_config", return_value={"default_scene": "/scene.usda"}), patch.object(tools.jobs, "submit") as submit:
            tools.dispatch("pipeline", args)
            self.assertEqual(submit.call_args.args, ("pipeline", {**args["config"], "scene_path": "/scene.usda"}))
        self.assertEqual(args, original)

    def test_explicit_scene_does_not_need_runtime_default(self):
        with patch.object(tools.settings, "load_runtime_config", side_effect=AssertionError("unused")), patch.object(tools.jobs, "submit") as submit:
            tools.dispatch("sample", {"config": {"scene_path": "/chosen.usda"}})
            submit.assert_called_once_with("sample", {"scene_path": "/chosen.usda"})

    def test_missing_scene_has_actionable_error(self):
        with patch.object(tools.settings, "load_runtime_config", return_value={}):
            with self.assertRaisesRegex(ValueError, "runtime default_scene"):
                tools.dispatch("pipeline", {"config": {}})

    def test_get_config_is_read_only(self):
        runtime = {"default_scene": "/scene.usda", "batch_size": 16}
        with patch.object(tools.settings, "load_runtime_config", return_value=runtime), patch.object(tools.jobs, "submit") as submit:
            result = tools.dispatch("get_config", {})
            self.assertEqual(result["runtime"], runtime)
            self.assertEqual(result["sampling"]["n_maximum"], 10000)
            self.assertEqual(result["sampling"]["pitch_deg_default"], [-5, 10])
            self.assertEqual(result["camera"]["horizontal_fov_deg_default"], 90)
            submit.assert_not_called()

    def test_template_camera_schema_is_public(self):
        config = {"scene_path": "/scene.usda", "camera": {"template_prim": "/World/Camera", "resolution": [1280, 720]}}
        with patch.object(tools.jobs, "submit") as submit:
            tools.dispatch("sample", {"config": config})
            submit.assert_called_once_with("sample", config)
        for path in ("World/Camera", "/", "/World.camera", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                tools.dispatch("sample", {"config": {**config, "camera": {"template_prim": path}}})

    def test_config_rejects_nonfinite_boolean_and_unknown_fields(self):
        configs = [{"n": True}, {"n": 10001}, {"pitch_deg": [0, float("nan")]},
                   {"camera": {"horizontal_fov_deg": float("inf")}},
                   {"camera": {"unexpected": 1}}, {"render": {}},
                   {"split_ratios": [0, 0, 0]}, {"split_ratios": [1e308, 1e308, 1e308]}]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                tools.dispatch("pipeline", {"config": {"scene_path": "/scene.usda", **config}})

    def test_handler_serializes_errors_and_busy(self):
        handle = tools.handler("resume")
        self.assertEqual(json.loads(handle({"job_id": "../other"}))["error"], "invalid_arguments")
        with patch.object(tools.jobs, "resume", side_effect=tools.jobs.JobBusyError("already running")):
            self.assertEqual(json.loads(handle({"job_id": self.job_id}))["error"], "job_busy")

    def test_unknown_action_and_extra_fields_rejected(self):
        for action, args in (("unsupported", {}), ("resume", {"job_id": self.job_id, "config": {}}),
                             ("get_config", {"job_id": self.job_id})):
            with self.subTest(action=action), self.assertRaises(ValueError):
                tools.dispatch(action, args)


if __name__ == "__main__":
    unittest.main()
