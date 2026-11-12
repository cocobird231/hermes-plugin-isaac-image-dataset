"""Public camera pool contract and sampling invariance, CPU/OpenUSD only."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import tools
from dataset.sampling import generate_plan, normalize_config
import test_dataset_sampling as fixtures


class MulticamPublicTests(unittest.TestCase):
    def test_sample_and_pipeline_publish_bounded_camera_count(self):
        for action in ("sample", "pipeline"):
            with self.subTest(action=action):
                config = tools.parameters(action)["properties"]["config"]
                schema = config["properties"]["camera_count"]
                self.assertEqual(schema["type"], "integer")
                self.assertEqual((schema["minimum"], schema["maximum"], schema["default"]), (1, 8, 1))
                self.assertIn("concurrent reusable cameras", schema["description"])
                self.assertIn("runtime.batch_size", schema["description"])

    def test_valid_counts_forwarded_without_mutating_caller(self):
        for action in ("sample", "pipeline"):
            for count in (1, 2, 4, 8):
                args = {"config": {"scene_path": "/scene.usda", "n": 3, "camera_count": count}}
                original = copy.deepcopy(args)
                with self.subTest(action=action, count=count), patch.object(tools.jobs, "submit") as submit:
                    tools.dispatch(action, args)
                    submit.assert_called_once_with(action, original["config"])
                self.assertEqual(args, original)

    def test_invalid_counts_rejected_before_job_creation(self):
        invalid = (0, 9, -1, True, False, 1.0, 2.5, "4", None, float("nan"), float("inf"), 10**1000)
        with patch.object(tools.jobs, "submit") as submit:
            for action in ("sample", "pipeline"):
                for count in invalid:
                    with self.subTest(action=action, count=str(count)[:30]), self.assertRaises(ValueError):
                        tools.dispatch(action, {"config": {"scene_path": "/scene.usda", "camera_count": count}})
            submit.assert_not_called()

    def test_camera_count_is_top_level_config_not_intrinsic_or_runtime_override(self):
        invalid = ({"camera": {"camera_count": 4}}, {"batch_size": 4})
        with patch.object(tools.jobs, "submit") as submit:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    tools.dispatch("pipeline", {"config": {"scene_path": "/scene.usda", **config}})
            submit.assert_not_called()

    def test_get_config_explains_count_without_creating_job(self):
        runtime = {"batch_size": 20}
        with patch.object(tools.settings, "load_runtime_config", return_value=runtime), patch.object(tools.jobs, "submit") as submit:
            result = tools.dispatch("get_config", {})
            camera = result["camera"]
            self.assertEqual((camera["camera_count_minimum"], camera["camera_count_maximum"], camera["camera_count_default"]), (1, 8, 1))
            self.assertEqual(result["runtime"], {"batch_size": 20})
            self.assertIn("config.n is total images", camera["camera_count_semantics"])
            self.assertIn("runtime.batch_size", camera["camera_count_semantics"])
            submit.assert_not_called()

    def test_collect_and_resume_cannot_override_immutable_camera_count(self):
        with patch.object(tools.jobs, "submit") as submit, patch.object(tools.jobs, "resume") as resume:
            for action in ("collect", "resume"):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    tools.dispatch(action, {"job_id": "a"*32, "camera_count": 4})
            submit.assert_not_called()
            resume.assert_not_called()


class MulticamSamplingTests(unittest.TestCase):
    setUp = fixtures.SamplingTests.setUp
    fixture = fixtures.SamplingTests.fixture

    def test_omission_normalizes_to_one_without_changing_input(self):
        config = self.fixture()
        original = copy.deepcopy(config)
        self.assertEqual(normalize_config(config)["camera_count"], 1)
        self.assertEqual(config, original)
        self.assertEqual(generate_plan(config), generate_plan({**config, "camera_count": 1}))

    def test_normalization_rejects_non_integer_and_out_of_bounds(self):
        config = self.fixture()
        for count in (0, 9, -1, True, False, 1.0, 2.5, "4", None, float("nan"), float("inf"), 10**1000):
            with self.subTest(count=str(count)[:30]), self.assertRaisesRegex(ValueError, "camera_count"):
                normalize_config({**config, "camera_count": count})
        for count in (1, 2, 4, 8):
            self.assertEqual(normalize_config({**config, "camera_count": count})["camera_count"], count)

    def test_pool_size_never_changes_rng_pose_grouping_or_total_count(self):
        config = {**self.fixture(obstacle=True), "n": 11, "min_distance_m": 0.4,
                  "group_distance_m": 3., "clearance_m": .5, "max_attempts": 200}
        baseline = generate_plan({**config, "camera_count": 1})
        self.assertTrue(baseline["complete"])
        self.assertEqual(len(baseline["samples"]), 11)
        for count in (2, 4, 8):
            with self.subTest(count=count):
                plan = generate_plan({**config, "camera_count": count})
                self.assertTrue(plan["complete"])
                self.assertEqual(plan["samples"], baseline["samples"])
                self.assertEqual(plan["statistics"], baseline["statistics"])
                self.assertEqual(plan["scene"], baseline["scene"])
                self.assertEqual(plan["config"]["camera_count"], count)

    def test_fewer_samples_than_cameras_does_not_multiply_images(self):
        config = {**self.fixture(), "n": 3, "camera_count": 8}
        plan = generate_plan(config)
        self.assertTrue(plan["complete"])
        self.assertEqual(len(plan["samples"]), 3)
        self.assertEqual(plan["statistics"]["requested_count"], 3)
        self.assertEqual(plan["config"]["camera_count"], 8)


if __name__ == "__main__":
    unittest.main()
