"""CPU/OpenUSD render policy, scheduling and immutable-plan compatibility tests."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.quality import DEFAULT_RENDER_QUALITY, normalize_render_quality, render_policy
from dataset.sampling import generate_plan, normalize_config
import test_dataset_render as render_fixtures
import test_dataset_sampling as sampling_fixtures
from test_multicam_backend import Runtime, backend, poses


class QualitySamplingTests(unittest.TestCase):
    setUp = sampling_fixtures.SamplingTests.setUp
    fixture = sampling_fixtures.SamplingTests.fixture

    def test_new_plan_default_is_normal_and_input_is_unchanged(self):
        config = self.fixture()
        original = copy.deepcopy(config)
        normalized = normalize_config(config)
        self.assertEqual(DEFAULT_RENDER_QUALITY, "normal")
        self.assertEqual(normalized["render_quality"], "normal")
        self.assertEqual(config, original)
        self.assertEqual(generate_plan(config), generate_plan({**config, "render_quality": "normal"}))

    def test_presets_do_not_change_rng_geometry_poses_or_groups(self):
        config = {**self.fixture(obstacle=True), "camera_count": 4, "n": 11,
                  "min_distance_m": .4, "group_distance_m": 3., "clearance_m": .5,
                  "max_attempts": 200}
        baseline = generate_plan(config)
        self.assertTrue(baseline["complete"])
        for mode, updates in (("efficiency", 0), ("normal", 1), ("quality", 10)):
            with self.subTest(mode=mode):
                plan = generate_plan({**config, "render_quality": mode})
                self.assertEqual(plan["samples"], baseline["samples"])
                self.assertEqual(plan["statistics"], baseline["statistics"])
                self.assertEqual(plan["scene"], baseline["scene"])
                self.assertEqual(render_policy(plan["config"])["shared_updates"], updates)

    def test_reject_invalid_values_without_coercion(self):
        config = self.fixture()
        for value in (None, True, 0, 1, 10, "NORMAL", "", "fast", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "render_quality"):
                normalize_config({**config, "render_quality": value})
            with self.assertRaisesRegex(ValueError, "render_quality"):
                normalize_render_quality(value)


class QualityCompatibilityTests(unittest.TestCase):
    def test_old_pool_plan_remains_ten_updates_without_inserting_default(self):
        plan = render_fixtures.plan()
        plan["config"]["camera_count"] = 4
        original = copy.deepcopy(plan)
        render_fixtures.worker.validate_plan(plan)
        self.assertEqual(plan, original)
        self.assertEqual(render_policy(plan["config"]), {
            "backend": "shared_render_products", "render_quality": "quality",
            "render_quality_source": "legacy_camera_pool", "shared_updates": 10})

    def test_old_single_camera_plan_keeps_legacy_backend(self):
        plan = render_fixtures.plan()
        original = copy.deepcopy(plan)
        render_fixtures.worker.validate_plan(plan)
        policy = render_policy(plan["config"])
        self.assertEqual(policy["backend"], "legacy_camera_renderer")
        self.assertEqual(policy["render_quality"], "legacy")
        self.assertIsNone(policy["shared_updates"])
        self.assertEqual(plan, original)

    def test_explicit_presets_validate_without_changing_committed_plan(self):
        for mode, updates in (("efficiency", 0), ("normal", 1), ("quality", 10)):
            with self.subTest(mode=mode):
                plan = render_fixtures.plan()
                plan["config"].update(camera_count=4, render_quality=mode)
                original = copy.deepcopy(plan)
                render_fixtures.worker.validate_plan(plan)
                self.assertEqual(plan, original)
                self.assertEqual(render_policy(plan["config"])["shared_updates"], updates)
                self.assertEqual(render_policy(plan["config"])["render_quality_source"], "plan")

    def test_invalid_or_incompatible_committed_mode_is_rejected(self):
        for mode in (None, 0, True, "invalid"):
            plan = render_fixtures.plan()
            plan["config"].update(camera_count=4, render_quality=mode)
            with self.assertRaisesRegex(ValueError, "render_quality"):
                render_fixtures.worker.validate_plan(plan)
        plan = render_fixtures.plan()
        plan["config"]["render_quality"] = "normal"
        with self.assertRaisesRegex(ValueError, "requires camera_count"):
            render_fixtures.worker.validate_plan(plan)


class QualitySchedulingTests(unittest.TestCase):
    def test_each_preset_updates_entire_four_camera_group_after_one_step(self):
        for mode, updates in (("efficiency", 0), ("normal", 1), ("quality", 10)):
            with self.subTest(mode=mode), Runtime(4) as runtime:
                policy = render_policy({"camera_count": 4, "render_quality": mode})
                pool = backend.BatchCameraRenderer(runtime.stage, runtime, runtime.paths,
                                                   (3, 2), runtime.check_time,
                                                   post_step_updates=policy["shared_updates"])
                try:
                    self.assertEqual(sum(e[0] == "update" for e in runtime.events), 5)
                    self.assertEqual(pool.init_render_calls, 1)
                    runtime.events.clear()
                    frames, timing = pool.capture(poses(11, 22, 33, 44), 12)
                    self.assertEqual(sum(e[0] == "step" for e in runtime.events), 1)
                    self.assertEqual(sum(e[0] == "update" for e in runtime.events), updates)
                    self.assertEqual(timing["post_step_updates_per_attempt"], updates)
                    self.assertEqual(timing["post_step_update_calls"], updates)
                    self.assertEqual(pool.post_step_update_calls, updates)
                    first_read = next(i for i, event in enumerate(runtime.events) if event[0] == "read")
                    self.assertTrue(all(e[0] not in ("step", "update") for e in runtime.events[first_read:]))
                    for frame, x in zip(frames, (11, 22, 33, 44)):
                        np.testing.assert_array_equal(frame["rgb"][0, 0], [x, x+1, 7])
                finally:
                    pool.close()

    def test_retry_counts_actual_updates_and_keeps_requested_mode(self):
        for updates in (0, 1, 10):
            with self.subTest(updates=updates), Runtime(4) as runtime:
                pool = backend.BatchCameraRenderer(runtime.stage, runtime, runtime.paths,
                                                   (3, 2), runtime.check_time,
                                                   post_step_updates=updates)
                try:
                    runtime.events.clear()
                    runtime.missing_reads = 1
                    frames, timing = pool.capture(poses(11, 22, 33, 44), 12)
                    self.assertEqual(len(frames), 4)
                    self.assertEqual(timing["attempts"], 2)
                    self.assertEqual(sum(e[0] == "step" for e in runtime.events), 2)
                    self.assertEqual(sum(e[0] == "update" for e in runtime.events), 2*updates)
                    self.assertEqual(timing["post_step_update_calls"], 2*updates)
                    self.assertEqual(pool.post_step_update_calls, 2*updates)
                    pool.capture(poses(55, 66, 77, 88), 12)
                    self.assertEqual(pool.post_step_update_calls, 3*updates)
                finally:
                    pool.close()

    def test_invalid_update_count_fails_before_allocating_camera_resources(self):
        for value in (True, False, None, -1, 2, 10., "1"):
            with Runtime(4) as runtime:
                with self.assertRaisesRegex(ValueError, "post_step_updates"):
                    backend.BatchCameraRenderer(runtime.stage, runtime, runtime.paths,
                                                (3, 2), runtime.check_time, post_step_updates=value)
                self.assertEqual(runtime.targets, [])


if __name__ == "__main__":
    unittest.main()
