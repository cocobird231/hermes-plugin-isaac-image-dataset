"""Exact spatial-index equivalence, large bounded sampling, and camera semantics."""
import copy
import hashlib
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.sampling import _XYSpacingIndex, generate_plan, normalize_config, position_groups
import test_dataset_sampling as fixtures


def samples_at(points):
    return [{"pose_xyzw": [float(x), float(y), 1.5, 0., 0., 0., 1.]} for x, y in points]


def reference_groups(samples, distance):
    parent = list(range(len(samples)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, sample in enumerate(samples):
        for j in range(i):
            if np.linalg.norm(np.array(sample["pose_xyzw"][:2])-samples[j]["pose_xyzw"][:2]) <= distance+1e-10:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(root(i), f"group_{len(groups):06d}")
        sample["group_id"] = groups[root(i)]
    return len(groups)


class SamplingBulkTests(unittest.TestCase):
    setUp = fixtures.SamplingTests.setUp
    fixture = fixtures.SamplingTests.fixture

    def test_grouping_matches_independent_all_pairs(self):
        rng = np.random.default_rng(67)
        for distance in (0., 1e-9, .1, 1., 100.):
            points = np.r_[rng.uniform(-3, 3, (180, 2)), [[0., 0.], [0., 0.], [distance, 0.], [-distance, 0.]]]
            fast = samples_at(points)
            slow = copy.deepcopy(fast)
            self.assertEqual(position_groups(fast, distance), reference_groups(slow, distance))
            self.assertEqual(fast, slow)

    def test_grouping_inclusive_tolerance_and_transitivity(self):
        for gap, expected in ((1., 1), (1.+.5e-10, 1), (1.+2e-10, 3)):
            samples = samples_at([[0., 0.], [gap, 0.], [2*gap, 0.]])
            self.assertEqual(position_groups(samples, 1.), expected)
        self.assertEqual(position_groups([], 0), 0)

    def test_10000_sparse_and_dense_groups(self):
        started = time.monotonic()
        grid = samples_at([(float(i % 100)*3, float(i // 100)*3) for i in range(10000)])
        self.assertEqual(position_groups(grid, 2.), 10000)
        self.assertEqual(position_groups(grid, 3.), 1)
        dense = samples_at(np.zeros((10000, 2)))
        self.assertEqual(position_groups(dense, 0.), 1)
        self.assertLess(time.monotonic()-started, 30., "10000-position spatial groups must stay practical")

    def test_spacing_matches_all_accepted_points(self):
        rng = np.random.default_rng(89)
        for distance in (0., .1, 1., 4.):
            index, accepted = _XYSpacingIndex(distance), []
            points = np.r_[rng.uniform(-5, 5, (400, 2)), [[0., 0.], [distance, 0.], [0., 0.]]]
            for point in points:
                expected = any(np.linalg.norm(other-point) < distance for other in accepted)
                self.assertEqual(index.too_close(point), expected)
                if not expected:
                    index.add(point)
                    accepted.append(point)

    def test_10000_plan_is_bounded_reproducible_and_source_preserving(self):
        config = {**self.fixture(), "n": 10000, "max_attempts": 10000, "min_distance_m": 0., "group_distance_m": 0.}
        paths = (self.root/"source.usda", self.root/"scene.usda")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        plan = generate_plan(config)
        self.assertTrue(plan["complete"])
        self.assertEqual(plan["statistics"]["accepted_count"], 10000)
        self.assertEqual(plan["statistics"]["attempts"], 10000)
        self.assertEqual(plan["statistics"]["group_count"], 10000)
        self.assertEqual(plan["config"]["pitch_deg"], [-5., 10.])
        self.assertEqual(plan["config"]["camera"]["horizontal_fov_deg"], 90.)
        short = generate_plan({**config, "n": 30})
        self.assertEqual(plan["samples"][:30], short["samples"])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_template_camera_clipping_and_explicit_overrides(self):
        from dataset.render_worker import configured_camera
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        camera = UsdGeom.Camera.Define(stage, "/World/Template")
        camera.CreateFocalLengthAttr().Set(24.)
        camera.CreateHorizontalApertureAttr().Set(36.)
        camera.CreateVerticalApertureAttr().Set(20.25)
        camera.CreateClippingRangeAttr().Set((.2, 800.))
        stage.GetRootLayer().Save()
        normalized = normalize_config({**config, "camera": {"template_prim": "/World/Template", "horizontal_fov_deg": 120}})
        self.assertNotIn("clipping_range", normalized["camera"])
        values = configured_camera(stage, normalized["camera"])
        self.assertAlmostEqual(values["clipping_range"][0], .2, places=6)
        self.assertEqual(values["clipping_range"][1], 800.)
        self.assertEqual(values["focal_length_mm"], 24.)
        self.assertLess(values["horizontal_fov_deg"], 90.)
        self.assertTrue(generate_plan(normalized)["complete"])
        normalized = normalize_config({**config, "camera": {"template_prim": "/World/Template", "clipping_range": [.5, 100.]}})
        self.assertEqual(configured_camera(stage, normalized["camera"])["clipping_range"], [.5, 100.])

    def test_template_must_exist_be_camera_and_perspective(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        camera = UsdGeom.Camera.Define(stage, "/World/Ortho")
        camera.CreateProjectionAttr().Set("orthographic")
        stage.GetRootLayer().Save()
        for path in ("/Missing", "/World/Ground", "/World/Ortho"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                generate_plan({**config, "camera": {"template_prim": path}})
        for path in ("", "World/Camera", "/World.camera", "/", None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                normalize_config({**config, "camera": {"template_prim": path}})

    def test_invalid_geometry_index_values_are_rejected(self):
        with self.assertRaises(ValueError):
            position_groups(samples_at([[float("nan"), 0.]]), 1.)
        for distance in (-1, True, float("inf"), 10**1000):
            with self.subTest(distance=str(distance)[:30]), self.assertRaises(ValueError):
                position_groups([], distance)


if __name__ == "__main__":
    unittest.main()
