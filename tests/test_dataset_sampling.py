"""CPU fixtures independently exercise schema binding, geometry and optical poses."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.sampling import generate_plan, normalize_config, position_groups, triangle_box_intersects
from dataset.pose_math import optical_rotation, matrix_to_xyzw, pose_from_angles


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, obstacle=False, layers=False, hole=False, slope=0., translation=(10., 20., 7.)):
        source = self.root / "source.usda"
        stage = Usd.Stage.CreateNew(str(source))
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        world = UsdGeom.Xform.Define(stage, "/World")
        world.AddTranslateOp().Set(Gf.Vec3d(*translation))
        coordinates = []
        for x0, x1 in ((0., 4.), (8., 12.)) if hole else ((0., 12.),):
            coordinates.extend([[x0, 0., slope*x0], [x1, 0., slope*x1],
                                [x1, 12., slope*x1], [x0, 12., slope*x0]])
        faces = [[i, i+1, i+2] for i in range(0, len(coordinates), 4)]
        faces += [[i, i+2, i+3] for i in range(0, len(coordinates), 4)]
        if layers:
            start, old_count = len(coordinates), len(faces)
            coordinates += [[p[0], p[1], p[2]+3.] for p in coordinates]
            faces += [[v+start for v in f] for f in faces[:old_count]]
        mesh = UsdGeom.Mesh.Define(stage, "/World/Ground")
        mesh.CreatePointsAttr().Set(coordinates)
        mesh.CreateFaceVertexCountsAttr().Set([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set(np.asarray(faces).reshape(-1).tolist())
        if obstacle:
            obs = UsdGeom.Mesh.Define(stage, "/World/Wall")
            obs.CreatePointsAttr().Set([[6., 0, .05], [6., 12, .05], [6., 12, 4.], [6., 0, 4.]])
            obs.CreateFaceVertexCountsAttr().Set([3, 3])
            obs.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        stage.GetRootLayer().Save()
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        semantic = self.root / "scene.usda"
        overlay = Usd.Stage.CreateNew(str(semantic))
        UsdGeom.SetStageUpAxis(overlay, "Z")
        UsdGeom.SetStageMetersPerUnit(overlay, 1.)
        overlay.GetRootLayer().subLayerPaths = ["source.usda"]
        scope = UsdGeom.Scope.Define(overlay, "/DatasetSemantics").GetPrim()
        subset_path = "/World/Ground/dataset_semantics_road"
        region = {"id": "road", "class": "road_surface", "sampling_eligible": True,
                  "face_count": len(faces), "subset_path": subset_path,
                  "geometry_support": {"verified": True}}
        annotation = {"schema_version": 2, "source_sha256": source_sha, "source_path": str(source),
                      "source_ground_prim": "/World/Ground", "source_face_count": len(faces),
                      "regions": [region]}
        scope.CreateAttribute("dataset:annotationJson", Sdf.ValueTypeNames.String).Set(json.dumps(annotation))
        region_prim = UsdGeom.Scope.Define(overlay, "/DatasetSemantics/road").GetPrim()
        attrs = {"dataset:sourceFaceIndices": (Sdf.ValueTypeNames.IntArray, list(range(len(faces)))),
                 "dataset:representation": (Sdf.ValueTypeNames.Token, "source_mesh_faces"),
                 "dataset:sourceGroundPrim": (Sdf.ValueTypeNames.String, "/World/Ground"),
                 "dataset:samplingEligible": (Sdf.ValueTypeNames.Bool, True),
                 "dataset:subsetPath": (Sdf.ValueTypeNames.String, subset_path),
                 "semantics:labels:class": (Sdf.ValueTypeNames.TokenArray, ["road_surface"])}
        for name, (kind, value) in attrs.items():
            region_prim.CreateAttribute(name, kind).Set(value)
        subset = UsdGeom.Subset.Define(overlay, subset_path)
        subset.CreateElementTypeAttr().Set("face")
        subset.CreateFamilyNameAttr().Set("dataset_semantics_v2")
        subset.CreateIndicesAttr().Set(list(range(len(faces))))
        subset.GetPrim().CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set(["road_surface"])
        overlay.GetRootLayer().Save()
        return {"scene_path": str(semantic), "n": 10, "min_distance_m": .5, "max_attempts": 100}

    def test_nonzero_transformed_ground_and_determinism(self):
        config = self.fixture()
        a, b = generate_plan(config), generate_plan(config)
        self.assertTrue(a["complete"])
        self.assertEqual(a, b)
        json.dumps(a, allow_nan=False)
        self.assertEqual(a["statistics"]["accepted_count"], 10)
        for sample in a["samples"]:
            x, y, z, *q = sample["pose_xyzw"]
            self.assertTrue(10 <= x <= 22 and 20 <= y <= 32)
            self.assertAlmostEqual(z, 8.5)
            self.assertAlmostEqual(sample["ground_z_m"], 7.)
            self.assertAlmostEqual(np.linalg.norm(q), 1.)
            self.assertGreaterEqual(q[3], 0)
            self.assertTrue(0 <= sample["angles_deg"]["yaw"] < 360)
            self.assertTrue(-5 <= sample["angles_deg"]["pitch"] <= 10)
            self.assertTrue(-5 <= sample["angles_deg"]["roll"] <= 5)
        distances = [np.linalg.norm(np.array(a["samples"][i]["pose_xyzw"][:2])-a["samples"][j]["pose_xyzw"][:2])
                     for i in range(10) for j in range(i)]
        self.assertGreaterEqual(min(distances), .5)
        self.assertEqual(set(a["scene"]["source_dependencies"]), {str(self.root/"source.usda"), str(self.root/"scene.usda")})

    def test_seed_changes_positions(self):
        config = self.fixture()
        self.assertNotEqual(generate_plan(config)["samples"], generate_plan({**config, "seed": 43})["samples"])

    def test_hole_never_filled(self):
        plan = generate_plan({**self.fixture(hole=True), "n": 50, "min_distance_m": 0.})
        self.assertTrue(plan["complete"])
        self.assertTrue(all(s["pose_xyzw"][0] <= 14 or s["pose_xyzw"][0] >= 18 for s in plan["samples"]))

    def test_ground_height_follows_slope(self):
        plan = generate_plan(self.fixture(slope=.1))
        for sample in plan["samples"]:
            self.assertAlmostEqual(sample["ground_z_m"], 7+.1*(sample["pose_xyzw"][0]-10), places=6)
            self.assertAlmostEqual(sample["pose_xyzw"][2]-sample["ground_z_m"], 1.5)

    def test_steep_faces_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_slope"):
            generate_plan(self.fixture(slope=1.))

    def test_vertical_wall_clearance(self):
        plan = generate_plan({**self.fixture(obstacle=True), "n": 50, "min_distance_m": 0., "clearance_m": 1.})
        self.assertTrue(plan["complete"])
        self.assertGreater(plan["statistics"]["rejected"]["mesh_clearance_obstacle"], 0)
        self.assertTrue(all(abs(s["pose_xyzw"][0]-16) > 1. for s in plan["samples"]))

    def test_multiple_ground_layers_explicit_partial(self):
        plan = generate_plan({**self.fixture(layers=True), "n": 1, "max_attempts": 5})
        self.assertFalse(plan["complete"])
        self.assertEqual(plan["samples"], [])
        self.assertEqual(plan["statistics"]["rejected"], {"ambiguous_multiple_ground_layers": 5})
        self.assertEqual(plan["statistics"]["failure_reason"], "max_attempts_exhausted")

    def test_impossible_spacing_is_bounded(self):
        plan = generate_plan({**self.fixture(), "min_distance_m": 100, "max_attempts": 10})
        self.assertFalse(plan["complete"])
        self.assertEqual(len(plan["samples"]), 1)
        self.assertEqual(plan["statistics"]["attempts"], 10)
        self.assertEqual(plan["statistics"]["rejected"]["minimum_xy_spacing"], 9)

    def test_source_hash_tamper_rejected(self):
        config = self.fixture()
        with (self.root/"source.usda").open("a") as stream:
            stream.write("\n# changed bytes\n")
        with self.assertRaisesRegex(ValueError, "sublayer SHA"):
            generate_plan(config)

    def test_scene_hash_mismatch(self):
        config = self.fixture()
        with self.assertRaisesRegex(ValueError, "Scene SHA"):
            generate_plan({**config, "scene_sha256": "0"*64})

    def test_subset_mismatch(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        stage.GetPrimAtPath("/World/Ground/dataset_semantics_road").GetAttribute("indices").Set([0])
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            generate_plan(config)

    def test_unknown_class_rejected(self):
        config = self.fixture()
        with self.assertRaises(ValueError):
            generate_plan({**config, "classes": ["unknown"]})

    def test_no_matching_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "No eligible"):
            generate_plan({**self.fixture(), "classes": ["lane"]})

    def test_schema_one_rejected(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        attr = stage.GetPrimAtPath("/DatasetSemantics").GetAttribute("dataset:annotationJson")
        annotation = json.loads(attr.Get())
        annotation["schema_version"] = 1
        attr.Set(json.dumps(annotation))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "schema-2"):
            generate_plan(config)

    def test_bad_unit_rejected(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        UsdGeom.SetStageMetersPerUnit(stage, .01)
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "metre Z-up"):
            generate_plan(config)

    def test_invalid_config(self):
        config = self.fixture()
        for key, value in [("n", True), ("seed", -1), ("sensor_height_m", float("nan")),
                           ("max_attempts", 1), ("pitch_deg", [31, -31]), ("classes", []),
                           ("split_ratios", [0, 0, 0]), ("unexpected", 1), ("clearance_m", 2.),
                           ("camera", {"resolution": [0, 720]}), ("yaw_deg", [0, 720])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_config({**config, key: value})

    def test_overlay_geometry_override_rejected(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Ground"))
        mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get())
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "override source mesh"):
            generate_plan(config)

    def test_overlay_transform_override_rejected(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        stage.GetPrimAtPath("/World").GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "override source transforms"):
            generate_plan(config)

    def test_unmeshed_primitive_rejected(self):
        config = self.fixture()
        stage = Usd.Stage.Open(config["scene_path"])
        UsdGeom.Cube.Define(stage, "/ObstacleCube")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "Non-mesh"):
            generate_plan(config)

    def test_low_height_cannot_invert_clearance_box(self):
        with self.assertRaisesRegex(ValueError, "too low"):
            normalize_config({**self.fixture(), "sensor_height_m": .05, "clearance_m": .001})

    def test_camera_clipping_validation(self):
        with self.assertRaises(ValueError):
            normalize_config({**self.fixture(), "camera": {"clipping_range": [20., 1.]}})

    def test_public_camera_boundaries_match_render_contract(self):
        from dataset.render_worker import validate_plan
        config = self.fixture()
        pose, matrix = pose_from_angles([12., 24., 8.5], 0., 0., 0.)
        sample = {"sample_id": "000000", "pose_xyzw": pose, "T_world_camera": matrix,
                  "group_id": "group_000000", "source_face_id": 0, "region_id": "road",
                  "ground_z_m": 7., "angles_deg": {"yaw": 0., "pitch": 0., "roll": 0.}}
        for fov in (2., 175.):
            normalized = normalize_config({**config, "n": 1,
                                           "camera": {"resolution": [4096, 4096], "horizontal_fov_deg": fov}})
            plan = {"schema_version": 1, "complete": True, "samples": [sample],
                    "scene": {"path": normalized["scene_path"], "sha256": "0"*64}, "config": normalized}
            self.assertIs(validate_plan(plan), plan)
        for camera in ({"horizontal_fov_deg": 1.}, {"resolution": [8192, 8192]}):
            with self.assertRaises(ValueError):
                normalize_config({**config, "camera": camera})

    def test_grouping_is_transitive(self):
        samples = [{"pose_xyzw": [x, 0., 0., 0., 0., 0., 1.]} for x in (0., 1., 2., 7.)]
        self.assertEqual(position_groups(samples, 1.1), 2)
        self.assertEqual(samples[0]["group_id"], samples[2]["group_id"])
        self.assertNotEqual(samples[2]["group_id"], samples[3]["group_id"])

    def test_triangle_box_exact_avoids_bbox_false_positive(self):
        tri = np.array([[[0., 2., 0], [2., 0., 0], [2., 2., 0]],
                        [[0., -.5, -.5], [0., .5, -.5], [0., 0, .5]]])
        hit = triangle_box_intersects(tri, np.array([-.1]*3), np.array([.1]*3))
        self.assertEqual(hit.tolist(), [False, True])


class PoseTests(unittest.TestCase):
    def test_identity_angles_optical_axes(self):
        rotation = optical_rotation(0, 0, 0)
        np.testing.assert_allclose(rotation[:, 2], [1, 0, 0])
        np.testing.assert_allclose(rotation[:, 0], [0, -1, 0])
        np.testing.assert_allclose(rotation[:, 1], [0, 0, -1])

    def test_yaw_pitch_analytic_forward(self):
        for yaw, pitch in [(0., 30.), (90., 0.), (130., -25.)]:
            y, p = np.radians([yaw, pitch])
            np.testing.assert_allclose(optical_rotation(yaw, pitch, 12)[:, 2],
                                       [np.cos(y)*np.cos(p), np.sin(y)*np.cos(p), np.sin(p)], atol=1e-12)

    def test_quaternion_matches_independent_scipy(self):
        from scipy.spatial.transform import Rotation
        for yaw, pitch, roll in [(0., 0., 0.), (180., 0., 0.), (271., 30., -5.), (35., -27., 4.)]:
            pose, transform = pose_from_angles([1., 2., 3.], yaw, pitch, roll)
            expected = Rotation.from_euler("Z", yaw, degrees=True) * Rotation.from_euler("Y", -pitch, degrees=True) * Rotation.from_euler("X", roll, degrees=True)
            body_from_optical = np.array([[0., 0, 1], [-1., 0, 0], [0, -1., 0]])
            np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), expected.as_matrix() @ body_from_optical, atol=1e-12)
            np.testing.assert_allclose(np.asarray(transform)[:3, 3], [1., 2., 3.])

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            matrix_to_xyzw(np.diag([1., 1., -1.]))


if __name__ == "__main__":
    unittest.main()
