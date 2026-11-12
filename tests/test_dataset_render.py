"""CPU-only contract checks; never start SimulationApp."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

WORKER = Path(__file__).resolve().parents[1]/"dataset/render_worker.py"
spec = importlib.util.spec_from_file_location("dataset_capture_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def sample(rotation=None):
    optical = np.eye(4)
    optical[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]) if rotation is None else rotation
    optical[:3, 3] = [137.2, -83.4, 1.25]
    pose = [*optical[:3, 3], *worker.matrix_quaternion(optical[:3, :3])]
    return {"sample_id": "000000", "pose_xyzw": pose, "T_world_camera": optical.tolist(),
            "group_id": "p0", "source_face_id": 123, "region_id": "road0", "ground_z_m": -.25,
            "angles_deg": {"yaw": 0, "pitch": 0, "roll": 0}}


def plan():
    return {"schema_version": 1, "complete": True, "samples": [sample()],
            "scene": {"path": "/tmp/test.usda", "sha256": "a"*64},
            "config": {"camera": {"resolution": [1280, 720], "horizontal_fov_deg": 90,
                                   "clipping_range": [.1, 2000]}}}


def calibration(s):
    _, world = worker.optical_to_usd(s)
    w, h, focal, ax, ay, near, far = 1280, 720, 18., 36., 20.25, .1, 2000.
    projection = np.array([[2*focal/ax, 0., 0., 0.],
                           [0., 2*focal/ay, 0., 0.],
                           [0., 0., near/(far-near), -1.],
                           [0., 0., near*far/(far-near), 0.]])
    params = {"cameraModel": "pinhole", "renderProductResolution": [w, h],
              "metersPerSceneUnit": 1., "cameraFocalLength": focal,
              "cameraAperture": [ax, ay], "cameraApertureOffset": [0., 0.],
              "cameraNearFar": [near, far], "cameraViewTransform": np.linalg.inv(world),
              "cameraProjection": projection}
    camera = {"resolution": [w, h], "focal_length_mm": focal, "aperture_mm": [ax, ay],
              "clipping_range": [near, far]}
    return world, params, camera


class CaptureContractTests(unittest.TestCase):
    def test_known_optical_axes_convert_to_usd(self):
        usd_pose, rows = worker.optical_to_usd(sample())
        rotation = worker.quaternion_matrix(usd_pose[3:])
        np.testing.assert_allclose(rotation @ [0, 0, -1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(rows.T @ worker.AXES, sample()["T_world_camera"], atol=1e-12)

    def test_random_rotations_and_half_turn_roundtrip(self):
        rng = np.random.default_rng(42)
        quaternions = [np.array([1., 0, 0, 0]), np.array([0., 1, 0, 0]), np.array([0., 0, 1, 0])]
        quaternions += [q/np.linalg.norm(q) for q in rng.normal(size=(100, 4))]
        for q in quaternions:
            rotation = worker.quaternion_matrix(q)
            measured = worker.matrix_quaternion(rotation)
            np.testing.assert_allclose(worker.quaternion_matrix(measured), rotation, atol=1e-12)
            self.assertGreaterEqual(measured[3], -1e-14)

    def test_reject_pose_matrix_disagreement(self):
        changed = sample()
        changed["T_world_camera"][0][3] += 1
        with self.assertRaisesRegex(ValueError, "translation"):
            worker.optical_to_usd(changed)

    def test_reject_nonfinite_or_nonunit_quaternion(self):
        for bad in ([0, 0, 0, 2], [0, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                worker.quaternion_matrix(bad)

    def test_projection_of_known_rays_at_rotated_translated_camera(self):
        for q in ([.5, .5, .5, .5], [0., 0., 1., 0.], [.1, -.2, .3, .4]):
            q = np.array(q)/np.linalg.norm(q)
            world, params, camera = calibration(sample(worker.quaternion_matrix(q)))
            evidence = worker.validate_readback(params, world, world, camera)
            self.assertTrue(evidence["passed"])
            self.assertLess(evidence["known_optical_ray_projection_max_error_px"], 1e-9)
            np.testing.assert_allclose(evidence["intrinsics_K"], [[640, 0, 640], [0, 640, 360], [0, 0, 1]])

    def test_stale_annotation_rejected(self):
        world, params, camera = calibration(sample())
        params["cameraViewTransform"] = np.eye(4)
        with self.assertRaisesRegex(ValueError, "stale view"):
            worker.validate_readback(params, world, world, camera)

    def test_wrong_sensor_intrinsics_rejected(self):
        for key, value in (("renderProductResolution", [640, 360]), ("cameraFocalLength", 24.),
                           ("cameraApertureOffset", [1., 0.]), ("cameraModel", "fisheye"),
                           ("cameraNearFar", [1., 2000.])):
            world, params, camera = calibration(sample())
            params[key] = value
            with self.assertRaises(ValueError, msg=key):
                worker.validate_readback(params, world, world, camera)

    def test_plan_incomplete_duplicate_and_unsafe_ids_rejected(self):
        worker.validate_plan(plan())
        for field, value in (("complete", False), ("schema_version", 2)):
            changed = plan()
            changed[field] = value
            with self.assertRaises(ValueError):
                worker.validate_plan(changed)
        for sid in ("../outside", "x.png", "", "/absolute"):
            changed = plan()
            changed["samples"][0]["sample_id"] = sid
            with self.assertRaises(ValueError):
                worker.validate_plan(changed)
        changed = plan()
        changed["samples"].append(copy.deepcopy(changed["samples"][0]))
        with self.assertRaises(ValueError):
            worker.validate_plan(changed)

    def test_invalid_camera_and_render_limits_rejected(self):
        for key, value in (("resolution", [True, 720]), ("horizontal_fov_deg", 180),
                           ("clipping_range", [100, .1])):
            changed = plan()
            changed["config"]["camera"][key] = value
            with self.assertRaises(ValueError):
                worker.validate_plan(changed)
        changed = plan()
        changed["config"]["render"] = {"discard_frames": 0}
        with self.assertRaises(ValueError):
            worker.validate_plan(changed)

    def test_template_intrinsics_copied_without_moving_original(self):
        from pxr import Gf, Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Sensor")
        camera.CreateFocalLengthAttr(24)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        camera.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        before = stage.GetRootLayer().ExportToString()
        config = worker.configured_camera(stage, {"resolution": [1280, 720], "template_prim": "/Sensor"})
        self.assertEqual(config["mode"], "existing_pinhole_template")
        self.assertEqual(config["focal_length_mm"], 24)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        camera.CreateHorizontalApertureOffsetAttr(1)
        with self.assertRaisesRegex(ValueError, "principal point"):
            worker.configured_camera(stage, {"resolution": [1280, 720], "template_prim": "/Sensor"})

    def test_invalid_template_clipping_rejected(self):
        from pxr import Gf, Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Sensor")
        camera.CreateClippingRangeAttr(Gf.Vec2f(10, 1))
        with self.assertRaisesRegex(ValueError, "clipping range"):
            worker.configured_camera(stage, {"resolution": [1280, 720], "template_prim": "/Sensor"})

    def test_distorted_template_explicitly_rejected(self):
        from pxr import Sdf, Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Sensor")
        camera.GetPrim().CreateAttribute("camera:distortionModel", Sdf.ValueTypeNames.String).Set("fisheye")
        with self.assertRaisesRegex(ValueError, "distortion"):
            worker.configured_camera(stage, {"resolution": [1280, 720], "template_prim": "/Sensor"})


class WorkerExitPolicyTests(unittest.TestCase):
    @staticmethod
    def receipt():
        return {"ok": True, "app_closed": True, "source_unchanged": True,
                "native_nurec_crop_unchanged": True, "phase": "complete",
                "captured_count": 32, "requested_count": 32}

    def test_success_requires_completed_durable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            worker.atomic_json(path, self.receipt())
            self.assertEqual(worker.validated_process_exit_code(0, path), 0)
            self.assertEqual(json.loads(path.read_text()), self.receipt())

    def test_incomplete_success_is_forced_to_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            for key, value in (("ok", False), ("app_closed", False),
                               ("source_unchanged", False), ("native_nurec_crop_unchanged", False),
                               ("phase", "closing_simulation_app"), ("captured_count", 31),
                               ("captured_count", 0), ("captured_count", True),
                               ("requested_count", True)):
                result = self.receipt()
                result[key] = value
                worker.atomic_json(path, result)
                self.assertEqual(worker.validated_process_exit_code(0, path), 1, key)
            for body in ("not json", "null"):
                path.write_text(body)
                self.assertEqual(worker.validated_process_exit_code(0, path), 1)
            path.unlink()
            self.assertEqual(worker.validated_process_exit_code(0, path), 1)

    def test_failure_codes_are_never_promoted(self):
        for code in (1, 2, 7, 139, 255):
            self.assertEqual(worker.validated_process_exit_code(code, "/missing/result.json"), code)
        for code in (True, False, -11, None, 256):
            self.assertEqual(worker.validated_process_exit_code(code, "/missing/result.json"), 1)

    def test_symlink_success_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original, linked = Path(directory)/"real.json", Path(directory)/"result.json"
            worker.atomic_json(original, self.receipt())
            linked.symlink_to(original)
            self.assertEqual(worker.validated_process_exit_code(0, linked), 1)

    def test_explicit_exit_flushes_stdio_skips_native_atexit_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/"result.json"
            marker = Path(directory)/"atexit-ran.txt"
            worker.atomic_json(receipt, self.receipt())
            for requested, expected in ((0, 0), (7, 7)):
                script = (
                    "import atexit,importlib.util;from pathlib import Path;"
                    f"spec=importlib.util.spec_from_file_location('capture_exit',{str(WORKER)!r});"
                    "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
                    f"atexit.register(lambda:Path({str(marker)!r}).write_text('unexpected'));"
                    "print('explicit-close-and-receipt-finished',end='');"
                    f"m.exit_after_finalization({requested},{str(receipt)!r})"
                )
                child = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                       text=True, timeout=15, check=False)
                self.assertEqual(child.returncode, expected, child.stderr)
                self.assertEqual(child.stdout, "explicit-close-and-receipt-finished")
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
