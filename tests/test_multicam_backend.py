"""CPU tests for multi-camera scheduling, buffer lifetime and cleanup contracts."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from pxr import Usd, UsdGeom

SPEC = importlib.util.spec_from_file_location(
    "dataset_multicam_test", Path(__file__).resolve().parents[1]/"dataset/multicam.py")
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


class Runtime:
    """Fake Kit with real USD transforms and deliberately reused image buffers."""

    def __init__(self, count=3, fail_attach=None):
        self.stage = Usd.Stage.CreateInMemory()
        self.paths = ["/Camera"+str(i) for i in range(count)]
        for path in self.paths:
            UsdGeom.Camera.Define(self.stage, path)
        self.events, self.targets, self.annotations = [], [], []
        self.fail_attach = fail_attach
        self.fail_detach = False
        self.missing_reads = 0
        self.current_time = 0.
        self.checks = 0
        runtime = self

        class Texture:
            def __init__(self, target):
                self.target = target

            def set_updates_enabled(self, enabled):
                self.target.enabled = enabled
                runtime.events.append(("enabled", self.target.path, enabled))

        class Target:
            def __init__(self, path):
                self.path = path
                self.enabled, self.released = True, False
                self.attach_target = self
                self.render_product = types.SimpleNamespace(hydra_texture=Texture(self))
                self.rgb = np.zeros((2, 3, 4), dtype=np.uint8)
                self.params = {"nested": {"camera_position": np.zeros(3)}}
                self.attached = []

            def release(self):
                self.released = True
                runtime.events.append(("release", self.path))

        class Factory:
            def __init__(self, has_spg, resolution):
                assert has_spg is False
                assert resolution == (3, 2)

            def create(self, stage, name, camera_path):
                target = Target(camera_path)
                runtime.targets.append(target)
                return target

        class Annotator:
            def __init__(self, name):
                self.name, self.target = name, None
                self.detached = False
                runtime.annotations.append(self)

            def attach(self, target):
                self.target = target
                target.attached.append(self)
                if runtime.fail_attach == (target.path, self.name):
                    raise RuntimeError("injected attach failure")

            def detach(self):
                self.detached = True
                runtime.events.append(("detach", self.target.path, self.name))
                if runtime.fail_detach and self.name == "camera_params":
                    raise RuntimeError("injected detach failure")

            def get_data(self, do_array_copy):
                assert do_array_copy is True
                runtime.events.append(("read", self.target.path, self.name))
                if self.name == "LdrColor":
                    if runtime.missing_reads:
                        runtime.missing_reads -= 1
                        return None
                    # Deliberately ignore the requested copy to expose buffer aliasing.
                    return {"data": self.target.rgb, "info": {}}
                return self.target.params

        modules = {}
        for name in ("omni", "omni.replicator", "omni.replicator.core", "omni.timeline",
                     "isaacsim", "isaacsim.replicator", "isaacsim.replicator.nurec_utils",
                     "isaacsim.replicator.nurec_utils.render"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
            if "." in name:
                parent, key = name.rsplit(".", 1)
                setattr(modules[parent], key, module)
        modules["omni.replicator.core"].AnnotatorRegistry = types.SimpleNamespace(
            get_annotator=lambda name: Annotator(name))
        modules["omni.replicator.core"].orchestrator = types.SimpleNamespace(step=self.step)
        modules["omni.timeline"].get_timeline_interface = lambda: self
        native = modules["isaacsim.replicator.nurec_utils.render"]
        native.RenderTargetFactory = Factory
        native.ensure_identity_exposure = lambda stage, path: None
        self.module_patch = patch.dict(sys.modules, modules)

    def __enter__(self):
        self.module_patch.start()
        return self

    def __exit__(self, *_):
        self.module_patch.stop()

    def set_auto_update(self, enabled):
        self.auto_update = enabled

    def set_current_time(self, value):
        self.current_time = value

    def check_time(self):
        self.checks += 1

    def update(self):
        self.events.append(("update",))

    def positions(self):
        return [np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(t.path))
                           .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3].tolist()
                for t in self.targets if not t.released and t.enabled]

    def step(self, **kwargs):
        self.events.append(("step", self.positions(), kwargs))
        for target in self.targets:
            if target.released or not target.enabled:
                continue
            position = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(target.path))
                                  .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3]
            target.rgb[:] = [int(position[0]) % 255, int(position[1]) % 255, 7, 255]
            target.params["nested"]["camera_position"][:] = position

    def renderer(self):
        return backend.BatchCameraRenderer(self.stage, self, self.paths, (3, 2), self.check_time)


def poses(*xs):
    return [[x, x+1, 1, 0, 0, 0, 1] for x in xs]


class MultiCameraBackendTests(unittest.TestCase):
    def test_all_poses_share_one_step_and_timeline_snapshot(self):
        with Runtime() as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            self.assertEqual(pool.init_render_calls, 1)
            runtime.events.clear()
            frames, timing = pool.capture(poses(11, 22, 33), 12)
            steps = [e for e in runtime.events if e[0] == "step"]
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0][1], [[11, 12, 1], [22, 23, 1], [33, 34, 1]])
            self.assertEqual(steps[0][2], dict(delta_time=0., pause_timeline=True, wait_for_render=True))
            self.assertEqual(sum(e[0] == "update" for e in runtime.events), 10)
            self.assertEqual(runtime.current_time, 12/runtime.stage.GetTimeCodesPerSecond())
            self.assertFalse(runtime.auto_update)
            self.assertEqual(pool.render_calls, 1)
            self.assertEqual(len(frames), 3)
            for frame, x in zip(frames, (11, 22, 33)):
                np.testing.assert_array_equal(frame["rgb"][0, 0], [x, x+1, 7])
                np.testing.assert_array_equal(frame["actual_row"][3, :3], [x, x+1, 1])
            for key in ("pose_setup_seconds", "orchestrator_seconds", "update_seconds",
                        "readback_seconds", "total_seconds"):
                self.assertGreaterEqual(timing[key], 0.)

    def test_returned_pixels_and_nested_params_survive_reused_buffers(self):
        with Runtime(2) as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            first, _ = pool.capture(poses(11, 22), 12)
            second, _ = pool.capture(poses(44, 55), 12)
            np.testing.assert_array_equal(first[0]["rgb"][0, 0], [11, 12, 7])
            np.testing.assert_array_equal(first[1]["params"]["nested"]["camera_position"], [22, 23, 1])
            np.testing.assert_array_equal(second[0]["rgb"][0, 0], [44, 45, 7])
            self.assertFalse(np.shares_memory(first[0]["rgb"], runtime.targets[0].rgb))

    def test_partial_tail_detaches_before_disable_and_cannot_regrow(self):
        with Runtime() as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            pool.capture(poses(11, 22, 33), 12)
            runtime.events.clear()
            frames, _ = pool.capture(poses(44), 12)
            self.assertEqual(len(frames), 1)
            self.assertEqual(len(pool.slots), 1)
            for target in runtime.targets[1:]:
                self.assertTrue(target.released)
                disable_index = runtime.events.index(("enabled", target.path, False))
                for kind in ("LdrColor", "camera_params"):
                    self.assertLess(runtime.events.index(("detach", target.path, kind)), disable_index)
            with self.assertRaises(ValueError):
                pool.capture(poses(55, 66), 12)

    def test_retry_recaptures_whole_group_before_returning(self):
        with Runtime(2) as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            runtime.missing_reads = 1
            frames, timing = pool.capture(poses(11, 22), 12)
            self.assertEqual(timing["attempts"], 2)
            self.assertEqual(pool.render_calls, 2)
            self.assertEqual(len(frames), 2)
            np.testing.assert_array_equal(frames[1]["rgb"][0, 0], [22, 23, 7])

    def test_persistent_missing_output_is_bounded(self):
        with Runtime(2) as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            runtime.missing_reads = 100
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                pool.capture(poses(11, 22), 12)
            self.assertEqual(pool.render_calls, 3)

    def test_partial_constructor_failure_releases_every_allocated_target(self):
        with Runtime(3, fail_attach=("/Camera1", "camera_params")) as runtime:
            with self.assertRaisesRegex(RuntimeError, "injected attach failure"):
                runtime.renderer()
            self.assertEqual(len(runtime.targets), 2)
            self.assertTrue(all(t.released for t in runtime.targets))
            self.assertTrue(all(a.detached for a in runtime.annotations))
            for path in runtime.paths:
                xform = UsdGeom.Xformable(runtime.stage.GetPrimAtPath(path))
                self.assertFalse(xform.GetResetXformStack())
                self.assertEqual(len(xform.GetOrderedXformOps()), 0)

    def test_cleanup_error_does_not_skip_other_resources_and_close_is_idempotent(self):
        with Runtime(3) as runtime:
            pool = runtime.renderer()
            runtime.fail_detach = True
            with self.assertRaisesRegex(RuntimeError, "injected detach failure"):
                pool.close()
            self.assertTrue(all(t.released for t in runtime.targets))
            self.assertTrue(all(a.detached for a in runtime.annotations))
            before = list(runtime.events)
            pool.close()
            self.assertEqual(before, runtime.events)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.capture(poses(11), 12)

    def test_bad_input_does_not_release_tail_or_render(self):
        with Runtime(2) as runtime:
            pool = runtime.renderer()
            self.addCleanup(pool.close)
            for invalid in ([], [[1, 2, 3]], [[1, 2, 3, 0, 0, 0, 2]],
                            [[float("nan"), 2, 3, 0, 0, 0, 1]]):
                with self.assertRaises(ValueError):
                    pool.capture(invalid, 12)
            with self.assertRaises(ValueError):
                pool.capture(poses(11), float("nan"))
            self.assertEqual(pool.render_calls, 0)
            self.assertEqual(len(pool.slots), 2)
            self.assertFalse(any(t.released for t in runtime.targets))


if __name__ == "__main__":
    unittest.main()
