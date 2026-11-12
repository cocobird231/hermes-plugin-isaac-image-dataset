"""Synchronous NuRec camera pool; one render step captures a group of poses.

Kit imports are delayed until construction. The owner supplies already-created
cameras with equal intrinsics and owns the application, stage and output files.
"""
from copy import deepcopy
import math
import time

import numpy as np


class BatchCameraRenderer:
    """Reuse ordinary RenderProducts without advancing the static scene time.

    Capture groups may shrink, but cannot grow after an unused tail was released.
    Returned arrays own their storage and are safe across later Kit updates.
    """

    def __init__(self, stage, app, camera_paths, resolution, check_time, post_step_updates=10):
        if type(post_step_updates) is not int or post_step_updates not in (0, 1, 10):
            raise ValueError("post_step_updates must be 0, 1 or 10")
        self.post_step_updates = post_step_updates
        self.post_step_update_calls = 0
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import Gf, Usd, UsdGeom
        from isaacsim.replicator.nurec_utils.render import RenderTargetFactory, ensure_identity_exposure

        if not camera_paths or len(set(camera_paths)) != len(camera_paths):
            raise ValueError("Camera paths must be nonempty and unique")
        self.stage, self.app, self.check_time = stage, app, check_time
        self.rep, self.Gf, self.Usd, self.UsdGeom = rep, Gf, Usd, UsdGeom
        self.width, self.height = resolution
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = float(stage.GetTimeCodesPerSecond())
        if not math.isfinite(self.tcps) or self.tcps <= 0:
            raise ValueError("Stage time codes per second must be positive")
        self.slots = []
        self.closed = False
        self.render_calls = 0
        self.init_render_calls = 0
        self.timeline.set_auto_update(False)
        factory = RenderTargetFactory(False, resolution=tuple(resolution))
        try:
            for index, path in enumerate(camera_paths):
                check_time()
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsA(UsdGeom.Camera):
                    raise ValueError("Pool path is not a Camera: " + str(path))
                xform = UsdGeom.Xformable(prim)
                slot = {"path": str(path), "target": None, "rgb": None, "params": None,
                        "xform": xform, "saved_order": xform.GetXformOpOrderAttr().Get(),
                        "saved_reset": xform.GetResetXformStack(), "pose_op": None}
                self.slots.append(slot)
                ensure_identity_exposure(stage, path)
                xform.ClearXformOpOrder()
                slot["pose_op"] = xform.AddTransformOp()
                xform.SetResetXformStack(True)
                target = factory.create(stage, "localization_pool_"+str(index), camera_path=path)
                slot["target"] = target
                target.render_product.hydra_texture.set_updates_enabled(False)
                for key, name in (("rgb", "LdrColor"), ("params", "camera_params")):
                    slot[key] = rep.AnnotatorRegistry.get_annotator(name)
                    slot[key].attach(target.attach_target)
                target.render_product.hydra_texture.set_updates_enabled(True)
            for _ in range(5):
                check_time()
                app.update()
            check_time()
            self.init_render_calls += 1
            rep.orchestrator.step(delta_time=0.0, pause_timeline=True, wait_for_render=True)
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note("Pool construction cleanup: " + str(cleanup))
            raise

    @staticmethod
    def _release_slot(slot):
        errors = []
        for key in ("params", "rgb"):
            annotator = slot[key]
            slot[key] = None
            if annotator is not None:
                try:
                    annotator.detach()
                except Exception as exc:
                    errors.append(key + ": " + str(exc))
        target, slot["target"] = slot["target"], None
        if target is not None:
            try:
                target.render_product.hydra_texture.set_updates_enabled(False)
            except Exception as exc:
                errors.append("disable target: " + str(exc))
            try:
                target.release()
            except Exception as exc:
                errors.append("release target: " + str(exc))
        if slot["pose_op"] is not None:
            slot["pose_op"] = None
            try:
                xform = slot["xform"]
                xform.ClearXformOpOrder()
                xform.SetResetXformStack(slot["saved_reset"])
                if slot["saved_order"]:
                    xform.GetXformOpOrderAttr().Set(slot["saved_order"])
            except Exception as exc:
                errors.append("restore camera: " + str(exc))
        return errors

    def capture(self, usd_poses, time_code):
        """Capture all supplied world poses at one USD time code.

        Missing/invalid RGB gets at most three whole-group render attempts. All
        arrays and metadata are read before returning or performing another tick.
        Calibration and durable per-sample commits remain the caller's concern.
        """
        if self.closed:
            raise RuntimeError("Camera pool is closed")
        poses = np.asarray(usd_poses, dtype=float)
        if (poses.ndim != 2 or poses.shape[1:] != (7,) or not 0 < len(poses) <= len(self.slots)
                or not np.isfinite(poses).all()
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., rtol=0, atol=1e-6)):
            raise ValueError("Expected one finite unit-quaternion pose per active camera")
        if not math.isfinite(float(time_code)):
            raise ValueError("USD time code must be finite")
        started = time.perf_counter()
        timing = {"pose_setup_seconds": 0., "orchestrator_seconds": 0.,
                  "update_seconds": 0., "readback_seconds": 0.,
                  "post_step_updates_per_attempt": self.post_step_updates,
                  "post_step_update_calls": 0}
        setup_started = time.perf_counter()
        errors = []
        while len(self.slots) > len(poses):
            errors.extend(self._release_slot(self.slots.pop()))
        if errors:
            raise RuntimeError("Unused camera cleanup failed: " + "; ".join(errors))
        for slot, pose in zip(self.slots, poses):
            self.check_time()
            tx, ty, tz, qx, qy, qz, qw = map(float, pose)
            matrix = self.Gf.Matrix4d().SetRotate(self.Gf.Quatd(qw, self.Gf.Vec3d(qx, qy, qz)))
            matrix.SetTranslateOnly(self.Gf.Vec3d(tx, ty, tz))
            slot["pose_op"].Set(matrix)
        self.timeline.set_current_time(float(time_code)/self.tcps)
        timing["pose_setup_seconds"] = time.perf_counter()-setup_started
        for attempt in range(1, 4):
            self.check_time()
            stamp = time.perf_counter()
            self.render_calls += 1
            self.rep.orchestrator.step(delta_time=0.0, pause_timeline=True, wait_for_render=True)
            timing["orchestrator_seconds"] += time.perf_counter()-stamp
            stamp = time.perf_counter()
            for _ in range(self.post_step_updates):
                self.check_time()
                self.app.update()
                self.post_step_update_calls += 1
                timing["post_step_update_calls"] += 1
            timing["update_seconds"] += time.perf_counter()-stamp
            stamp = time.perf_counter()
            frames = []
            valid = True
            for slot in self.slots:
                self.check_time()
                payload = slot["rgb"].get_data(do_array_copy=True)
                pixels = payload.get("data") if isinstance(payload, dict) else payload
                arr = np.asarray(pixels) if pixels is not None else None
                if arr is not None and arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[..., :3]
                rgb = np.array(arr, copy=True) if arr is not None else None
                params = deepcopy(slot["params"].get_data(do_array_copy=True))
                actual_row = np.array(slot["xform"].ComputeLocalToWorldTransform(
                    self.Usd.TimeCode(float(time_code))), dtype=float, copy=True)
                frames.append({"rgb": rgb, "params": params, "actual_row": actual_row})
                if rgb is None or rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                    valid = False
            timing["readback_seconds"] += time.perf_counter()-stamp
            if valid:
                timing.update(total_seconds=time.perf_counter()-started, attempts=attempt)
                return frames, timing
        raise RuntimeError("Camera pool did not produce complete RGB group after 3 attempts")

    def close(self):
        """Release every resource, then report any cleanup failure."""
        if self.closed:
            return
        self.closed = True
        errors = []
        for slot in reversed(self.slots):
            errors.extend(self._release_slot(slot))
        self.slots.clear()
        if errors:
            raise RuntimeError("Camera pool cleanup failed: " + "; ".join(errors))
