---
name: isaac-image-dataset
description: Configure and collect image-localization datasets using reusable cameras, efficiency/normal/quality rendering, checkpoints and recovery.
---

# Image localization dataset

Use `isaac_dataset_*` tools for this independent plugin. It launches its own headless Isaac Sim renderer; do not start/stop other Isaac or RViz plugins. Leave unrelated GPU work intact.

## First use and persistent configuration

1. Call `isaac_dataset_get_config` before asking for paths. Reuse the saved values; do not infer an account, hostname, install directory or scene location.
2. If `isaac_python` is missing, ask for the user's Isaac Python executable or Isaac `python.sh` entry point. This is not the Hermes Python. Ask for a semantic scene when no default or per-request scene was supplied, and for output location only if the user wants a nondefault directory.
3. Save user-supplied defaults with `isaac_dataset_configure`, using `{"settings":{"isaac_python":"/path/to/isaac-python","default_scene":"/path/to/semantic.usda","run_root":"/path/to/runs"}}`. Supply only known fields. Paths are validated and saved outside the plugin repository. On later requests, read and reuse them without asking again. A temporary scene override may instead be supplied as `config.scene_path`; do not silently replace a saved default.
4. A successful configuration write is not an Isaac/GPU capability test. Preserve errors and ask for corrected paths when necessary. Never install credentials into this repository.

## Plan and capture

- Read total sample count N, camera count, camera parameters, constraints and quality. N is the total image count, not N multiplied by the number of cameras. `seed` controls the whole reproducible sampling sequence, not one seed per sample.
- Translate the user's quality into `config.render_quality`: 效率/efficiency → `efficiency` (0 extra updates); 一般/normal → `normal` (1, default); 品質/quality → `quality` (10). Do not silently upgrade quality or override the user's choice. These select post-step updates, not resolution or a guaranteed image-quality threshold.
- `config.camera_count` is an integer1–8, default1. `runtime.batch_size` separately limits images per Isaac process. Do not clamp an unsupported request silently.
- Unless specified, use N32, seed42, 1280×720, horizontal FOV90°, height1.5m, pitch[-5,10]°, roll[-5,5]°, yaw[0,360]°, split[4,2,2]. For requested wide-angle without a precise FOV, state the120° rectilinear pinhole assumption; it is not fisheye or ZED. An existing supported perspective Camera may provide intrinsics via `camera.template_prim`.
- Require a schema2 semantic USD scene with eligible source mesh faces. `road_surface` is not a lane annotation. Ordinary unlabeled USDZ is not a supported sampling input. Do not rewrite source assets or claim semantic correctness beyond their labels.
- Submit `isaac_dataset_pipeline` once, retaining its `job_id`. Alternatively use sample→collect→split and wait for sampled/captured/completed respectively. Do not create a duplicate job while one is pending.

Example using saved paths, four reusable cameras and the default normal quality:

```json
{
  "config": {
    "n": 60,
    "camera_count": 4,
    "render_quality": "normal",
    "seed": 42,
    "camera": {"resolution": [1280, 720], "horizontal_fov_deg": 120},
    "split_ratios": [4, 2, 2]
  }
}
```

The renderer sets all active camera poses, calls one shared Replicator step, performs the chosen0/1/10 extra updates, reads all outputs and commits each image durably. Initialization is retained; step itself can execute internal updates/subframes. The static scene is paused after simulation startup. This is not a driven vehicle sequence. More cameras or more updates do not guarantee proportional performance or better localization accuracy.

## Status, recovery and completion

Poll `isaac_dataset_status` with `wait_seconds:30` until terminal. Pending means accepted, not complete. Report completed only with N verified images, split manifests and actual exit0. Include job_id, dataset location, chosen quality, actual camera intrinsics, camera count and split counts. Spatial groups remain intact; resulting ratios can differ from the target.

For transient process/resource failure, inspect the error and resume the SAME job at most twice. Do not resume after a user-requested stop without a later resume request. Permanent geometry, source, checkpoint or camera errors require correction. Unknown state is not permission to launch a competing attempt. Cancel only the recorded job using `isaac_dataset_cancel`.

Quality, camera count, geometry and intrinsics are bound to the immutable plan. Runtime batch/timeout settings can change for a future attempt; never rewrite a plan or checkpoint to change quality or manufacture completion. Existing pooled plans without render_quality keep10 extra updates; older plans without camera_count keep the legacy renderer. A new quality or camera count requires a new job. Already completed jobs must not be rerendered.

After an upgrade, start a new Hermes conversation and verify `isaac_dataset_configure` is available and get_config reports `render_quality_default=normal`. If the active gateway still caches the old interface, reload/restart it through the installed Hermes version's supported controls; do not silently ignore quality or configuration requests.

## 設定異動

`default_scene: null`可清除舊預設場景。已有執行中或待恢復job時保持 `run_root` 不變；變更此值不會搬移或停止舊job，必須使用回傳的 `previous_run_root` 恢復查詢位置才能控制舊job。不要因新根目錄查無job就重新提交pipeline。
