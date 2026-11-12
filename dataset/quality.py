"""Render policy for new requests and immutable plans from earlier versions.

Presets control extra Kit updates after one synchronous group capture step.
They do not replace scene initialization or promise a particular image quality.
"""

DEFAULT_RENDER_QUALITY = "normal"
RENDER_QUALITY_UPDATES = {"efficiency": 0, "normal": 1, "quality": 10}


def normalize_render_quality(value):
    if not isinstance(value, str) or value not in RENDER_QUALITY_UPDATES:
        raise ValueError("render_quality must be efficiency, normal or quality")
    return value


def render_policy(config):
    """Resolve a committed plan without inserting defaults or changing its bytes."""
    if "render_quality" in config:
        normalize_render_quality(config["render_quality"])
        if "camera_count" not in config:
            raise ValueError("render_quality requires camera_count in a committed plan")
    if "camera_count" not in config:
        return {"backend": "legacy_camera_renderer", "render_quality": "legacy",
                "render_quality_source": "legacy_single_camera", "shared_updates": None}
    mode = config.get("render_quality", "quality")
    return {"backend": "shared_render_products", "render_quality": mode,
            "render_quality_source": "plan" if "render_quality" in config else "legacy_camera_pool",
            "shared_updates": RENDER_QUALITY_UPDATES[mode]}
