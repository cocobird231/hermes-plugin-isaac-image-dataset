"""Independent Hermes image-localization dataset plugin."""
from pathlib import Path
from .dataset.tools import register as register_tools


def register(ctx):
    register_tools(ctx)
    ctx.register_skill("isaac-image-dataset", Path(__file__).parent/"skills/isaac-image-dataset/SKILL.md",
                       description="Collect semantic-scene RGB and camera-pose datasets with batching and recovery.")
