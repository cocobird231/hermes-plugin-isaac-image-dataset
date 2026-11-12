"""Install this plugin and persist optional machine-specific settings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

NAME = "isaac-image-dataset"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SETTINGS_OPTIONS = (
    ("isaac_python", "isaac_python"),
    ("scene", "default_scene"),
    ("run_root", "run_root"),
    ("gpu_lock_path", "gpu_lock_path"),
    ("batch_size", "batch_size"),
    ("startup_timeout_seconds", "startup_timeout_seconds"),
    ("per_sample_timeout_seconds", "per_sample_timeout_seconds"),
    ("sampling_timeout_seconds", "sampling_timeout_seconds"),
)
PATH_SETTINGS = {"isaac_python", "default_scene", "run_root", "gpu_lock_path"}


def install(args):
    from dataset.settings import config_path as runtime_config_path, configure_runtime_config

    default_config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / NAME / "config.json"
    config_path = Path(args.config or runtime_config_path()).expanduser().absolute()
    home = args.hermes_home.expanduser().absolute()
    destination = home / "plugins" / NAME
    # Native Hermes installation is an ordinary directory; re-running its installer
    # only configures it. An unrelated existing copy must never be replaced implicitly.
    if (destination.exists() or destination.is_symlink()) and destination.resolve() != ROOT:
        raise FileExistsError("Existing plugin location differs; keep it or choose another Hermes home: " + str(destination))

    command = args.hermes or shutil.which("hermes")
    fallback = Path.home() / ".local/bin/hermes"
    if not command and fallback.is_file():
        command = str(fallback)
    if args.enable and not command:
        raise ValueError("Supply --hermes PATH, or omit --enable and enable later from hermes plugins")

    patch = {}
    for argument, setting in SETTINGS_OPTIONS:
        value = getattr(args, argument, None)
        if value is not None:
            if setting in PATH_SETTINGS:
                # Preserve the virtual-environment symlink entry, never its resolved target.
                value = str(Path(value).expanduser().absolute())
            patch[setting] = value
    saved = configure_runtime_config(patch, path=config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and not destination.is_symlink():
        destination.symlink_to(ROOT, target_is_directory=True)
    if args.enable:
        subprocess.run([command, "plugins", "enable", NAME], check=True,
                       env={**os.environ, "HERMES_HOME": str(home)})
    missing = [key for key in ("isaac_python",) if not saved.get(key)]
    print(json.dumps({"installed": True, "plugin": NAME, "plugin_path": str(destination),
                      "source": str(ROOT), "config": str(config_path),
                      "enabled_requested": args.enable, "missing_required_settings": missing},
                     ensure_ascii=False, indent=2))
    if missing:
        print("Before sampling, provide the Isaac Sim Python executable through isaac_dataset_configure.")
    if config_path != default_config:
        print("Set ISAAC_DATASET_CONFIG to this config file whenever starting Hermes: " + str(config_path))
    if not args.enable:
        print("Enable isaac-image-dataset using hermes plugins, then start a new Hermes session.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", help="Isaac Sim environment Python entry point; may be configured later")
    parser.add_argument("--scene", help="Default semantic USD/USDZ scene; may be supplied in a tool call instead")
    parser.add_argument("--config", type=Path, help="Runtime config file; default ~/.config/isaac-image-dataset/config.json")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--startup-timeout-seconds", type=float)
    parser.add_argument("--per-sample-timeout-seconds", type=float)
    parser.add_argument("--sampling-timeout-seconds", type=float)
    parser.add_argument("--hermes-home", type=Path, default=Path(os.environ.get("HERMES_HOME", "~/.hermes")))
    parser.add_argument("--hermes", help="Hermes CLI executable when it is not on PATH")
    parser.add_argument("--enable", action="store_true", help="Enable only this plugin through the Hermes CLI")
    args = parser.parse_args()
    try:
        install(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
