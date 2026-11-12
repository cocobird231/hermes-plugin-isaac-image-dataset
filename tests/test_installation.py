"""Installation isolation and portable runtime paths; no host Hermes mutations."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("installer", ROOT/"scripts/install_plugin.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root/"hermes"
        self.config = self.root/"config.json"
        self.existing = self.home/"plugins/isaac-sim"
        self.existing.mkdir(parents=True)
        (self.existing/"keep.txt").write_text("untouched")
        (self.home/"config.yaml").write_text("existing configuration stays byte identical\n")
        self.python = self.root/"python"
        self.python.symlink_to(sys.executable)

    def command(self, *extra):
        return [sys.executable, str(ROOT/"scripts/install_plugin.py"), "--config", str(self.config),
                "--hermes-home", str(self.home), "--isaac-python", str(self.python), *extra]

    def test_install_and_reinstall_preserve_other_plugin_and_venv_path(self):
        for _ in range(2):
            result = subprocess.run(self.command(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        target = self.home/"plugins/isaac-image-dataset"
        self.assertTrue(target.is_symlink())
        self.assertEqual(target.resolve(), ROOT)
        config = json.loads(self.config.read_text())
        self.assertEqual(config["isaac_python"], str(self.python))
        self.assertEqual((self.existing/"keep.txt").read_text(), "untouched")
        self.assertEqual((self.home/"config.yaml").read_text(), "existing configuration stays byte identical\n")
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)

    def test_install_without_paths_does_not_select_host_python(self):
        command = [sys.executable, str(ROOT/"scripts/install_plugin.py"), "--config", str(self.config),
                   "--hermes-home", str(self.home)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(self.config.read_text())
        self.assertIsNone(config["isaac_python"])
        self.assertIsNone(config["default_scene"])
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)
        self.assertIn("isaac_dataset_configure", result.stdout)

    def test_reinstall_preserves_configuration_and_accepts_runtime_limits(self):
        scene = self.root/"semantic scene.usda"
        scene.write_text("#usda 1.0\n")
        args = self.command("--scene", str(scene), "--batch-size", "36",
                            "--run-root", str(self.root/"dataset output"),
                            "--startup-timeout-seconds", "900",
                            "--per-sample-timeout-seconds", "120",
                            "--sampling-timeout-seconds", "800")
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        first = json.loads(self.config.read_text())
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.config.read_text()), first)
        self.assertEqual(first["default_scene"], str(scene))
        self.assertEqual(first["startup_timeout_seconds"], 900)
        self.assertEqual(first["batch_size"], 36)

    def test_native_install_directory_can_be_configured_in_place(self):
        import argparse
        target = self.home/"plugins/isaac-image-dataset"
        target.mkdir()
        marker = target/"plugin.yaml"
        marker.write_text("name: isaac-image-dataset\n")
        args = argparse.Namespace(config=self.config, isaac_python=str(self.python), scene=None,
                                  run_root=None, gpu_lock_path=None, batch_size=None,
                                  hermes_home=self.home, hermes=None, enable=False)
        with patch.object(installer, "ROOT", target):
            installer.install(args)
        self.assertFalse(target.is_symlink())
        self.assertEqual(marker.read_text(), "name: isaac-image-dataset\n")
        self.assertEqual(json.loads(self.config.read_text())["isaac_python"], str(self.python))

    def test_xdg_config_location_is_used_without_explicit_config(self):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("ISAAC_DATASET_")}
        env.update({"XDG_CONFIG_HOME": str(self.root/"xdg-config"),
                    "XDG_DATA_HOME": str(self.root/"xdg-data"),
                    "XDG_CACHE_HOME": str(self.root/"xdg-cache")})
        command = [sys.executable, str(ROOT/"scripts/install_plugin.py"),
                   "--hermes-home", str(self.home)]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root/"xdg-config/isaac-image-dataset/config.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["run_root"], str(self.root/"xdg-data/isaac-image-dataset/runs"))
        self.assertEqual(config["gpu_lock_path"], str(self.root/"xdg-cache/isaac-image-dataset/gpu.lock"))

    def test_nonexecutable_interpreter_is_rejected(self):
        fake = self.root/"not-executable"
        fake.write_text("placeholder\n")
        fake.chmod(0o600)
        command = [sys.executable, str(ROOT/"scripts/install_plugin.py"), "--config", str(self.config),
                   "--hermes-home", str(self.home), "--isaac-python", str(fake)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.config.exists())
        self.assertFalse((self.home/"plugins/isaac-image-dataset").exists())

    def test_configuration_symlink_is_not_followed(self):
        external = self.root/"outside.json"
        external.write_text("{}")
        self.config.symlink_to(external)
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(external.read_text(), "{}")
        self.assertFalse((self.home/"plugins/isaac-image-dataset").exists())

    def test_environment_cannot_mask_invalid_persisted_config(self):
        self.config.write_text(json.dumps({"run_root":"relative-invalid"}))
        original = self.config.read_bytes()
        result = subprocess.run(self.command(), capture_output=True, text=True,
                                env={**os.environ,"ISAAC_DATASET_RUN_ROOT":str(self.root/"valid")})
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.config.read_bytes(),original)
        self.assertFalse((self.home/"plugins/isaac-image-dataset").exists())

    def test_invalid_limits_do_not_install(self):
        result = subprocess.run(self.command("--batch-size", "0"), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.config.exists())
        self.assertFalse((self.home/"plugins/isaac-image-dataset").exists())

    def test_existing_destination_is_not_overwritten(self):
        target = self.home/"plugins/isaac-image-dataset"
        target.mkdir()
        (target/"owned").write_text("keep")
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((target/"owned").read_text(), "keep")
        self.assertFalse(self.config.exists())

    def test_missing_scene_fails_before_write(self):
        result = subprocess.run(self.command("--scene", str(self.root/"missing.usda")), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.config.exists())

    def test_enable_uses_cli_only_for_own_plugin(self):
        import argparse
        args = argparse.Namespace(config=self.config, isaac_python=str(self.python), scene=None,
                                  run_root=None, gpu_lock_path=None, batch_size=None,
                                  hermes_home=self.home, hermes="/opt/hermes", enable=True)
        with patch.object(installer.subprocess, "run") as run:
            installer.install(args)
        self.assertEqual(run.call_args.args[0], ["/opt/hermes", "plugins", "enable", "isaac-image-dataset"])


if __name__ == "__main__":
    unittest.main()
