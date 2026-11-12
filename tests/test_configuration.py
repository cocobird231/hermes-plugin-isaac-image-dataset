"""Persistent first-use configuration, independent of the developer machine."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import jobs, settings, tools


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"ISAAC_DATASET_CONFIG", "ISAAC_DATASET_RUN_ROOT"}}
        environment.update(XDG_CONFIG_HOME=str(self.root / 'config'),
                           XDG_DATA_HOME=str(self.root / 'data'), XDG_CACHE_HOME=str(self.root / 'cache'))
        self.env = patch.dict(os.environ, environment, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_clean_install_readable_without_falling_back_to_hermes_python(self):
        result = tools.dispatch('get_config', {})
        self.assertIsNone(result['runtime']['isaac_python'])
        self.assertFalse(result['configuration_complete'])
        self.assertEqual(result['missing_required_settings'], ['isaac_python'])
        self.assertEqual(result['rendering']['render_quality_default'], 'normal')
        self.assertFalse(settings.config_path().exists())
        with patch.object(jobs, '_launch') as launch:
            with self.assertRaisesRegex(ValueError, 'isaac_dataset_configure'):
                jobs.submit('sample', {'scene_path': '/scene.usda'})
            launch.assert_not_called()
        self.assertFalse(Path(result['runtime']['run_root']).exists())

    def test_configuration_is_persistent_private_and_user_paths_survive(self):
        scene = self.root / 'semantic.usda'; scene.write_text('#usda 1.0\n')
        entry = self.root / 'python-entry'; entry.symlink_to(sys.executable)
        args = {'settings': {'isaac_python': str(entry), 'default_scene': str(scene)}}
        original = json.loads(json.dumps(args))
        result = tools.dispatch('configure', args)
        self.assertTrue(result['saved'])
        self.assertTrue(result['configuration_complete'])
        self.assertEqual(args, original)
        self.assertEqual(result['runtime']['isaac_python'], str(entry))
        self.assertEqual(stat.S_IMODE(settings.config_path().stat().st_mode), 0o600)
        later = tools.dispatch('get_config', {})
        self.assertEqual(later['runtime'], result['runtime'])
        self.assertTrue(later['default_scene_configured'])
        with patch.object(jobs, 'submit') as submit:
            tools.dispatch('sample', {'config': {'n': 4, 'render_quality': 'efficiency'}})
            self.assertEqual(submit.call_args.args[1]['scene_path'], str(scene))

    def test_agent_can_clear_deleted_optional_default_scene(self):
        scene = self.root / 'semantic.usda'; scene.write_text('#usda 1.0\n')
        tools.dispatch('configure', {'settings': {'isaac_python': sys.executable, 'default_scene': str(scene)}})
        scene.unlink()
        result = tools.dispatch('configure', {'settings': {'default_scene': None}})
        self.assertIsNone(result['runtime']['default_scene'])
        self.assertFalse(result['default_scene_configured'])
        tools.dispatch('configure', {'settings': {'batch_size': 12}})
        with patch.object(jobs, 'submit') as submit:
            tools.dispatch('sample', {'config': {'scene_path': '/per-job.usda'}})
            self.assertEqual(submit.call_args.args[1]['scene_path'], '/per-job.usda')
        with self.assertRaises(ValueError):
            tools.dispatch('configure', {'settings': {'isaac_python': None}})

    def test_root_change_reports_previous_root_without_moving_jobs(self):
        first = tools.dispatch('configure', {'settings': {'isaac_python': sys.executable}})
        old = Path(first['runtime']['run_root']); old.mkdir(parents=True)
        marker = old / 'existing'; marker.write_text('unchanged')
        result = tools.dispatch('configure', {'settings': {'run_root': str(self.root / 'new-runs')}})
        self.assertEqual(result['previous_run_root'], str(old))
        self.assertIn('restore', result['job_lookup_note'])
        self.assertEqual(marker.read_text(), 'unchanged')
        self.assertFalse((self.root / 'new-runs').exists())

    def test_environment_override_is_not_copied_into_saved_config(self):
        first = settings.configure_runtime_config({'isaac_python': sys.executable})
        with patch.dict(os.environ, {'ISAAC_DATASET_RUN_ROOT': str(self.root / 'temporary-run')}):
            result = settings.configure_runtime_config({'batch_size': 16})
            self.assertEqual(result['run_root'], first['run_root'])
            self.assertNotEqual(settings.load_runtime_config()['run_root'], first['run_root'])
        self.assertEqual(settings.load_runtime_config()['batch_size'], 16)

    def test_concurrent_patches_do_not_lose_unrelated_fields(self):
        settings.configure_runtime_config({'isaac_python': sys.executable})
        changes = [{'batch_size': 12}, {'sampling_timeout_seconds': 700},
                   {'per_sample_timeout_seconds': 100}, {'startup_timeout_seconds': 800}]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(settings.configure_runtime_config, changes))
        result = settings.load_runtime_config()
        for change in changes:
            for key, value in change.items():
                self.assertEqual(result[key], value)

    def test_invalid_update_preserves_saved_bytes(self):
        settings.configure_runtime_config({'isaac_python': sys.executable})
        before = settings.config_path().read_bytes()
        for change in ({'isaac_python': str(self.root / 'missing')}, {'default_scene': str(self.root / 'missing.usda')},
                       {'batch_size': True}, {'api_key': 'not-a-config-field'}, {'run_root': 'relative/path'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings.configure_runtime_config(change)
            self.assertEqual(settings.config_path().read_bytes(), before)

    def test_malformed_config_is_never_overwritten(self):
        settings.config_path().parent.mkdir(parents=True)
        settings.config_path().write_text('{broken')
        with self.assertRaises(ValueError):
            settings.configure_runtime_config({'batch_size': 12})
        self.assertEqual(settings.config_path().read_text(), '{broken')

    def test_config_symlink_rejected(self):
        target = self.root / 'other.json'; target.write_text('{}')
        settings.config_path().parent.mkdir(parents=True)
        settings.config_path().symlink_to(target)
        with self.assertRaises(ValueError):
            settings.configure_runtime_config({'batch_size': 12})
        self.assertEqual(target.read_text(), '{}')

    def test_custom_config_path_and_empty_tool_patch(self):
        destination = self.root / 'custom.json'
        result = settings.configure_runtime_config({}, path=destination)
        self.assertIsNone(result['isaac_python'])
        self.assertTrue(destination.is_file())
        self.assertFalse(settings.config_path().exists())
        for payload in ({'settings': {}}, {'settings': {'render_quality': 'normal'}},
                        {'settings': {'isaac_python': ''}}, {'settings': {}, 'extra': 1}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                tools.dispatch('configure', payload)

    def test_quality_schema_rejects_invalid_values_before_dispatch(self):
        for value in ('fast', None, 1, True):
            with self.subTest(value=value), patch.object(jobs, 'submit') as submit, self.assertRaises(ValueError):
                tools.dispatch('sample', {'config': {'scene_path': '/scene.usda', 'render_quality': value}})
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
