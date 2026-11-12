"""Recover an interrupted legacy plan publication only from its exact committed SHA."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import jobs, sampling, worker


def encoded(plan):
    return (json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode("utf-8")


class MulticamLegacyPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"n": 2, "scene_path": "/fixture.usda"}
        self.request = {"job_id": "a"*32, "configuration": self.config}
        jobs._write(self.root, "config.json", self.config)
        self.generated = {"schema_version": 1, "complete": True,
                          "config": {**self.config, "camera_count": 1},
                          "samples": [{"sample_id": "0", "pose": [1, 2, 3]},
                                      {"sample_id": "1", "pose": [4, 5, 6]}]}
        self.legacy = copy.deepcopy(self.generated)
        self.legacy["config"].pop("camera_count")

    def commit_gap(self, plan):
        write = jobs._write
        def interrupt_publication(job, name, value, **kwargs):
            if name == "plan.json":
                raise OSError("injected interruption after receipt")
            return write(job, name, value, **kwargs)
        with patch.object(jobs, "_write", side_effect=interrupt_publication):
            with self.assertRaisesRegex(OSError, "after receipt"):
                worker.commit_plan(self.root, self.request, plan)
        self.assertFalse((self.root/"plan.json").exists())
        return (self.root/"plan-commit.json").read_bytes()

    def execute(self):
        with patch.object(sampling, "generate_plan", return_value=self.generated):
            return worker.sample_plan(self.root, self.request)

    def assert_rejected_without_publish(self, receipt):
        with self.assertRaises(ValueError):
            self.execute()
        self.assertFalse((self.root/"plan.json").exists())
        self.assertEqual((self.root/"plan-commit.json").read_bytes(), receipt)

    def test_interrupted_legacy_plan_recovers_exact_bytes_without_mutation(self):
        receipt = self.commit_gap(self.legacy)
        generated_before = copy.deepcopy(self.generated)
        recovered = self.execute()
        self.assertEqual(recovered, self.legacy)
        self.assertEqual((self.root/"plan.json").read_bytes(), encoded(self.legacy))
        self.assertEqual((self.root/"plan-commit.json").read_bytes(), receipt)
        self.assertEqual(self.generated, generated_before)
        self.assertEqual(worker.validate_plan_commit(self.root, self.request), self.legacy)

    def test_fresh_default_count_is_kept(self):
        plan = self.execute()
        self.assertEqual(plan["config"]["camera_count"], 1)
        receipt = jobs._read(self.root, "plan-commit.json")
        self.assertEqual(receipt["plan_sha256"], hashlib.sha256(encoded(self.generated)).hexdigest())

    def test_new_publication_gap_keeps_explicit_normalized_count(self):
        receipt = self.commit_gap(self.generated)
        self.assertEqual(self.execute(), self.generated)
        self.assertEqual((self.root/"plan-commit.json").read_bytes(), receipt)

    def test_explicit_camera_configuration_never_uses_legacy_fallback(self):
        self.config["camera_count"] = 1
        jobs._write(self.root, "config.json", self.config)
        receipt = self.commit_gap(self.legacy)
        self.assert_rejected_without_publish(receipt)

    def test_nondefault_generated_camera_count_is_not_stripped(self):
        receipt = self.commit_gap(self.legacy)
        self.generated["config"]["camera_count"] = 2
        self.assert_rejected_without_publish(receipt)

    def test_other_plan_changes_are_not_authorized_by_legacy_fallback(self):
        receipt = self.commit_gap(self.legacy)
        self.generated["samples"][0]["pose"][0] += 1
        self.assert_rejected_without_publish(receipt)

    def test_changed_configuration_on_disk_cannot_use_legacy_receipt(self):
        receipt = self.commit_gap(self.legacy)
        jobs._write(self.root, "config.json", {**self.config, "n": 99})
        self.assert_rejected_without_publish(receipt)

    def test_corrupt_receipt_identity_and_hash_are_rejected(self):
        original = self.commit_gap(self.legacy)
        for field, value in (("job_id", "b"*32), ("configuration_sha256", "0"*64),
                             ("plan_sha256", "1"*64)):
            with self.subTest(field=field):
                corrupt = json.loads(original)
                corrupt[field] = value
                jobs._write(self.root, "plan-commit.json", corrupt)
                self.assert_rejected_without_publish((self.root/"plan-commit.json").read_bytes())

    def test_unbound_existing_plan_is_never_adopted(self):
        jobs._write(self.root, "plan.json", self.legacy)
        before = (self.root/"plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot adopt"):
            self.execute()
        self.assertFalse((self.root/"plan-commit.json").exists())
        self.assertEqual((self.root/"plan.json").read_bytes(), before)

    def test_existing_bound_legacy_plan_is_not_replaced(self):
        worker.commit_plan(self.root, self.request, self.legacy)
        before = (self.root/"plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "differs"):
            self.execute()
        self.assertEqual((self.root/"plan.json").read_bytes(), before)
        self.assertEqual(worker.validate_plan_commit(self.root, self.request), self.legacy)


if __name__ == "__main__":
    unittest.main()
