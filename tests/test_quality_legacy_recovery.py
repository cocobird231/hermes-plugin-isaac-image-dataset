"""Exact-byte recovery of plans published before rendering presets existed."""
import copy
import unittest

from dataset import jobs, worker
import test_multicam_legacy_recovery as legacy_fixtures


class QualityLegacyPublicationTests(unittest.TestCase):
    commit_gap = legacy_fixtures.MulticamLegacyPublicationTests.commit_gap
    execute = legacy_fixtures.MulticamLegacyPublicationTests.execute
    assert_rejected_without_publish = legacy_fixtures.MulticamLegacyPublicationTests.assert_rejected_without_publish

    def setUp(self):
        legacy_fixtures.MulticamLegacyPublicationTests.setUp(self)
        self.old_pool = copy.deepcopy(self.generated)
        self.generated["config"]["render_quality"] = "normal"

    def assert_recovery(self, old_plan):
        receipt = self.commit_gap(old_plan)
        original = copy.deepcopy(self.generated)
        self.assertEqual(self.execute(), old_plan)
        self.assertEqual((self.root/"plan.json").read_bytes(), legacy_fixtures.encoded(old_plan))
        self.assertEqual((self.root/"plan-commit.json").read_bytes(), receipt)
        self.assertEqual(self.generated, original)
        self.assertEqual(worker.validate_plan_commit(self.root, self.request), old_plan)

    def test_interrupted_old_pool_plan_recovers_ten_update_policy_bytes(self):
        self.assert_recovery(self.old_pool)

    def test_interrupted_old_single_camera_plan_recovers_without_both_new_defaults(self):
        self.assert_recovery(self.legacy)

    def test_old_explicit_four_camera_plan_recovers_missing_quality_only(self):
        self.config["camera_count"] = 4
        jobs._write(self.root, "config.json", self.config)
        self.generated["config"]["camera_count"] = 4
        self.old_pool["config"]["camera_count"] = 4
        self.assert_recovery(self.old_pool)

    def test_explicit_requested_quality_never_stripped_to_match_old_receipt(self):
        self.config["render_quality"] = "normal"
        jobs._write(self.root, "config.json", self.config)
        receipt = self.commit_gap(self.old_pool)
        self.assert_rejected_without_publish(receipt)

    def test_nondefault_generated_quality_never_stripped(self):
        receipt = self.commit_gap(self.old_pool)
        self.generated["config"]["render_quality"] = "efficiency"
        self.assert_rejected_without_publish(receipt)

    def test_quality_fallback_does_not_authorize_different_pose(self):
        receipt = self.commit_gap(self.old_pool)
        self.generated["samples"][0]["pose"][0] += 1
        self.assert_rejected_without_publish(receipt)

    def test_new_publication_preserves_explicit_normal_quality(self):
        receipt = self.commit_gap(self.generated)
        self.assertEqual(self.execute(), self.generated)
        self.assertEqual((self.root/"plan-commit.json").read_bytes(), receipt)

    def test_bound_old_plan_is_never_rewritten(self):
        worker.commit_plan(self.root, self.request, self.old_pool)
        original = (self.root/"plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "differs"):
            self.execute()
        self.assertEqual((self.root/"plan.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
