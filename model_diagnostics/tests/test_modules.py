import unittest
from collections import defaultdict

try:
    from diagnostics.engine import Dataset, ImageInfo, Instance
    from diagnostics.modules import overlap_analysis, resolve_modules
except ModuleNotFoundError:
    from model_diagnostics.diagnostics.engine import Dataset, ImageInfo, Instance
    from model_diagnostics.diagnostics.modules import overlap_analysis, resolve_modules


class ModuleSelectionTests(unittest.TestCase):
    def test_only_selects_one_module(self):
        values = resolve_modules({}, only=["diagnostics.overlap"])
        self.assertTrue(values["diagnostics.overlap"])
        self.assertFalse(values["diagnostics.missed"])

    def test_conflicting_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_modules({}, enable=["diagnostics.overlap"], disable=["diagnostics.overlap"])

    def test_partial_new_style_config_turns_omitted_modules_off(self):
        values = resolve_modules({"modules": {"diagnostics.overlap": True}})
        self.assertTrue(values["diagnostics.overlap"])
        self.assertFalse(values["diagnostics.missed"])

    def test_images_require_bad_case_module(self):
        with self.assertRaises(ValueError):
            resolve_modules({"modules": {"diagnostics.overlap": True, "output.images": True}})


class OverlapTests(unittest.TestCase):
    def test_target_count_is_deduplicated_and_distinct_targets_are_not_duplicates(self):
        dataset = Dataset(
            images={"a": ImageInfo("a")},
            gt={"a": [Instance("a", (0, 0, 10, 10), 0, instance_id="g0"), Instance("a", (2, 0, 12, 10), 0, instance_id="g1"), Instance("a", (4, 0, 14, 10), 0, instance_id="g2")]},
            predictions=defaultdict(list, {"a": [Instance("a", (0, 0, 10, 10), 0, 0.9, "p0"), Instance("a", (2, 0, 12, 10), 0, 0.8, "p1")]}),
        )
        dataset.diagnostic_per_gt = [{"image_id": "a", "gt_id": "g0", "matched": True}, {"image_id": "a", "gt_id": "g1", "matched": True}, {"image_id": "a", "gt_id": "g2", "matched": True}, {"image_id": "a", "prediction_id": "p0", "matched_gt_id": "g0"}, {"image_id": "a", "prediction_id": "p1", "matched_gt_id": "g1"}]
        result = overlap_analysis(dataset, {"score_threshold": 0.25, "overlap": {"iou_threshold": 0.3, "smaller_area_threshold": 0.7, "criterion": "either"}})
        self.assertEqual(result["summary"]["gt_overlapping_target_count"], 3)
        self.assertEqual(result["summary"]["prediction_duplicate_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
