import unittest
from collections import defaultdict

try:
    from diagnostics.engine import Dataset, ImageInfo, Instance, evaluate_operating, validate_config
    from diagnostics.modules import bad_case_rows, overlap_analysis, resolve_modules
except ModuleNotFoundError:
    from model_diagnostics.diagnostics.engine import Dataset, ImageInfo, Instance, evaluate_operating, validate_config
    from model_diagnostics.diagnostics.modules import bad_case_rows, overlap_analysis, resolve_modules


class ModuleSelectionTests(unittest.TestCase):
    def test_invalid_diagnostic_thresholds_are_rejected(self):
        invalid = (
            {"score_threshold": 1.1},
            {"candidate_threshold": 0.5, "score_threshold": 0.25},
            {"candidate_iou_thresholds": []},
            {"score_sweep": [-0.1]},
            {"fp_budgets_per_image": [-1]},
            {"bootstrap_images": -1},
            {"bad_cases": {"top_k": -1}},
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

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


class ErrorEventTests(unittest.TestCase):
    def test_wrong_class_prediction_and_missed_gt_are_one_event(self):
        dataset = Dataset(
            images={"long/path/image.jpg": ImageInfo("long/path/image.jpg")},
            gt={"long/path/image.jpg": [Instance("long/path/image.jpg", (0, 0, 10, 10), 0, instance_id="g0")]},
            predictions={"long/path/image.jpg": [Instance("long/path/image.jpg", (0, 0, 10, 10), 1, 0.9, "p0")]},
        )
        result = evaluate_operating(dataset, {"score_threshold": 0.25, "match_iou": 0.5, "localization_iou_floor": 0.1})
        self.assertEqual(result["aggregate"]["error_event_count"], 1)
        self.assertEqual(result["aggregate"]["error_counts"], {"classification": 1})
        rows = bad_case_rows(result)
        self.assertEqual(rows[0]["image_category"], "classification")
        self.assertEqual(rows[0]["error_count"], 1)
        self.assertNotIn("error_event_ids", rows[0])
        self.assertNotIn("classification_count", rows[0])

    def test_mixed_image_is_indexed_once(self):
        dataset = Dataset(
            images={"a": ImageInfo("a")},
            gt={"a": [Instance("a", (0, 0, 10, 10), 0, instance_id="g0")]},
            predictions={"a": [Instance("a", (20, 20, 30, 30), 0, 0.9, "p0")]},
        )
        result = evaluate_operating(dataset, {"score_threshold": 0.25, "match_iou": 0.5, "localization_iou_floor": 0.1})
        self.assertEqual(result["aggregate"]["error_event_count"], 2)
        self.assertEqual(result["aggregate"]["error_image_count"], 1)
        rows = bad_case_rows(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_category"], "mixed")


if __name__ == "__main__":
    unittest.main()
