import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from PIL import Image

from model_diagnostics.diagnostics.engine import (
    Dataset, ImageInfo, Instance, _summary_findings, evaluate_operating, load_ground_truth,
    load_predictions, validate_config,
)
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

    def test_explicit_empty_only_disables_all_modules(self):
        for only in ([], [""], [" , "]):
            with self.subTest(only=only):
                self.assertFalse(any(resolve_modules({"modules": {"diagnostics.missed": True}}, only=only).values()))
        with self.assertRaises(ValueError):
            resolve_modules({}, only=[""], enable=["diagnostics.missed"])

    def test_reserved_validation_does_not_silently_succeed(self):
        with self.assertRaisesRegex(ValueError, "预留模块"):
            resolve_modules({"modules": {"validation.ultralytics": True}})

    def test_foreign_speed_module_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知模块"):
            resolve_modules({"modules": {"speed.pytorch_call": True}})

    def test_zero_true_positives_have_zero_f1(self):
        from model_diagnostics.diagnostics.engine import _prf
        self.assertEqual(_prf(0, 1, 1)["f1"], 0.0)

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


class GroundTruthInputTests(unittest.TestCase):
    def test_explicit_canonical_gt_keeps_xyxy_in_image_and_annotation_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.json"
            path.write_text(json.dumps({
                "images": [{"id": "sample"}],
                "annotations": [{"image_id": "sample", "class_id": 0, "bbox": [10, 10, 20, 20]}],
            }), encoding="utf-8")

            dataset = load_ground_truth(path, "canonical")
            dataset.predictions["sample"] = [Instance("sample", (10, 10, 20, 20), 0, 0.9)]
            metrics = evaluate_operating(dataset, {})["aggregate"]
            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))
            self.assertEqual(load_ground_truth(path, "auto").gt["sample"][0].bbox, (10, 10, 30, 30))

    def test_binary_yolo_label_reports_its_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "val"
            labels = root / "labels" / "val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(images / "sample.png")
            label = labels / "sample.txt"
            label.write_bytes(b"\x00\x48\xc7\x04")
            data = root / "dataset.yaml"
            data.write_text("path: .\nval: images/val\nnames: [ball]\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "YOLO 标签不是 UTF-8 文本") as error:
                load_ground_truth(data, "ultralytics", split="val")
            self.assertIn(str(label), str(error.exception))


class OverlapTests(unittest.TestCase):
    def test_duplicate_predictions_are_matched_within_their_image(self):
        box = (0, 0, 10, 10)
        dataset = Dataset(
            gt={"a": [Instance("a", box, 0, instance_id="g0")],
                "b": [Instance("b", box, 0, instance_id="g0"), Instance("b", box, 0, instance_id="g1")]},
            predictions={key: [Instance(key, box, 0, 0.9, "p0"), Instance(key, box, 0, 0.8, "p1")]
                         for key in ("a", "b")},
        )
        operating = evaluate_operating(dataset, {})
        dataset.diagnostic_per_gt = operating["per_gt"]
        dataset.diagnostic_per_prediction = operating["per_prediction"]
        pairs = overlap_analysis(dataset, {})["prediction_pairs"]
        self.assertEqual({pair["image_id"]: pair["suspected_duplicate"] for pair in pairs}, {"a": True, "b": False})

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
    def test_flat_canonical_predictions_keep_xyxy_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            box = (10, 10, 20, 20)
            path.write_text(json.dumps([{"image_id": "sample", "bbox": box, "score": 0.9, "class_id": 0}]), encoding="utf-8")
            dataset = Dataset(gt={"sample": [Instance("sample", box, 0)]})
            for fmt in ("canonical", "auto"):
                with self.subTest(fmt=fmt):
                    dataset.predictions, _ = load_predictions(path, dataset.images, fmt)
                    self.assertEqual(evaluate_operating(dataset, {})["aggregate"]["tp"], 1)
            coco, _ = load_predictions(path, dataset.images, "coco")
            self.assertEqual(coco["sample"][0].bbox, (10, 10, 30, 30))

    def test_coco_crowd_ignores_contained_same_class_predictions_after_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.json"
            path.write_text(json.dumps({
                "images": [{"id": "sample"}],
                "annotations": [
                    {"image_id": "sample", "bbox": [0, 0, 100, 100], "category_id": 0, "iscrowd": 1},
                    {"image_id": "sample", "bbox": [10, 10, 10, 10], "category_id": 0},
                ],
            }), encoding="utf-8")
            dataset = load_ground_truth(path, "coco")
            dataset.predictions["sample"] = [
                Instance("sample", (10, 10, 20, 20), 0, 0.9, "tp"),
                Instance("sample", (30, 30, 40, 40), 0, 0.8, "ignored"),
                Instance("sample", (30, 30, 40, 40), 1, 0.8, "fp"),
            ]
            result = evaluate_operating(dataset, {})
            aggregate = result["aggregate"]
            self.assertEqual((aggregate["tp"], aggregate["fp"], aggregate["fn"]), (1, 1, 0))
            self.assertEqual(aggregate["prediction_count_at_threshold"], 2)
            self.assertEqual(aggregate["below_threshold_prediction_count"], 0)
            self.assertEqual([event["prediction_id"] for event in result["error_events"]], ["fp"])

    def test_zero_iou_threshold_with_empty_gt_keeps_background_prediction(self):
        dataset = Dataset(predictions={"sample": [Instance("sample", (0, 0, 10, 10), 0, 0.9, "p")]})
        result = evaluate_operating(dataset, {"match_iou": 0})
        self.assertEqual(result["aggregate"]["fp"], 1)
        self.assertEqual(result["aggregate"]["tp"], 0)
        self.assertEqual(result["error_events"][0]["error_type"], "background")

    def test_findings_count_low_score_candidates_and_unique_confused_targets(self):
        box = (0, 0, 10, 10)
        dataset = Dataset(
            images={key: ImageInfo(key) for key in ("a", "b")},
            gt={key: [Instance(key, box, 0, instance_id="g0")] for key in ("a", "b")},
            predictions={
                "a": [Instance("a", box, 1, 0.9, "p0"), Instance("a", box, 1, 0.8, "p1")],
                "b": [Instance("b", box, 0, 0.1, "p0")],
            },
            class_names={0: "A", 1: "B"},
        )
        result = evaluate_operating(dataset, {"score_threshold": 0.25})
        findings = _summary_findings(result, dataset.class_names, resolve_modules({}))
        self.assertEqual(result["aggregate"]["error_counts"]["classification"], 2)
        self.assertEqual(findings, [
            {
                "type": "missed", "class_id": 0, "class_name": "A", "density_group": "gt_per_image_1_4",
                "gt_count": 2, "missed_count": 2, "low_confidence_candidate_count": 1,
            },
            {
                "type": "classification", "class_id": 0, "class_name": "A",
                "predicted_class_id": 1, "predicted_class_name": "B",
                "gt_count": 2, "confused_gt_count": 1,
            },
        ])
        self.assertEqual(_summary_findings(result, dataset.class_names, {}), [])

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
        self.assertTrue(rows[0]["case_name"].startswith("classification_"))
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
        self.assertTrue(rows[0]["case_name"].startswith("mixed_"))


if __name__ == "__main__":
    unittest.main()
