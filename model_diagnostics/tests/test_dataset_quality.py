"""End-to-end fixtures for the read-only dataset audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from model_diagnostics.check_dataset import audit, main


class DatasetQualityTests(unittest.TestCase):
    def test_rounding_at_image_boundary_is_not_reported_or_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (64, 64), "white").save(image_dir / "edge.png")
            label = label_dir / "edge.txt"
            label.write_text("0 0.992135 0.946960 0.015731 0.076855\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text("path: .\nval: images\nnames: [cat]\n", encoding="utf-8")
            result = audit(data, root / "reports", sample_count=0)

            self.assertEqual(result["class_counts"], {"0": 1})
            self.assertNotIn("out_of_bounds_box", result["issue_counts"])

    def test_report_preserves_inputs_and_finds_cross_split_problems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            train = root / "images" / "train" / "a.png"
            validation = root / "images" / "val" / "copy.png"
            source_image = Image.new("RGB", (64, 64), "white")
            source_image.putpixel((0, 0), (0, 0, 0))
            source_image.save(train)
            validation.write_bytes(train.read_bytes())
            bad = root / "images" / "train" / "bad.png"
            bad.write_bytes(b"not an image")
            missing = root / "images" / "val" / "missing.png"
            Image.new("RGB", (64, 64), "blue").save(missing)
            label = root / "labels" / "train" / "a.txt"
            original = "0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.2 0.2\n0 0.9 0.9 0.4 0.4\n"
            label.write_text(original, encoding="utf-8")
            data = root / "data.yaml"
            data.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [cat]\n", encoding="utf-8")
            output = root / "reports" / "run1"
            output.mkdir(parents=True)

            result = audit(data, output, sample_count=1)

            self.assertEqual(result["status"], "issues_found")
            self.assertEqual(result["splits"], {"train": 2, "val": 2})
            self.assertEqual(result["class_counts"], {"0": 1})
            self.assertEqual(result["box_short_side"]["<16px"], 1)
            self.assertEqual(label.read_text(encoding="utf-8"), original)
            with (output / "issues.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            codes = {row["code"] for row in rows}
            self.assertTrue({"invalid_class", "out_of_bounds_box", "corrupt_image",
                             "missing_label", "split_leak_duplicate"}.issubset(codes))
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "issues_found")
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "samples" / "train_001.png").is_file())
            with Image.open(output / "samples" / "train_001.png") as preview:
                self.assertEqual(preview.getpixel((29, 29)), (255, 0, 0))

    def test_cli_exit_code_and_unique_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "val"
            labels = root / "labels" / "val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            clean_image = Image.new("RGB", (32, 32), "red")
            clean_image.putpixel((0, 0), (0, 0, 0))
            clean_image.save(images / "a.png")
            (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text("path: .\nval: images/val\nnames: [cat]\n", encoding="utf-8")
            argv = ["--data", str(data), "--output-root", str(root / "reports"), "--sample-count", "0"]
            self.assertEqual(main(argv), 0)
            self.assertEqual(main(argv), 0)
            self.assertEqual(json.loads((root / "reports" / "run1" / "summary.json").read_text())["status"], "passed")
            self.assertTrue((root / "reports" / "run2" / "summary.json").is_file())
            (labels / "a.txt").write_text("7 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            self.assertEqual(main(argv), 1)
            self.assertEqual(json.loads((root / "reports" / "run3" / "summary.json").read_text())["status"],
                             "issues_found")
            data.write_text("task: segment\nval: images/val\nnames: [cat]\n", encoding="utf-8")
            self.assertEqual(main(argv), 2)
            self.assertEqual(json.loads((root / "reports" / "run4" / "summary.json").read_text())["status"],
                             "failed")


if __name__ == "__main__":
    unittest.main()
