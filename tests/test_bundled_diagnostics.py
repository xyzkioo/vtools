"""Saved-prediction diagnostics must run without importing model frameworks."""

import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

import run_tools


class SavedPredictionDiagnosticsTests(unittest.TestCase):
    def test_report_and_bad_case_images_do_not_import_torch_or_ultralytics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "val" / "sample.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(image)
            label = root / "labels" / "val" / "sample.txt"
            label.parent.mkdir(parents=True)
            label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"path": str(root), "val": "images/val", "names": ["object"]}), encoding="utf-8")
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps({"images": [{"image_id": "images/val/sample.png", "predictions": []}]}), encoding="utf-8")
            config = root / "diagnostics.yaml"
            config.write_text(yaml.safe_dump({
                "project": {"root": str(root)}, "mode": "predictions_file",
                "predictions_file": {"name": "saved", "predictions": str(predictions)},
                "dataset": {"data": str(data)},
                "modules": {"diagnostics.missed": True, "output.bad_cases": True, "output.images": True, "output.html": True},
            }), encoding="utf-8")
            output = root / "report"
            real_import = builtins.__import__

            def import_without_frameworks(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "ultralytics"}:
                    raise AssertionError(f"Saved predictions must not import {name}")
                return real_import(name, *args, **kwargs)

            with patch.object(sys, "argv", ["run_tools.py", "--tool", "diagnostics", "--config", str(config), "--run-dir", str(output)]), \
                    patch.object(builtins, "__import__", side_effect=import_without_frameworks):
                self.assertEqual(run_tools.main(), 0)
            summary = json.loads((output / "saved" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["error_counts"], {"missed": 1})
            self.assertEqual(len(list((output / "saved" / "images").rglob("*.jpg"))), 1)
            self.assertTrue((output / "saved" / "index.html").is_file())
