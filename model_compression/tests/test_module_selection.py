import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from model_compression.core.module_selection import resolve_compression_modules


class CompressionModuleSelectionTests(unittest.TestCase):
    def test_effective_config_records_cli_enabled_module(self):
        from model_compression import run_model_compression as entry

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "project:\n  root: .\nrun:\n  root: runs\nmodel:\n  task: classify\n"
                "modules:\n  compression.prune.structured: false\n", encoding="utf-8")
            with patch("sys.argv", ["compression", "--config", str(config_path),
                                    "--enable", "compression.prune.structured"]):
                self.assertEqual(entry.main(), 1)  # No weights: execution fails after recording selection.
            run_dir = root / "runs" / "run1"
            effective = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(effective["modules"]["compression.prune.structured"])
            self.assertEqual(summary["executed_modules"], ["compression.prune.structured"])

    def test_only_selection_disables_omitted_modules(self):
        values = resolve_compression_modules({}, only=["model.parameters"])
        self.assertTrue(values["model.parameters"])
        self.assertFalse(values["baseline.evaluate"])

    def test_explicit_empty_only_disables_all_modules(self):
        for only in ([], [""], [" , "]):
            with self.subTest(only=only):
                self.assertFalse(any(resolve_compression_modules({"modules": {"model.parameters": True}}, only=only).values()))
        with self.assertRaises(ValueError):
            resolve_compression_modules({}, only=[""], enable=["model.parameters"])

    def test_partial_config_is_an_allow_list(self):
        values = resolve_compression_modules({"modules": {"compression.prune.unstructured": True}})
        self.assertTrue(values["compression.prune.unstructured"])
        self.assertFalse(values["compression.quantize.dynamic_int8"])

    def test_conflicting_cli_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_compression_modules({}, enable=["model.parameters"], disable=["model.parameters"])

    def test_metric_comparison_leaves_missing_values_unknown(self):
        from model_compression.core.metrics import compare_to_baseline

        values = compare_to_baseline({"size_bytes": 100}, {"size_bytes": 50})
        self.assertEqual(values["compression_ratio"], 2.0)
        self.assertIsNone(values["accuracy_delta"])


if __name__ == "__main__":
    unittest.main()


class ExplicitTaskTests(unittest.TestCase):
    def test_explicit_classification_overrides_dataset_inference(self):
        from model_compression.core.detection import is_detection
        self.assertFalse(is_detection({'model': {'task': 'classify'}, 'dataset': {'data': 'old_detection.yaml'}}))
        self.assertTrue(is_detection({'model': {'task': 'detect'}}))
