import unittest

from model_compression.core.module_selection import resolve_compression_modules


class CompressionModuleSelectionTests(unittest.TestCase):
    def test_only_selection_disables_omitted_modules(self):
        values = resolve_compression_modules({}, only=["branch.create"])
        self.assertTrue(values["branch.create"])
        self.assertFalse(values["baseline.evaluate"])

    def test_partial_config_is_an_allow_list(self):
        values = resolve_compression_modules({"modules": {"compression.prune.unstructured": True}})
        self.assertTrue(values["compression.prune.unstructured"])
        self.assertFalse(values["compression.quantize.dynamic_int8"])

    def test_conflicting_cli_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_compression_modules({}, enable=["branch.create"], disable=["branch.create"])

    def test_metric_comparison_leaves_missing_values_unknown(self):
        from model_compression.core.metrics import compare_to_baseline

        values = compare_to_baseline({"size_bytes": 100}, {"size_bytes": 50})
        self.assertEqual(values["compression_ratio"], 2.0)
        self.assertIsNone(values["accuracy_delta"])


if __name__ == "__main__":
    unittest.main()
