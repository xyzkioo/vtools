"""Falsey YAML scalars/lists must not become default model configurations."""

import tempfile
import unittest
from pathlib import Path

from model_compression.core.config import load_config as load_compression
from model_visualization.core.common import load_config as load_visualization
from speed_test.core.vision_benchmark_config import load_config as load_benchmark
from vtools_ui.webapp.core import parse_yaml


class ConfigShapeTests(unittest.TestCase):
    def test_ui_rejects_falsey_non_mapping_yaml(self):
        for value in ("[]", "false", "0", "''"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "顶层"):
                parse_yaml(value)
        self.assertEqual(parse_yaml(""), {})

    def test_model_readers_reject_falsey_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            for value in ("[]", "false", "0", "''"):
                config.write_text(value, encoding="utf-8")
                for reader in (load_compression, load_visualization, load_benchmark):
                    with self.subTest(value=value, reader=reader.__module__), self.assertRaisesRegex((ValueError, TypeError), "顶层"):
                        reader(config)
