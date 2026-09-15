import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from model_visualization.core.capture import _aggregate, _channel_indices
from model_visualization.core.common import append_html_index, load_config, parse_size, write_image


class VisualizationOutput(unittest.TestCase):
    def test_input_size_must_be_positive(self):
        for value in (0, -1, [640, 0], "640x-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_size(value)

    def test_detection_limits_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "image.jpg"
            source.touch()
            for line in ("conf: .nan", "iou: 1.1", "max_det: 0"):
                config = root / "config.yaml"
                config.write_text(
                    f"project:\n  root: {root}\ninput:\n  source: {source}\ndetection:\n  {line}\n",
                    encoding="utf-8",
                )
                with self.subTest(line=line), self.assertRaises(ValueError):
                    load_config(config)

    def test_unknown_feature_modes_are_rejected(self):
        array = np.zeros((2, 3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            _channel_indices(array, {"mode": "typo"})
        with self.assertRaises(ValueError):
            _channel_indices(array, {"count": 0})
        with self.assertRaises(ValueError):
            _aggregate(array, "typo")

    def test_failed_image_encoding_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "image.png"
            target.write_bytes(b"original")

            def fail(path, image):
                Path(path).write_bytes(b"partial")
                return False

            with patch("model_visualization.core.common.cv2.imwrite", side_effect=fail):
                with self.assertRaises(OSError):
                    write_image(target, np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_html_index_escapes_names_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_html_index(root, [{"image": "<script>x</script>", "links": ['x\" onmouseover=\"bad']}])
            text = (root / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", text)
            self.assertNotIn('onmouseover="bad', text)
            self.assertIn("&lt;script&gt;", text)


if __name__ == "__main__":
    unittest.main()
