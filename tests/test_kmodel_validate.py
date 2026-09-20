import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from transform_tools import PY2KM_validate as validate


class RuntimeTensor:
    def __init__(self, value):
        self.value = value

    def to_numpy(self):
        return self.value


class KmodelValidation(unittest.TestCase):
    def test_directory_resolution_returns_sorted_supported_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.JPG").touch()
            (root / "a.png").touch()
            (root / "notes.txt").touch()
            (root / "nested").mkdir()
            self.assertEqual(
                [path.name for path in validate.resolve_image_paths(root)],
                ["a.png", "z.JPG"],
            )

    def test_empty_tensor_is_not_considered_equal(self):
        empty = np.asarray([], dtype=np.float32)
        self.assertFalse(validate.compare([empty], [RuntimeTensor(empty)]))

    def test_invalid_numeric_options_are_rejected_before_file_access(self):
        cases = (
            ["validate", "--size", "0"],
            ["validate", "--color", "0", "256", "0"],
            ["validate", "--cosine-threshold", "1.1"],
            ["validate", "--mae-threshold", "-0.1"],
        )
        for argv in cases:
            with self.subTest(argv=argv), patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
                validate.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
