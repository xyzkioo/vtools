import tempfile
import unittest
from pathlib import Path

from PIL import Image

from model_compression.core.model_runtime import build_dataloaders


class ClassificationMappingTests(unittest.TestCase):
    def test_missing_validation_class_cannot_shift_label_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, categories in (("train", ("A", "B")), ("val", ("B",))):
                for category in categories:
                    target = root / split / category
                    target.mkdir(parents=True)
                    Image.new("RGB", (2, 2)).save(target / "sample.png")
            with self.assertRaisesRegex(ValueError, "类别目录必须一致"):
                build_dataloaders({"dataset": {"train": str(root / "train"), "val": str(root / "val")}})


if __name__ == "__main__":
    unittest.main()
