from pathlib import Path
import unittest

from model_visualization.run_visualization import _logical_image_id
from model_diagnostics.diagnostics.engine import _ultralytics_image_id
from model_diagnostics.diagnostics.engine import _find_image


class ReviewRegressions(unittest.TestCase):
    def test_image_identity_across_subsets_and_layouts(self):
        for folder in ('images', 'photos'):
            root = Path('/dataset')
            for suffix in ('.jpg', '.png'):
                image = root / folder / ('a' + suffix)
                single = _logical_image_id(image, image, root)
                directory = _logical_image_id(image, image.parent, root)
                diagnostic = _ultralytics_image_id(image, root, set())
                self.assertEqual(single, directory)
                self.assertEqual(single, diagnostic)
                self.assertEqual(single, folder + '/a' + suffix)

    def test_custom_layout_requires_root(self):
        self.assertEqual(
            _logical_image_id(Path('/dataset/photos/a.jpg'), Path('/dataset/photos')),
            'a.jpg',
        )

    def test_image_lookup_accepts_canonical_id_with_extension(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'images' / 'val' / 'a.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            self.assertEqual(_find_image(root, 'images/val/a.jpg'), image)
