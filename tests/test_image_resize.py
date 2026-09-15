import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from others import image_resize


class ResizeSafety(unittest.TestCase):
    def test_conversion_collision_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input'; source.mkdir(); output=root/'output'
            for suffix in ('.png', '.jpg'):
                Image.new('RGB', (4,4)).save(source/('a'+suffix))
            with patch('sys.argv', ['image_resize', '--input', str(source), '--output-dir', str(output), '--output-format', 'jpg', '--overwrite']):
                self.assertEqual(image_resize.main(), 1)
            self.assertFalse(output.exists())

    def test_failed_encoding_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.png'; output=root/'output'; output.mkdir()
            Image.new('RGB', (4,4)).save(source); target=output/'source.png'; target.write_bytes(b'original')
            def fail(image, path, *args):
                path.write_bytes(b'partial')
                raise OSError('disk full')
            with patch('sys.argv', ['image_resize', '--input', str(source), '--output-dir', str(output), '--overwrite']), patch.object(image_resize, 'save_image', side_effect=fail):
                self.assertEqual(image_resize.main(), 1)
            self.assertEqual(target.read_bytes(), b'original')
            self.assertEqual(list(output.iterdir()), [target])

    def test_palette_transparency_is_composited_on_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Image.new('P',(8,8),0); image.putpalette([255,0,0] + [0,0,0]*255); image.info['transparency']=0
            output=Path(tmp)/'output.jpg'; image_resize.save_image(image, output, 'jpg', 100)
            with Image.open(output) as saved:
                self.assertEqual(saved.getpixel((0,0)), (255,255,255))
