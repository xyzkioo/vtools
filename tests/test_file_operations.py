import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from others.filename_transform.image_filename_converter import (
    RenameItem, CocoUpdate, apply_plan, rollback_plan, write_coco_with_backup,
)


class FileOperations(unittest.TestCase):
    def test_swap_and_rollback_preserve_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.jpg', Path(tmp)/'b.jpg'
            a.write_bytes(b'A'); b.write_bytes(b'B')
            plan = [RenameItem(a, b), RenameItem(b, a)]
            apply_plan(plan)
            self.assertEqual((a.read_bytes(), b.read_bytes()), (b'B', b'A'))
            rollback_plan(plan)
            self.assertEqual((a.read_bytes(), b.read_bytes()), (b'A', b'B'))

    def test_incomplete_rollback_does_not_move_first_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); moved = root/'new.jpg'; moved.write_bytes(b'keep')
            with self.assertRaises(FileNotFoundError):
                rollback_plan([RenameItem(root/'old.jpg', moved), RenameItem(root/'old2.jpg', root/'missing.jpg')])
            self.assertEqual(moved.read_bytes(), b'keep')

    def test_repeated_coco_updates_preserve_each_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'annotations.json'
            first = {'images': [{'file_name': 'a.jpg'}]}
            path.write_text(json.dumps(first))
            backup1 = write_coco_with_backup(path, first, [CocoUpdate(0, 'a.jpg', 'b.jpg')])
            second = json.loads(path.read_text())
            backup2 = write_coco_with_backup(path, second, [CocoUpdate(0, 'b.jpg', 'c.jpg')])
            self.assertNotEqual(backup1, backup2)
            self.assertEqual(json.loads(backup1.read_text()), first)
            self.assertEqual(json.loads(backup2.read_text()), second)

    def test_coco_write_failure_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'annotations.json'; path.write_text('{"images": [{"file_name": "a.jpg"}]}')
            before = path.read_bytes()
            with patch('others.filename_transform.image_filename_converter.json.dump', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    write_coco_with_backup(path, json.loads(before), [CocoUpdate(0,'a.jpg','b.jpg')])
            self.assertEqual(path.read_bytes(), before)
