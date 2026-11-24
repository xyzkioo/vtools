import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from others.filename_transform.image_filename_converter import (
    RenameItem, CocoUpdate, apply_plan, load_coco_json, rollback_plan, validate_plan,
    main as rename_main, write_coco_with_backup,
)


class FileOperations(unittest.TestCase):
    def test_yolo_rename_cannot_assign_leftover_label_to_background_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            image = images / "a.jpg"
            image.write_bytes(b"image")
            label = labels / "08-01.txt"
            label.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
            args = ["rename", "--dir", str(root), "--dataset-format", "yolo", "--apply", "--yes"]
            with patch.object(sys, "argv", args):
                self.assertEqual(rename_main(), 1)
            self.assertEqual(image.read_bytes(), b"image")
            self.assertFalse((images / "08-01.jpg").exists())
            self.assertEqual(label.read_text(encoding="utf-8"), "0 0.5 0.5 0.3 0.3\n")

            # If the same label belongs to an image being moved out of the way,
            # the background image can safely reuse that stem.
            image.rename(images / "00.jpg")
            label.rename(labels / "01.txt")
            (images / "01.jpg").write_bytes(b"labeled")
            args += ["--template", "{index:02d}"]
            with patch.object(sys, "argv", args):
                self.assertEqual(rename_main(), 0)
            self.assertEqual((images / "01.jpg").read_bytes(), b"image")
            self.assertEqual((images / "02.jpg").read_bytes(), b"labeled")
            self.assertFalse((labels / "01.txt").exists())
            self.assertTrue((labels / "02.txt").is_file())

    def test_rename_dry_run_only_writes_a_new_non_conflicting_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            image.write_bytes(b"original")
            future_name = root / "08-01.jpg"
            csv = root / "plan.csv"
            for output, expected in ((image, 1), (future_name, 1), (csv, 0), (csv, 1)):
                with self.subTest(output=output, expected=expected), patch.object(sys, "argv", [
                    "rename", "--dir", str(root), "--dry-run", "--plan", str(output),
                ]):
                    self.assertEqual(rename_main(), expected)
            self.assertEqual(image.read_bytes(), b"original")
            self.assertFalse(future_name.exists())
            self.assertIn("08-01.jpg", csv.read_text(encoding="utf-8-sig"))

    def test_coco_json_with_utf8_bom_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text('{"images": [{"file_name": "a.jpg"}]}', encoding="utf-8-sig")
            self.assertEqual(load_coco_json(path)["images"][0]["file_name"], "a.jpg")

    def test_duplicate_rename_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "same.jpg"
            with self.assertRaisesRegex(ValueError, "目标文件名重复"):
                validate_plan([RenameItem(Path(tmp) / "a.jpg", target),
                               RenameItem(Path(tmp) / "b.jpg", target)])

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
