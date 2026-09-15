import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui.runner import append_history


class UIHistory(unittest.TestCase):
    def test_non_list_history_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text('{"unexpected": true}', encoding="utf-8")
            append_history(path, "task", 0, None)
            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task"], "task")

    def test_failed_publish_preserves_existing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text("[]", encoding="utf-8")
            original_replace = Path.replace

            def fail_for_history(source, target):
                if Path(target) == path:
                    raise OSError("disk full")
                return original_replace(source, target)

            with patch.object(Path, "replace", fail_for_history):
                with self.assertRaises(OSError):
                    append_history(path, "task", 0, None)
            self.assertEqual(path.read_text(encoding="utf-8"), "[]")
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
