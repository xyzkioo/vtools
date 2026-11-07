"""Permission failures must not escape Windows desktop initialization."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui import state
from vtools_ui import __main__ as entry


class DesktopStateTests(unittest.TestCase):
    def tearDown(self):
        state.desktop_state_dir.cache_clear()

    def test_frozen_windows_keeps_data_next_to_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ), patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "platform", "win32"), patch.object(sys, "executable", str(root / "vtools.exe")), \
                    patch.object(state.Path, "home", side_effect=AssertionError("must not access C: profile")):
                os.environ.pop("VTOOLS_USER_DATA", None)
                expected = root / "user-data"
                self.assertEqual(state.desktop_state_dir(), expected)
                self.assertTrue((expected / "logs").is_dir())

    def test_explicit_directory_failure_is_reported_without_fallback(self):
        with patch.dict(os.environ, {"VTOOLS_USER_DATA": "explicit-state"}), \
                patch.object(state, "_check_writable", side_effect=PermissionError("explicit denied")) as probe:
            with self.assertRaisesRegex(PermissionError, "explicit denied"):
                state.desktop_state_dir()
            probe.assert_called_once()

    @unittest.skipUnless(sys.platform == "win32", "Windows error dialog")
    def test_failure_before_log_open_shows_dialog_without_unhandled_exception(self):
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "argv", ["vtools.exe"]), \
                patch.object(sys, "stderr", None), \
                patch.object(state, "desktop_state_dir", side_effect=PermissionError("state denied")), \
                patch("ctypes.windll.user32.MessageBoxW") as dialog:
            self.assertEqual(entry.main(), 1)
        self.assertIn("state denied", dialog.call_args.args[1])
