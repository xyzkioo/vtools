"""Read-only integrity checks must report restricted execution accurately."""

import sys
import unittest
from unittest.mock import patch

from vtools_ui.windows_runtime import process_integrity_level, require_desktop_integrity


class WindowsRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows token API")
    def test_current_token_can_be_queried(self):
        self.assertGreaterEqual(process_integrity_level(), 0)

    def test_low_integrity_is_rejected_without_changing_it(self):
        with patch("vtools_ui.windows_runtime.process_integrity_level", return_value=4096):
            with self.assertRaisesRegex(RuntimeError, "低完整性"):
                require_desktop_integrity()

    def test_standard_user_is_accepted(self):
        with patch("vtools_ui.windows_runtime.process_integrity_level", return_value=8192):
            require_desktop_integrity()
