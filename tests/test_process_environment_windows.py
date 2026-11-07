"""Keep external interpreters independent of the frozen desktop ABI."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui.process_environment import build_tool_environment, external_dll_search_path


class ExternalPythonTests(unittest.TestCase):
    def test_windows_conda_uses_its_prefix_and_removes_frozen_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "app/_internal"
            sources = bundle / "tool-sources"
            prefix = root / "envs/vision"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "Library/bin").mkdir(parents=True)
            sources.mkdir(parents=True)
            inherited = {
                "PATH": os.pathsep.join([str(bundle / "PySide6"), "outside"]),
                "PYTHONPATH": os.pathsep.join([str(bundle), str(bundle / "PIL"), "user-modules"]),
                "PYTHONHOME": str(bundle), "_PYI_APPLICATION_HOME_DIR": str(bundle),
                "QT_PLUGIN_PATH": str(bundle / "PySide6/plugins"),
            }
            with patch.object(sys, "platform", "win32"), patch.object(sys, "_MEIPASS", str(bundle), create=True):
                env = build_tool_environment(prefix / "python.exe", sources, inherited, frozen=True)
            self.assertEqual(env["CONDA_PREFIX"], str(prefix))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [str(sources), "user-modules"])
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(prefix))
            self.assertNotIn(str(bundle / "PySide6"), env["PATH"].split(os.pathsep))
            for name in ("PYTHONHOME", "QT_PLUGIN_PATH", "_PYI_APPLICATION_HOME_DIR"):
                self.assertNotIn(name, env)

    @unittest.skipUnless(sys.platform == "win32", "Windows DLL search API")
    def test_dll_directory_is_restored_even_when_spawn_fails(self):
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
        kernel.GetDllDirectoryW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
        def current():
            buffer = ctypes.create_unicode_buffer(32768)
            kernel.GetDllDirectoryW(len(buffer), buffer)
            return buffer.value or None
        original = current()
        with tempfile.TemporaryDirectory() as directory:
            try:
                kernel.SetDllDirectoryW(directory)
                with patch.object(sys, "frozen", True, create=True):
                    with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                        with external_dll_search_path(Path(directory) / "external-python.exe"):
                            self.assertIsNone(current())
                            raise RuntimeError("spawn failed")
                self.assertEqual(current(), directory)
            finally:
                kernel.SetDllDirectoryW(original)
