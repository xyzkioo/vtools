"""Windows desktop and worker dispatch contracts."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui.webapp import core
from vtools_ui.process_environment import build_tool_environment


class WindowsDesktopTests(unittest.TestCase):
    def test_windows_bundled_task_requires_console_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gui = root / "vtools.exe"
            worker = root / "vtools-worker.exe"
            gui.touch()
            request = {"tool_id": "data", "variant": "filename", "values": {"dir": directory, "dataset": "images", "apply": False}}
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "platform", "win32"), \
                    patch.object(sys, "executable", str(gui)), \
                    patch.object(core, "read_settings", return_value={"python_executable": "missing", "results_root": directory}):
                with self.assertRaisesRegex(ValueError, "vtools-worker.exe"):
                    core.build_command(request)
                worker.touch()
                executable, args, _ = core.build_command(request)
                self.assertEqual(Path(executable), worker)
                self.assertEqual(args[0], "--run-script")
                env = build_tool_environment(worker, root, {"CONDA_PREFIX": "parent"})
                self.assertEqual(env["CONDA_PREFIX"], "parent")

