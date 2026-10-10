"""Exercise clipboard dispatch from the pywebview worker to Qt's GUI thread."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


class DesktopClipboardTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("qtpy"), "需要 Qt 桌面依赖")
    def test_unicode_log_is_copied_from_worker_thread(self) -> None:
        script = '''
import threading
from qtpy import QtCore, QtGui
from vtools_ui.webapp.clipboard import create_qt_clipboard_writer
from vtools_ui.webapp.server import DesktopBridge
writer = create_qt_clipboard_writer()
app = QtGui.QGuiApplication([])
bridge = DesktopBridge(writer)
text = "日志：剪枝前评估\\nCUDA out of memory\\n"
result = {}
def work():
    try:
        result["copied"] = bridge.copy_text(text)
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["done"] = True
worker = threading.Thread(target=work)
worker.start()
timer = QtCore.QTimer()
timer.timeout.connect(lambda: app.quit() if result.get("done") else None)
timer.start(10)
QtCore.QTimer.singleShot(8000, app.quit)
app.exec()
worker.join(timeout=1)
assert result.get("copied") is True, result
assert app.clipboard().text() == text
'''
        result = subprocess.run([sys.executable, "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
