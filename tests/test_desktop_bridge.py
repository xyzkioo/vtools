"""Exercise the real React/WebChannel/native Qt path in a subprocess."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DesktopBridgeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("PySide6"), "需要 PySide6")
    def test_react_bridge_clipboard_dialog_and_navigation(self):
        script = r'''
import os, threading, json
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QEvent, QTimer, QUrl
from PySide6.QtWidgets import QApplication, QFileDialog
from vtools_ui.webapp.server import WorkbenchServer
from vtools_ui.webapp.tasks import TaskManager
from vtools_ui.webapp.desktop import create_window, DesktopBridge
app = QApplication([])
manager = TaskManager()
server = WorkbenchServer(manager)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
result = {}
directory = os.environ["VTOOLS_QT_TEST_STATE"]
url = f'http://127.0.0.1:{server.server_port}/?token={server.token}'
window = create_window(url, Path(directory), manager.close)
view, page, profile, channel, bridge = window._desktop_objects
from PySide6.QtWebEngineCore import QWebEnginePage
assert not page.acceptNavigationRequest(QUrl('https://example.com'), QWebEnginePage.NavigationType.NavigationTypeOther, True)
original_picker = QFileDialog.getOpenFileName
QFileDialog.getOpenFileName = lambda *args: ('D:/测试/image.png', '')
try:
    assert bridge.pick_path('file')['value'] == 'D:/测试/image.png'
finally:
    QFileDialog.getOpenFileName = original_picker
assert 'error' in bridge.pick_path('invalid')
text = '日志：任务完成\n中文剪贴板'
def observed(value):
    if isinstance(value, str):
        value = json.loads(value) if value else {}
    if not isinstance(value, dict) or not value.get('ready'):
        return
    if value.get('done'):
        result.update(value)
        result['clipboard'] = app.clipboard().text()
        window.close()
        app.quit()
    elif not result.get('started'):
        result['started'] = True
        page.runJavaScript("window.vtoolsDesktop.api.copy_text('日志：任务完成\\n中文剪贴板').then(v=>window.__nativeProbe=v)")
timer = QTimer()
timer.timeout.connect(lambda: page.runJavaScript("JSON.stringify({ready:!!window.vtoolsDesktop,done:window.__nativeProbe===true,react:!!document.querySelector('#root')?.textContent.includes('常用工具')})", observed))
timer.start(100)
QTimer.singleShot(20000, app.quit)
window.show()
app.exec()
timer.stop()
assert result.get('done'), result
assert result.get('react'), result
assert result['clipboard'] == text, result
view.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
profile.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
server.shutdown()
server.server_close()
manager.close()
print('React + WebChannel + Qt native integration OK')
'''
        with tempfile.TemporaryDirectory() as directory:
            worker = os.environ.get("VTOOLS_QT_TEST_WORKER")
            if worker:
                test_script = Path(directory) / "qt-test.py"
                test_script.write_text(script, encoding="utf-8")
                command = [worker, "--run-script", str(test_script)]
            else:
                command = [sys.executable, "-c", script]
            completed = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "VTOOLS_QT_TEST_STATE": directory,
                     "VTOOLS_USER_DATA": str(Path(directory) / "state"),
                     "QT_QPA_PLATFORM": "windows" if os.name == "nt" else "offscreen"},
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=35,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
