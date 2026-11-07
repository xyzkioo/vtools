"""Qt WebEngine desktop host for the React workbench."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QCoreApplication, QEvent, QFile, QIODevice, QObject, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox


class DesktopBridge(QObject):
    """Only these two explicit slots are exposed to the local frontend."""

    @Slot(str, result="QVariantMap")
    def copy_text(self, text: str) -> dict:
        QApplication.clipboard().setText(text)
        return {"value": QApplication.clipboard().text() == text}

    @Slot(str, result="QVariantMap")
    def pick_path(self, kind: str) -> dict:
        parent = self.parent()
        if kind == "file":
            value, _ = QFileDialog.getOpenFileName(parent, "选择文件")
        elif kind == "dir":
            value = QFileDialog.getExistingDirectory(parent, "选择目录")
        elif kind == "save_file":
            value, _ = QFileDialog.getSaveFileName(parent, "选择保存位置")
        else:
            return {"error": f"不支持的文件选择类型：{kind}"}
        return {"value": value}


class WorkbenchPage(QWebEnginePage):
    def __init__(self, profile: QWebEngineProfile, origin: QUrl, parent: QObject) -> None:
        super().__init__(profile, parent)
        self._origin = origin

    def acceptNavigationRequest(self, url: QUrl, kind, main_frame: bool) -> bool:
        local = (url.scheme(), url.host(), url.port()) == (
            self._origin.scheme(), self._origin.host(), self._origin.port(),
        )
        if local or url.toString() == "about:blank":
            return True
        if main_frame and kind == QWebEnginePage.NavigationType.NavigationTypeLinkClicked and url.scheme() in {"https", "http"}:
            QDesktopServices.openUrl(url)
        return False


def create_window(url: str, state: Path, on_close: Callable[[], None], *, quiet: bool = False) -> QMainWindow:
    class WorkbenchWindow(QMainWindow):
        def closeEvent(self, event) -> None:
            on_close()
            super().closeEvent(event)

    window = WorkbenchWindow()
    window.setWindowTitle("vtools 工作台")
    window.resize(1280, 800)
    window.setMinimumSize(1080, 680)
    icon = (Path(sys.executable).parent / "vtools.ico") if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2] / "packaging/vtools.ico"
    if icon.is_file():
        window.setWindowIcon(QIcon(str(icon)))
    view = QWebEngineView(window)
    window.setCentralWidget(view)
    # Explicit paths keep Chromium storage out of C: when installed on D:.
    profile = QWebEngineProfile("vtools", window)
    profile.setPersistentStoragePath(str(state / "webengine/storage"))
    profile.setCachePath(str(state / "webengine/cache"))
    page = WorkbenchPage(profile, QUrl(url), view)
    page.setBackgroundColor(QColor("#f4f4f6"))
    view.setPage(page)
    channel = QWebChannel(page)
    bridge = DesktopBridge(window)
    channel.registerObject("desktop", bridge)
    page.setWebChannel(channel)
    resource = QFile(":/qtwebchannel/qwebchannel.js")
    if not resource.open(QIODevice.OpenModeFlag.ReadOnly):
        raise RuntimeError("缺少 Qt WebChannel JavaScript 资源")
    library = bytes(resource.readAll()).decode("utf-8")
    resource.close()
    parsed = QUrl(url)
    origin = f"{parsed.scheme()}://{parsed.host()}:{parsed.port()}"
    script = QWebEngineScript()
    script.setName("vtools-desktop")
    script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    script.setRunsOnSubFrames(False)
    script.setSourceCode(library + "\n" + """
if (location.origin === ORIGIN) {
  new QWebChannel(qt.webChannelTransport, channel => {
    const invoke = (method, value) => new Promise((resolve, reject) => {
      channel.objects.desktop[method](value, result => {
        if (result.error) reject(new Error(result.error));
        else resolve(result.value);
      });
    });
    window.vtoolsDesktop = {api: {
      pick_path: kind => invoke('pick_path', kind),
      copy_text: text => invoke('copy_text', text),
    }};
    window.dispatchEvent(new Event('vtoolsdesktopready'));
  });
}
""".replace("ORIGIN", json.dumps(origin)))
    page.scripts().insert(script)
    view.loadFinished.connect(lambda ok: print(f"vtools 页面加载：{'成功' if ok else '失败'}", flush=True))
    def render_stopped(status, code: int) -> None:
        print(f"vtools 渲染进程退出：status={status}, code={code}", flush=True)
        if not quiet:
            QMessageBox.critical(window, "vtools", "页面渲染进程已退出。详细错误已记录到 user-data/logs/desktop.log。")
    page.renderProcessTerminated.connect(render_stopped)
    view.load(QUrl(url))
    # Keep Python wrappers alive alongside their Qt parents.
    window._desktop_objects = (view, page, profile, channel, bridge)
    return window


def run_desktop(url: str, state: Path, on_close: Callable[[], None], *, verify: bool = False) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("vtools")
    window = create_window(url, state, on_close, quiet=verify)
    verification = [1]
    poll = QTimer(window)
    timeout = QTimer(window)
    if verify:
        from vtools_ui import __version__
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        def finish(ok: bool) -> None:
            verification[0] = 0 if ok else 1
            poll.stop()
            timeout.stop()
            print(f"安装启动验证：{'通过' if ok else '失败'}", flush=True)
            window.close()
            app.quit()
        def observed(value) -> None:
            if value is True:
                finish(True)
        probe = f"Boolean(window.vtoolsDesktop?.api && document.querySelector('#root')?.textContent.includes({json.dumps(__version__)}))"
        poll.timeout.connect(lambda: window.centralWidget().page().runJavaScript(probe, observed))
        poll.start(100)
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: finish(False))
        timeout.start(30000)
    window.show()
    print("vtools Qt WebEngine 工作台窗口已显示。", flush=True)
    result = app.exec()
    poll.stop()
    timeout.stop()
    # Destroy pages before their disk profile is released.
    view, page, profile, *_ = window._desktop_objects
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    profile.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.deleteLater()
    app.processEvents()
    return verification[0] if verify else result
