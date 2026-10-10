"""Write the desktop clipboard on Qt's GUI thread."""

from __future__ import annotations

import threading
from typing import Callable


def create_qt_clipboard_writer() -> Callable[[str], bool]:
    from qtpy import QtCore, QtGui

    class ClipboardWriter(QtCore.QObject):
        requested = QtCore.Signal(str, object)

        def __init__(self) -> None:
            super().__init__()
            self.requested.connect(self.write, QtCore.Qt.ConnectionType.QueuedConnection)

        @QtCore.Slot(str, object)
        def write(self, text: str, request: dict) -> None:
            try:
                app = QtGui.QGuiApplication.instance()
                if app is None:
                    raise RuntimeError("桌面剪贴板尚未就绪")
                clipboard = app.clipboard()
                clipboard.setText(text)
                if clipboard.text() != text:
                    raise RuntimeError("系统剪贴板写入校验失败")
            except Exception as exc:
                request["error"] = str(exc)
            finally:
                request["done"].set()

        def copy_text(self, text: str) -> bool:
            request = {"done": threading.Event()}
            if QtCore.QThread.currentThread() == self.thread():
                self.write(text, request)
            else:
                self.requested.emit(text, request)
                if not request["done"].wait(5):
                    raise RuntimeError("桌面剪贴板写入超时")
            if request.get("error"):
                raise RuntimeError(request["error"])
            return True

    return ClipboardWriter().copy_text
