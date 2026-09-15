"""桌面 runner 的进程交互回归：失败补发 finished、UTF-8 分块解码、无缓冲输出。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication

from vtools_ui.runner import ToolRunner

ROOT = Path(__file__).resolve().parents[1]


class RunnerBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_failed_to_start_emits_finished(self) -> None:
        runner = ToolRunner(ROOT)
        seen: list[tuple[int, str, bool]] = []
        runner.finished.connect(lambda code, name, stopped: seen.append((code, name, stopped)))
        runner._process_error(QProcess.ProcessError.FailedToStart)
        self.assertEqual(seen, [(127, "", False)])
        runner.deleteLater()

    def test_child_environment_is_unbuffered_utf8(self) -> None:
        runner = ToolRunner(ROOT)
        environment = runner.process.processEnvironment()
        self.assertEqual(environment.value("PYTHONUNBUFFERED"), "1")
        self.assertEqual(environment.value("PYTHONIOENCODING"), "utf-8")
        runner.deleteLater()

    def test_process_output_survives_multibyte_chunks(self) -> None:
        runner = ToolRunner(ROOT)
        chunks: list[str] = []
        finished: list[int] = []
        runner.output.connect(chunks.append)
        runner.finished.connect(lambda code, _name, _stopped: finished.append(code))
        runner.process.setWorkingDirectory(str(ROOT))
        runner.process.start(
            sys.executable,
            ["-c", "import sys; sys.stdout.write('中文输出 ok\\n'); sys.stdout.flush()"],
        )
        loop = QEventLoop()
        runner.process.finished.connect(loop.quit)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(30000)
        loop.exec()
        runner._read_output(final=True)
        self.assertEqual(finished, [0])
        self.assertIn("中文输出 ok", "".join(chunks))
        self.assertNotIn("\ufffd", "".join(chunks))
        runner.deleteLater()


if __name__ == "__main__":
    unittest.main()
