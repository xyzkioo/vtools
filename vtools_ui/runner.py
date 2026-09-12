"""Cross-platform process runner used by the desktop workbench."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal


@dataclass(frozen=True)
class ToolSpec:
    key: str
    title: str
    description: str
    tool: str | None = None
    default_config: Path | None = None
    script: Path | None = None


class ToolRunner(QObject):
    """Run one vtools entry point without blocking the Qt event loop."""

    output = Signal(str)
    state_changed = Signal(str)
    finished = Signal(int, str, bool)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.python_executable = Path(sys.executable)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(lambda: self.state_changed.emit("运行中"))
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)
        self._task_name = ""
        self._started_at = ""
        self._stopping = False

    def set_python_executable(self, executable: Path) -> None:
        self.python_executable = executable.expanduser().resolve()

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        spec: ToolSpec,
        config_path: Path | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError("已有任务正在运行")
        extra_args = list(extra_args or [])
        if spec.script:
            script = spec.script if spec.script.is_absolute() else self.project_root / spec.script
            if not script.exists():
                raise FileNotFoundError(f"找不到入口文件：{script}")
            args = [str(script), *extra_args]
        else:
            if not spec.tool:
                raise ValueError("该页面暂未接入运行入口")
            script = self.project_root / "run_tools.py"
            if not script.exists():
                raise FileNotFoundError(f"找不到入口文件：{script}")
            config = config_path or spec.default_config
            args = [str(script), "--tool", spec.tool]
            if config:
                args.extend(["--config", str(config)])
            args.extend(extra_args)

        self._task_name = spec.title
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._stopping = False
        self.output.emit(f"$ {self._format_command(args)}")
        self.output.emit(f"开始执行：{spec.title}\n")
        self.process.setWorkingDirectory(str(self.project_root))
        # Keep QProcess attached to the actual Python process.  The stop path
        # walks its descendants, so no wrapper process can outlive the UI.
        self.process.start(str(self.python_executable), args)

    @staticmethod
    def _descendants(root_pid: int) -> list[int]:
        """Return descendant PIDs on Unix without depending on psutil."""

        if os.name == "nt":
            return []
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,ppid="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        children: dict[int, list[int]] = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                pid, parent = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            children.setdefault(parent, []).append(pid)
        result: list[int] = []
        pending = list(children.get(root_pid, []))
        while pending:
            pid = pending.pop()
            result.append(pid)
            pending.extend(children.get(pid, []))
        return result

    @classmethod
    def _signal_unix_tree(cls, root_pid: int, sig: signal.Signals) -> None:
        # Kill children first so a parent cannot immediately respawn them.
        for pid in list(reversed(cls._descendants(root_pid))) + [root_pid]:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue

    def stop(self) -> None:
        if not self.is_running:
            return
        self.output.emit("\n正在停止任务…")
        self.state_changed.emit("正在停止")
        self._stopping = True
        # During the short Starting state Qt may not have published a PID yet.
        # Never pass 0 to os.kill (on Unix that means the whole process group).
        pid = int(self.process.processId())
        if os.name == "nt":
            if pid > 0:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self.process.kill()
        else:
            if pid > 0:
                self._signal_unix_tree(pid, signal.SIGTERM)
            # Also terminate the QProcess-owned Python process immediately.
            # The tree walk handles workers; this guarantees the UI receives
            # finished() even when the main process ignores SIGTERM.
            self.process.kill()
        if not self.process.waitForFinished(1500):
            if os.name == "nt":
                if pid > 0:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    self.process.kill()
            else:
                if pid > 0:
                    self._signal_unix_tree(pid, signal.SIGKILL)
                self.process.kill()

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if data:
            self.output.emit(data)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.output.emit("无法启动 Python 进程，请检查当前环境。")
        self.state_changed.emit("启动失败")

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        was_stopped = self._stopping
        status = "已停止" if was_stopped else ("成功" if exit_code == 0 else f"失败（退出码 {exit_code}）")
        self.state_changed.emit(status)
        self.finished.emit(exit_code, self._task_name, was_stopped)
        self._stopping = False

    def _format_command(self, args: list[str]) -> str:
        command = [str(self.python_executable), *args]
        if os.name == "nt":
            return " ".join(f'"{item}"' if " " in item else item for item in command)
        return " ".join(item.replace(" ", "\\ ") for item in command)


def append_history(path: Path, task_name: str, exit_code: int, config: Path | None, status_override: str | None = None) -> None:
    """Keep a small, human-readable task history next to the UI settings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            records = []
    records.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "task": task_name,
            "status": status_override or ("成功" if exit_code == 0 else "失败"),
            "config": str(config) if config else "默认配置",
        },
    )
    path.write_text(json.dumps(records[:40], ensure_ascii=False, indent=2), encoding="utf-8")
