"""Legacy Qt process runner kept for compatibility.

The new web workbench uses :mod:`vtools_ui.webapp.tasks`; this runner remains
available for older imports and the legacy Qt regression tests.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


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
        self.last_run_dir: Path | None = None
        self._run_dir_buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.process.setProcessEnvironment(self._child_environment())

    @staticmethod
    def _child_environment() -> QProcessEnvironment:
        """Force line-buffered, UTF-8 output so the log pane stays readable."""

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        return environment

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
        self.last_run_dir = None
        self._run_dir_buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
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
        known_descendants = self._descendants(pid) if pid > 0 and os.name != "nt" else []
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
                    for child_pid in known_descendants:
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                self.process.kill()
        # QProcess may report that the parent exited while a worker remains
        # alive.  Always clean up the descendants observed before stopping;
        # only those PIDs are targeted, so unrelated processes are untouched.
        if os.name != "nt":
            for child_pid in known_descendants:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _read_output(self, *, final: bool = False) -> None:
        raw = bytes(self.process.readAllStandardOutput())
        data = self._decoder.decode(raw, final) if (raw or final) else ""
        if data:
            self._capture_run_dir(data)
            self.output.emit(data)

    def _capture_run_dir(self, data: str) -> None:
        """Remember the result directory printed by vtools entry points."""

        self._run_dir_buffer = (self._run_dir_buffer + data)[-8192:]
        for line in self._run_dir_buffer.splitlines():
            text = line.strip()
            match = re.search(r"(?:本次运行目录|运行目录)\s*[:：]\s*(.+?)\s*$", text)
            if match:
                value = match.group(1).strip().strip('"')
                if value:
                    self.last_run_dir = Path(value).expanduser().resolve()
                continue
            saved = re.search(
                r"(?:可视化报告已保存|汇总结果已保存|模块列表已保存|完整改名清单已保存|权重检查完成；结果已保存|结果已保存|输出目录)\s*[:：]\s*(.+?)\s*$",
                text,
            )
            if saved:
                self._remember_saved_path(saved.group(1), directory=text.startswith("输出目录"))
                continue
            completed = re.search(r"完成\s*[:：].*?->\s*(.+?)(?:[，,（(].*)?$", text)
            if completed:
                self._remember_saved_path(completed.group(1), directory=True)
                continue
            generated = re.search(r"(?:生成|导出|简化)\s*[:：]\s*(.+?)\s*$", text)
            if generated:
                self._remember_saved_path(generated.group(1))

    def _remember_saved_path(self, value: str, *, directory: bool = False) -> None:
        value = value.strip().strip('"')
        if value:
            path = Path(value).expanduser().resolve()
            self.last_run_dir = path if directory or path.is_dir() else path.parent

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.output.emit("无法启动 Python 进程，请检查当前环境。")
            self.state_changed.emit("启动失败")
            # QProcess 在启动失败时不会发出 finished()；这里补发，保持任务
            # 记录、按钮状态和失败提示与其他失败路径一致。
            self.finished.emit(127, self._task_name, False)
            return
        self.state_changed.emit("启动失败")

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output(final=True)
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


def append_history(
    path: Path,
    task_name: str,
    exit_code: int,
    config: Path | None,
    status_override: str | None = None,
    limit: int = 40,
) -> None:
    """Keep a small, human-readable task history next to the UI settings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
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
    payload = json.dumps(records[: max(1, int(limit))], ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
