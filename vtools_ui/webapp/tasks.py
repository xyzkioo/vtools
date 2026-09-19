"""One-at-a-time subprocess runner with streaming events."""

from __future__ import annotations

import codecs
import os
import re
import signal
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import ROOT
from .core import STATE_DIR, add_history, build_command
from ..process_environment import build_tool_environment


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._process: subprocess.Popen[bytes] | None = None
        self._state: dict[str, Any] | None = None
        self._events: list[dict[str, Any]] = []
        self._stopping = False

    def state(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._state) if self._state else None

    def _emit(self, kind: str, payload: Any) -> None:
        with self._lock:
            self._events.append({"id": len(self._events), "kind": kind, "payload": payload})
            self._lock.notify_all()

    def events_after(self, cursor: int, timeout: float = 15) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            if len(self._events) <= cursor and self._state and self._state["status"] in {"starting", "running", "stopping"}:
                self._lock.wait(timeout)
            return self._events[cursor:], bool(self._state and self._state["status"] in {"succeeded", "succeeded_with_issues", "failed", "stopped"})

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        executable, args, config = build_command(request)
        with self._lock:
            if self._state and self._state["status"] in {"starting", "running", "stopping"}:
                raise RuntimeError("已有任务正在运行")
            run_id = uuid.uuid4().hex
            tool_id = str(request["tool_id"])
            self._state = {"id": run_id, "tool_id": tool_id, "status": "starting", "started_at": datetime.now().isoformat(timespec="seconds"), "exit_code": None, "result_dir": None, "config": config}
            self._events = []
            self._stopping = False
        self._emit("state", self.state())
        worker = threading.Thread(target=self._run, args=(run_id, executable, args), daemon=True)
        worker.start()
        return self.state() or {}

    def _run(self, run_id: str, executable: str, args: list[str]) -> None:
        log_dir = STATE_DIR / "logs"
        log_file = log_dir / f"{run_id}.log"
        command_display = subprocess.list2cmdline([executable, *args])
        self._emit("log", f"$ {command_display}\n")
        process: subprocess.Popen[bytes] | None = None
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {"cwd": ROOT, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "bufsize": 0}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            environment = build_tool_environment(executable, ROOT)
            process = subprocess.Popen([executable, *args], env=environment, **kwargs)
            with self._lock:
                self._process = process
                if self._state and self._state["id"] == run_id:
                    self._state["status"] = "stopping" if self._stopping else "running"
            self._emit("state", self.state())
            if self._stopping:
                self.stop()
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            line_buffer = ""
            with log_file.open("w", encoding="utf-8") as log:
                while True:
                    chunk = process.stdout.read(4096) if process.stdout else b""
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    if text:
                        log.write(text)
                        log.flush()
                        self._emit("log", text)
                        line_buffer = self._capture_result(line_buffer + text)
                tail = decoder.decode(b"", final=True)
                if tail:
                    log.write(tail)
                    self._emit("log", tail)
                self._capture_result(line_buffer + tail + "\n")
            exit_code = process.wait()
        except (OSError, ValueError) as exc:
            exit_code = 127
            if process is not None:
                self._terminate_process(process)
            self._emit("log", f"任务执行或日志写入失败：{exc}\n")
        finally:
            if process is not None:
                if process.stdout:
                    process.stdout.close()
                if process.poll() is None:
                    self._terminate_process(process)
                process.wait()
        with self._lock:
            stopped = self._stopping
            if self._state and self._state["id"] == run_id:
                self._state["exit_code"] = exit_code
                if stopped:
                    status = "stopped"
                elif exit_code == 0:
                    status = "succeeded"
                elif self._state["tool_id"] in {"dataset_quality", "benchmark"} and exit_code == 1:
                    # These tools use exit code 1 for a complete run with
                    # findings. Real execution errors use exit code 2.
                    status = "succeeded_with_issues"
                else:
                    status = "failed"
                self._state["status"] = status
                self._state["log_path"] = str(log_file)
                final = dict(self._state)
            else:
                return
            self._process = None
        try:
            issue_label = "完成（有差异）" if final["tool_id"] == "benchmark" else "检查完成（有问题）"
            add_history({"id": run_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "task": final["tool_id"], "status": {"succeeded": "成功", "succeeded_with_issues": issue_label, "failed": "失败", "stopped": "已停止"}[final["status"]], "config": final["config"] or "默认配置", "exit_code": exit_code, "result_dir": final["result_dir"], "log_path": str(log_file)})
        except OSError as exc:
            self._emit("log", f"任务记录写入失败：{exc}\n")
        self._emit("state", final)

    def _capture_result(self, buffer: str) -> str:
        lines = buffer.splitlines(keepends=True)
        remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            remainder = lines.pop()
        for line in lines:
            message = line.strip()
            match = re.search(r"(?:本次运行目录|运行目录|结果目录|输出目录|可视化报告已保存|汇总结果已保存|模块列表已保存|完整改名清单已保存|权重检查完成；结果已保存|结果已保存|生成|导出|简化)\s*[:：]\s*(.+?)\s*$", message)
            if not match:
                match = re.search(r"完成\s*[:：].*?->\s*(.+?)(?:[，,（(].*)?$", message)
            if match:
                path = Path(match.group(1).strip().strip('"')).expanduser()
                path = (path if path.is_absolute() else ROOT / path).resolve()
                if not path.exists():
                    continue
                if path.is_file():
                    path = path.parent
                with self._lock:
                    if self._state:
                        self._state["result_dir"] = str(path)
        return remainder[-8192:]

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if not self._state or self._state["status"] not in {"starting", "running", "stopping"}:
                return
            self._stopping = True
            self._state["status"] = "stopping"
        self._emit("state", self.state())
        if process:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()

    def close(self) -> None:
        self.stop()
