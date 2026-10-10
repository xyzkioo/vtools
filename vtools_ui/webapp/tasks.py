"""One-at-a-time subprocess runner with streaming events."""

from __future__ import annotations

import codecs
import json
import os
import re
import signal
import subprocess
import threading
import uuid
from datetime import datetime, timezone
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
            log_file.write_text(f"$ {command_display}\n", encoding="utf-8")
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
            with log_file.open("a", encoding="utf-8") as log:
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
            error_line = f"任务执行或日志写入失败：{exc}\n"
            self._emit("log", error_line)
            if log_file.is_file():
                try:
                    with log_file.open("a", encoding="utf-8") as log:
                        log.write(error_line)
                except OSError:
                    pass
        finally:
            if process is not None:
                if process.stdout:
                    process.stdout.close()
                if process.poll() is None:
                    self._terminate_process(process)
                process.wait()
        with self._lock:
            current = self._state if self._state and self._state["id"] == run_id else None
            tool_id = str(current.get("tool_id", "")) if current else ""
            was_stopped = self._stopping
            started_at = str(current.get("started_at", "")) if current else ""
        if tool_id == "compression" and exit_code != 0 and not was_stopped:
            recovered = self._write_interrupted_compression_summary(
                run_id, args, exit_code, log_file, started_at
            )
            if recovered:
                message = (
                    f"父进程检测到压缩任务异常退出（退出码 {exit_code}）；"
                    f"已生成失败摘要：{recovered['summary_path']}\n"
                )
                if recovered.get("signal"):
                    message = (
                        f"父进程检测到压缩任务被信号 {recovered['signal']} 终止"
                        f"（退出码 {exit_code}）；已生成失败摘要：{recovered['summary_path']}\n"
                    )
                self._emit("log", message)
                try:
                    with log_file.open("a", encoding="utf-8") as log:
                        log.write(message)
                except OSError:
                    pass
                self._capture_result(f"本次运行目录：{recovered['run_dir']}\n")
        with self._lock:
            stopped = self._stopping
            if self._state and self._state["id"] == run_id:
                self._state["exit_code"] = exit_code
                if stopped:
                    status = "stopped"
                elif exit_code == 0:
                    status = "succeeded"
                elif exit_code == 1 and self._completed_with_issues(self._state["tool_id"], log_file):
                    status = "succeeded_with_issues"
                else:
                    status = "failed"
                self._state["status"] = status
                self._state["log_path"] = str(log_file) if log_file.is_file() else None
                final = dict(self._state)
            else:
                return
            self._process = None
        try:
            issue_label = "完成（有差异）" if final["tool_id"] == "benchmark" else "检查完成（有问题）"
            add_history({"id": run_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "task": final["tool_id"], "status": {"succeeded": "成功", "succeeded_with_issues": issue_label, "failed": "失败", "stopped": "已停止"}[final["status"]], "config": final["config"] or "默认配置", "exit_code": exit_code, "result_dir": final["result_dir"], "log_path": final["log_path"]})
        except OSError as exc:
            self._emit("log", f"任务记录写入失败：{exc}\n")
        self._emit("state", final)

    @staticmethod
    def _completed_with_issues(tool_id: str, log_file: Path) -> bool:
        """Recognize a completed report with findings, not a failed process."""

        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if tool_id == "dataset_quality":
            return "状态：issues_found" in text or "issues found" in text.lower()
        if tool_id != "benchmark":
            return False
        if "Traceback (most recent call last)" in text or "阶段失败" in text or "错误：" in text:
            return False
        if "一致性总状态：error" in text:
            return False
        return any(
            f"一致性总状态：{status}" in text
            for status in ("failed", "warning", "inconclusive", "issues_found")
        )

    @staticmethod
    def _write_interrupted_compression_summary(
        run_id: str, args: list[str], exit_code: int, log_file: Path, started_at: str,
    ) -> dict[str, str] | None:
        """Write a parent-owned summary when compression dies before its own report."""

        try:
            run_dir_arg = args[args.index("--run-dir") + 1]
        except (ValueError, IndexError):
            return None
        run_dir = Path(run_dir_arg).expanduser()
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            return None
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            return None

        module_status: dict[str, str] = {}
        current_module: str | None = None
        active_stage: str | None = None
        last_stage: str | None = None
        if log_file.is_file():
            try:
                with log_file.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        module_match = re.match(
                            r"^\[([A-Za-z][A-Za-z0-9_.-]*)\]\s*(开始|完成|失败|跳过)(?:：.*)?",
                            line.strip(),
                        )
                        if module_match:
                            module_id, state = module_match.groups()
                            current_module = module_id
                            module_status[module_id] = {
                                "开始": "running", "完成": "succeeded",
                                "失败": "failed", "跳过": "skipped",
                            }[state]
                            if state in {"完成", "失败", "跳过"}:
                                current_module = None
                        stage_match = re.match(
                            r"^\[结构化剪枝\]\s*(开始|完成)：(.+?)\s*$", line.strip()
                        )
                        if stage_match:
                            state, stage_name = stage_match.groups()
                            last_stage = stage_name
                            active_stage = stage_name if state == "开始" else None
            except OSError:
                pass

        signal_name: str | None = None
        if exit_code < 0:
            try:
                signal_name = signal.Signals(-exit_code).name
            except ValueError:
                signal_name = f"SIG{-exit_code}"
        elif os.name == "posix" and 128 < exit_code <= 255:
            try:
                signal_name = signal.Signals(exit_code - 128).name
            except ValueError:
                pass

        if active_stage:
            last_stage = active_stage
        if current_module:
            module_status[current_module] = "failed"
        try:
            config: dict[str, Any] = {}
            config_path = run_dir / "effective_config.json"
            if config_path.is_file():
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config = loaded
        except (OSError, json.JSONDecodeError):
            config = {}

        model = config.get("model") if isinstance(config.get("model"), dict) else {}
        modules = config.get("modules") if isinstance(config.get("modules"), dict) else {}
        modules = dict(modules)
        if "--only" in args:
            try:
                only_values = args[args.index("--only") + 1].split(",")
                modules = {str(key): str(key) in only_values for key in modules}
                modules.update({name: True for name in only_values if name})
            except IndexError:
                pass
        for index, argument in enumerate(args[:-1]):
            if argument in {"--enable", "--disable"}:
                modules[args[index + 1]] = argument == "--enable"
        executed_modules = [str(key) for key, enabled in modules.items() if enabled]
        explanation = f"子进程未能正常完成，退出码 {exit_code}。"
        if signal_name:
            explanation = f"子进程被 {signal_name} 终止（退出码 {exit_code}）。"
        failure: dict[str, Any] = {
            "exit_code": exit_code,
            "signal": signal_name,
            "last_module": current_module,
            "last_stage": last_stage,
            "reason": explanation,
        }
        summary: dict[str, Any] = {
            "run_id": run_dir.name or run_id,
            "started_at": started_at or None,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "model": {
                "name": model.get("name", "model"),
                "task": model.get("task", "classify"),
                "adapter": model.get("adapter", "torch"),
                "source_weights": model.get("weights"),
            },
            "executed_modules": executed_modules,
            "errors": [explanation],
            "module_status": module_status,
            "failure_details": failure,
        }
        if current_module == "compression.prune.structured":
            summary["structured_pruning"] = {
                "status": "failed",
                "module": "compression.prune.structured",
                "error": explanation,
                "last_stage": last_stage,
            }

        temporary = summary_path.with_name(f".{summary_path.name}.{run_id}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, summary_path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return {
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "signal": signal_name or "",
        }

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
