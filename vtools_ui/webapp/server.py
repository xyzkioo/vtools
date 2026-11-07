"""Loopback-only HTTP API and desktop host."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from vtools_ui import __version__
from vtools_ui.process_environment import external_dll_search_path

from .catalog import ROOT, public_catalog
from .core import (
    IMAGE_SUFFIXES, TEXT_SUFFIXES, STATE_DIR, clear_history, environments, flatten_config, history, list_results,
    load_config, parse_yaml, patch_config, preview_text, project_path, read_settings,
    result_file, save_config, save_settings,
)
from .tasks import TaskManager
from .updater import check_update, install_update

DIST = Path(__file__).resolve().parent / "frontend" / "dist"


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, manager: TaskManager) -> None:
        super().__init__(("127.0.0.1", 0), ApiHandler)
        self.manager = manager
        self.token = secrets.token_urlsafe(32)
        self.results_root = project_path(str(read_settings()["results_root"]))
        if getattr(sys, "frozen", False):
            self.results_root.mkdir(parents=True, exist_ok=True)


class ApiHandler(BaseHTTPRequestHandler):
    server: WorkbenchServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, exc: Exception, status: int = 400) -> None:
        self._json({"error": str(exc)}, status)

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        return self.headers.get("X-Vtools-Token") == self.server.token or query.get("token", [""])[0] == self.server.token

    def _body(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 0 or size > 2 * 1024 * 1024:
            raise ValueError("请求过大")
        payload = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("请求内容必须是对象")
        return payload

    @staticmethod
    def _first(query: dict[str, list[str]], name: str, default: str = "") -> str:
        return query.get(name, [default])[0]

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path.startswith("/api/"):
            if not self._authorized(query):
                self._json({"error": "会话已失效，请重新启动工作台"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                self._api_get(parsed.path, query)
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                self._error(exc)
            return
        self._static(parsed.path)

    def _api_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/bootstrap":
            self._json({"catalog": public_catalog(), "settings": read_settings(), "history": history(), "state": self.server.manager.state(), "version": __version__})
        elif path == "/api/update/check":
            self._json(check_update())
        elif path == "/api/environments":
            self._json({"environments": environments()})
        elif path == "/api/config":
            self._json(load_config(self._first(query, "path")))
        elif path == "/api/history":
            self._json({"history": history()})
        elif path == "/api/runs/log":
            run_id = self._first(query, "id")
            expected = self._run_log_path(run_id)
            limit = 2 * 1024 * 1024
            with expected.open("rb") as handle:
                size = expected.stat().st_size
                if size > limit:
                    handle.seek(size - limit)
                content = handle.read().decode("utf-8", errors="replace")
            self._json({"text": content, "truncated": size > limit, "path": str(expected)})
        elif path == "/api/runs/current":
            self._json({"state": self.server.manager.state()})
        elif path == "/api/runs/events":
            run_id = self._first(query, "id")
            current = self.server.manager.state()
            if not current or current["id"] != run_id:
                raise ValueError("任务不存在")
            cursor = max(0, int(self._first(query, "cursor", "0")))
            events, done = self.server.manager.events_after(cursor)
            self._json({"events": events, "cursor": cursor + len(events), "done": done})
        elif path == "/api/results":
            raw = self._first(query, "root", str(self.server.results_root))
            root = project_path(raw)
            files = list_results(root)
            self.server.results_root = root
            self._json({"root": str(root), "files": files})
        elif path == "/api/results/preview":
            target = result_file(self.server.results_root, self._first(query, "path"))
            if target.suffix.lower() in TEXT_SUFFIXES:
                self._json(preview_text(target))
            elif target.suffix.lower() in IMAGE_SUFFIXES:
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"text": "模型文件请从所在目录打开。", "truncated": False})
        else:
            self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/api/") or not self._authorized(parse_qs(parsed.query)):
            self._json({"error": "未授权"}, HTTPStatus.UNAUTHORIZED)
            return
        try:
            body = self._body()
            path = parsed.path
            if path == "/api/settings":
                settings = save_settings(body)
                self.server.results_root = project_path(str(settings["results_root"]))
                self._json({"settings": settings})
            elif path == "/api/history/clear":
                clear_history()
                self._json({"history": []})
            elif path == "/api/config/parse":
                data = parse_yaml(str(body.get("text", "")))
                self._json({"data": data, "fields": flatten_config(data)})
            elif path == "/api/config/patch":
                self._json(patch_config(
                    str(body.get("text", "")), str(body.get("path", "")), body.get("value"),
                    from_form=body.get("from_form") is True,
                ))
            elif path == "/api/config/save":
                self._json(save_config(str(body.get("path", "")), str(body.get("text", ""))))
            elif path == "/api/update/install":
                state = self.server.manager.state()
                if state and state.get("status") in {"starting", "running", "stopping"}:
                    raise ValueError("请先结束正在运行的任务再安装更新")
                self._json(install_update())
            elif path == "/api/runs":
                self._json({"state": self.server.manager.start(body)})
            elif path == "/api/runs/stop":
                self.server.manager.stop()
                self._json({"state": self.server.manager.state()})
            elif path == "/api/runs/log/open":
                target = self._run_log_path(str(body.get("id", "")))
                self._open_local(target.parent if body.get("folder", False) else target)
                self._json({"ok": True})
            elif path == "/api/results/open":
                target = result_file(self.server.results_root, str(body.get("path", "")))
                self._open_local(target.parent if body.get("folder", True) else target)
                self._json({"ok": True})
            else:
                self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            self._error(exc)

    def _run_log_path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("运行日志不存在")
        current = self.server.manager.state()
        record = next((item for item in history() if item.get("id") == run_id), None)
        if record is None and current and current.get("id") == run_id:
            record = current
        expected = (STATE_DIR / "logs" / f"{run_id}.log").resolve()
        if not record or not record.get("log_path") or Path(record["log_path"]).resolve() != expected or not expected.is_file():
            raise ValueError("运行日志不存在")
        return expected

    @staticmethod
    def _open_local(path: Path) -> None:
        if os.name == "nt":
            with external_dll_search_path(path):
                os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _static(self, path: str) -> None:
        target = (DIST / (path.lstrip("/") or "index.html")).resolve()
        if not target.is_relative_to(DIST.resolve()) or not target.is_file():
            target = DIST / "index.html"
        if not target.is_file():
            self._error(ValueError("前端资源未构建，请先运行 npm install 和 npm run build"), 503)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vtools 浅色桌面工作台")
    parser.add_argument("--browser", action="store_true", help="在浏览器中打开，供开发预览")
    parser.add_argument("--install-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manager = TaskManager()
    server = WorkbenchServer(manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/?token={server.token}"
    try:
        if args.browser:
            print(f"vtools 工作台：{url}")
            webbrowser.open(url)
            try:
                thread.join()
            except KeyboardInterrupt:
                pass
        else:
            from .desktop import run_desktop
            return run_desktop(url, STATE_DIR, manager.close, verify=args.install_check)
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
    return 0
