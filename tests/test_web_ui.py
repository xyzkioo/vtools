"""Backend contract checks for the browser-based workbench."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vtools_ui.webapp import core
from vtools_ui.webapp.server import DesktopBridge, WorkbenchServer
from vtools_ui.webapp.tasks import TaskManager


class WebWorkbenchTests(unittest.TestCase):
    def test_desktop_picker_uses_native_dialog_for_file_and_directory(self) -> None:
        class FileDialog(IntEnum):
            OPEN = 10
            FOLDER = 20
            SAVE = 30

        bridge = DesktopBridge()
        bridge.window = Mock()
        bridge.window.create_file_dialog.side_effect = [("/tmp/model.pt",), ("/tmp/images",), None]
        with patch.dict(sys.modules, {"webview": SimpleNamespace(FileDialog=FileDialog)}):
            self.assertEqual(bridge.pick_path("file"), "/tmp/model.pt")
            self.assertEqual(bridge.pick_path("dir"), "/tmp/images")
            self.assertEqual(bridge.pick_path("save_file"), "")
            with self.assertRaises(ValueError):
                bridge.pick_path("file_or_dir")
        self.assertEqual(
            [call.args[0] for call in bridge.window.create_file_dialog.call_args_list],
            [FileDialog.OPEN, FileDialog.FOLDER, FileDialog.SAVE],
        )

    def test_config_views_preserve_unknown_and_dotted_modules(self) -> None:
        raw = "modules:\n  diagnostics.missed: true\n  predict.generate: false\ncustom:\n  keep: [a, b]\n"
        changed = core.patch_config(raw, "modules.diagnostics.missed", False)
        self.assertFalse(changed["data"]["modules"]["diagnostics.missed"])
        self.assertFalse(changed["data"]["modules"]["predict.generate"])
        self.assertEqual(changed["data"]["custom"]["keep"], ["a", "b"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text(raw, encoding="utf-8")
            core.save_config(str(target), changed["text"])
            self.assertEqual(target.with_suffix(".yaml.bak").read_text(encoding="utf-8"), raw)
            self.assertFalse(core.load_config(str(target))["data"]["modules"]["diagnostics.missed"])

    def test_module_edits_are_incremental_and_empty_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "diagnostics.yaml"
            target.write_text("modules:\n  predict.generate: true\n  diagnostics.missed: true\n", encoding="utf-8")
            request = {"tool_id": "diagnostics", "config_path": str(target), "python_executable": core.sys.executable, "module_edits": {"diagnostics.missed": False}, "values": {}}
            _executable, args, _config = core.build_command(request)
            self.assertIn("--disable", args)
            self.assertIn("diagnostics.missed", args)
            self.assertNotIn("--only", args)
            self.assertNotIn("predict.generate", args)
            request["module_edits"] = {"predict.generate": False, "diagnostics.missed": False}
            _executable, args, _config = core.build_command(request)
            self.assertEqual(args.count("--disable"), 2)

    def test_structured_commands_cover_converters_and_benchmark_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tool_id, variant, values, expected in (
                ("transform", "ndjson", {"input": str(root / "data.ndjson"), "output": str(root / "yolo")}, "ndjson_to_yolo.py"),
                ("transform", "kmodel", {"pt": str(root / "model.pt"), "calib": str(root / "images")}, "py2kmodel.py"),
                ("transform", "validate", {"onnx": str(root / "model.onnx"), "kmodel": str(root / "model.kmodel"), "image": str(root / "image.jpg"), "size": "320"}, "PY2KM_validate.py"),
                ("data", "video", {"input": str(root / "video.mp4"), "every_n": "8"}, "video_frame_extractor.py"),
                ("data", "resize", {"input": str(root / "image.jpg"), "width": "1024", "height": "1024"}, "image_resize.py"),
                ("data", "filename", {"dir": str(root), "dataset": "auto", "apply": False}, "image_filename_converter.py"),
            ):
                _executable, args, _config = core.build_command({"tool_id": tool_id, "variant": variant, "values": values})
                self.assertEqual(Path(args[0]).name, expected)
            _executable, args, _config = core.build_command({"tool_id": "benchmark", "values": {"backend": "all"}})
            self.assertEqual(Path(args[0]).name, "run_all.py")
            _executable, args, _config = core.build_command({"tool_id": "benchmark", "values": {"backend": "checkpoint"}})
            self.assertEqual(Path(args[0]).name, "inspect_checkpoint.py")

    def test_result_preview_does_not_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            root.mkdir()
            inside = root / "summary.json"
            inside.write_text('{}', encoding="utf-8")
            outside = Path(directory) / "secret.txt"
            outside.write_text("private", encoding="utf-8")
            self.assertEqual(core.result_file(root.resolve(), str(inside)), inside)
            with self.assertRaises(ValueError):
                core.result_file(root.resolve(), str(outside))

    def test_project_result_scan_excludes_source_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("source", encoding="utf-8")
            report = root / "reports" / "run1" / "summary.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            with patch.object(core, "ROOT", root):
                files = core.list_results(root)
            self.assertEqual([item["path"] for item in files], [str(report)])

    def test_api_requires_session_token(self) -> None:
        server = WorkbenchServer(TaskManager())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/api/bootstrap"
        try:
            with self.assertRaises(HTTPError) as rejected:
                urlopen(url, timeout=5)
            self.assertEqual(rejected.exception.code, 401)
            response = urlopen(Request(url, headers={"X-Vtools-Token": server.token}), timeout=5)
            payload = json.load(response)
            self.assertIn("catalog", payload)
            self.assertIn("settings", payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_run_start_failure_is_visible(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=("/missing/python", ["unused.py"], None)), patch("vtools_ui.webapp.tasks.add_history"):
            state = manager.start({"tool_id": "diagnostics"})
            cursor = 0
            while True:
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(state["status"], "starting")
        self.assertEqual(manager.state()["status"], "failed")
        self.assertEqual(manager.state()["exit_code"], 127)

    def test_running_process_can_be_stopped(self) -> None:
        manager = TaskManager()
        command = ["-c", "import time; print('ready', flush=True); time.sleep(30)"]
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, command, None)), patch("vtools_ui.webapp.tasks.add_history"):
            manager.start({"tool_id": "environment"})
            cursor = 0
            while manager.state()["status"] == "starting":
                events, _done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
            manager.stop()
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "stopped")
        self.assertNotEqual(manager.state()["exit_code"], 0)

    def test_successful_process_reports_logs_and_exit_status(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "print('finished')"], None)), patch("vtools_ui.webapp.tasks.add_history"):
            manager.start({"tool_id": "environment"})
            cursor = 0
            output = []
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                output.extend(str(event["payload"]) for event in events if event["kind"] == "log")
                if done:
                    break
        self.assertEqual(manager.state()["status"], "succeeded")
        self.assertEqual(manager.state()["exit_code"], 0)
        self.assertIn("finished", "".join(output))

    def test_result_directory_detected_from_file_report_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "summary.json"
            report.write_text("{}", encoding="utf-8")
            manager = TaskManager()
            manager._state = {"result_dir": None}
            manager._capture_result(f"汇总结果已保存：{report}\n")
            self.assertEqual(manager.state()["result_dir"], directory)


if __name__ == "__main__":
    unittest.main()
