"""Backend contract checks for the browser-based workbench."""

from __future__ import annotations

import json
import os
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
from vtools_ui.process_environment import build_tool_environment


class WebWorkbenchTests(unittest.TestCase):
    def test_selected_python_receives_packaged_runtime_and_environment_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "_internal"
            prefix = Path(directory) / "envs" / "vision"
            executable = prefix / "bin" / "python3.10"
            (prefix / "lib").mkdir(parents=True)
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
            (prefix / "lib" / "libstdc++.so.6").touch()
            (prefix / "conda-meta").mkdir()
            inherited = {
                "PATH": "/usr/bin",
                "PYTHONPATH": "/existing/modules",
                "LD_LIBRARY_PATH": "/opt/vtools/_internal",
                "LD_LIBRARY_PATH_ORIG": "/system/lib",
            }
            environment = build_tool_environment(executable, root, inherited, frozen=True)

        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(executable.parent.resolve()))
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0], str(root.resolve()))
        self.assertEqual(environment["LD_LIBRARY_PATH"].split(os.pathsep), [str((prefix / "lib").resolve()), "/system/lib"])
        self.assertEqual(environment["CONDA_PREFIX"], str(prefix.resolve()))
        self.assertEqual(environment["CONDA_DEFAULT_ENV"], "vision")

    def test_frozen_environment_scan_finds_conda_without_path_and_hides_workbench(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            conda = home / "miniconda3/bin/conda"
            yolo = home / "miniconda3/envs/yolo/bin/python"
            conda.parent.mkdir(parents=True)
            yolo.parent.mkdir(parents=True)
            conda.touch()
            yolo.touch()
            environments_file = home / ".conda/environments.txt"
            environments_file.parent.mkdir()
            environments_file.write_text(str(yolo.parent.parent) + "\n", encoding="utf-8")
            with patch.object(core.sys, "frozen", True, create=True), \
                    patch.object(core.sys, "executable", "/opt/vtools/vtools"), \
                    patch.object(core.Path, "home", return_value=home), \
                    patch.object(core.shutil, "which", return_value=None), \
                    patch.object(core, "read_settings", return_value={"python_executable": "/opt/vtools/vtools"}), \
                    patch.object(core.subprocess, "run", side_effect=OSError("unavailable")):
                found = core.environments()

        self.assertEqual(found, [{"label": "yolo", "path": str(yolo.resolve())}])

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
            self.assertFalse(target.with_suffix(".yaml.bak").exists())
            self.assertFalse(core.load_config(str(target))["data"]["modules"]["diagnostics.missed"])

    def test_frozen_config_is_copied_to_user_directory_before_editing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            source = root / "model_compression" / "config" / "config.yaml"
            source.parent.mkdir(parents=True)
            source.write_text("model:\n  task: detect\n", encoding="utf-8")
            user_dir = Path(directory) / "user"
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", user_dir), patch.object(core.sys, "frozen", True, create=True):
                loaded = core.load_config("model_compression/config/config.yaml")
            copied = user_dir / "configs" / "model_compression" / "config" / "config.yaml"
            self.assertEqual(Path(loaded["path"]), copied)
            self.assertEqual(copied.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))

    def test_frozen_config_save_also_redirects_direct_bundle_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            source = root / "config.yaml"
            source.parent.mkdir(parents=True)
            source.write_text("value: old\n", encoding="utf-8")
            user_dir = Path(directory) / "user"
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", user_dir), patch.object(core.sys, "frozen", True, create=True):
                core.save_config(str(source), "value: new\n")
            copied = user_dir / "configs" / "config.yaml"
            self.assertEqual(copied.read_text(encoding="utf-8"), "value: new\n")
            self.assertEqual(source.read_text(encoding="utf-8"), "value: old\n")

    def test_frozen_settings_move_old_bundle_results_root_to_user_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            state = Path(directory) / "user"
            settings_file = state / "settings.json"
            settings_file.parent.mkdir(parents=True)
            settings_file.write_text(json.dumps({"results_root": str(root / "runs")}), encoding="utf-8")
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", state), patch.object(core, "SETTINGS_PATH", settings_file), patch.object(core.sys, "frozen", True, create=True):
                settings = core.read_settings()
            self.assertEqual(Path(settings["results_root"]), state / "runs")

    def test_frozen_model_run_uses_user_copy_of_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            config = root / "model_diagnostics" / "config" / "config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("modules:\n  diagnostics.missed: true\n", encoding="utf-8")
            (root / "run_tools.py").touch()
            user_dir = Path(directory) / "user"
            results = Path(directory) / "user-runs"
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", user_dir), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": str(results)}), patch.object(core.sys, "frozen", True, create=True):
                _executable, args, _config = core.build_command({"tool_id": "diagnostics", "values": {}})
            self.assertIn("--config", args)
            self.assertEqual(Path(args[args.index("--config") + 1]), user_dir / "configs" / "model_diagnostics" / "config" / "config.yaml")

    def test_frozen_dataset_quality_default_output_is_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "model_diagnostics").mkdir(parents=True)
            (root / "model_diagnostics" / "check_dataset.py").touch()
            results = Path(directory) / "user-runs"
            with patch.object(core, "ROOT", root), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": str(results)}), patch.object(core.sys, "frozen", True, create=True):
                _executable, args, _config = core.build_command({"tool_id": "dataset_quality", "values": {"data": str(Path(directory) / "data.yaml")}})
            output = Path(args[args.index("--output-root") + 1])
            self.assertEqual(output, results / "dataset_quality")
            self.assertNotIn(root, output.parents)

    def test_frozen_kmodel_defaults_write_to_user_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "transform_tools").mkdir(parents=True)
            (root / "transform_tools" / "py2kmodel.py").touch()
            results = Path(directory) / "user-runs"
            with patch.object(core, "ROOT", root), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": str(results)}), patch.object(core.sys, "frozen", True, create=True):
                _executable, args, _config = core.build_command({"tool_id": "transform", "variant": "kmodel", "values": {"pt": str(Path(directory) / "best.pt"), "calib": str(Path(directory) / "images")}})
            output = Path(args[args.index("--output-dir") + 1])
            self.assertEqual(output.parent.parent.parent, results)
            self.assertNotIn(root, output.parents)

    def test_kmodel_uses_one_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "converted"
            _executable, args, _config = core.build_command({
                "tool_id": "transform",
                "variant": "kmodel",
                "values": {"pt": str(root / "best.pt"), "output": str(output), "calib": str(root / "images")},
            })
        self.assertEqual(args[args.index("--output-dir") + 1], str(output.resolve()))
        self.assertNotIn("--onnx", args)
        self.assertNotIn("--kmodel", args)

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

    def test_dataset_quality_is_an_independent_ui_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            report_root = root / "quality-reports"
            data.write_text("names: [cat]\nval: images/val\n", encoding="utf-8")
            _executable, args, config = core.build_command({
                "tool_id": "dataset_quality",
                "values": {"data": str(data), "output_root": str(report_root), "sample_count": 0},
            })
        self.assertEqual(Path(args[0]).name, "check_dataset.py")
        self.assertIn("--data", args)
        self.assertEqual(args[args.index("--data") + 1], str(data.resolve()))
        self.assertEqual(args[args.index("--output-root") + 1], str(report_root.resolve()))
        self.assertEqual(args[args.index("--sample-count") + 1], "0")
        self.assertIsNone(config)

    def test_dataset_quality_requires_data_yaml(self) -> None:
        with self.assertRaisesRegex(ValueError, "数据集 YAML"):
            core.build_command({"tool_id": "dataset_quality", "values": {}})

    def test_frozen_workbench_wraps_tool_scripts(self) -> None:
        with patch.object(core.sys, "frozen", True, create=True), patch.object(core.sys, "executable", sys.executable):
            executable, args, _config = core.build_command({"tool_id": "environment", "values": {"skip_model": True}})
        self.assertEqual(Path(executable).resolve(), Path(sys.executable).resolve())
        self.assertEqual(args[0], "--run-script")
        self.assertEqual(Path(args[1]).name, "check_install.py")

    def test_environment_check_leaves_source_automatic_when_empty(self) -> None:
        _executable, args, _config = core.build_command({"tool_id": "environment", "values": {"skip_model": True}})
        self.assertNotIn("--source", args)
        self.assertIn("--skip-model", args)

    def test_environment_check_passes_explicit_source(self) -> None:
        _executable, args, _config = core.build_command({"tool_id": "environment", "values": {"source": "../ultralytics-ezcn"}})
        self.assertEqual(args[args.index("--source") + 1], str((core.ROOT.parent / "ultralytics-ezcn").resolve()))

    def test_frozen_model_run_uses_writable_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core.sys, "frozen", True, create=True), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": directory}):
                _executable, args, _config = core.build_command({"tool_id": "diagnostics", "values": {}})
            run_dir = Path(args[args.index("--run-dir") + 1])
            self.assertEqual(run_dir.parent, Path(directory) / "diagnostics")
            self.assertTrue(run_dir.parent.is_dir())
            self.assertFalse(run_dir.exists())

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
            self.assertIn("dataset_quality", payload["catalog"])
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

    def test_log_open_failure_reaps_started_process(self) -> None:
        manager = TaskManager()
        started = []
        from vtools_ui.webapp import tasks
        real_popen = tasks.subprocess.Popen
        real_open = Path.open

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        def fail_log(path, *args, **kwargs):
            if path.suffix == ".log":
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "import time; time.sleep(30)"], None)), patch("vtools_ui.webapp.tasks.add_history"), patch.object(tasks.subprocess, "Popen", side_effect=capture), patch.object(Path, "open", fail_log):
            manager.start({"tool_id": "environment"})
            cursor = 0
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "failed")
        self.assertEqual(manager.state()["exit_code"], 127)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].poll())

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

    def test_dataset_quality_findings_are_completed_with_warning(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "print('issues found'); raise SystemExit(1)"], None)), patch("vtools_ui.webapp.tasks.add_history") as add_history_mock:
            manager.start({"tool_id": "dataset_quality"})
            cursor = 0
            completed = False
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    completed = True
                    break
        self.assertEqual(manager.state()["status"], "succeeded_with_issues")
        self.assertEqual(manager.state()["exit_code"], 1)
        self.assertTrue(completed)
        self.assertEqual(add_history_mock.call_args.args[0]["status"], "检查完成（有问题）")

    def test_benchmark_difference_is_completed_with_warning(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "raise SystemExit(1)"], None)), patch("vtools_ui.webapp.tasks.add_history") as add_history_mock:
            manager.start({"tool_id": "benchmark"})
            cursor = 0
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "succeeded_with_issues")
        self.assertEqual(add_history_mock.call_args.args[0]["status"], "完成（有差异）")

    def test_benchmark_execution_error_remains_failed(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "raise SystemExit(2)"], None)), patch("vtools_ui.webapp.tasks.add_history"):
            manager.start({"tool_id": "benchmark"})
            cursor = 0
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "failed")

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
