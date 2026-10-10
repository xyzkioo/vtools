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

from vtools_ui.webapp import core, server
from vtools_ui.webapp.server import DesktopBridge, WorkbenchServer
from vtools_ui.webapp.tasks import TaskManager
from vtools_ui.process_environment import build_tool_environment


class WebWorkbenchTests(unittest.TestCase):
    def test_history_recovers_from_invalid_content_and_writes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text('{"unexpected": true}', encoding="utf-8")
            with patch.object(core, "HISTORY_PATH", path), \
                    patch.object(core, "read_settings", return_value={"history_limit": 40}):
                core.add_history({"task": "task"})
            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(records, [{"task": "task"}])

    def test_failed_history_publish_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text("[]", encoding="utf-8")
            original_replace = Path.replace

            def fail_for_history(source: Path, target: Path) -> Path:
                if Path(target) == path:
                    raise OSError("disk full")
                return original_replace(source, target)

            with patch.object(core, "HISTORY_PATH", path), \
                    patch.object(core, "read_settings", return_value={"history_limit": 40}), \
                    patch.object(Path, "replace", fail_for_history):
                with self.assertRaises(OSError):
                    core.add_history({"task": "task"})
            self.assertEqual(path.read_text(encoding="utf-8"), "[]")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_selected_python_receives_packaged_runtime_and_environment_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "_internal"
            prefix = Path(directory) / "envs" / "vision"
            executable = prefix / "bin" / "python3.10"
            (prefix / "lib").mkdir(parents=True)
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
            (prefix / "lib" / "libstdc++.so.6").touch()
            (prefix.parent.parent / "lib" / "dotnet" / "host" / "fxr").mkdir(parents=True)
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
        self.assertEqual(environment["DOTNET_ROOT"], str((prefix.parent.parent / "lib" / "dotnet").resolve()))
        self.assertEqual(environment["DOTNET_ROOT_X64"], environment["DOTNET_ROOT"])

    def test_bundled_tool_child_keeps_packaged_library_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "vtools"
            executable.touch()
            inherited = {
                "PATH": "/usr/bin",
                "LD_LIBRARY_PATH": "/opt/vtools/_internal",
                "LD_LIBRARY_PATH_ORIG": "/system/lib",
            }
            with patch.object(core.sys, "executable", str(executable)):
                environment = build_tool_environment(executable, directory, inherited, frozen=True)

        self.assertEqual(environment["LD_LIBRARY_PATH"], "/opt/vtools/_internal")
        self.assertEqual(environment["LD_LIBRARY_PATH_ORIG"], "/system/lib")

    def test_uv_job_does_not_inherit_previous_conda_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / ".venv"
            executable = prefix / "bin" / "python"
            executable.parent.mkdir(parents=True)
            executable.touch()
            (prefix / "pyvenv.cfg").touch()
            environment = build_tool_environment(executable, directory, {
                "CONDA_PREFIX": "/previous/conda", "CONDA_DEFAULT_ENV": "old",
                "VIRTUAL_ENV": "/previous/venv",
            }, frozen=False)
        self.assertNotIn("CONDA_PREFIX", environment)
        self.assertNotIn("CONDA_DEFAULT_ENV", environment)
        self.assertEqual(environment["VIRTUAL_ENV"], str(prefix))

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

    def test_environment_scan_lists_uv_projects_from_python_envs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            project = home / "python-envs" / "vtools"
            executable = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True)
            try:
                executable.symlink_to(sys.executable)
            except OSError as exc:
                self.skipTest(f"无法创建解释器符号链接：{exc}")
            (project / "pyproject.toml").write_text("[project]\nname = 'vtools'\n", encoding="utf-8")
            (project / "uv.lock").touch()
            with patch.object(core.sys, "frozen", True, create=True), \
                    patch.object(core.sys, "executable", "/opt/vtools/vtools"), \
                    patch.object(core.Path, "home", return_value=home), \
                    patch.object(core.shutil, "which", return_value=None), \
                    patch.object(core, "read_settings", return_value={"python_executable": "/opt/vtools/vtools"}), \
                    patch.object(core.subprocess, "run", side_effect=OSError("unavailable")), \
                    patch.dict(os.environ, {}, clear=True):
                found = core.environments()

        self.assertEqual(
            found,
            [{"label": "vtools (uv)", "path": str(executable)}],
        )

    def test_frozen_default_prefers_vtools_uv_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            project = home / "python-envs" / "vtools"
            executable = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True)
            executable.touch()
            (project / "pyproject.toml").write_text("[project]\nname = 'vtools'\n", encoding="utf-8")
            with patch.object(core.sys, "frozen", True, create=True), \
                    patch.object(core.Path, "home", return_value=home), \
                    patch.object(core.shutil, "which", return_value="/usr/bin/python3"), \
                    patch.dict(os.environ, {}, clear=True):
                selected = core._default_python_executable()

        self.assertEqual(selected, str(executable))

    def test_python_executable_path_preserves_uv_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True)
            try:
                executable.symlink_to(sys.executable)
            except OSError as exc:
                self.skipTest(f"无法创建解释器符号链接：{exc}")

            selected = core.python_executable_path(str(executable))
            environment = build_tool_environment(executable, directory, {"PATH": "/usr/bin"}, frozen=False)

        self.assertEqual(selected, executable)
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(executable.parent))

    def test_webview_fallback_probes_uv_environment_not_shared_base_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ".venv" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            try:
                executable.symlink_to(sys.executable)
            except OSError as exc:
                self.skipTest(f"无法创建解释器符号链接：{exc}")
            with patch.object(server, "environments", return_value=[{"path": str(executable)}]), \
                    patch.object(server.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as probe:
                self.assertEqual(server._find_webview_python(), str(executable))
            self.assertEqual(probe.call_args.args[0][0], str(executable))

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

    def test_form_sizes_save_as_integers_and_repair_previously_stringified_values(self) -> None:
        for initial in ("[832, 832]", "['832', '832']", "'832,832'", "832"):
            raw = f"dataset:\n  input_size: {initial}\ncustom:\n  keep: text\n"
            for entered in ("640,640", "[640, 640]", "640"):
                with self.subTest(initial=initial, entered=entered):
                    changed = core.patch_config(raw, "dataset.input_size", entered, from_form=True)
                    self.assertEqual(changed["data"]["dataset"]["input_size"], [640, 640])
                    self.assertTrue(all(type(item) is int for item in changed["data"]["dataset"]["input_size"]))
                    self.assertEqual(core.parse_yaml(changed["text"]), changed["data"])
                    self.assertEqual(changed["data"]["custom"]["keep"], "text")
                    with tempfile.TemporaryDirectory() as directory:
                        target = Path(directory) / "config.yaml"
                        core.save_config(str(target), changed["text"])
                        loaded = core.load_config(str(target))
                        self.assertEqual(loaded["data"]["dataset"]["input_size"], [640, 640])
                        size_field = next(field for field in loaded["fields"] if field["path"] == "dataset.input_size")
                        self.assertEqual(size_field["kind"], "size")
                        self.assertIn("高、宽", size_field["hint"])

    def test_form_rejects_invalid_sizes_and_scalar_training_dimensions(self) -> None:
        raw = "dataset:\n  input_size: [832, 832]\ncompression:\n  structured:\n    train_input_size: 1024\n    pruning_input_size: 640\n"
        for entered in ("", "640,abc", "640,0", "-1,640", "640.5,640", "[640]", "640,640,3", "true,640"):
            with self.subTest(entered=entered), self.assertRaises(ValueError):
                core.patch_config(raw, "dataset.input_size", entered, from_form=True)
        for name in ("train_input_size", "pruning_input_size"):
            path = f"compression.structured.{name}"
            for entered in ("320", "320,320", "[320,320]"):
                self.assertEqual(core.config_get(core.patch_config(raw, path, entered, from_form=True)["data"], path), 320)
            for entered in ("640,320", "0", "", "1.5"):
                with self.subTest(path=path, entered=entered), self.assertRaises(ValueError):
                    core.patch_config(raw, path, entered, from_form=True)

    def test_form_lists_keep_numbers_and_literal_strings(self) -> None:
        raw = "thresholds: [0.1, 0.5]\npaths: [old]\nindices: []\nenabled: true\ncount: 2\nrate: 0.1\nproject:\n  python_paths: []\n"
        for path, entered, expected in (
            ("thresholds", "0.25,0.75", [0.25, 0.75]),
            ("paths", '001,true,null,"a,b"', ["001", "true", "null", "a,b"]),
            ("indices", "1,2", [1, 2]),
            ("indices", "", []),
            ("enabled", False, False),
            ("count", "3", 3),
            ("rate", "1e-3", 0.001),
            ("project.python_paths", "001,true,null", ["001", "true", "null"]),
        ):
            with self.subTest(path=path, entered=entered):
                changed = core.patch_config(raw, path, entered, from_form=True)
                actual = core.config_get(changed["data"], path)
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), type(expected))
        for path, entered in (("thresholds", "0.25,nope"), ("count", "1.5"), ("rate", "inf")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                core.patch_config(raw, path, entered, from_form=True)

    def test_config_help_is_ui_metadata_and_covers_each_model_in_benchmark(self) -> None:
        raw = "models:\n  - weights: first.pt\n  - weights: second.pt\nmodules:\n  compression.prune.structured: true\n"
        fields = {field["path"]: field for field in core.flatten_config(core.parse_yaml(raw))}
        self.assertEqual(fields["models.0.weights"]["hint"], fields["models.1.weights"]["hint"])
        self.assertIn("权重", fields["models.1.weights"]["hint"])
        self.assertIn("结构化", fields["modules.compression.prune.structured"]["hint"])
        changed = core.patch_config(raw, "modules.compression.prune.structured", False, from_form=True)
        self.assertNotIn("结构化", changed["text"])
        self.assertNotIn("hint", changed["data"])

    def test_config_patch_api_parses_raw_form_input_and_returns_error_for_invalid_size(self) -> None:
        web = WorkbenchServer(TaskManager())
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{web.server_port}/api/config/patch"
            raw = "dataset:\n  input_size: [832, 832]\n"
            for entered in ("640,640", "640,bad"):
                body = json.dumps({"text": raw, "path": "dataset.input_size", "value": entered, "from_form": True}).encode()
                request = Request(url, data=body, headers={"X-Vtools-Token": web.token, "Content-Type": "application/json"})
                if entered == "640,640":
                    with urlopen(request, timeout=5) as response:
                        payload = json.load(response)
                    self.assertEqual(payload["data"]["dataset"]["input_size"], [640, 640])
                    self.assertIn("hint", payload["fields"][0])
                else:
                    with self.assertRaises(HTTPError) as rejected:
                        urlopen(request, timeout=5)
                    self.assertEqual(rejected.exception.code, 400)
        finally:
            web.shutdown()
            web.server_close()
            thread.join(timeout=5)

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

    def test_copying_packaged_template_preserves_original_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            source = root / "model_diagnostics" / "config" / "config.yaml"
            source.parent.mkdir(parents=True)
            source.write_text("project:\n  root: ../..\nmode: predictions_file\npredictions_file:\n  predictions: data/predictions.json\n", encoding="utf-8")
            user_dir = Path(directory) / "user"
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", user_dir), patch.object(core.sys, "frozen", True, create=True):
                loaded = core.load_config(str(source))
                self.assertEqual(loaded["data"]["project"]["root"], str(root))
                self.assertEqual(loaded["data"]["predictions_file"]["predictions"], "data/predictions.json")
                core.save_config(loaded["path"], "project:\n  root: /user/custom\n")
                self.assertEqual(core.load_config(str(source))["data"]["project"]["root"], "/user/custom")
            self.assertIn("root: ../..", source.read_text(encoding="utf-8"))

    def test_json_config_remains_json_after_copy_and_graphical_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            source = root / "config.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"project": {"root": "."}, "custom": {"value": 1}}), encoding="utf-8")
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", Path(directory) / "user"), patch.object(core.sys, "frozen", True, create=True):
                loaded = core.load_config(str(source))
                copied = Path(loaded["path"])
                self.assertEqual(json.loads(copied.read_text())["project"]["root"], str(root))
                patched = core.patch_config(loaded["text"], "custom.value", 2)
                core.save_config(str(copied), patched["text"])
                self.assertEqual(json.loads(copied.read_text())["custom"]["value"], 2)
            self.assertEqual(json.loads(source.read_text())["custom"]["value"], 1)

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
            with patch.object(core, "ROOT", root), patch.object(core, "STATE_DIR", user_dir), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": str(results)}), patch.object(core.sys, "frozen", True, create=True), patch.object(core.sys, "executable", "/opt/vtools/vtools"):
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
                executable, args, _config = core.build_command({"tool_id": "dataset_quality", "values": {"data": str(Path(directory) / "data.yaml")}})
            self.assertEqual(executable, core.sys.executable)
            self.assertEqual(args[0], "--run-script")
            output = Path(args[args.index("--output-root") + 1])
            self.assertEqual(output, results / "dataset_quality")
            self.assertNotIn(root, output.parents)

    def test_frozen_lightweight_data_tools_use_bundled_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = (
                ("filename", {"dir": str(root), "dataset": "images", "apply": False}, "image_filename_converter.py"),
                ("resize", {"input": str(root / "image.jpg"), "width": "640", "height": "480"}, "image_resize.py"),
                ("video", {"input": str(root / "video.mp4"), "every_n": "8"}, "video_frame_extractor.py"),
            )
            with patch.object(core, "read_settings", return_value={"python_executable": str(root / "missing-python"), "results_root": str(root / "results")}), \
                    patch.object(core.sys, "frozen", True, create=True):
                for variant, values, script_name in requests:
                    executable, args, _config = core.build_command({
                        "tool_id": "data", "variant": variant, "values": values,
                    })
                    self.assertEqual(executable, core.sys.executable)
                    self.assertEqual(args[0], "--run-script")
                    self.assertEqual(Path(args[1]).name, script_name)

    def test_frozen_model_tools_still_require_selected_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_python = str(Path(directory) / "missing-python")
            with patch.object(core, "read_settings", return_value={"python_executable": missing_python, "results_root": directory}), \
                    patch.object(core.sys, "frozen", True, create=True):
                with self.assertRaisesRegex(ValueError, "Python 解释器不存在"):
                    core.build_command({"tool_id": "diagnostics", "values": {}})

    def test_frozen_diagnostics_dispatch_follows_prediction_and_ground_truth_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled_python = root / "vtools"
            bundled_python.touch()
            external_python = sys.executable
            config = root / "diagnostics.yaml"
            for mode, resource, ground_truth, bundled in (
                ("ultralytics_model", "predictions.json", "data.yaml", True),
                ("predictions_file", "", "data.yaml", True),
                ("predictions_file", "model.pt", "data.yaml", False),
                ("predictions_file", "model.engine", "data.yaml", False),
                ("predictions_file", "predictions.json", "data.ndjson", False),
                ("ultralytics_model", "", "data.yaml", False),
            ):
                config.write_text(f"mode: {mode}\ndataset:\n  data: {ground_truth}\nmodules: {{}}\n", encoding="utf-8")
                with self.subTest(mode=mode, resource=resource, ground_truth=ground_truth), \
                        patch.object(core.sys, "frozen", True, create=True), \
                        patch.object(core.sys, "executable", str(bundled_python)), \
                        patch.object(core, "read_settings", return_value={"python_executable": external_python, "results_root": str(root / "reports")}):
                    executable, args, _config = core.build_command({
                        "tool_id": "diagnostics", "config_path": str(config), "values": {"resource": resource},
                    })
                self.assertEqual(executable, str(bundled_python) if bundled else external_python)
                self.assertEqual(args[0] == "--run-script", bundled)
                if resource.endswith(".engine"):
                    self.assertIn("--weights", args)
                    self.assertNotIn("--predictions", args)

    def test_frozen_predictions_diagnostics_does_not_require_external_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "diagnostics.yaml"
            config.write_text("mode: predictions_file\nmodules: {}\n", encoding="utf-8")
            with patch.object(core.sys, "frozen", True, create=True), \
                    patch.object(core, "read_settings", return_value={"python_executable": str(root / "missing-python"), "results_root": str(root / "reports")}):
                executable, args, _config = core.build_command({"tool_id": "diagnostics", "config_path": str(config), "values": {}})
            self.assertEqual(executable, sys.executable)
            self.assertEqual(args[0], "--run-script")

    def test_frozen_external_tools_reject_workbench_as_model_interpreter(self) -> None:
        with patch.object(core.sys, "frozen", True, create=True), \
                patch.object(core, "read_settings", return_value={"python_executable": sys.executable}):
            with self.assertRaisesRegex(ValueError, "需要外部 Python 模型环境"):
                core.build_command({"tool_id": "environment", "values": {"skip_model": True}})

    def test_frozen_kmodel_defaults_write_to_user_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "transform_tools").mkdir(parents=True)
            (root / "transform_tools" / "py2kmodel.py").touch()
            results = Path(directory) / "user-runs"
            with patch.object(core, "ROOT", root), patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": str(results)}), patch.object(core.sys, "frozen", True, create=True), patch.object(core.sys, "executable", "/opt/vtools/vtools"):
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
        expected = sys.executable
        with patch.object(core.sys, "frozen", True, create=True), \
                patch.object(core.sys, "executable", expected), \
                patch.object(core, "read_settings", return_value={
                    "python_executable": expected,
                    "results_root": str(core.ROOT),
                    "history_limit": 40,
                }):
            executable, args, _config = core.build_command({"tool_id": "data", "variant": "filename", "values": {"dir": str(core.ROOT), "dataset": "images"}})
        self.assertEqual(Path(executable).resolve(), Path(expected).resolve())
        self.assertEqual(args[0], "--run-script")
        self.assertEqual(Path(args[1]).name, "image_filename_converter.py")

    def test_environment_check_leaves_source_automatic_when_empty(self) -> None:
        _executable, args, _config = core.build_command({"tool_id": "environment", "values": {"skip_model": True}})
        self.assertNotIn("--source", args)
        self.assertIn("--skip-model", args)

    def test_environment_check_passes_explicit_source(self) -> None:
        _executable, args, _config = core.build_command({"tool_id": "environment", "values": {"source": "../ultralytics-ezcn"}})
        self.assertEqual(args[args.index("--source") + 1], str((core.ROOT.parent / "ultralytics-ezcn").resolve()))

    def test_frozen_model_run_uses_writable_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core, "read_settings", return_value={"python_executable": sys.executable, "results_root": directory}), patch.object(core.sys, "frozen", True, create=True), patch.object(core.sys, "executable", "/opt/vtools/vtools"):
                _executable, args, _config = core.build_command({"tool_id": "diagnostics", "values": {}})
            run_dir = Path(args[args.index("--run-dir") + 1])
            self.assertEqual(run_dir.parent, Path(directory) / "diagnostics")
            self.assertRegex(run_dir.name, r"^run1-[a-z]{5}$")
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

    def test_benchmark_execution_error_is_not_reported_as_warning(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(
                sys.executable, ["-c", "print('一致性总状态：error'); raise SystemExit(1)"], None)), \
                patch("vtools_ui.webapp.tasks.add_history"):
            manager.start({"tool_id": "benchmark"})
            cursor = 0
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "failed")

    def test_benchmark_findings_are_reported_as_warning(self) -> None:
        manager = TaskManager()
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(
                sys.executable, ["-c", "print('一致性总状态：failed'); raise SystemExit(1)"], None)), \
                patch("vtools_ui.webapp.tasks.add_history"):
            manager.start({"tool_id": "benchmark"})
            cursor = 0
            for _ in range(10):
                events, done = manager.events_after(cursor, timeout=2)
                cursor += len(events)
                if done:
                    break
        self.assertEqual(manager.state()["status"], "succeeded_with_issues")

    def test_log_open_failure_prevents_process_start(self) -> None:
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
        self.assertEqual(len(started), 0)

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

    @unittest.skipUnless(os.name == "posix", "子进程信号退出只在 POSIX 上验证")
    def test_compression_signal_exit_writes_failure_summary(self) -> None:
        from vtools_ui.webapp import tasks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            run_dir = root / "runs" / "run-signal"
            run_dir.mkdir(parents=True)
            (run_dir / "effective_config.json").write_text(json.dumps({
                "model": {"name": "detector", "task": "detect", "weights": "model.pt"},
                "modules": {"baseline.evaluate": True, "compression.prune.structured": True},
            }), encoding="utf-8")
            child = (
                "import os, signal; "
                "print('[baseline.evaluate] 开始', flush=True); "
                "print('[baseline.evaluate] 完成', flush=True); "
                "print('[compression.prune.structured] 开始', flush=True); "
                "print('[结构化剪枝] 开始：依赖图追踪与通道剪枝', flush=True); "
                "os.kill(os.getpid(), signal.SIGKILL)"
            )
            command = ["-c", child, "--run-dir", str(run_dir)]
            with patch.object(tasks, "STATE_DIR", state_dir), \
                    patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, command, None)), \
                    patch("vtools_ui.webapp.tasks.add_history"):
                manager = TaskManager()
                manager.start({"tool_id": "compression"})
                cursor = 0
                for _ in range(10):
                    events, done = manager.events_after(cursor, timeout=2)
                    cursor += len(events)
                    if done:
                        break

            self.assertEqual(manager.state()["status"], "failed")
            self.assertEqual(manager.state()["result_dir"], str(run_dir.resolve()))
            report_path = run_dir / "summary.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure_details"]["signal"], "SIGKILL")
            self.assertEqual(report["failure_details"]["last_module"], "compression.prune.structured")
            self.assertEqual(report["failure_details"]["last_stage"], "依赖图追踪与通道剪枝")
            self.assertEqual(report["module_status"]["baseline.evaluate"], "succeeded")
            self.assertEqual(report["module_status"]["compression.prune.structured"], "failed")

    def test_saved_log_contains_command_and_is_readable_by_run_id(self) -> None:
        from vtools_ui.webapp import tasks
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            history_path = state_dir / "history.json"
            with patch.object(tasks, "STATE_DIR", state_dir), \
                    patch.object(server, "STATE_DIR", state_dir), \
                    patch.object(core, "HISTORY_PATH", history_path), \
                    patch("vtools_ui.webapp.tasks.build_command", return_value=(
                        sys.executable, ["-c", "print('saved output')"], None)):
                manager = TaskManager()
                manager.start({"tool_id": "environment"})
                cursor = 0
                for _ in range(10):
                    events, done = manager.events_after(cursor, timeout=2)
                    cursor += len(events)
                    if done:
                        break
                run_id = manager.state()["id"]
                web = WorkbenchServer(manager)
                thread = threading.Thread(target=web.serve_forever, daemon=True)
                thread.start()
                try:
                    url = f"http://127.0.0.1:{web.server_port}/api/runs/log?id={run_id}"
                    with urlopen(Request(url, headers={"X-Vtools-Token": web.token}), timeout=5) as response:
                        payload = json.load(response)
                    self.assertIn("saved output", payload["text"])
                    self.assertIn("$ ", payload["text"])
                    with patch.object(server.ApiHandler, "_open_local") as open_local:
                        for folder in (False, True):
                            body = json.dumps({"id": run_id, "folder": folder}).encode()
                            request = Request(url.split("?")[0] + "/open", data=body,
                                              headers={"X-Vtools-Token": web.token, "Content-Type": "application/json"})
                            with urlopen(request, timeout=5) as response:
                                self.assertTrue(json.load(response)["ok"])
                            expected = Path(payload["path"])
                            self.assertEqual(open_local.call_args.args[0], expected.parent if folder else expected)
                    with self.assertRaises(HTTPError) as rejected:
                        urlopen(Request(url.replace(run_id, 'unknown'), headers={"X-Vtools-Token": web.token}), timeout=5)
                    self.assertEqual(rejected.exception.code, 400)
                finally:
                    web.shutdown()
                    web.server_close()
                    thread.join(timeout=5)

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
        with patch("vtools_ui.webapp.tasks.build_command", return_value=(sys.executable, ["-c", "print('一致性总状态：failed'); raise SystemExit(1)"], None)), patch("vtools_ui.webapp.tasks.add_history") as add_history_mock:
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
