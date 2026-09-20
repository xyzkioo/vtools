"""无 GPU 回归测试：python -m unittest discover -s tests -v。"""

import csv
import importlib.util
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from checks.vision_consistency_metrics import (
    aggregate_status, align_outputs, combine_status, compare_array, compare_detections,
)
from checks.vision_consistency import _sources, save_reports
from core.vision_benchmark_config import apply_python_paths, load_config
from run_all import _invoke, _merge_results, main as run_all_main
from core.vision_run_manager import apply_run_directory, prepare_run_directory, resolve_cli_path
from core.vision_runtime import _with_library_dir


class TensorTests(unittest.TestCase):
    def check(self, a, b, **kwargs):
        return compare_array(a, b, atol=kwargs.get("atol", 0.001), rtol=kwargs.get("rtol", 0.01))

    def test_close_float(self):
        r = self.check(np.array([0., 2.]), np.array([0.0005, 2.01]))
        self.assertEqual(r["status"], "passed")
        self.assertEqual(r["close_fraction"], 1.)

    def test_clear_mismatch(self):
        r = self.check([1., 2.], [1., 3.])
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["max_abs_error"], 1.)
        self.assertEqual(r["mismatched_elements"], 1)
        self.assertEqual(r["worst_index"], [1])

    def test_no_broadcast(self):
        self.assertEqual(self.check([[1., 2.]], [1., 2.])["reason"], "shape_mismatch")

    def test_scalar(self):
        self.assertEqual(self.check(np.array(1.), np.array(1.))["worst_index"], [])

    def test_nan_and_inf_fail_even_if_equal(self):
        for value in (np.nan, np.inf, -np.inf):
            r = self.check([value], [value])
            self.assertEqual(r["status"], "failed")
            self.assertEqual(r["reason"], "nan_or_inf")

    def test_empty_is_not_pass(self):
        self.assertEqual(self.check([], [])["status"], "inconclusive")

    def test_integer_and_bool_exact(self):
        self.assertEqual(self.check([1000], [1001], atol=5)["status"], "failed")
        self.assertEqual(self.check([True], [False], atol=5)["status"], "failed")

    def test_integer_no_float_precision_loss(self):
        self.assertEqual(self.check(np.array([2**60]), np.array([2**60 + 1]))["status"], "failed")

    def test_tolerance_validation(self):
        for atol in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.check([1.], [1.], atol=atol)

    def test_align_by_name(self):
        rows = align_outputs({"a": 1, "b": 2}, {"b": 20, "a": 10})
        self.assertEqual(rows, [("a", 1, 10), ("b", 2, 20)])

    def test_missing_extra_or_no_outputs_rejected(self):
        for actual in ({"a": 1}, {"a": 1, "b": 2, "c": 3}, {"x": 1, "y": 2}):
            with self.assertRaises(ValueError):
                align_outputs({"a": 1, "b": 2}, actual)
        with self.assertRaises(ValueError):
            align_outputs({}, {})


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[[1., 2., 11., 12., .8, 0.], [20., 20., 30., 30., .7, 1.]]])

    def compare(self, a, b):
        return compare_detections(a, b, conf=.25, iou_threshold=.95, score_atol=.01)

    def test_reordered_detections_match(self):
        b = self.a[:, ::-1, :].copy()
        r = self.compare(self.a, b)
        self.assertEqual(r["status"], "passed")
        raw = compare_array(self.a, b, atol=.001, rtol=.01)
        self.assertEqual(combine_status([raw], r), "warning")

    def test_detection_only_uses_task_output_for_acceptance(self):
        b = self.a[:, ::-1, :].copy()
        detection = self.compare(self.a, b)
        raw = compare_array(self.a, b, atol=.001, rtol=.01)
        self.assertEqual(raw["status"], "failed")
        self.assertEqual(
            combine_status([raw], detection, acceptance_mode="detection_only"),
            "passed",
        )

    def test_detection_only_does_not_hide_hard_tensor_error(self):
        detection = self.compare(self.a, self.a)
        self.assertEqual(
            combine_status(
                [{"status": "failed", "reason": "shape_mismatch"}],
                detection,
                acceptance_mode="detection_only",
            ),
            "failed",
        )

    def test_class_error_and_score_error_fail(self):
        for column, value in ((5, 2.), (4, .4), (0, 4.)):
            b = self.a.copy()
            b[0, 0, column] = value
            self.assertEqual(self.compare(self.a, b)["status"], "failed")

    def test_one_side_extra_box_fails(self):
        r = self.compare(self.a, self.a[:, :1, :])
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["images"][0]["unmatched_pytorch"], 1)

    def test_empty_detections_inconclusive(self):
        empty = self.a.copy()
        empty[:, :, 4] = .01
        self.assertEqual(self.compare(empty, empty)["status"], "inconclusive")

    def test_invalid_shape_and_boxes(self):
        self.assertEqual(self.compare(self.a[:, :, :5], self.a)["status"], "failed")
        b = self.a.copy()
        b[0, 0, 2] = 0
        self.assertEqual(self.compare(b, b)["status"], "failed")

    def test_nan_below_threshold_still_fails(self):
        b = self.a.copy()
        b[0, 0, 4] = .01
        b[0, 0, 0] = np.nan
        self.assertEqual(self.compare(b, b)["reason"], "nan_or_inf")

    def test_matching_uses_augmenting_path(self):
        from checks.vision_consistency_metrics import _matching
        edges = np.array([[True, True], [True, False]])
        ious = np.array([[1., .99], [1., 0.]])
        self.assertEqual(_matching(edges, ious), [(0, 1), (1, 0)])

    def test_hard_tensor_error_not_hidden_by_boxes(self):
        r = self.compare(self.a, self.a)
        self.assertEqual(combine_status([{"status": "failed", "reason": "nan_or_inf"}], r), "failed")

    def test_mixed_status_aggregation(self):
        self.assertEqual(aggregate_status(["passed", "error"]), "error")
        self.assertEqual(aggregate_status(["passed", "warning"]), "warning")
        self.assertEqual(aggregate_status([]), "inconclusive")


class WorkflowTests(unittest.TestCase):
    def test_cli_model_path_participates_in_fork_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fork = base / "workspace" / "ultralytics-cn"
            (fork / "ultralytics").mkdir(parents=True)
            (fork / "ultralytics/__init__.py").touch()
            weights = base / "workspace/project/runs/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.touch()
            config = {
                "project": {"root": str(base / "opt/vtools"), "python_paths": []},
                "benchmark": {"adapter": "ultralytics", "task": "detect"},
                "models": [],
            }
            original = list(sys.path)
            try:
                apply_python_paths(config, [str(weights)])
                self.assertEqual(sys.path[0], str(fork))
            finally:
                sys.path[:] = original

    def test_conda_library_path_is_prepended_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conda_lib = root / "conda" / "lib"
            other = root / "other"
            conda_lib.mkdir(parents=True)
            other.mkdir()
            environment = _with_library_dir(
                {"LD_LIBRARY_PATH": os.pathsep.join([str(other), str(conda_lib), str(other)])},
                conda_lib,
            )
            parts = environment["LD_LIBRARY_PATH"].split(os.pathsep)
            self.assertEqual(parts[0], str(conda_lib.resolve()))
            self.assertEqual(sum(Path(item).resolve() == conda_lib.resolve() for item in parts), 1)

    def test_cli_dependency_failure_returns_error_report(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("本测试针对无 torch 的环境；CUDA 实测另行执行")
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"project": {"root": directory}, "models": [{"name": "demo", "weights": "x.pt"}],
                      "run": {"enabled": False},
                      "consistency": {"output": "result.csv", "json_output": "result.json"}}
            config_path = path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "check_consistency.py"),
                       "--config", str(config_path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            report = json.loads((path / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(report["status"], "error")
        self.assertIn("torch", report["models"][0]["error"])

    def test_no_image_fallback_to_random(self):
        with self.assertRaises(ValueError):
            _sources({"input_mode": "image"}, {})
        self.assertEqual(_sources({"input_mode": "adapter"}, {}), [None])

    def test_dataset_root_prefers_validation_images_over_generated_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "val" / "images" / "original.jpg"
            generated = root / "run1" / "samples" / "report.png"
            validation.parent.mkdir(parents=True)
            generated.parent.mkdir(parents=True)
            validation.touch()
            generated.touch()
            self.assertEqual(
                _sources({"input_mode": "image", "source": str(root), "max_images": 10}, {}),
                [validation],
            )

    def test_ultralytics_detect_export_and_validation_keep_primary_output(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("需要 torch")
        import torch
        from core.vision_benchmark_common import _TupleOutputWrapper, UltralyticsAdapter

        class Demo(torch.nn.Module):
            def forward(self, value):
                return value + 1, {"aux": value + 2}

        value = torch.zeros(1)
        outputs = _TupleOutputWrapper(Demo(), primary_only=True)(value)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(outputs[0], value + 1))
        adapter = UltralyticsAdapter(task="detect")
        self.assertTrue(adapter.primary_export_output())
        self.assertEqual(adapter.validation_outputs((value + 1, value + 2), ["detections"]), {"detections": value + 1})

    def test_ultralytics_native_export_is_confined_to_run_directory(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("需要 torch")
        import torch
        from core.vision_benchmark_common import InputBundle, UltralyticsAdapter

        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pt_path = "/private/input/best.pt"

        class Model:
            def __init__(self):
                self.model = Target()
                self.arguments = None

            def export(self, **kwargs):
                self.arguments = kwargs
                exported = Path(self.model.pt_path).with_suffix(".onnx")
                exported.write_bytes(b"onnx")
                return str(exported)

        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            output = Path(directory) / "artifacts" / "model.onnx"
            result = UltralyticsAdapter(task="detect").export_onnx(
                model,
                output,
                InputBundle.from_value(torch.zeros(1, 3, 320, 640)),
                opset=17,
                dynamic=False,
            )
            self.assertEqual(result, output.resolve())
            self.assertEqual(model.model.pt_path, "/private/input/best.pt")
            self.assertEqual(model.arguments["imgsz"], [320, 640])
            self.assertEqual(model.arguments["device"], "cpu")
            self.assertEqual(model.arguments["quantize"], 32)

    def test_yaml_paths_and_preserved_settings(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "benchmark_config.yaml")
        self.assertTrue(cfg["modules"]["speed.pytorch_call"])
        self.assertTrue(cfg["modules"]["speed.tensorrt_call"])
        self.assertFalse(cfg["modules"]["consistency.tensor"])
        self.assertTrue(cfg["modules"]["consistency.detection"])
        self.assertFalse(cfg["modules"]["speed.pytorch_pipeline"])
        self.assertNotIn("enabled", cfg["consistency"])
        self.assertTrue(Path(cfg["consistency"]["output"]).is_absolute())
        self.assertEqual(cfg["benchmark"]["input_size"], "832x832")

    def test_run_directories_increment_and_group_outputs(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "project": {"root": directory},
                "run": {"enabled": True, "root": "runs-profile", "name": "auto"},
                "models": [{"name": "demo", "weights": "x.pt"}],
                "pytorch": {"output": "runs-profile/vision_speed_v2.csv"},
                "tensorrt": {"engine_dir": "runs-profile/engines", "onnx_dir": "runs-profile/onnx"},
                "run_all": {"summary_output": "runs-profile/summary.csv"},
            }), encoding="utf-8")
            first = load_config(config_path)
            first_dir = prepare_run_directory(first)
            second = load_config(config_path)
            second_dir = prepare_run_directory(second)
            self.assertEqual(first_dir.name, "run1")
            self.assertEqual(second_dir.name, "run2")
            self.assertEqual(Path(first["pytorch"]["output"]), first_dir / "vision_speed_v2.csv")
            self.assertEqual(Path(first["tensorrt"]["engine_dir"]), root / "runs-profile" / "engines")
            self.assertEqual(Path(first["tensorrt"]["onnx_dir"]), root / "runs-profile" / "onnx")
            self.assertEqual(Path(first["run_all"]["summary_output"]), first_dir / "summary.csv")

    def test_run_dir_override_is_reused_by_child_stage(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "project": {"root": directory},
                "run": {"enabled": True, "root": "runs-profile"},
                "pytorch": {"output": "runs-profile/pytorch.csv"},
            }), encoding="utf-8")
            first = load_config(config_path)
            allocated = prepare_run_directory(first)
            child = load_config(config_path)
            reused = prepare_run_directory(child, allocated)
            self.assertEqual(reused, allocated)
            self.assertEqual(Path(child["pytorch"]["output"]), allocated / "pytorch.csv")

    def test_explicit_run_dir_moves_build_artifacts_to_writable_directory(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "project": {"root": "/opt/vtools"},
                "run": {"enabled": True, "root": "runs-profile"},
                "tensorrt": {"engine_dir": "runs-profile/engines", "onnx_dir": "runs-profile/onnx"},
            }), encoding="utf-8")
            config = load_config(config_path)
            writable = root / "desktop-run"
            prepare_run_directory(config, writable)
            self.assertEqual(Path(config["tensorrt"]["engine_dir"]), writable / "engines")
            self.assertEqual(Path(config["tensorrt"]["onnx_dir"]), writable / "onnx")

    def test_relative_outputs_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "_raw_config": {"pytorch": {"output": "../../outside.csv"}},
                "project": {"root": str(root)},
                "run": {"enabled": True, "root": str(root / "runs")},
                "pytorch": {"output": str(root / "outside.csv")},
            }
            with self.assertRaisesRegex(ValueError, "跳出运行目录"):
                apply_run_directory(config, root / "runs" / "run1")
            with self.assertRaisesRegex(ValueError, "跳出运行目录"):
                resolve_cli_path("../outside.csv", root / "runs" / "run1", root)

    def test_false_success_regression(self):
        rows = [{"backend": "tensorrt", "status": "failed", "error": "ABI error"}]
        argv = sys.argv[:]
        with patch("run_all.importlib.import_module", return_value=SimpleNamespace(main=lambda: rows)):
            status, returned = _invoke("tensorrt", "fake_module", "config.yaml")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(returned, rows)
        self.assertEqual(sys.argv, argv)

    def test_import_failure_captured(self):
        with patch("run_all.importlib.import_module", side_effect=ImportError("missing dependency")):
            status, rows = _invoke("tensorrt", "fake_module", "config.yaml")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(rows, [])

    def test_consistency_warning_not_success(self):
        with patch("run_all.importlib.import_module", return_value=SimpleNamespace(main=lambda: {"status": "warning"})):
            status, rows = _invoke("consistency", "fake_module", "config.yaml")
        self.assertEqual(status["status"], "issues_found")
        self.assertEqual(rows, [])

    def test_consistency_difference_is_not_execution_failure(self):
        with patch("run_all.importlib.import_module", return_value=SimpleNamespace(main=lambda: {"status": "failed", "json_output": "report.json"})):
            status, _rows = _invoke("consistency", "fake_module", "config.yaml")
        self.assertEqual(status, {"backend": "consistency", "status": "issues_found", "error": "", "report": "report.json"})

    def test_merge_uses_current_rows_and_labels_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _merge_results(path, [{"model": "fresh", "median_ms": 1.}], [{"backend": "tensorrt", "status": "failed"}])
            with path.open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["record_type"], "measurement")
        self.assertEqual(rows[1]["record_type"], "stage_summary")
        self.assertEqual(rows[1]["status"], "failed")

    def test_failed_tensorrt_skips_consistency(self):
        config = {"_config_path": "dummy.yaml", "modules": {
            "speed.tensorrt_call": True,
            "consistency.tensor": True,
        }}
        with tempfile.TemporaryDirectory() as directory:
            config["run_all"] = {"summary_output": str(Path(directory) / "summary.csv")}
            with patch("run_all.load_config", return_value=config), patch("sys.argv", ["run_all.py"]), \
                    patch("run_all._invoke", return_value=({"backend": "tensorrt", "status": "failed", "error": "bad"}, [])) as invoke:
                run_all_main()
                self.assertEqual(invoke.call_count, 1)

    def test_run_all_only_routes_pytorch_module_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "project": {"root": str(root)},
                "run": {"enabled": True, "root": str(root / "runs")},
                "run_all": {"summary_output": str(root / "summary.csv")},
                "modules": {"speed.pytorch_call": True},
            }
            with patch("run_all.load_config", return_value=config), patch("sys.argv", ["run_all.py", "--only", "speed.pytorch_call"]), \
                    patch("run_all._invoke", return_value=({"backend": "pytorch", "status": "succeeded"}, [])) as invoke:
                run_all_main()
            self.assertEqual(invoke.call_count, 1)
            self.assertEqual(invoke.call_args.args[0], "pytorch")

    def test_error_report_written_without_cuda(self):
        report = {"status": "error", "models": [{"name": "test", "status": "error", "error": "No CUDA"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_reports(report, path / "a.csv", path / "a.json")
            saved = json.loads((path / "a.json").read_text(encoding="utf-8"))
            with (path / "a.csv").open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(saved["status"], "error")
        self.assertEqual(rows[0]["error"], "No CUDA")

    def test_report_paths_must_differ(self):
        with self.assertRaises(ValueError):
            save_reports({"models": []}, Path("x.csv"), Path("x.csv"))


if __name__ == "__main__":
    unittest.main()
