"""Check runtime capabilities without pinning incidental versions or models."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from env_test import check_install
from env_test.check_install import Reporter, run_dependency_check


class DependencyCheckTests(unittest.TestCase):
    def test_uv_check_targets_selected_interpreter(self):
        reporter = Reporter()
        with patch("env_test.check_install.shutil.which", return_value="/tools/uv"), \
                patch("env_test.check_install.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as check:
            run_dependency_check(reporter, strict=True)
        self.assertEqual(check.call_args.args[0], ["/tools/uv", "pip", "check", "--python", sys.executable])
        self.assertEqual(reporter.events[-1].status, "PASS")

    def test_checker_exception_fails_only_in_strict_mode(self):
        for strict, status in ((True, "FAIL"), (False, "WARN")):
            for error in (OSError("missing tool"), subprocess.TimeoutExpired("pip", 120)):
                with self.subTest(strict=strict, error=error):
                    reporter = Reporter()
                    with patch("env_test.check_install.shutil.which", return_value=None), \
                            patch("env_test.check_install.subprocess.run", side_effect=error):
                        run_dependency_check(reporter, strict=strict)
                    self.assertEqual(reporter.events[-1].status, status)


class CompatibilityCheckTests(unittest.TestCase):
    def run_check(self, arguments=(), *, info=None, model=None, exports=None, compression=None):
        info = info if info is not None else {
            "module_file": str(Path("installed/ultralytics/__init__.py").resolve()),
            "module_version": "8.2.0",
            "distribution_version": "8.2.0",
            "has_yolo": True,
            "cuda_available": False,
            "imports": {name: {"ok": True} for name in
                        ("torch", "torchvision", "numpy", "pyyaml", "opencv", "pillow")},
        }
        reporter = Reporter()
        with patch.object(sys, "argv", ["check_install.py", *arguments]), \
                patch.object(check_install, "Reporter", return_value=reporter), \
                patch.object(check_install, "run_import_probe", return_value=(info, "")) as imported, \
                patch.object(check_install, "run_dependency_check"), \
                patch.object(check_install, "run_model_probe", return_value=(model, "")) as modeled, \
                patch.object(check_install, "run_export_probe", return_value=(exports or {}, "")), \
                patch.object(check_install, "run_compression_probe", return_value=(compression or {}, "")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = check_install.main()
        return result, reporter, imported, modeled

    def test_installed_package_needs_no_fork_or_fixed_version_even_in_strict_mode(self):
        for arguments in ([], ["--strict"]):
            result, reporter, imported, modeled = self.run_check(arguments)
            self.assertEqual(result, 0)
            self.assertEqual(imported.call_args.args[2], None)
            self.assertFalse(any(e.status == "WARN" and e.name in {"版本", "本地源码目录"}
                                 for e in reporter.events))
            self.assertTrue(any(e.name == "模型构建/前向" and "未执行" in e.message
                                for e in reporter.events))
            modeled.assert_not_called()

    def test_explicit_version_is_warning_unless_strict(self):
        for strict, expected in (([], 0), (["--strict"], 1)):
            result, reporter, _, _ = self.run_check(["--expected-version", "8.4.128", *strict])
            self.assertEqual(result, expected)
            version = next(e for e in reporter.events if e.name == "版本")
            self.assertEqual(version.status, "FAIL" if strict else "WARN")

    def test_missing_runtime_dependencies_still_fail(self):
        info = {"import_error": "No module named ultralytics", "imports": {
            name: {"ok": False, "error": f"No module named {name}"}
            for name in ("torch", "torchvision")}}
        result, reporter, _, modeled = self.run_check(info=info)
        self.assertEqual(result, 1)
        self.assertEqual(reporter.failures, 3)
        modeled.assert_not_called()

    def test_explicit_invalid_source_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, reporter, _, _ = self.run_check(["--source", directory])
        self.assertEqual(result, 1)
        self.assertTrue(any(e.name == "本地源码目录" and e.status == "FAIL" for e in reporter.events))

    def test_selected_weights_do_not_also_require_yolo26_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "ultralytics"
            package.mkdir()
            (package / "__init__.py").touch()
            model_yaml = package / "cfg/models/26/yolo26.yaml"
            model_yaml.parent.mkdir(parents=True)
            model_yaml.touch()
            weights = root / "custom.pt"
            weights.touch()
            result, _, _, modeled = self.run_check(
                ["--source", str(root), "--no-local-check", "--weights", str(weights)],
                model={"weights": {"ok": True, "result_count": 1}},
            )
            self.assertEqual(result, 0)
            self.assertIsNone(modeled.call_args.args[1])
            self.assertEqual(modeled.call_args.args[2], weights.resolve())

    def test_model_failure_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "unsupported.pt"
            weights.touch()
            result, reporter, _, _ = self.run_check(
                ["--weights", str(weights)], model={"error": "Unsupported model layer"})
        self.assertEqual(result, 1)
        self.assertTrue(any(e.name == "模型构建/前向" and e.status == "FAIL" for e in reporter.events))

    def test_optional_exports_warn_unless_strict(self):
        for strict, expected in (([], 0), (["--strict"], 1)):
            result, _, _, _ = self.run_check(["--check-export", *strict], exports={
                "onnx": {"ok": False}, "tensorrt": {"ok": False}})
            self.assertEqual(result, expected)

    def test_missing_compression_dependency_warns_unless_strict(self):
        for strict, expected in (([], 0), (["--strict"], 1)):
            with self.subTest(strict=bool(strict)):
                result, reporter, _, _ = self.run_check(
                    ["--check-compression", *strict],
                    compression={"ok": False, "error": "ModuleNotFoundError: No module named 'torch_pruning'"},
                )
                self.assertEqual(result, expected)
                event = next(e for e in reporter.events if e.name == "压缩环境")
                self.assertEqual(event.status, "FAIL" if strict else "WARN")
                self.assertIn("torch-pruning", event.detail)

    def test_outdated_compression_dependency_warns(self):
        result, reporter, _, _ = self.run_check(
            ["--check-compression"],
            compression={"ok": True, "version": "1.2.0", "file": "torch_pruning/__init__.py"},
        )
        self.assertEqual(result, 0)
        event = next(e for e in reporter.events if e.name == "压缩环境")
        self.assertEqual(event.status, "WARN")
        self.assertIn("1.4.1", event.message)

    def test_satisfied_compression_dependency_passes(self):
        result, reporter, _, _ = self.run_check(
            ["--check-compression"],
            compression={"ok": True, "version": "1.6.0", "file": "torch_pruning/__init__.py"},
        )
        self.assertEqual(result, 0)
        event = next(e for e in reporter.events if e.name == "压缩环境")
        self.assertEqual(event.status, "PASS")

    def test_compression_check_is_skipped_by_default(self):
        result, reporter, _, _ = self.run_check()
        self.assertEqual(result, 0)
        event = next(e for e in reporter.events if e.name == "压缩环境")
        self.assertEqual(event.status, "SKIP")
