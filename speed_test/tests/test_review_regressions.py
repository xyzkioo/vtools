import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backends import pytorch_vision_speed_benchmark_v2 as pytorch_backend
from backends import pytorch_vision_tensorrt_benchmark_v2 as tensorrt_backend
from core.vision_benchmark_config import load_config


class BenchmarkReviewRegressions(unittest.TestCase):
    def test_obsolete_checkpoint_switch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("benchmark:\n  inspect_checkpoint: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已移除 benchmark 旧权重检查字段"):
                load_config(path)

    def _base_config(self, root: Path, *, stop_on_failure: bool) -> dict:
        return {
            "project": {"root": str(root)},
            "benchmark": {"device": "cpu", "fail_on_checkpoint_inspection": stop_on_failure},
            "models": [],
        }

    def test_failed_checkpoint_stops_both_backends_before_model_load(self):
        failed = SimpleNamespace(
            name="sample", architecture_status="unknown", adapter_load_status="skipped",
            raw_error="unreadable checkpoint", adapter_error="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            for backend in (pytorch_backend, tensorrt_backend):
                with self.subTest(backend=backend.__name__), \
                     patch.object(backend, "load_config", return_value=self._base_config(Path(tmp), stop_on_failure=True)), \
                     patch.object(backend, "prepare_run_directory", return_value=None), \
                     patch.object(backend, "apply_python_paths"), \
                     patch.object(backend, "get_model_entries", return_value=[{"name": "sample", "weights": str(Path(tmp) / "model.pt"), "adapter": "checkpoint"}]), \
                     patch.object(backend, "inspect_entries", return_value=[failed]), \
                     patch.object(backend, "save_rows") as saved, \
                     patch.object(backend, "create_adapter") as adapter, \
                     patch.object(sys, "argv", ["benchmark", "--only", "checkpoint.inspect,model.parameters"]):
                    rows = backend.main()
                    self.assertEqual(rows[0]["status"], "failed")
                    saved.assert_called_once()
                    adapter.assert_not_called()

    def test_pipeline_without_real_source_fails_instead_of_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pytorch_backend, "load_config", return_value=self._base_config(Path(tmp), stop_on_failure=False)), \
             patch.object(pytorch_backend, "prepare_run_directory", return_value=None), \
             patch.object(pytorch_backend, "apply_python_paths"), \
             patch.object(pytorch_backend, "get_model_entries", return_value=[{"name": "sample", "weights": str(Path(tmp) / "model.pt"), "adapter": "checkpoint"}]), \
             patch.object(pytorch_backend, "create_adapter", return_value=SimpleNamespace(supports_real_predict=True)), \
             patch.object(pytorch_backend, "save_rows") as saved, \
             patch.object(sys, "argv", ["benchmark", "--only", "speed.pytorch_pipeline"]):
            with self.assertRaisesRegex(ValueError, "需要 --source"):
                pytorch_backend.main()
            saved.assert_not_called()

    def test_tensorrt_rejects_explicit_pytorch_module(self):
        with patch.object(sys, "argv", ["benchmark", "--only", "speed.pytorch_call,export.onnx"]):
            with self.assertRaisesRegex(ValueError, "TensorRT 入口不执行"):
                tensorrt_backend.main()


if __name__ == "__main__":
    unittest.main()
