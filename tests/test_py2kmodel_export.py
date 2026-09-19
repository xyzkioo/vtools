"""A failed ONNX rebuild must leave the prior output intact."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transform_tools.py2kmodel import export_onnx, load_yolo, output_paths


class ExportOnnxTests(unittest.TestCase):
    def test_load_yolo_discovers_sibling_fork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "vtools"
            fork_package = root / "ultralytics-ezcn" / "ultralytics"
            fork_package.mkdir(parents=True)
            (fork_package / "__init__.py").write_text("class YOLO: pass\n", encoding="utf-8")
            script = project / "transform_tools" / "py2kmodel.py"
            script.parent.mkdir(parents=True)
            script.touch()
            with patch("transform_tools.py2kmodel.__file__", str(script)), patch.dict("sys.modules", {"ultralytics": None}):
                # A None cache entry deliberately simulates a previous failed import.
                import sys
                sys.modules.pop("ultralytics", None)
                yolo = load_yolo(project / "model.pt")
                sys.modules.pop("ultralytics", None)
                sys.path.remove(str(fork_package.parent))
        self.assertEqual(yolo.__name__, "YOLO")

    def test_output_directory_uses_weight_stem_for_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            onnx, kmodel = output_paths("/models/custom.pt", directory)
        self.assertEqual(Path(onnx).name, "custom.onnx")
        self.assertEqual(Path(kmodel).name, "custom.kmodel")

    def test_simplification_failure_preserves_previous_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.pt"
            weights.write_bytes(b"weights")
            output = root / "model.onnx"
            output.write_bytes(b"old onnx")

            class FakeYOLO:
                def __init__(self, path: str) -> None:
                    self.path = Path(path)

                def export(self, **_kwargs: object) -> str:
                    exported = self.path.with_suffix(".onnx")
                    exported.write_bytes(b"new onnx")
                    return str(exported)

            model = SimpleNamespace(graph=SimpleNamespace(input=[SimpleNamespace(name="images")]))
            onnx = SimpleNamespace(load=lambda _path: model, shape_inference=SimpleNamespace(infer_shapes=lambda value: value))
            onnxsim = SimpleNamespace(simplify=lambda value, **_kwargs: (value, False))
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=FakeYOLO), "onnx": onnx, "onnxsim": onnxsim}):
                with self.assertRaisesRegex(RuntimeError, "简化校验失败"):
                    export_onnx(str(weights), str(output), rebuild=True)
            self.assertEqual(output.read_bytes(), b"old onnx")
            self.assertEqual(weights.read_bytes(), b"weights")
