"""A failed ONNX rebuild must leave the prior output intact."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transform_tools.py2kmodel import export_onnx


class ExportOnnxTests(unittest.TestCase):
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
