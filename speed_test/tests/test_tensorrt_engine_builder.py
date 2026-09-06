"""TensorRT engine builder 的无 GPU 回归测试。"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backends.tensorrt_engine_builder import (
    _major_version,
    build_engine_from_onnx,
    build_engine_with_python,
)


class _FakeLogger:
    WARNING = 1

    def __init__(self, *_args):
        pass


class _FakeNetwork:
    pass


class _FakeParser:
    num_errors = 0

    def __init__(self, _network, _logger):
        pass

    def parse_from_file(self, _path):
        return True


class _FakeBuilderConfig:
    def __init__(self):
        self.workspace = None
        self.flags = []

    def set_memory_pool_limit(self, _pool, value):
        self.workspace = value

    def set_flag(self, flag):
        self.flags.append(flag)


class _FakeBuilder:
    last = None

    def __init__(self, _logger):
        self.flags = None
        self.config = None
        _FakeBuilder.last = self

    def create_network(self, flags):
        self.flags = flags
        return _FakeNetwork()

    def create_builder_config(self):
        self.config = _FakeBuilderConfig()
        return self.config

    def build_serialized_network(self, _network, _config):
        return b"fake-engine"


def _fake_tensorrt():
    module = types.SimpleNamespace(
        __version__="11.2.1.2",
        Logger=_FakeLogger,
        Builder=_FakeBuilder,
        OnnxParser=_FakeParser,
        NetworkDefinitionCreationFlag=types.SimpleNamespace(STRONGLY_TYPED=0),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
    )
    return module


class BuilderTests(unittest.TestCase):
    def test_major_version(self):
        self.assertEqual(_major_version("TensorRT 11.2.1.2"), 11)
        self.assertEqual(_major_version("unknown"), 0)

    def test_python_builder_uses_strong_typing_and_atomic_output(self):
        fake_tensorrt = _fake_tensorrt()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "model.onnx"
            engine_path = root / "nested" / "model.engine"
            onnx_path.write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"tensorrt": fake_tensorrt}):
                result = build_engine_with_python(onnx_path, engine_path, "fp16", 2)
            self.assertEqual(result, engine_path)
            self.assertEqual(engine_path.read_bytes(), b"fake-engine")
            self.assertEqual(_FakeBuilder.last.flags, 1)
            self.assertEqual(_FakeBuilder.last.config.workspace, 2 * 1024 * 1024)
            self.assertEqual(_FakeBuilder.last.config.flags, [])

    def test_invalid_builder_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_engine_from_onnx(Path("model.onnx"), Path("model.engine"), builder="unknown")


if __name__ == "__main__":
    unittest.main()
