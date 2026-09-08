"""TensorRT 构建元数据的无 GPU 回归测试。"""

import tempfile
import unittest
from pathlib import Path

from core.build_metadata import (
    collect_build_metadata,
    compare_build_request,
    metadata_path,
    read_build_metadata,
    write_build_metadata,
)


class BuildMetadataTests(unittest.TestCase):
    def test_sidecar_round_trip_and_parameter_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "model.engine"
            engine.write_bytes(b"engine")
            payload = collect_build_metadata(precision="fp16", batch=1, input_size="832x832")
            sidecar = write_build_metadata(engine, payload)
            self.assertEqual(sidecar, metadata_path(engine))
            self.assertEqual(read_build_metadata(engine)["precision"], "fp16")
            self.assertEqual(compare_build_request(read_build_metadata(engine), {"precision": "fp32"})[0].split(":", 1)[0], "precision")

    def test_missing_sidecar_is_informational(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "model.engine"
            engine.write_bytes(b"engine")
            self.assertIsNone(read_build_metadata(engine))
            self.assertTrue(compare_build_request(None, {}))


if __name__ == "__main__":
    unittest.main()
