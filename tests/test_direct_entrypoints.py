"""Direct script launches must find the repository's shared runtime."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DirectEntrypointTests(unittest.TestCase):
    def test_public_scripts_start_without_pythonpath(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        cases = (
            ("model_diagnostics/run_model_diagnostics.py", "--list-modules"),
            ("model_visualization/run_visualization.py", "--list-modules"),
            ("speed_test/benchmark_pytorch.py", "--list-modules"),
            ("speed_test/benchmark_tensorrt.py", "--list-modules"),
            ("speed_test/run_all.py", "--list-modules"),
            ("transform_tools/ndjson_to_yolo.py", "--help"),
        )
        for script, option in cases:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / script), option],
                    cwd=ROOT, env=environment, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
