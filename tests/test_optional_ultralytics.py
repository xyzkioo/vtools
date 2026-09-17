from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_runtime.ultralytics import find_repo


class OptionalUltralyticsDiscoveryTests(unittest.TestCase):
    def test_discovers_sibling_fork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vtools"
            fork = Path(directory) / "ultralytics-cn" / "ultralytics"
            (root / "placeholder").mkdir(parents=True)
            fork.mkdir(parents=True)
            (fork / "__init__.py").touch()
            self.assertEqual(find_repo({}, root), fork.parent)

    def test_discovers_ezcn_named_sibling_fork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vtools"
            fork = Path(directory) / "ultralytics-ezcn" / "ultralytics"
            (root / "placeholder").mkdir(parents=True)
            fork.mkdir(parents=True)
            (fork / "__init__.py").touch()
            self.assertEqual(find_repo({}, root), fork.parent)

    def test_explicit_path_beats_sibling_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vtools"
            explicit = Path(directory) / "my-fork" / "ultralytics"
            sibling = Path(directory) / "ultralytics-cn" / "ultralytics"
            for package in (explicit, sibling):
                package.mkdir(parents=True)
                (package / "__init__.py").touch()
            with patch.dict(os.environ, {"VTOOLS_ULTRALYTICS_REPO": str(sibling.parent)}):
                self.assertEqual(find_repo({"project": {"ultralytics_repo": str(explicit.parent)}}, root), explicit.parent)

    def test_missing_checkout_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(find_repo({}, Path(directory) / "vtools"))


if __name__ == "__main__":
    unittest.main()
