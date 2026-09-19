from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_runtime.ultralytics import add_repo_to_path, find_repo


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

    def test_packaged_project_discovers_checkout_near_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            packaged_root = base / "opt" / "vtools"
            packaged_root.mkdir(parents=True)
            dataset = base / "workspace" / "lychee_grow" / "dataset_v2" / "data.yaml"
            dataset.parent.mkdir(parents=True)
            dataset.touch()
            fork = base / "workspace" / "ultralytics-cn" / "ultralytics"
            fork.mkdir(parents=True)
            (fork / "__init__.py").touch()
            self.assertEqual(find_repo({}, packaged_root, (dataset,)), fork.parent)

    def test_add_repo_infers_workspace_from_nested_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fork = base / "workspace" / "ultralytics-cn" / "ultralytics"
            weights = base / "workspace" / "project" / "runs" / "weights" / "best.pt"
            fork.mkdir(parents=True)
            (fork / "__init__.py").touch()
            weights.parent.mkdir(parents=True)
            weights.touch()
            original = list(sys.path)
            try:
                found = add_repo_to_path(
                    {"model": {"weights": str(weights)}},
                    base / "opt" / "vtools" / "_internal",
                )
                self.assertEqual(found, fork.parent)
                self.assertEqual(sys.path[0], str(fork.parent))
            finally:
                sys.path[:] = original


if __name__ == "__main__":
    unittest.main()
