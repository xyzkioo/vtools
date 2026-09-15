"""分支输入解析：连续模块必须消费 head，同时拒绝分支之外的权重。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_compression.core.registry import ModelRegistry


class BranchInputResolution(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.original = self.root / "original.pt"
        self.original.write_bytes(b"original-weights")
        self.pruned = self.root / "pruned.pt"
        self.pruned.write_bytes(b"pruned-weights")
        self.unrelated = self.root / "unrelated.pt"
        self.unrelated.write_bytes(b"unrelated-weights")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _branch_with_history(self) -> ModelRegistry:
        registry = ModelRegistry(self.root / "registry.json")
        registry.ensure_branch("main")
        base = registry.add_version(
            name="base", artifact=self.original, parent_version_id=None, run_id="r1", method="baseline"
        )
        registry.advance_branch("main", base["id"])
        pruned = registry.add_version(
            name="pruned", artifact=self.pruned, parent_version_id=base["id"], run_id="r2", method="unstructured_l1"
        )
        registry.advance_branch("main", pruned["id"])
        return registry

    def test_empty_branch_uses_explicit_weights(self) -> None:
        registry = ModelRegistry(self.root / "registry.json")
        registry.ensure_branch("main")
        version_id, path = registry.resolve_input_version("main", self.original)
        self.assertIsNone(version_id)
        self.assertEqual(path, str(self.original.resolve()))

    def test_head_is_used_for_sequential_modules(self) -> None:
        registry = self._branch_with_history()
        # 配置里仍然写着原始权重，但剪枝后的 head 才是一次运行的真实输入。
        version_id, path = registry.resolve_input_version("main", self.original)
        head = registry.get_branch("main")["head_version_id"]
        self.assertEqual(version_id, head)
        self.assertEqual(path, str(self.pruned.resolve()))

    def test_head_without_explicit_weights(self) -> None:
        registry = self._branch_with_history()
        version_id, path = registry.resolve_input_version("main", None)
        self.assertEqual(version_id, registry.get_branch("main")["head_version_id"])
        self.assertEqual(path, str(self.pruned.resolve()))

    def test_ancestor_weights_stay_valid(self) -> None:
        registry = self._branch_with_history()
        version_id, _ = registry.resolve_input_version("main", self.pruned)
        self.assertEqual(version_id, registry.get_branch("main")["head_version_id"])

    def test_unrelated_weights_are_rejected(self) -> None:
        registry = self._branch_with_history()
        with self.assertRaisesRegex(ValueError, "不是同一份权重"):
            registry.resolve_input_version("main", self.unrelated)

    def test_missing_explicit_weights_are_reported(self) -> None:
        registry = self._branch_with_history()
        with self.assertRaises(FileNotFoundError):
            registry.resolve_input_version("main", self.root / "missing.pt")


if __name__ == "__main__":
    unittest.main()
