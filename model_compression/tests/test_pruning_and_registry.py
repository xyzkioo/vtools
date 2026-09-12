import tempfile
import unittest
from pathlib import Path


class PruningAndRegistryTests(unittest.TestCase):
    def test_pruning_reports_actual_sparsity(self):
        import torch

        from model_compression.modules.pruning import prune_unstructured

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        pruned, metadata = prune_unstructured(model, {"sparsity": 0.5})
        self.assertGreater(metadata["actual_sparsity"], 0.0)
        self.assertEqual(metadata["zero_parameter_count"], sum(int((p == 0).sum()) for p in pruned.parameters()))

    def test_registry_keeps_lineage_and_branch_head(self):
        from model_compression.core.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.pt"
            source.write_bytes(b"model")
            registry = ModelRegistry(Path(directory) / "registry.json")
            registry.ensure_branch("main")
            first = registry.add_version(name="base", artifact=source, parent_version_id=None, run_id="r1", method="import")
            registry.advance_branch("main", first["id"])
            second = registry.add_version(name="compressed", artifact=source, parent_version_id=first["id"], run_id="r2", method="dynamic_int8")
            registry.advance_branch("main", second["id"])
            self.assertEqual(registry.get_branch("main")["head_version_id"], second["id"])
            self.assertEqual([item["id"] for item in registry.lineage(second["id"])], [second["id"], first["id"]])
            self.assertEqual(registry.data["relations"][0]["role"], "parent")


if __name__ == "__main__":
    unittest.main()
