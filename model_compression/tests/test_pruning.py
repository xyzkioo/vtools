import unittest


class PruningTests(unittest.TestCase):
    def test_pruning_reports_actual_sparsity(self):
        import torch

        from model_compression.modules.pruning import prune_unstructured

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        pruned, metadata = prune_unstructured(model, {"sparsity": 0.5})
        self.assertGreater(metadata["actual_sparsity"], 0.0)
        self.assertEqual(metadata["zero_parameter_count"], sum(int((p == 0).sum()) for p in pruned.parameters()))


if __name__ == "__main__":
    unittest.main()
