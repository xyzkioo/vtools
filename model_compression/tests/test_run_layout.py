import json
import tempfile
import unittest
from pathlib import Path

from model_compression.core.run_manager import prepare_run_directory
from model_compression.run_model_compression import _build_summary


class RunLayoutTests(unittest.TestCase):
    def test_prepare_run_creates_only_shared_layout_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = prepare_run_directory({"run": {"root": tmp, "name": "auto"}})
            self.assertIsNotNone(run_dir)
            assert run_dir is not None
            self.assertTrue((run_dir / "artifacts" / "structured_pruning").is_dir())
            self.assertTrue((run_dir / "artifacts" / "unstructured_pruning").is_dir())
            self.assertFalse((run_dir / "run_info.json").exists())
            self.assertFalse((run_dir / "registry.json").exists())

    def test_summary_is_flat_and_omits_internal_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            artifact = run_dir / "artifacts" / "unstructured_pruning" / "model.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"model")
            summary = _build_summary(
                {"model": {"name": "demo", "task": "classify", "adapter": "torch", "weights": "/input/model.pt"}},
                run_dir,
                "run1",
                {"baseline.evaluate": True, "compression.prune.unstructured": True},
                [
                    {"module": "baseline.evaluate", "status": "succeeded", "metrics": {"evaluation_accuracy": 0.8}},
                    {"module": "compression.prune.unstructured", "status": "succeeded",
                     "version_id": "v-1", "parent_version_id": "v-0",
                     "artifact": {"path": str(artifact), "size_bytes": 5, "sha256": "secret"},
                     "metadata": {"actual_sparsity": 0.3, "version_id": "v-1"}},
                ],
                [],
                "now",
            )
            payload = json.dumps(summary, ensure_ascii=False)
            self.assertIn("unstructured_pruning", summary)
            self.assertIn("artifacts/unstructured_pruning/model.pt", payload)
            self.assertNotIn("sha256", payload)
            self.assertNotIn("parent_version_id", payload)
            self.assertNotIn("version_id", payload)


if __name__ == "__main__":
    unittest.main()
