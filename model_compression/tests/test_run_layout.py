import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from model_compression.core.run_manager import prepare_run_directory, write_json
from model_compression.core.model_runtime import resolve_device, save_model
from model_compression.core.detection import _isolated_validation_data
from model_compression.run_model_compression import _RunState, _build_summary


class RunLayoutTests(unittest.TestCase):
    def test_run_state_uses_latest_artifact_for_sequential_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.pt"
            compressed = root / "compressed.pt"
            original.write_bytes(b"original")
            compressed.write_bytes(b"compressed")
            state = _RunState()
            base = state.add_version(
                name="base", artifact=original, parent_version_id=None,
                run_id="run1", method="baseline",
            )
            state.advance(base["id"])
            latest = state.add_version(
                name="compressed", artifact=compressed, parent_version_id=base["id"],
                run_id="run1", method="unstructured_l1",
            )
            state.advance(latest["id"])

            version_id, artifact = state.current(original)

            self.assertEqual(version_id, latest["id"])
            self.assertEqual(artifact, str(compressed.resolve()))
            self.assertEqual([item["id"] for item in state.lineage()], [latest["id"], base["id"]])

    def test_validation_dataset_mirror_keeps_label_cache_inside_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "dataset" / "images" / "val"
            labels = root / "dataset" / "labels" / "val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "a.jpg").write_bytes(b"image")
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            data = root / "dataset" / "data.yaml"
            data.write_text("path: .\nval: images/val\nnames: [cat]\n", encoding="utf-8")
            run_dir = root / "run1"

            isolated = _isolated_validation_data(data, run_dir, "val")

            self.assertEqual(isolated, run_dir / "_input" / "data.yaml")
            self.assertTrue((run_dir / "_input" / "dataset" / "images" / "val").exists())
            self.assertTrue((run_dir / "_input" / "dataset" / "labels" / "val").exists())
            self.assertFalse((labels.parent / "val.cache").exists())

    def test_explicit_run_directory_cannot_overwrite_old_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run1"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_run_directory({"run": {"root": tmp}}, target)

    def test_absolute_run_directory_does_not_create_configured_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            configured_root = Path(tmp) / "bundle" / "runs"
            target = Path(tmp) / "user" / "run1"
            self.assertEqual(prepare_run_directory({"run": {"root": str(configured_root)}}, target), target)
            self.assertFalse(configured_root.exists())

    def test_failed_json_write_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "summary.json"
            target.write_text("old", encoding="utf-8")
            with patch("model_compression.core.run_manager.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_json(target, {"new": True})
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_failed_model_save_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.pt"
            target.write_bytes(b"old")

            def fail_after_partial(_model, path):
                Path(path).write_bytes(b"partial")
                raise OSError("disk full")

            with patch("model_compression.core.model_runtime.require_torch", return_value=type("Torch", (), {"save": staticmethod(fail_after_partial)})()):
                with self.assertRaises(OSError):
                    save_model(object(), target)
            self.assertEqual(target.read_bytes(), b"old")

    def test_explicit_cuda_does_not_silently_use_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_device("cuda:0")

    def test_prepare_run_creates_only_shared_layout_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = prepare_run_directory({"run": {"root": tmp, "name": "auto"}})
            self.assertIsNotNone(run_dir)
            assert run_dir is not None
            self.assertTrue((run_dir / "artifacts" / "structured_pruning").is_dir())
            self.assertTrue((run_dir / "artifacts" / "unstructured_pruning").is_dir())
            self.assertFalse((run_dir / "run_info.json").exists())

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
