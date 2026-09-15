import tempfile
import unittest
from pathlib import Path
from model_compression.core.registry import ModelRegistry


class RegistryConcurrency(unittest.TestCase):
    def test_stale_new_branch_cannot_overwrite_existing_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.json'
            stale = ModelRegistry(path)
            current = ModelRegistry(path)
            current.ensure_branch('main')
            current.data['branches']['main']['head_version_id'] = 'advanced-version'
            current.save()
            with self.assertRaisesRegex(RuntimeError, '并发创建冲突'):
                stale.ensure_branch('main')
            self.assertEqual(ModelRegistry(path).get_branch('main')['head_version_id'], 'advanced-version')

    def test_separate_branch_creation_preserves_existing_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.json'
            stale = ModelRegistry(path)
            current = ModelRegistry(path)
            current.ensure_branch('main')
            current.data['branches']['main']['head_version_id'] = 'advanced-version'
            current.save()
            stale.ensure_branch('other')
            self.assertEqual(ModelRegistry(path).get_branch('main')['head_version_id'], 'advanced-version')

    def test_run_id_argument_cannot_be_overridden_by_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(Path(tmp) / 'registry.json')
            registry.add_run('run1', {'id': 'wrong', 'status': 'running'})
            registry.add_run('run1', {'status': 'succeeded'})
            self.assertEqual(len(registry.data['runs']), 1)
            self.assertEqual(registry.data['runs'][0]['status'], 'succeeded')
