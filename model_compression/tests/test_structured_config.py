import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from model_compression.modules.dependency_pruning import _structured_section, _input_size
from model_compression.modules.structured import _ultralytics_amp_weights_dir
from model_compression.core.detection import _structured_stage


class StructuredConfigTests(unittest.TestCase):
    def test_oom_error_identifies_stage(self):
        with patch('model_compression.core.detection._cuda_memory_note', side_effect=lambda **_kwargs: ''):
            with self.assertRaisesRegex(RuntimeError, '剪枝前测速.*降低评估 batch_size'):
                _structured_stage('剪枝前测速', lambda: (_ for _ in ()).throw(RuntimeError('CUDA out of memory')))

    def test_structured_settings_use_canonical_nested_section(self):
        config = {'compression': {'structured': {'channel_sparsity': .3, 'pruning_input_size': 128}},
                  'structured': {'channel_sparsity': .1}, 'dataset': {'input_size': 640}}
        self.assertEqual(_structured_section(config)['channel_sparsity'], .3)
        self.assertEqual(_input_size(config), 128)

    def test_top_level_structured_settings_are_ignored(self):
        self.assertEqual(_structured_section({'compression': {'structured': {}},
                                             'structured': {'channel_sparsity': .9}}), {})

    def test_amp_check_weights_use_run_directory_and_are_restored(self):
        previous = Path("original-weights")
        utils = types.ModuleType("ultralytics.utils")
        utils.WEIGHTS_DIR = previous
        utils.SETTINGS = {"weights_dir": str(previous)}
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.utils = utils
        modules = {
            "ultralytics": ultralytics,
            "ultralytics.utils": utils,
        }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            run_dir = Path(tmp) / "run"
            with _ultralytics_amp_weights_dir(run_dir) as cache_dir:
                self.assertEqual(cache_dir, (run_dir / "ultralytics-cache" / "weights").resolve())
                self.assertEqual(utils.WEIGHTS_DIR, cache_dir)
                self.assertEqual(utils.SETTINGS["weights_dir"], str(cache_dir))
                self.assertTrue(cache_dir.is_dir())
            self.assertEqual(utils.WEIGHTS_DIR, previous)
            self.assertEqual(utils.SETTINGS["weights_dir"], str(previous))
