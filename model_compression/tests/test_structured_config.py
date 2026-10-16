import os
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

    @staticmethod
    def _mock_ultralytics():
        previous = Path("original-weights")
        utils = types.ModuleType("ultralytics.utils")
        utils.WEIGHTS_DIR = previous
        utils.SETTINGS = {"weights_dir": str(previous)}
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.utils = utils
        return utils, {"ultralytics": ultralytics, "ultralytics.utils": utils}

    def test_amp_check_weights_are_reused_and_settings_restored(self):
        utils, modules = self._mock_ultralytics()
        previous = utils.WEIGHTS_DIR
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
            with _ultralytics_amp_weights_dir() as first:
                self.assertEqual(first, (Path(tmp) / "ultralytics" / "weights").resolve())
                self.assertEqual(utils.WEIGHTS_DIR, first)
                self.assertEqual(utils.SETTINGS["weights_dir"], str(first))
                (first / "yolo26n.pt").write_bytes(b"cached-check-weights")
            self.assertEqual(utils.WEIGHTS_DIR, previous)
            self.assertEqual(utils.SETTINGS["weights_dir"], str(previous))
            with _ultralytics_amp_weights_dir() as second:
                self.assertEqual(second, first)
                self.assertEqual((second / "yolo26n.pt").read_bytes(), b"cached-check-weights")

    def test_amp_cache_defaults_to_user_home(self):
        utils, modules = self._mock_ultralytics()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {"XDG_CACHE_HOME": ""}), \
                patch.object(Path, "home", return_value=Path(tmp)):
            with _ultralytics_amp_weights_dir() as cache_dir:
                self.assertEqual(cache_dir, Path(tmp) / ".cache" / "ultralytics" / "weights")

    def test_amp_settings_are_restored_after_training_error(self):
        utils, modules = self._mock_ultralytics()
        previous = utils.WEIGHTS_DIR
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with _ultralytics_amp_weights_dir():
                    raise RuntimeError("training failed")
            self.assertEqual(utils.WEIGHTS_DIR, previous)
            self.assertEqual(utils.SETTINGS["weights_dir"], str(previous))
