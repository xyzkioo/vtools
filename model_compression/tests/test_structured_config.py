import unittest
from model_compression.modules.dependency_pruning import _structured_section, _input_size


class StructuredConfigTests(unittest.TestCase):
    def test_nested_settings_override_legacy_settings(self):
        config = {'compression': {'structured': {'channel_sparsity': .3, 'pruning_input_size': 128}},
                  'structured': {'channel_sparsity': .1}, 'dataset': {'input_size': 640}}
        self.assertEqual(_structured_section(config)['channel_sparsity'], .3)
        self.assertEqual(_input_size(config), 128)

    def test_empty_nested_settings_do_not_use_stale_legacy_values(self):
        self.assertEqual(_structured_section({'compression': {'structured': {}},
                                             'structured': {'channel_sparsity': .9}}), {})
