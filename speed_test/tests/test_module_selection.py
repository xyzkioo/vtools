import unittest

from core.module_selection import resolve_speed_modules


class SpeedModuleSelectionTests(unittest.TestCase):
    def test_only_selection_disables_omitted_modules(self):
        values = resolve_speed_modules({}, only=["speed.pytorch_call"])
        self.assertTrue(values["speed.pytorch_call"])
        self.assertFalse(values["speed.pytorch_pipeline"])
        self.assertFalse(values["model.parameters"])

    def test_partial_new_style_config_is_an_allow_list(self):
        values = resolve_speed_modules({"modules": {"export.onnx": True}})
        self.assertTrue(values["export.onnx"])
        self.assertFalse(values["speed.pytorch_call"])


if __name__ == "__main__":
    unittest.main()
