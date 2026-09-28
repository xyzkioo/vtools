import unittest

from core.module_selection import resolve_speed_modules


class SpeedModuleSelectionTests(unittest.TestCase):
    def test_only_selection_disables_omitted_modules(self):
        values = resolve_speed_modules({}, only=["speed.pytorch_call"])
        self.assertTrue(values["speed.pytorch_call"])
        self.assertFalse(values["speed.pytorch_pipeline"])
        self.assertFalse(values["model.parameters"])

    def test_explicit_empty_only_disables_all_modules(self):
        for only in ([], [""], [" , "]):
            with self.subTest(only=only):
                self.assertFalse(any(resolve_speed_modules({"modules": {"speed.pytorch_call": True}}, only=only).values()))
        with self.assertRaises(ValueError):
            resolve_speed_modules({}, only=[""], enable=["speed.pytorch_call"])

    def test_partial_new_style_config_is_an_allow_list(self):
        values = resolve_speed_modules({"modules": {"export.onnx": True}})
        self.assertTrue(values["export.onnx"])
        self.assertFalse(values["speed.pytorch_call"])


if __name__ == "__main__":
    unittest.main()
