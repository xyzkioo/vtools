import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_tools


class UnifiedEntry(unittest.TestCase):
    def test_invalid_yaml_does_not_fall_back_to_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "tools.yaml"
            config.write_text("tool: [", encoding="utf-8")
            with patch.object(sys, "argv", ["run_tools", "--config", str(config)]):
                self.assertEqual(run_tools.main(), 2)

    def test_invalid_tool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "tools.yaml"
            config.write_text("tool: typo\n", encoding="utf-8")
            with patch.object(sys, "argv", ["run_tools", "--config", str(config)]):
                self.assertEqual(run_tools.main(), 2)

    def test_repository_tools_config_resolves_child_path_from_root(self):
        captured = []

        def child_main():
            captured.extend(sys.argv)
            return 0

        module = SimpleNamespace(main=child_main)
        config = run_tools.ROOT / "config" / "tools.yaml"
        with patch.object(sys, "argv", ["run_tools", "--tool", "diagnostics", "--config", str(config)]), \
                patch.object(run_tools.importlib, "import_module", return_value=module):
            self.assertEqual(run_tools.main(), 0)
        child_config = captured[captured.index("--config") + 1]
        self.assertEqual(Path(child_config), run_tools.ROOT / "model_diagnostics" / "config" / "pycharm_run.yaml")


if __name__ == "__main__":
    unittest.main()
