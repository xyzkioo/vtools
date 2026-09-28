"""Dependency checker failures must obey strict mode, including missing tools."""

import subprocess
import sys
import unittest
from unittest.mock import patch

from env_test.check_install import Reporter, run_dependency_check


class DependencyCheckTests(unittest.TestCase):
    def test_uv_check_targets_selected_interpreter(self):
        reporter = Reporter()
        with patch("env_test.check_install.shutil.which", return_value="/tools/uv"), \
                patch("env_test.check_install.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as check:
            run_dependency_check(reporter, strict=True)
        self.assertEqual(check.call_args.args[0], ["/tools/uv", "pip", "check", "--python", sys.executable])
        self.assertEqual(reporter.events[-1].status, "PASS")

    def test_checker_exception_fails_only_in_strict_mode(self):
        for strict, status in ((True, "FAIL"), (False, "WARN")):
            for error in (OSError("missing tool"), subprocess.TimeoutExpired("pip", 120)):
                with self.subTest(strict=strict, error=error):
                    reporter = Reporter()
                    with patch("env_test.check_install.shutil.which", return_value=None), \
                            patch("env_test.check_install.subprocess.run", side_effect=error):
                        run_dependency_check(reporter, strict=strict)
                    self.assertEqual(reporter.events[-1].status, status)
