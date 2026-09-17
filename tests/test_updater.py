"""Regression checks for the Debian update trust boundary."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui.webapp import updater


class UpdaterTests(unittest.TestCase):
    def test_no_release_is_reported_without_installing(self) -> None:
        with patch.object(updater.sys, "frozen", True, create=True), patch.object(updater, "_latest_release", return_value=None):
            self.assertEqual(updater.check_update()["status"], "unpublished")

    def test_new_release_requires_matching_deb_and_sha256(self) -> None:
        release = {
            "tag_name": "v0.3.2",
            "assets": [{
                "name": "vtools_0.3.2_amd64.deb",
                "browser_download_url": "https://github.com/xyzkioo/vtools/releases/download/v0.3.2/vtools_0.3.2_amd64.deb",
                "digest": "sha256:" + "a" * 64,
                "size": 123,
            }],
        }
        with patch.object(updater.sys, "frozen", True, create=True), patch.object(updater, "_latest_release", return_value=release), patch.object(updater, "_is_newer", return_value=True), patch.object(updater.subprocess, "check_output", return_value="amd64\n"):
            info = updater.check_update()
            self.assertEqual(info["status"], "available")
            release["assets"][0]["digest"] = ""
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                updater.check_update()

    def test_bad_download_is_rejected_and_temporary_file_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(updater, "STATE_DIR", Path(directory)), patch.object(updater, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    updater._download_package({
                        "asset_name": "vtools_0.3.2_amd64.deb",
                        "asset_url": "https://github.com/xyzkioo/vtools/releases/download/v0.3.2/vtools_0.3.2_amd64.deb",
                        "asset_digest": "a" * 64,
                        "asset_size": "5",
                    })
            self.assertEqual(list((Path(directory) / "updates").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
