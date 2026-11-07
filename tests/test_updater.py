"""更新器回归：版本比较、按平台选择资产、校验边界与 Windows 半自动安装。"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtools_ui.webapp import updater


def _asset(name: str, version: str, digest: str = "a" * 64, size: int = 123) -> dict:
    return {
        "name": name,
        "browser_download_url": f"https://github.com/xyzkioo/vtools/releases/download/v{version}/{name}",
        "digest": f"sha256:{digest}" if digest else "",
        "size": size,
    }


def _release(version: str, *assets: dict, draft: bool = False, prerelease: bool = False) -> dict:
    return {"tag_name": f"v{version}", "draft": draft, "prerelease": prerelease, "assets": list(assets)}


class VersionTests(unittest.TestCase):
    def test_version_key_accepts_tag_prefix(self) -> None:
        self.assertEqual(updater._version_key("v0.5.7"), (0, 5, 7, 0))
        self.assertIsNone(updater._version_key("nightly"))
        self.assertIsNone(updater._version_key(""))

    def test_is_newer_compares_numerically_not_lexically(self) -> None:
        self.assertTrue(updater._is_newer("0.5.10", "0.5.9"))
        self.assertFalse(updater._is_newer("0.5.7", "0.5.7"))
        self.assertFalse(updater._is_newer("0.5.6", "0.5.7"))


class DiscoveryTests(unittest.TestCase):
    def test_windows_and_deb_versions_are_resolved_independently(self) -> None:
        releases = [
            _release("9.9.9", _asset("vtools-9.9.9-windows-x64-setup.exe", "9.9.9")),
            _release("9.8.0", _asset("vtools_9.8.0_amd64.deb", "9.8.0")),
        ]
        rows = {
            key: updater._platform_row(key, releases, "amd64")
            for key in updater.PLATFORM_ORDER
        }
        self.assertEqual(rows["windows"]["latest_version"], "9.9.9")
        self.assertEqual(rows["linux"]["latest_version"], "9.8.0")
        # 每个平台各自比较版本，互不阻塞
        self.assertEqual(rows["windows"]["status"], "available")
        self.assertEqual(rows["linux"]["status"], "available")

    def test_linux_row_uses_deb_naming_with_underscores(self) -> None:
        releases = [_release("9.9.9", _asset("vtools_9.9.9_amd64.deb", "9.9.9"))]
        row = updater._platform_row("linux", releases, "amd64")
        self.assertEqual(row["asset_name"], "vtools_9.9.9_amd64.deb")
        self.assertEqual(row["status"], "available")

    def test_missing_asset_is_reported_per_platform(self) -> None:
        releases = [_release("9.9.9", _asset("vtools-9.9.9-windows-x64-setup.exe", "9.9.9"))]
        self.assertEqual(updater._platform_row("linux", releases, "amd64")["status"], "missing")

    def test_untrusted_asset_is_not_offered_for_install(self) -> None:
        releases = [_release("0.9.9", _asset("vtools-0.9.9-windows-x64-setup.exe", "0.9.9", digest=""))]
        row = updater._platform_row("windows", releases, "amd64")
        self.assertEqual(row["status"], "untrusted")
        self.assertFalse(row["can_install"])

    def test_draft_and_prerelease_versions_are_ignored(self) -> None:
        releases = [
            _release("9.9.0", _asset("vtools-9.9.0-windows-x64-setup.exe", "9.9.0"), draft=True),
            _release("9.8.0", _asset("vtools-9.8.0-windows-x64-setup.exe", "9.8.0"), prerelease=True),
            _release("9.7.0", _asset("vtools-9.7.0-windows-x64-setup.exe", "9.7.0")),
        ]
        row = updater._platform_row("windows", releases, "amd64")
        self.assertEqual(row["latest_version"], "9.7.0")


class CheckUpdateTests(unittest.TestCase):
    def test_source_run_is_unsupported(self) -> None:
        with patch.object(updater.sys, "frozen", False, create=True):
            self.assertEqual(updater.check_update()["status"], "unsupported")

    def test_no_release_is_reported_without_installing(self) -> None:
        with patch.object(updater.sys, "frozen", True, create=True), patch.object(updater, "_fetch_releases", return_value=[]):
            self.assertEqual(updater.check_update()["status"], "unpublished")

    def test_current_platform_row_drives_overall_status(self) -> None:
        releases = [
            _release("9.9.9", _asset("vtools-9.9.9-windows-x64-setup.exe", "9.9.9")),
            _release("9.8.0", _asset("vtools_9.8.0_amd64.deb", "9.8.0")),
        ]
        with patch.object(updater.sys, "frozen", True, create=True), patch.object(
            updater, "_fetch_releases", return_value=releases
        ), patch.object(updater, "current_platform", return_value="windows"):
            info = updater.check_update()
        self.assertEqual(info["platform"], "windows")
        self.assertEqual(info["latest_version"], "9.9.9")
        self.assertEqual(set(info["platforms"]), {"windows", "linux"})
        self.assertEqual(info["platforms"]["linux"]["latest_version"], "9.8.0")


class DownloadTests(unittest.TestCase):
    def test_bad_download_is_rejected_and_temporary_file_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(updater, "STATE_DIR", Path(directory)), patch.object(
                updater, "urlopen", return_value=io.BytesIO(b"wrong")
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    updater._download_package(
                        {
                            "asset_name": "vtools_0.3.2_amd64.deb",
                            "asset_url": "https://github.com/xyzkioo/vtools/releases/download/v0.3.2/vtools_0.3.2_amd64.deb",
                            "asset_digest": "a" * 64,
                            "asset_size": "5",
                        }
                    )
            self.assertEqual(list((Path(directory) / "updates").iterdir()), [])


class InstallTests(unittest.TestCase):
    def _available(self, platform_name: str) -> dict:
        return {
            "status": "available",
            "current_version": "0.5.7",
            "platform": platform_name,
            "latest_version": "0.5.8",
            "message": "发现新版本 0.5.8",
            "asset_name": "vtools-0.5.8-windows-x64-setup.exe" if platform_name == "windows" else "vtools_0.5.8_amd64.deb",
            "asset_url": "https://github.com/xyzkioo/vtools/releases/download/v0.5.8/asset",
            "asset_digest": "a" * 64,
            "asset_size": 5,
        }

    def test_windows_install_launches_wizard_without_apt(self) -> None:
        info = self._available("windows")
        with tempfile.TemporaryDirectory() as directory, patch.object(updater, "STATE_DIR", Path(directory)), patch.object(
            updater, "check_update", return_value=info
        ), patch.object(updater, "current_platform", return_value="windows"), patch.object(
            updater, "_download_package", return_value=Path(directory) / "setup.exe"
        ), patch.object(updater.subprocess, "Popen") as popen, patch.object(
            updater.subprocess, "run"
        ) as run:
            result = updater.install_update()
        popen.assert_called_once()
        run.assert_not_called()
        self.assertIn("安装向导", result["message"])

    def test_unavailable_update_is_refused(self) -> None:
        with patch.object(updater, "check_update", return_value={"status": "current", "message": "已是最新版本"}):
            with self.assertRaisesRegex(RuntimeError, "已是最新版本"):
                updater.install_update()


if __name__ == "__main__":
    unittest.main()