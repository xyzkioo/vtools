"""检查 GitHub Releases，按平台选择安装包并安装。

- 检查：扫描已发布的正式版本，分别找出 Windows 安装包与 Linux `.deb` 的最新可用版本，
  两个平台各自比较版本，互不阻塞。
- 安装：Windows 下载安装包、校验 SHA-256 后调起 Inno Setup 安装向导（半自动，用户确认向导）；
  Linux 沿用 pkexec + apt-get 安装 `.deb`。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vtools_ui import __version__

from .core import STATE_DIR


RELEASE_LIST_API = "https://api.github.com/repos/xyzkioo/vtools/releases?per_page=30"
ASSET_PREFIX = "https://github.com/xyzkioo/vtools/releases/download/"
MAX_RELEASE_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024

WINDOWS = "windows"
LINUX = "linux"
PLATFORM_ORDER = (WINDOWS, LINUX)
PLATFORM_LABELS = {WINDOWS: "Windows x64", LINUX: "Linux (.deb)"}
VERSION_PATTERN = re.compile(r"\d+(?:\.\d+){1,3}")


def current_platform() -> str:
    """本机平台；只有这两类安装包，其他平台按 Linux 处理。"""

    return WINDOWS if os.name == "nt" else LINUX


def _version_key(text: str) -> tuple[int, ...] | None:
    value = str(text or "").strip().removeprefix("v")
    if not VERSION_PATTERN.fullmatch(value):
        return None
    parts = [int(part) for part in value.split(".")]
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def _is_newer(candidate: str, baseline: str = __version__) -> bool:
    newer = _version_key(candidate)
    if newer is None:
        return False
    reference = _version_key(baseline)
    if reference is None:
        return True
    return newer > reference


def _linux_arch() -> str:
    if shutil.which("dpkg") is not None:
        result = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    machine = platform.machine().lower()
    return {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine or "amd64")


def _asset_name(plat: str, version: str, arch: str) -> str:
    """Windows 安装包用连字符，Debian 包用下划线，不能混用。"""

    if plat == WINDOWS:
        return f"vtools-{version}-windows-x64-setup.exe"
    return f"vtools_{version}_{arch}.deb"


def _request(url: str) -> Request:
    return Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": f"vtools/{__version__}"})


def _fetch_releases() -> list[dict]:
    try:
        with urlopen(_request(RELEASE_LIST_API), timeout=20) as response:
            payload = response.read(MAX_RELEASE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"检查更新失败：GitHub 返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"检查更新失败：{exc}") from exc
    if len(payload) > MAX_RELEASE_BYTES:
        raise RuntimeError("检查更新失败：发布信息过大")
    try:
        releases = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError("检查更新失败：发布信息不是有效 JSON") from exc
    if not isinstance(releases, list):
        raise RuntimeError("检查更新失败：发布信息格式错误")
    return [item for item in releases if isinstance(item, dict)]


def _trusted_asset(asset: dict) -> tuple[str, str, int] | None:
    """返回 (下载地址, sha256 摘要, 大小)；缺少可信信息时返回 None。"""

    url = str(asset.get("browser_download_url", ""))
    digest = str(asset.get("digest", ""))
    size = asset.get("size")
    if not url.startswith(ASSET_PREFIX):
        return None
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        return None
    if not isinstance(size, int) or not 0 < size <= MAX_PACKAGE_BYTES:
        return None
    return url, digest[7:].lower(), size


def _discover(releases: list[dict], plat: str, arch: str) -> dict | None:
    """在正式发布版里找该平台版本号最高、且带对应安装包的版本。"""

    best: dict | None = None
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        version = _version_key(str(release.get("tag_name", "")))
        if version is None or (best is not None and version <= best["version_key"]):
            continue
        assets = {
            str(asset.get("name", "")): asset
            for asset in release.get("assets") or []
            if isinstance(asset, dict)
        }
        asset = assets.get(_asset_name(plat, str(release["tag_name"]).strip().removeprefix("v"), arch))
        if asset is None:
            continue
        trusted = _trusted_asset(asset)
        best = {
            "version_key": version,
            "version": str(release["tag_name"]).strip().removeprefix("v"),
            "asset": asset,
            "trusted": trusted,
        }
    return best


def _platform_row(plat: str, releases: list[dict], arch: str) -> dict:
    found = _discover(releases, plat, arch)
    row: dict[str, object] = {
        "key": plat,
        "label": PLATFORM_LABELS[plat],
        "latest_version": "",
        "status": "missing",
        "message": f"{PLATFORM_LABELS[plat]} 尚无已发布的安装包",
        "can_install": False,
        "is_current_platform": plat == current_platform(),
    }
    if found is None:
        return row
    row["latest_version"] = found["version"]
    if found["trusted"] is None:
        row["status"] = "untrusted"
        row["message"] = f"{PLATFORM_LABELS[plat]} {found['version']} 缺少可信的下载地址或 SHA-256 摘要"
        return row
    url, digest, size = found["trusted"]
    row.update(
        {
            "asset_name": str(found["asset"].get("name", "")),
            "asset_url": url,
            "asset_digest": digest,
            "asset_size": size,
        }
    )
    if not _is_newer(found["version"]):
        row["status"] = "current"
        row["message"] = f"{PLATFORM_LABELS[plat]} 已是最新 {found['version']}"
        return row
    row["status"] = "available"
    row["message"] = f"{PLATFORM_LABELS[plat]} 最新 {found['version']}"
    row["can_install"] = plat == current_platform() and getattr(sys, "frozen", False)
    if plat != current_platform():
        row["message"] = f"{PLATFORM_LABELS[plat]} 最新 {found['version']}（仅该平台可安装）"
    return row


def check_update() -> dict:
    """返回当前平台状态，同时给出 Windows 与 deb 两个平台的最新版本。"""

    plat = current_platform()
    if not getattr(sys, "frozen", False):
        return {
            "status": "unsupported",
            "current_version": __version__,
            "platform": plat,
            "latest_version": "",
            "message": "更新器仅用于安装版；源码运行请直接更新代码",
            "platforms": {},
        }

    releases = _fetch_releases()
    if not releases:
        return {
            "status": "unpublished",
            "current_version": __version__,
            "platform": plat,
            "latest_version": "",
            "message": "GitHub 尚无发布版",
            "platforms": {},
        }
    linux_arch = _linux_arch()
    rows = {key: _platform_row(key, releases, linux_arch if key == LINUX else "amd64") for key in PLATFORM_ORDER}
    mine = rows[plat]
    info = {
        "status": mine["status"],
        "current_version": __version__,
        "platform": plat,
        "latest_version": mine["latest_version"],
        "message": mine["message"],
        "platforms": rows,
    }
    if mine["status"] == "available":
        for field in ("asset_name", "asset_url", "asset_digest", "asset_size"):
            info[field] = mine[field]
    elif mine["status"] == "missing":
        info["status"] = "missing_asset"
    return info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_package(info: dict) -> Path:
    updates = STATE_DIR / "updates"
    updates.mkdir(parents=True, exist_ok=True)
    target = updates / str(info["asset_name"])
    expected_digest = str(info["asset_digest"])
    expected_size = int(info["asset_size"])
    if target.is_file() and target.stat().st_size == expected_size:
        if _sha256_file(target) == expected_digest:
            return target

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=updates, prefix=f".{target.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            total = 0
            try:
                with urlopen(_request(str(info["asset_url"])), timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > expected_size or total > MAX_PACKAGE_BYTES:
                            raise RuntimeError("下载的安装包大小与发布信息不符")
                        handle.write(chunk)
                        digest.update(chunk)
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"下载安装包失败：{exc}") from exc
            if total != expected_size or digest.hexdigest() != expected_digest:
                raise RuntimeError("安装包 SHA-256 校验失败")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _launch_windows_installer(package: Path) -> None:
    """半自动：拉起安装向导，由用户在向导里确认；安装结束后重新打开工作台。"""

    try:
        subprocess.Popen([str(package)], cwd=str(package.parent), close_fds=True)
    except OSError as exc:
        raise RuntimeError(f"无法启动安装向导：{exc}") from exc


def _install_linux_package(info: dict, package: Path) -> dict[str, str]:
    if shutil.which("pkexec") is None:
        raise RuntimeError("缺少 pkexec，无法请求系统安装权限")
    field_text = subprocess.check_output(
        ["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"], text=True,
    )
    fields = dict(line.split(": ", 1) for line in field_text.splitlines() if ": " in line)
    architecture = _linux_arch()
    if fields != {"Package": "vtools", "Version": info["latest_version"], "Architecture": architecture}:
        raise RuntimeError("安装包名称、版本或架构与发布信息不符")
    result = subprocess.run(
        ["pkexec", "/usr/bin/apt-get", "install", "-y", str(package)],
        capture_output=True, text=True, timeout=900, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"安装更新失败：{detail[-1000:] or f'退出码 {result.returncode}'}")
    return {"message": f"已安装 {info['latest_version']}，请关闭并重新打开工作台"}


def install_update() -> dict[str, str]:
    info = check_update()
    if info["status"] != "available":
        raise RuntimeError(str(info["message"]))
    package = _download_package(info)
    if current_platform() == WINDOWS:
        _launch_windows_installer(package)
        return {
            "message": (
                f"已下载 {info['asset_name']} 并通过 SHA-256 校验，安装向导已启动；"
                "在向导里完成安装后重新打开工作台"
            ),
        }
    return _install_linux_package(info, package)