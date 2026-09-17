"""Check GitHub Releases and install a newer checksum-verified Debian asset."""

from __future__ import annotations

import hashlib
import json
import os
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


RELEASE_API = "https://api.github.com/repos/xyzkioo/vtools/releases/latest"
ASSET_PREFIX = "https://github.com/xyzkioo/vtools/releases/download/"
MAX_RELEASE_BYTES = 1024 * 1024
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024


def _request(url: str) -> Request:
    return Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": f"vtools/{__version__}"})


def _latest_release() -> dict | None:
    try:
        with urlopen(_request(RELEASE_API), timeout=15) as response:
            payload = response.read(MAX_RELEASE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"检查更新失败：GitHub 返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"检查更新失败：{exc}") from exc
    if len(payload) > MAX_RELEASE_BYTES:
        raise RuntimeError("检查更新失败：发布信息过大")
    try:
        release = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError("检查更新失败：发布信息不是有效 JSON") from exc
    if not isinstance(release, dict):
        raise RuntimeError("检查更新失败：发布信息格式错误")
    return release


def _is_newer(version: str) -> bool:
    result = subprocess.run(
        ["dpkg", "--compare-versions", version, "gt", __version__],
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("无法比较安装包版本")
    return result.returncode == 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_update() -> dict[str, str]:
    if not getattr(sys, "frozen", False) or sys.platform != "linux":
        return {"status": "unsupported", "current_version": __version__, "message": "更新器仅用于 Linux 安装版"}
    release = _latest_release()
    if release is None:
        return {"status": "unpublished", "current_version": __version__, "message": "GitHub 尚无发布版"}

    version = str(release.get("tag_name", "")).removeprefix("v")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version):
        raise RuntimeError("发布版版本号格式无效")
    info = {"current_version": __version__, "latest_version": version}
    if not _is_newer(version):
        return {**info, "status": "current", "message": "已是最新版本"}

    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    name = f"vtools_{version}_{architecture}.deb"
    for asset in release.get("assets", []):
        if not isinstance(asset, dict) or asset.get("name") != name:
            continue
        url = str(asset.get("browser_download_url", ""))
        digest = str(asset.get("digest", ""))
        size = asset.get("size")
        if not url.startswith(ASSET_PREFIX) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
            raise RuntimeError("发布版安装包缺少可信的下载地址或 SHA-256 摘要")
        if not isinstance(size, int) or not 0 < size <= MAX_PACKAGE_BYTES:
            raise RuntimeError("发布版安装包大小无效")
        return {
            **info, "status": "available", "message": f"发现新版本 {version}",
            "asset_name": name, "asset_url": url, "asset_digest": digest[7:].lower(),
            "asset_size": str(size),
        }
    return {**info, "status": "missing_asset", "message": f"发布版 {version} 尚无适用于 {architecture} 的 .deb"}


def _download_package(info: dict[str, str]) -> Path:
    updates = STATE_DIR / "updates"
    updates.mkdir(parents=True, exist_ok=True)
    target = updates / info["asset_name"]
    expected_digest = info["asset_digest"]
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
                with urlopen(_request(info["asset_url"]), timeout=60) as response:
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


def install_update() -> dict[str, str]:
    info = check_update()
    if info["status"] != "available":
        raise RuntimeError(info["message"])
    if shutil.which("pkexec") is None:
        raise RuntimeError("缺少 pkexec，无法请求系统安装权限")
    package = _download_package(info)
    field_text = subprocess.check_output(
        ["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"], text=True,
    )
    fields = dict(line.split(": ", 1) for line in field_text.splitlines() if ": " in line)
    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
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
