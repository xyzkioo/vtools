"""检查 GitHub Releases，按平台选择安装包并安装。

发现版本有两条路，优先用官方 API（它能给出 GitHub 服务端计算的 SHA-256 摘要）：

1. REST API：`/releases` 列表，资产带 `digest: sha256:...`；未认证时每小时 60 次/IP，
   超限返回 403。
2. 回退：`releases.atom`（不走 REST，不限流）取版本号，再对每个版本的资产做 HEAD
   探测大小，并读取同目录的 `<资产>.sha256` 旁证文件作为摘要。

Windows 安装为半自动：下载 → 校验 SHA-256 与大小 → 调起 Inno Setup 安装向导。
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
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vtools_ui import __version__

from .core import STATE_DIR


RELEASE_LIST_API = "https://api.github.com/repos/xyzkioo/vtools/releases?per_page=30"
ATOM_FEED = "https://github.com/xyzkioo/vtools/releases.atom"
DOWNLOAD_PREFIX = "https://github.com/xyzkioo/vtools/releases/download/"
MAX_RELEASE_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
ATOM_MAX_VERSIONS = 12
CACHE_TTL_SECONDS = 24 * 3600

WINDOWS = "windows"
LINUX = "linux"
PLATFORM_ORDER = (WINDOWS, LINUX)
PLATFORM_LABELS = {WINDOWS: "Windows x64", LINUX: "Linux (.deb)"}
VERSION_PATTERN = re.compile(r"\d+(?:\.\d+){1,3}")
SHA256_PATTERN = re.compile(r"\b([0-9a-fA-F]{64})\b")


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


def _is_newer(candidate: str, baseline: str | None = None) -> bool:
    newer = _version_key(candidate)
    if newer is None:
        return False
    reference = _version_key(baseline if baseline is not None else __version__)
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


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {"Accept": accept, "User-Agent": f"vtools/{__version__}"}


def _read(url: str, *, method: str = "GET", accept: str = "application/vnd.github+json", limit: int = MAX_RELEASE_BYTES):
    request = Request(url, headers=_headers(accept), method=method)
    with urlopen(request, timeout=20) as response:
        return response, response.read(limit)


# --------------------------------------------------------------------------------------
# 发现：REST API（带服务端摘要）
# --------------------------------------------------------------------------------------


def _fetch_releases() -> list[dict]:
    try:
        _, payload = _read(RELEASE_LIST_API)
    except HTTPError as exc:
        if exc.code in {403, 429}:
            raise RuntimeError("GitHub 接口访问次数受限（HTTP 403）") from exc
        raise RuntimeError(f"GitHub 返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"无法连接 GitHub：{exc}") from exc
    if len(payload) > MAX_RELEASE_BYTES:
        raise RuntimeError("发布信息过大")
    try:
        releases = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError("发布信息不是有效 JSON") from exc
    if not isinstance(releases, list):
        raise RuntimeError("发布信息格式错误")
    return [item for item in releases if isinstance(item, dict)]


def _trusted_asset(asset: dict) -> tuple[str, str, int] | None:
    """返回 (下载地址, sha256 摘要, 大小)；缺少可信信息时返回 None。"""

    url = str(asset.get("browser_download_url", ""))
    digest = str(asset.get("digest", ""))
    size = asset.get("size")
    if not url.startswith(DOWNLOAD_PREFIX):
        return None
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        return None
    if not isinstance(size, int) or not 0 < size <= MAX_PACKAGE_BYTES:
        return None
    return url, digest[7:].lower(), size


def _api_candidates(releases: list[dict], plat: str, arch: str) -> list[dict]:
    candidates: list[dict] = []
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = str(release.get("tag_name", "")).strip()
        key = _version_key(tag)
        if key is None:
            continue
        version = tag.removeprefix("v")
        assets = {
            str(asset.get("name", "")): asset
            for asset in release.get("assets") or []
            if isinstance(asset, dict)
        }
        asset = assets.get(_asset_name(plat, version, arch))
        if asset is None:
            continue
        candidates.append({"version": version, "key": key, "trusted": _trusted_asset(asset)})
    candidates.sort(key=lambda item: item["key"], reverse=True)
    return candidates


# --------------------------------------------------------------------------------------
# 发现回退：atom 版本列表 + 资产 HEAD + `<资产>.sha256` 旁证
# --------------------------------------------------------------------------------------


def _atom_versions() -> list[str]:
    try:
        _, payload = _read(ATOM_FEED, accept="application/atom+xml")
    except HTTPError as exc:
        raise RuntimeError(f"GitHub 返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"无法连接 GitHub：{exc}") from exc
    text = payload.decode("utf-8", "replace")
    versions: list[str] = []
    for match in re.finditer(r"/releases/tag/v(\d+(?:\.\d+){1,3})", text):
        version = match.group(1)
        if version not in versions:
            versions.append(version)
    versions.sort(key=lambda value: _version_key(value) or (0,), reverse=True)
    return versions[:ATOM_MAX_VERSIONS]


def _sidecar_digest(asset_url: str) -> str:
    """读取同目录的 `<资产>.sha256`，取其中的 64 位十六进制摘要。"""

    try:
        _, payload = _read(asset_url + ".sha256", accept="text/plain", limit=4096)
    except (HTTPError, URLError, TimeoutError, OSError):
        return ""
    match = SHA256_PATTERN.search(payload.decode("utf-8", "replace"))
    return match.group(1).lower() if match else ""


def _atom_candidates(versions: list[str], plat: str, arch: str, stop_after: int = 1) -> list[dict]:
    """按版本从高到低探测资产，命中 stop_after 个就停，避免逐个版本全扫。"""

    candidates: list[dict] = []
    for version in versions:
        name = _asset_name(plat, version, arch)
        url = f"{DOWNLOAD_PREFIX}v{version}/{name}"
        try:
            response, _ = _read(url, method="HEAD", accept="application/octet-stream", limit=0)
            size = int(response.headers.get("Content-Length") or 0)
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
        if not 0 < size <= MAX_PACKAGE_BYTES:
            continue
        digest = _sidecar_digest(url)
        trusted = (url, digest, size) if digest else None
        candidates.append({"version": version, "key": _version_key(version), "trusted": trusted})
        if len(candidates) >= stop_after:
            break
    return candidates


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------


def _platform_row(plat: str, candidates: list[dict]) -> dict:
    row: dict[str, object] = {
        "key": plat,
        "label": PLATFORM_LABELS[plat],
        "latest_version": "",
        "status": "missing",
        "message": f"{PLATFORM_LABELS[plat]} 尚无已发布的安装包",
        "can_install": False,
        "is_current_platform": plat == current_platform(),
    }
    if not candidates:
        return row
    best = candidates[0]
    row["latest_version"] = best["version"]
    trusted = best["trusted"]
    if trusted is None:
        row["status"] = "untrusted"
        row["message"] = f"{PLATFORM_LABELS[plat]} {best['version']} 缺少可信的 SHA-256 摘要"
        return row
    url, digest, size = trusted
    row.update(
        {
            "asset_name": _asset_name(plat, best["version"], "amd64"),
            "asset_url": url,
            "asset_digest": digest,
            "asset_size": size,
        }
    )
    if not _is_newer(best["version"]):
        row["status"] = "current"
        row["message"] = f"{PLATFORM_LABELS[plat]} 已是最新 {best['version']}"
        return row
    row["status"] = "available"
    row["message"] = f"{PLATFORM_LABELS[plat]} 最新 {best['version']}"
    row["can_install"] = plat == current_platform() and getattr(sys, "frozen", False)
    if plat != current_platform():
        row["message"] = f"{PLATFORM_LABELS[plat]} 最新 {best['version']}（仅该平台可安装）"
    return row


def _rows(plat: str, arch: str) -> tuple[dict, str, str]:
    """返回 (rows, digest_source, 备注)。API 优先，受限时回退 atom + 旁证，再退到本地缓存。"""

    api_error = ""
    try:
        releases = _fetch_releases()
    except RuntimeError as exc:
        api_error = str(exc)
    else:
        rows = {key: _platform_row(key, _api_candidates(releases, key, arch if key == LINUX else "amd64")) for key in PLATFORM_ORDER}
        return rows, "api", ""

    try:
        versions = _atom_versions()
        rows = {
            key: _platform_row(key, _atom_candidates(versions, key, arch if key == LINUX else "amd64"))
            for key in PLATFORM_ORDER
        }
    except RuntimeError as exc:
        cached = _load_cache()
        if cached is None:
            raise RuntimeError(f"{api_error}；{exc}") from exc
        rows, saved_at = cached
        age_hours = max(0.0, (time.time() - saved_at) / 3600)
        return rows, "cache", f"{api_error}；{exc}；显示 {age_hours:.1f} 小时前缓存的结果"

    note = f"{api_error}；已改用发布列表与 .sha256 校验文件，版本信息可能略滞后"
    return rows, "sha256-sidecar", note


def _platforms_cache_path() -> Path:
    return STATE_DIR / "update-cache.json"


def _load_cache() -> tuple[dict, float] | None:
    """读取最近一次成功结果；超过缓存有效期就丢弃。"""

    path = _platforms_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at = float(payload["saved_at"])
        rows = payload["platforms"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(rows, dict) or time.time() - saved_at > CACHE_TTL_SECONDS:
        return None
    return rows, saved_at


def _save_cache(rows: dict) -> None:
    path = _platforms_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"saved_at": time.time(), "platforms": rows}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


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

    arch = _linux_arch()
    rows, digest_source, note = _rows(plat, arch)
    if digest_source != "cache":
        _save_cache(rows)
    if not rows[WINDOWS]["latest_version"] and not rows[LINUX]["latest_version"]:
        info = {
            "status": "unpublished",
            "current_version": __version__,
            "platform": plat,
            "latest_version": "",
            "message": "GitHub 尚无发布版",
            "platforms": rows,
            "digest_source": digest_source,
        }
        if note:
            info["message"] = f"{info['message']}（{note}）"
        return info

    mine = rows[plat]
    info = {
        "status": mine["status"],
        "current_version": __version__,
        "platform": plat,
        "latest_version": mine["latest_version"],
        "message": mine["message"],
        "platforms": rows,
        "digest_source": digest_source,
    }
    if note:
        info["note"] = note
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
            request = Request(str(info["asset_url"]), headers=_headers("application/octet-stream"))
            try:
                with urlopen(request, timeout=60) as response:
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