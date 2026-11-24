"""Verified release-asset downloads with progress and bounded retries."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from http.client import HTTPException
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_SOCKET_TIMEOUT = 15
DOWNLOAD_STALL_SECONDS = 30
DOWNLOAD_MIN_PROGRESS_BYTES = 64 * 1024
DOWNLOAD_MAX_SECONDS = 20 * 60


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ignore_progress(_status: dict) -> None:
    pass


def download_package(
    target: Path, expected_size: int, expected_digest: str,
    make_request: Callable[[int], Request], report: Callable[[dict], None] = ignore_progress,
) -> Path:
    updates = target.parent
    updates.mkdir(parents=True, exist_ok=True)
    if not 0 < expected_size <= MAX_PACKAGE_BYTES:
        raise RuntimeError("发布信息中的安装包大小无效")
    if target.is_file() and target.stat().st_size == expected_size:
        report({"phase": "verifying"})
        if _sha256_file(target) == expected_digest:
            return target

    temporary: Path | None = None
    deadline = time.monotonic() + DOWNLOAD_MAX_SECONDS
    try:
        with tempfile.NamedTemporaryFile("wb", dir=updates, prefix=f".{target.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            last_error: Exception | None = None
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("下载安装包超过总时限")
                downloaded = handle.tell()
                status = {"phase": "downloading", "downloaded": downloaded, "total": expected_size,
                          "attempt": attempt, "max_attempts": DOWNLOAD_ATTEMPTS}
                report(status)
                request = make_request(downloaded)
                try:
                    response = urlopen(request, timeout=min(DOWNLOAD_SOCKET_TIMEOUT, remaining))
                except HTTPError as error:
                    error.close()
                    if error.code not in {408, 429} and not 500 <= error.code < 600:
                        raise RuntimeError(f"下载安装包失败，HTTP {error.code}") from error
                    last_error = error
                    continue
                except (URLError, OSError, HTTPException) as error:
                    last_error = error
                    continue
                with response:
                    if downloaded:
                        if response.status == 200:
                            handle.seek(0)
                            handle.truncate()
                            downloaded = 0
                            report({**status, "downloaded": 0})
                        elif (response.status != 206
                              or response.headers.get("Content-Range") != f"bytes {downloaded}-{expected_size - 1}/{expected_size}"):
                            raise RuntimeError("安装包续传范围与发布信息不符")
                    progress_at = time.monotonic()
                    progress_bytes = downloaded
                    while True:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("下载安装包超过总时限")
                        try:
                            chunk = response.read1(256 * 1024)
                        except (OSError, HTTPException) as error:
                            last_error = error
                            break
                        if not chunk:
                            if downloaded != expected_size:
                                last_error = ConnectionError("连接提前结束，安装包未下载完整")
                            break
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise RuntimeError("下载的安装包大小与发布信息不符")
                        handle.write(chunk)
                        report({**status, "downloaded": downloaded})
                        now = time.monotonic()
                        if downloaded - progress_bytes >= DOWNLOAD_MIN_PROGRESS_BYTES:
                            progress_at, progress_bytes = now, downloaded
                        elif now - progress_at >= DOWNLOAD_STALL_SECONDS:
                            last_error = TimeoutError("下载长时间没有足够进展")
                            break
                if downloaded == expected_size:
                    break
            else:
                raise RuntimeError(f"下载安装包失败，已尝试 {DOWNLOAD_ATTEMPTS} 次：{last_error}") from last_error
            report({"phase": "verifying"})
            handle.flush()
            os.fsync(handle.fileno())
        if _sha256_file(temporary) != expected_digest:
            raise RuntimeError("安装包 SHA-256 校验失败")
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
