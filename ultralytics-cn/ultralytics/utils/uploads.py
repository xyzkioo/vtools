# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics 上传工具，沿用 downloads.py 的实现模式。"""

from __future__ import annotations

import os
from pathlib import Path
from time import sleep

from ultralytics.utils import LOGGER, TQDM


class _ProgressReader:
    """报告读取进度、用于监控上传过程的文件包装器。"""

    def __init__(self, file_path, pbar):
        self.file = open(file_path, "rb")  # noqa: SIM115  # 流式上传读取期间保持打开
        self.pbar = pbar
        self._size = os.path.getsize(file_path)

    def read(self, size=-1):
        """读取数据并更新进度条。"""
        data = self.file.read(size)
        if data and self.pbar:
            self.pbar.update(len(data))
        return data

    def __len__(self):
        """返回文件大小，用于设置 Content-Length 标头。"""
        return self._size

    def close(self):
        """关闭文件。"""
        self.file.close()


def safe_upload(
    file: str | Path,
    url: str,
    headers: dict | None = None,
    retry: int = 2,
    timeout: int = 600,
    progress: bool = False,
) -> bool:
    """将文件上传到 URL，支持失败重试和可选进度条。

    参数：
        file (str | Path): 要上传的文件路径。
        url (str): 文件上传地址（例如签名后的 GCS URL）。
        headers (dict, 可选): 要添加到请求中的其他标头。
        retry (int, 可选): 失败时的重试次数（默认 2 次，即总共尝试 3 次）。
        timeout (int, 可选): 请求超时时间，单位为秒。
        progress (bool, 可选): 上传期间是否显示进度条。

    返回：
        (bool): 上传成功返回 True，否则返回 False。

    示例：
        >>> from ultralytics.utils.uploads import safe_upload
        >>> success = safe_upload("model.pt", "https://storage.googleapis.com/...", progress=True)
    """
    import requests

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    file_size = file.stat().st_size
    desc = f"Uploading {file.name}"

    # 准备请求标头（Content-Length 会根据文件大小自动设置）
    upload_headers = {"Content-Type": "application/octet-stream"}
    if headers:
        upload_headers.update(headers)

    last_error = None
    for attempt in range(retry + 1):
        pbar = None
        reader = None
        try:
            if progress:
                pbar = TQDM(total=file_size, desc=desc, unit="B", unit_scale=True, unit_divisor=1024)
            reader = _ProgressReader(file, pbar)

            r = requests.put(url, data=reader, headers=upload_headers, timeout=timeout)
            r.raise_for_status()
            reader.close()
            reader = None  # Prevent double-close in finally
            if pbar:
                pbar.close()
                pbar = None
            LOGGER.info(f"Uploaded {file.name} ✅")
            return True

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500 and status not in {408, 429}:
                LOGGER.warning(f"{desc} failed: {status} {getattr(e.response, 'reason', '')}")
                return False
            last_error = f"HTTP {status}"
        except Exception as e:
            last_error = str(e)
        finally:
            if reader:
                reader.close()
            if pbar:
                pbar.close()

        if attempt < retry:
            wait_time = 2 ** (attempt + 1)
            LOGGER.warning(f"{desc} failed ({last_error}), retrying {attempt + 1}/{retry} in {wait_time}s...")
            sleep(wait_time)

    LOGGER.warning(f"{desc} failed after {retry + 1} attempts: {last_error}")
    return False
