"""模型产物与运行环境元数据。"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any


def artifact_info(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"模型产物不存在：{target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(target), "size_bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def environment_info() -> dict[str, Any]:
    """记录可复现实验所需的轻量环境信息，不强制导入 PyTorch。"""

    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


__all__ = ["artifact_info", "environment_info"]
