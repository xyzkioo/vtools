"""Choose a writable state directory for the installed desktop process."""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


def _check_writable(path: Path) -> None:
    logs = path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Checking ACL bits alone does not account for a restricted process token.
    with tempfile.TemporaryFile(dir=logs):
        pass


@lru_cache(maxsize=1)
def desktop_state_dir() -> Path:
    explicit = os.environ.get("VTOOLS_USER_DATA")
    if explicit:
        path = Path(explicit).expanduser()
        _check_writable(path)
        return path
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        path = Path(sys.executable).resolve().parent / "user-data"
        try:
            _check_writable(path)
        except OSError as exc:
            raise OSError(f"安装目录不可写：{path}。请将程序安装到当前用户可写的位置，例如 D:\\Vtools；或设置 VTOOLS_USER_DATA。") from exc
        return path
    primary = Path.home() / ".vtools_ui"
    try:
        _check_writable(primary)
        return primary
    except OSError as original:
        if sys.platform != "win32" or not os.environ.get("LOCALAPPDATA"):
            raise
        fallback = Path(os.environ["LOCALAPPDATA"]) / "vtools"
        try:
            _check_writable(fallback)
        except OSError as exc:
            raise OSError(f"用户数据目录均不可写：{primary}；{fallback}") from exc
        # Record the switch; existing settings remain in the original directory.
        with (fallback / "logs" / "desktop.log").open("a", encoding="utf-8") as log:
            log.write(f"原用户目录不可写，改用 {fallback}。原错误：{original}\n")
        return fallback
