#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时动态库兼容处理。

PyCharm 使用 conda 解释器时，可能不会继承 shell 中的
``LD_LIBRARY_PATH``。某些 ONNX 二进制扩展因此会加载系统的
``libstdc++.so.6``，而不是当前 conda 环境中的版本，最终出现
``CXXABI_1.3.15 not found``。本模块只依赖 Python 标准库，并在入口
真正导入 torch/onnx 前自动重新 exec 当前 Python 进程。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional


_READY_ENV = "VISION_BENCHMARK_CONDA_LIB_READY"


def _candidate_library_dirs() -> list[Path]:
    """按优先级返回可能属于当前 Python 环境的动态库目录。"""

    candidates: list[Path] = []
    # sys.prefix 是当前实际 Python 解释器所在环境，优先于继承来的
    # CONDA_PREFIX；后者用于终端和 PyCharm 的 conda 环境补充识别。
    for value in (sys.prefix, os.environ.get("CONDA_PREFIX")):
        if not value:
            continue
        directory = (Path(value).expanduser() / "lib").resolve()
        if directory not in candidates and (directory / "libstdc++.so.6").is_file():
            candidates.append(directory)
    return candidates


def _with_library_dir(environment: Mapping[str, str], directory: Path) -> dict[str, str]:
    """返回把指定目录放到 LD_LIBRARY_PATH 最前面的环境副本。"""

    updated = dict(environment)
    current = updated.get("LD_LIBRARY_PATH", "")
    parts = [item for item in current.split(os.pathsep) if item]
    directory_text = str(directory)
    parts = [item for item in parts if Path(item).expanduser().resolve() != directory]
    updated["LD_LIBRARY_PATH"] = os.pathsep.join([directory_text, *parts])
    return updated


def _reexec_arguments() -> list[str]:
    """保留 ``-m``/``-c`` 等 Python 启动参数的原始形式。"""

    original = getattr(sys, "orig_argv", None)
    if isinstance(original, list) and len(original) >= 2:
        # sys.orig_argv 的第一个元素是 Python 可执行文件路径。
        return list(original[1:])
    return [sys.argv[0], *sys.argv[1:]]


def ensure_conda_library_path() -> Optional[Path]:
    """确保 conda 的 ``libstdc++`` 优先级高于系统库。

    返回使用的目录。若当前进程尚未带上该目录，则用相同参数重新启动
    当前入口；exec 成功后不会返回。没有找到 conda 动态库时不阻止运行，
    让后续代码给出原始依赖错误。
    """

    if not sys.platform.startswith("linux"):
        return None
    candidates = _candidate_library_dirs()
    if not candidates:
        return None
    directory = candidates[0]
    current_parts = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    already_first = bool(current_parts) and Path(current_parts[0]).expanduser().resolve() == directory
    if already_first:
        return directory

    # 防止 exec 失败或 PyCharm 特殊启动方式导致无限重启。
    if os.environ.get(_READY_ENV) == "1":
        os.environ.update(_with_library_dir(os.environ, directory))
        return directory

    environment = _with_library_dir(os.environ, directory)
    environment[_READY_ENV] = "1"
    print(
        f"[环境] PyCharm 未继承 conda 动态库路径，自动使用：{directory}",
        flush=True,
    )
    try:
        os.execvpe(sys.executable, [sys.executable, *_reexec_arguments()], environment)
    except OSError as error:
        # 不能重启时保留可用的环境变量，后续仍可能通过已加载库工作；
        # 同时不掩盖真正的 ONNX 导入错误。
        os.environ.update(environment)
        print(f"[环境警告] 自动重启失败：{type(error).__name__}: {error}", flush=True)
    return directory


__all__ = ["ensure_conda_library_path"]
