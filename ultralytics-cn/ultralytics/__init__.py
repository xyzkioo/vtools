# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.4.128"

import importlib
import os
import sys
from typing import TYPE_CHECKING

# 设置环境变量（必须放在导入其他模块之前）
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # 训练期间默认减少 CPU 占用

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR", "LLM")
PLATFORM_EXPORTS = ("Platform", "AsyncPlatform", "APIError", "APIConnectionError")

__all__ = (  # noqa: PLE0604
    "__version__",
    "ASSETS",
    *MODELS,
    *(PLATFORM_EXPORTS if sys.version_info >= (3, 11) else ()),
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # 为类型检查器启用类型提示
    from ultralytics.models import LLM, YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa
    from ultralytics_platform import APIConnectionError, APIError, AsyncPlatform, Platform  # noqa: F401


def __getattr__(name: str):
    """首次访问公共类时再延迟导入。"""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    if name in PLATFORM_EXPORTS:
        if sys.version_info < (3, 11):
            raise ImportError("Ultralytics Platform requires Python 3.11 or newer.")
        return getattr(importlib.import_module("ultralytics_platform"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """扩展 dir() 结果，使其包含可延迟加载的公共名称，便于 IDE 自动补全。"""
    return sorted(set(globals()) | set(MODELS) | set(PLATFORM_EXPORTS))


if __name__ == "__main__":
    print(__version__)
