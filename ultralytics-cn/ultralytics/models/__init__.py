# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .fastsam import FastSAM
from .llm import LLM
from .nas import NAS
from .rtdetr import RTDETR
from .yolo import YOLO, YOLOE, YOLOWorld

__all__ = "LLM", "NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"  # allow simpler import


def __getattr__(name):
    """延迟导入 SAM，避免标准 YOLO 导入加载可选的 torchvision 内部模块。"""
    if name == "SAM":
        # 为提高 ultralytics 导入速度，SAM 会加载依赖 torchvision 的可选模块。
        from .sam import SAM

        return SAM
    raise AttributeError(f"module {__name__} has no attribute {name}")
