"""可独立运行的压缩与蒸馏模块。"""

from .distillation import distillation_loss, train_distillation, validate_distillation_config
from .dependency_pruning import prune_detector_dependency_aware
from .pruning import prune_unstructured
from .quantization import quantize_dynamic_int8
from .structured import build_structured_detector, finetune_structured_detector

__all__ = [
    "distillation_loss",
    "prune_detector_dependency_aware",
    "prune_unstructured",
    "quantize_dynamic_int8",
    "build_structured_detector",
    "finetune_structured_detector",
    "train_distillation",
    "validate_distillation_config",
]
