"""可独立运行的压缩与蒸馏模块。"""

from .distillation import distillation_loss, train_distillation, validate_distillation_config
from .pruning import prune_unstructured
from .quantization import quantize_dynamic_int8

__all__ = [
    "distillation_loss",
    "prune_unstructured",
    "quantize_dynamic_int8",
    "train_distillation",
    "validate_distillation_config",
]
