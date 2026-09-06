"""Compatibility types used by vtools-style model adapters.

The names intentionally match ``vtools/speed_test/core/vision_benchmark_common``
so an adapter can be copied between the two projects without changing its
function signatures.  The diagnostics tool only needs the input bundle and the
basic inference configuration; runtime benchmarking remains in vtools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputBundle:
    """Represents ``model(*args, **kwargs)`` inputs."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    description: str = "custom"

    @classmethod
    def from_value(cls, value: Any, batch_size: int = 1, description: str = "custom") -> "InputBundle":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "args" in value:
            raw_args = value.get("args", ())
            if isinstance(raw_args, tuple):
                args = raw_args
            elif isinstance(raw_args, list):
                # A list in the dictionary form is a single model argument in
                # vtools (useful for torchvision detection models).
                args = (raw_args,)
            else:
                args = (raw_args,)
            return cls(
                tuple(args),
                dict(value.get("kwargs", {})),
                int(value.get("batch_size", batch_size)),
                str(value.get("description", description)),
            )
        if isinstance(value, tuple):
            return cls(value, {}, batch_size, description)
        return cls((value,), {}, batch_size, description)


@dataclass
class BenchmarkConfig:
    """Inference settings with the same attributes used by vtools adapters."""

    device: Any
    batch_size: int
    height: int
    width: int
    precision: str = "fp32"
    warmup: int = 30
    repeats: int = 100
    fuse: bool = False
    conf: float = 0.001
    iou: float = 0.70
    max_det: int = 3000

    @property
    def image_size(self) -> tuple[int, int]:
        return self.height, self.width


def dtype_for_precision(precision: str, device: Any) -> Any:
    """Return the torch dtype expected by the adapter template."""
    import torch

    aliases = {
        "32": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    key = str(precision).lower().strip()
    if key not in aliases:
        raise ValueError(f"不支持的精度：{precision}，可选 fp32、fp16、bf16")
    precision = aliases[key]
    if precision == "fp16":
        if getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
            raise ValueError("FP16 诊断推理需要 CUDA；CPU 请使用 fp32")
        return torch.float16
    if precision == "bf16":
        if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 CUDA 设备不支持 BF16")
        return torch.bfloat16
    return torch.float32


def resolve_device(value: Any) -> Any:
    """Resolve the same ``auto``/``cuda:0``/``0`` spellings as vtools."""
    import torch

    text = str(value or "auto").strip().lower()
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前 PyTorch 没有可用的 CUDA 设备")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA 设备 {index} 不存在；当前设备数为 {torch.cuda.device_count()}")
    return device


def normalize_precision(value: Any) -> str:
    """Normalize precision aliases accepted by the vtools config."""
    text = str(value or "fp32").strip().lower()
    aliases = {
        "32": "fp32", "float32": "fp32", "fp32": "fp32",
        "16": "fp16", "float16": "fp16", "half": "fp16", "fp16": "fp16",
        "bf16": "bf16", "bfloat16": "bf16",
    }
    if text not in aliases:
        raise ValueError(f"不支持的精度：{value}，可选 fp32、fp16、bf16")
    return aliases[text]
