"""训练后动态 INT8 量化。"""

from __future__ import annotations

import copy
from typing import Any, Mapping


def quantize_dynamic_int8(model: Any, config: Mapping[str, Any] | None = None):
    """复制模型后对 Linear/LSTM/GRU 做动态 INT8 量化。

    动态量化通常对 CPU 推理更有帮助；模块会把实际处理的层数量写入元数据，
    不把“调用了量化 API”误报成性能提升。
    """

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("动态量化需要 PyTorch") from exc
    values = dict(config or {})
    supported = {torch.nn.Linear}
    for name in ("LSTM", "GRU", "LSTMCell", "GRUCell"):
        item = getattr(torch.nn, name, None)
        if item is not None:
            supported.add(item)
    source = copy.deepcopy(model).cpu().eval()
    before = sum(1 for module in source.modules() if type(module) in supported)
    result = torch.ao.quantization.quantize_dynamic(
        source,
        qconfig_spec=supported,
        dtype=getattr(torch, str(values.get("dtype", "qint8")), torch.qint8),
        inplace=False,
    )
    dynamic_linear = getattr(getattr(torch.nn, "quantized", None), "dynamic", None)
    quantized_types = tuple(
        item for item in (
            getattr(dynamic_linear, "Linear", None) if dynamic_linear else None,
            getattr(dynamic_linear, "LSTM", None) if dynamic_linear else None,
            getattr(dynamic_linear, "GRU", None) if dynamic_linear else None,
        ) if isinstance(item, type)
    )
    after = sum(1 for module in result.modules() if quantized_types and isinstance(module, quantized_types))
    return result, {"method": "dynamic_int8", "candidate_layer_count": before, "quantized_layer_count": after, "device": "cpu"}


__all__ = ["quantize_dynamic_int8"]
