"""非结构化幅值剪枝。"""

from __future__ import annotations

import copy
from typing import Any, Mapping


def prune_unstructured(model: Any, config: Mapping[str, Any] | None = None):
    """按全局 L1 幅值剪枝，并移除重参数化以便导出。"""

    try:
        import torch  # type: ignore
        from torch.nn.utils import prune  # type: ignore
    except ImportError as exc:
        raise RuntimeError("剪枝需要 PyTorch") from exc
    values = dict(config or {})
    amount = float(values.get("sparsity", values.get("amount", 0.3)))
    if not 0 <= amount < 1:
        raise ValueError("pruning sparsity/amount 必须在 [0, 1) 范围内")
    exclude = {str(item) for item in values.get("exclude", []) or []}
    source = copy.deepcopy(model)
    parameters = []
    for module_name, module in source.named_modules():
        if any(module_name == item or module_name.startswith(f"{item}.") for item in exclude):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            parameters.append((module, "weight"))
    if not parameters:
        raise ValueError("模型没有可剪枝的 Linear/Conv 权重")
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    for module, name in parameters:
        prune.remove(module, name)
    total = zero = 0
    for module, _ in parameters:
        tensor = module.weight.detach()
        total += int(tensor.numel())
        zero += int((tensor == 0).sum().item())
    return source, {
        "method": "unstructured_l1",
        "requested_sparsity": amount,
        "actual_sparsity": zero / total if total else 0.0,
        "pruned_parameter_count": total,
        "zero_parameter_count": zero,
    }


__all__ = ["prune_unstructured"]
