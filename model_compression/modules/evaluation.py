"""参数统计和分类评测模块的薄封装。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.model_runtime import evaluate_classification, parameter_metrics


def evaluate(model: Any, loader: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    values = config.get("benchmark") if isinstance(config.get("benchmark"), Mapping) else config
    return evaluate_classification(model, loader, device=values.get("device", "auto"))


def model_metrics(model: Any) -> dict[str, int]:
    return parameter_metrics(model)


__all__ = ["evaluate", "model_metrics"]
