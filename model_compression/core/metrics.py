"""模型版本对比的通用指标口径。"""

from __future__ import annotations

from typing import Any, Mapping


def compare_to_baseline(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """计算可定义的相对指标；缺少数据时返回 ``None`` 而不是 0。"""

    baseline_size = baseline.get("size_bytes")
    current_size = current.get("size_bytes")
    baseline_accuracy = baseline.get("accuracy")
    current_accuracy = current.get("accuracy")
    baseline_latency = baseline.get("latency_ms")
    current_latency = current.get("latency_ms")
    return {
        "compression_ratio": baseline_size / current_size if baseline_size and current_size else None,
        "accuracy_delta": current_accuracy - baseline_accuracy if baseline_accuracy is not None and current_accuracy is not None else None,
        "speedup": baseline_latency / current_latency if baseline_latency and current_latency else None,
    }


__all__ = ["compare_to_baseline"]
