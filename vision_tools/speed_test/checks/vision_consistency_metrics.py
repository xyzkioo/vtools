#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""与 GPU 无关的一致性指标；可用 NumPy 和 unittest 单独测试。"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_tolerances(atol: float, rtol: float) -> None:
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("atol/rtol 必须为有限且非负的数值")


def compare_array(reference: Any, actual: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    """以 PyTorch 输出为 reference，逐元素要求 |actual-ref| <= atol+rtol*|ref|。

    不广播形状，不把两边同时为 NaN/Inf 视为相等；整数/布尔输出严格相等。
    空张量不自动验收通过，返回 inconclusive。
    """
    validate_tolerances(atol, rtol)
    ref, got = np.asarray(reference), np.asarray(actual)
    result: dict[str, Any] = {
        "status": "failed", "reference_shape": list(ref.shape),
        "actual_shape": list(got.shape), "reference_dtype": str(ref.dtype),
        "actual_dtype": str(got.dtype), "elements": int(ref.size),
        "atol": atol, "rtol": rtol,
    }
    if ref.shape != got.shape:
        return {**result, "reason": "shape_mismatch"}
    if ref.dtype.kind not in "biuf" or got.dtype.kind not in "biuf":
        return {**result, "reason": "unsupported_dtype"}
    result["reference_nonfinite"] = int((~np.isfinite(ref)).sum())
    result["actual_nonfinite"] = int((~np.isfinite(got)).sum())
    if result["reference_nonfinite"] or result["actual_nonfinite"]:
        return {**result, "reason": "nan_or_inf"}
    if ref.size == 0:
        return {**result, "status": "inconclusive", "reason": "empty_tensor"}
    exact = ref.dtype.kind in "biu" or got.dtype.kind in "biu"
    if exact and ref.dtype.kind != got.dtype.kind:
        return {**result, "reason": "discrete_dtype_kind_mismatch"}
    if exact:
        # 不转 float64：大于 2**53 的整数转换后可能变成同一个数。
        close = got == ref
        passed = bool(np.all(close))
        return {**result, "status": "passed" if passed else "failed",
                "reason": "exact_match" if passed else "integer_mismatch",
                "close_fraction": float(np.mean(close)), "mismatched_elements": int((~close).sum())}
    with np.errstate(over="ignore", invalid="ignore"):
        r, a = ref.astype(np.float64), got.astype(np.float64)
        difference = np.abs(a - r)
        relative = difference / np.maximum(np.abs(r), 1e-12)
        close = np.isfinite(difference) & (difference <= (atol + rtol * np.abs(r)))
        rmse = float(np.sqrt(np.mean(difference ** 2)))
    worst = int(np.argmax(difference))
    result.update({
        "status": "passed" if bool(np.all(close)) else "failed",
        "reason": "elementwise_tolerance",
        "close_fraction": float(np.mean(close)), "mismatched_elements": int((~close).sum()),
        "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()),
        "rmse": rmse, "max_rel_error": float(relative.max()),
        "worst_index": [int(x) for x in np.unravel_index(worst, ref.shape)],
        "reference_at_worst": float(r.flat[worst]), "actual_at_worst": float(a.flat[worst]),
    })
    return result


def align_outputs(reference: dict[str, Any], actual: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """必须完整按名称对齐；数量/名字不一致就报错，不用 zip 静默丢输出。"""
    if not reference or set(reference) != set(actual):
        raise ValueError(f"output_schema_mismatch: PyTorch={list(reference)}, TensorRT={list(actual)}")
    return [(name, reference[name], actual[name]) for name in reference]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    intersection = np.prod(np.maximum(rb - lt, 0), axis=-1)
    area_a = np.prod(np.maximum(a[:, 2:4] - a[:, :2], 0), axis=-1)
    area_b = np.prod(np.maximum(b[:, 2:4] - b[:, :2], 0), axis=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _matching(edges: np.ndarray, ious: np.ndarray) -> list[tuple[int, int]]:
    """最大基数二分图匹配，允许框重排；不依赖 SciPy，也不复用已匹配的框。"""
    targets: dict[int, int] = {}
    neighbors = [
        sorted(np.flatnonzero(row).tolist(), key=lambda j: (-ious[i, j], j))
        for i, row in enumerate(edges)
    ]

    def augment(i: int, seen: set[int]) -> bool:
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in targets or augment(targets[j], seen):
                targets[j] = i
                return True
        return False

    for index in range(len(neighbors)):
        augment(index, set())
    return sorted((i, j) for j, i in targets.items())


def compare_detections(
    reference: Any, actual: Any, *, conf: float, iou_threshold: float,
    score_atol: float, max_boxes: int = 500,
) -> dict[str, Any]:
    """仅接受 [B,N,6]、xyxy/conf/class、输入图像坐标的检测结果。

    只比较 conf 阈值之上的框；覆盖率是相对 PyTorch 输出，不是数据集 Recall。
    两侧都没有框返回 inconclusive，防止空结果被当成检测准确率通过。
    """
    if not 0 <= conf <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("conf 和 iou_threshold 必须位于 [0,1]")
    validate_tolerances(score_atol, 0)
    ref, got = np.asarray(reference), np.asarray(actual)
    result: dict[str, Any] = {"status": "failed", "conf": conf, "iou_threshold": iou_threshold,
                              "score_atol": score_atol, "images": []}
    if ref.ndim != 3 or got.ndim != 3 or ref.shape[-1] != 6 or got.shape[-1] != 6:
        return {**result, "reason": "expected_B_N_6_xyxy_conf_class"}
    if ref.shape[0] != got.shape[0] or ref.shape[0] == 0:
        return {**result, "reason": "batch_mismatch_or_empty"}
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        return {**result, "reason": "nan_or_inf"}
    for ref_image, got_image in zip(ref, got):
        a = ref_image[ref_image[:, 4] >= conf].astype(np.float64)
        b = got_image[got_image[:, 4] >= conf].astype(np.float64)
        for boxes in (a, b):
            if (np.any(boxes[:, 2:4] <= boxes[:, :2]) or
                    np.any(boxes[:, 4] > 1) or np.any(boxes[:, 5] < 0) or
                    np.any(boxes[:, 5] != np.rint(boxes[:, 5]))):
                return {**result, "reason": "invalid_xyxy_score_or_class"}
        if max(len(a), len(b)) > max_boxes:
            return {**result, "reason": "too_many_boxes_for_matching"}
        ious = _iou_matrix(a, b)
        score_delta = np.abs(a[:, None, 4] - b[None, :, 4])
        edges = (a[:, None, 5] == b[None, :, 5]) & (ious >= iou_threshold) & (score_delta <= score_atol)
        pairs = _matching(edges, ious)
        image_status = ("inconclusive" if len(a) == len(b) == 0 else
                        "passed" if len(pairs) == len(a) == len(b) else "failed")
        result["images"].append({
            "status": image_status, "pytorch_count": len(a), "tensorrt_count": len(b),
            "matched_count": len(pairs), "unmatched_pytorch": len(a) - len(pairs),
            "unmatched_tensorrt": len(b) - len(pairs),
            "min_matched_iou": min((float(ious[i, j]) for i, j in pairs), default=None),
            "max_matched_score_diff": max((float(score_delta[i, j]) for i, j in pairs), default=None),
            "pairs": [[int(i), int(j)] for i, j in pairs],
        })
    states = [item["status"] for item in result["images"]]
    result["status"] = "failed" if "failed" in states else "inconclusive" if "inconclusive" in states else "passed"
    result["reason"] = "empty_detections_are_not_evidence" if "inconclusive" in states else "class_iou_score_matching"
    return result


def combine_status(tensors: list[dict[str, Any]], detections: dict[str, Any] | None = None) -> str:
    if not tensors:
        return "error"
    # 形状、数据类型、NaN/Inf 等硬错误不能被框匹配覆盖。
    hard = {"shape_mismatch", "unsupported_dtype", "nan_or_inf", "discrete_dtype_kind_mismatch", "integer_mismatch"}
    if any(item.get("reason") in hard for item in tensors):
        return "failed"
    if detections is not None and detections["status"] == "failed":
        return "failed"
    if any(item["status"] == "inconclusive" for item in tensors):
        return "inconclusive"
    if detections is not None and detections["status"] == "inconclusive":
        return "inconclusive"
    if all(item["status"] == "passed" for item in tensors):
        return "passed"
    # 可能只是 top-k 顺序变化，但保留警告，不悄悄把张量失败改成通过。
    return "warning" if detections is not None and detections["status"] == "passed" else "failed"


def aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "inconclusive"
    for status in ("error", "failed", "warning", "inconclusive", "skipped"):
        if status in statuses:
            return status
    return "passed" if all(status == "passed" for status in statuses) else "error"
