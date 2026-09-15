"""独立的检测诊断模块。

模块只接收已经规范化的数据和共享匹配结果，不负责解析命令行或加载模型。
这样同一份预测可以被多个诊断功能复用，也方便 PyCharm 和终端入口使用同一套实现。
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODULE_DEFAULTS: dict[str, bool] = {
    "checkpoint.inspect": False,
    "checkpoint.load_check": False,
    "model.parameters": False,
    "model.flops": False,
    "speed.pytorch_call": True,
    "speed.pytorch_pipeline": False,
    "speed.tensorrt_call": True,
    "memory.pytorch_peak": False,
    "export.onnx": False,
    "build.tensorrt": False,
    "consistency.tensor": True,
    "consistency.detection": False,
    "predict.generate": False,
    "validation.ultralytics": False,
    "diagnostics.missed": True,
    "diagnostics.classification": True,
    "diagnostics.background": True,
    "diagnostics.duplicate": True,
    "diagnostics.localization": True,
    "diagnostics.overlap": True,
    "diagnostics.groups": True,
    "diagnostics.threshold_sweep": False,
    "diagnostics.bootstrap": False,
    "diagnostics.stage_comparison": False,
    "output.bad_cases": True,
    "output.images": False,
    "output.html": False,
    "visualization.features": False,
    "visualization.cam": False,
    "visualization.stage_trace": False,
}


def available_modules() -> tuple[str, ...]:
    return tuple(MODULE_DEFAULTS)


def run(module_id: str, context: Any, options: Mapping[str, Any] | None = None) -> Any:
    """按模块 ID 调用一个独立诊断模块。

    ``context`` 至少需要包含 ``dataset``（重叠分析）或 ``summary``（差图清单）。
    入口负责共享资源和输出，模块本身不读取命令行。
    """
    values = options or {}
    if module_id == "diagnostics.overlap":
        return overlap_analysis(context["dataset"] if isinstance(context, Mapping) else context, values)
    if module_id == "output.bad_cases":
        summary = context["summary"] if isinstance(context, Mapping) else context
        return bad_case_rows(summary)
    raise ValueError(f"模块 {module_id} 没有独立 runner")


def _split_values(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def resolve_modules(
    config: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    enable: Iterable[str] | None = None,
    disable: Iterable[str] | None = None,
) -> dict[str, bool]:
    """合并 YAML 与命令行模块选择，并校验冲突和未知 ID。"""
    values = dict(MODULE_DEFAULTS)
    configured = config.get("modules")
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise ValueError("modules 必须是 module_id: true/false 的对象")
        unknown = sorted(set(str(key) for key in configured) - set(values))
        if unknown:
            raise ValueError(f"未知模块：{', '.join(unknown)}；请使用 --list-modules 查看可用模块")
        # Once a new-style modules section is present, omitted IDs are off.
        # This keeps a partial config from silently enabling extra work.
        values = {key: False for key in values}
        values.update({str(key): bool(value) for key, value in configured.items()})
    only_values = _split_values(only)
    enable_values = _split_values(enable)
    disable_values = _split_values(disable)
    if only_values and (enable_values or disable_values):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    requested = set(only_values) | set(enable_values) | set(disable_values)
    unknown = sorted(requested - set(values))
    if unknown:
        raise ValueError(f"未知模块：{', '.join(unknown)}；请使用 --list-modules 查看可用模块")
    if only_values:
        values = {key: key in set(only_values) for key in values}
    else:
        values.update({key: True for key in enable_values})
        values.update({key: False for key in disable_values})
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
    if values.get("output.images") and not any(values.get(key) for key in values if key.startswith("diagnostics.")):
        raise ValueError("output.images 需要至少启用一个 diagnostics.* 模块")
    if values.get("output.images") and not values.get("output.bad_cases"):
        raise ValueError("output.images 依赖 output.bad_cases")
    if values.get("output.html") and not values.get("output.bad_cases"):
        raise ValueError("output.html 依赖 output.bad_cases")
    return values


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _pair_metrics(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    try:
        from .engine import box_area, box_iou
    except ImportError:  # direct ``python diagnostics/engine.py`` compatibility
        from engine import box_area, box_iou

    inter = _intersection(a, b)
    smaller = min(box_area(tuple(a)), box_area(tuple(b)))
    return box_iou(tuple(a), tuple(b)), (inter / smaller if smaller > 0 else 0.0)


def overlap_analysis(dataset: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """分析 GT 框重叠和预测疑似重复框，返回可直接写入 CSV/JSON 的结果。"""
    options = config.get("overlap", {}) if isinstance(config.get("overlap"), Mapping) else {}
    iou_threshold = float(options.get("iou_threshold", config.get("overlap_iou_threshold", 0.30)))
    smaller_threshold = float(options.get("smaller_area_threshold", config.get("overlap_smaller_area_threshold", 0.70)))
    criterion = str(options.get("criterion", "either")).lower()
    class_scope = str(options.get("class_scope", "all")).lower()
    if criterion not in {"either", "both", "iou", "smaller_area"}:
        raise ValueError("diagnostics.overlap.criterion 必须是 either、both、iou 或 smaller_area")
    if class_scope not in {"all", "same", "same_class"}:
        raise ValueError("diagnostics.overlap.class_scope 必须是 all 或 same")
    if not 0 <= iou_threshold <= 1 or not 0 <= smaller_threshold <= 1:
        raise ValueError("重叠阈值必须位于 [0,1]")

    def selected(iou: float, ios: float) -> bool:
        if criterion == "iou":
            return iou >= iou_threshold
        if criterion == "smaller_area":
            return ios >= smaller_threshold
        if criterion == "both":
            return iou >= iou_threshold and ios >= smaller_threshold
        return iou >= iou_threshold or ios >= smaller_threshold

    gt_pairs: list[dict[str, Any]] = []
    pred_pairs: list[dict[str, Any]] = []
    gt_ids_by_image: dict[str, set[str]] = defaultdict(set)
    missed_gt_ids_by_image: dict[str, set[str]] = defaultdict(set)
    summary_rows = dataset_summary_rows(dataset)
    prediction_rows = list(getattr(dataset, "diagnostic_per_prediction", None) or [])
    if not prediction_rows:
        prediction_rows = [row for row in summary_rows if row.get("prediction_id") is not None]
    pred_gt_map: dict[str, str] = {
        str(row.get("prediction_id")): str(row.get("matched_gt_id", ""))
        for row in prediction_rows
    }
    gt_summary_map: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(row.get("image_id", "")), str(row.get("gt_id", ""))): row
        for row in summary_rows if row.get("gt_id") is not None
    }
    image_rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"image_id": "", "gt_pair_count": 0, "gt_overlapping_target_count": 0, "gt_overlapping_missed_count": 0, "prediction_pair_count": 0, "prediction_duplicate_pair_count": 0})
    for image_id in sorted(set(dataset.images) | set(dataset.gt) | set(dataset.predictions)):
        image_rows[image_id]["image_id"] = image_id
        gts = [
            item for item in dataset.gt.get(image_id, [])
            if not getattr(item, "ignore", False) and _box_area_for_overlap(item.bbox) > 0
        ]
        preds = [
            item for item in dataset.predictions.get(image_id, [])
            if float(getattr(item, "score", 0.0) or 0.0) >= float(config.get("score_threshold", 0.25))
            and _box_area_for_overlap(item.bbox) > 0
        ]
        for index, left in enumerate(gts):
            for right in gts[index + 1 :]:
                if class_scope in {"same", "same_class"} and left.class_id != right.class_id:
                    continue
                iou, ios = _pair_metrics(left.bbox, right.bbox)
                if not selected(iou, ios):
                    continue
                gt_rows = [gt_summary_map.get((image_id, left.instance_id)), gt_summary_map.get((image_id, right.instance_id))]
                gt_rows = [row for row in gt_rows if row is not None]
                missed = sum(not bool(row.get("matched")) for row in gt_rows)
                gt_pairs.append({"image_id": image_id, "left_gt_id": left.instance_id, "right_gt_id": right.instance_id, "left_class_id": left.class_id, "right_class_id": right.class_id, "iou": iou, "intersection_over_smaller": ios, "missed_target_count": missed})
                image_rows[image_id]["gt_pair_count"] += 1
                gt_ids_by_image[image_id].update((left.instance_id, right.instance_id))
                for row in gt_rows:
                    if not bool(row.get("matched")):
                        missed_gt_ids_by_image[image_id].add(str(row.get("gt_id")))
        for index, left in enumerate(preds):
            for right in preds[index + 1 :]:
                if class_scope in {"same", "same_class"} and left.class_id != right.class_id:
                    continue
                iou, ios = _pair_metrics(left.bbox, right.bbox)
                if not selected(iou, ios):
                    continue
                left_gt = pred_gt_map.get(left.instance_id, "")
                right_gt = pred_gt_map.get(right.instance_id, "")
                duplicate = left.class_id == right.class_id and (
                    bool(left_gt and right_gt and left_gt == right_gt)
                    or bool(not left_gt and not right_gt)
                )
                pred_pairs.append({"image_id": image_id, "left_prediction_id": left.instance_id, "right_prediction_id": right.instance_id, "left_class_id": left.class_id, "right_class_id": right.class_id, "left_score": float(left.score or 0.0), "right_score": float(right.score or 0.0), "iou": iou, "intersection_over_smaller": ios, "suspected_duplicate": duplicate})
                image_rows[image_id]["prediction_pair_count"] += 1
                image_rows[image_id]["prediction_duplicate_pair_count"] += int(duplicate)
        image_rows[image_id]["gt_overlapping_target_count"] = len(gt_ids_by_image[image_id])
        image_rows[image_id]["gt_overlapping_missed_count"] = len(missed_gt_ids_by_image[image_id])
    gt_target_count = sum(row["gt_overlapping_target_count"] for row in image_rows.values())
    gt_missed_count = sum(row["gt_overlapping_missed_count"] for row in image_rows.values())
    return {
        "config": {"iou_threshold": iou_threshold, "smaller_area_threshold": smaller_threshold, "criterion": criterion, "class_scope": class_scope},
        "gt_pairs": gt_pairs,
        "prediction_pairs": pred_pairs,
        "per_image": list(image_rows.values()),
        "summary": {"gt_pair_count": len(gt_pairs), "gt_overlapping_target_count": gt_target_count, "gt_overlapping_missed_count": gt_missed_count, "gt_overlapping_recall": (gt_target_count - gt_missed_count) / gt_target_count if gt_target_count else None, "prediction_pair_count": len(pred_pairs), "prediction_duplicate_pair_count": sum(bool(row["suspected_duplicate"]) for row in pred_pairs)},
    }


def dataset_summary_rows(dataset: Any) -> list[Mapping[str, Any]]:
    """读取共享匹配结果；由 runner 写入 Dataset.extra，缺失时返回空列表。"""
    rows = getattr(dataset, "diagnostic_per_gt", None)
    return list(rows or [])


def _box_area_for_overlap(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bad_case_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_image: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in summary.get("per_prediction", []) or []:
        image_id = str(item.get("image_id", ""))
        error_type = str(item.get("error_type", ""))
        if error_type and error_type != "TP":
            by_image[image_id][error_type] += 1
    for row in summary.get("per_image", []) or []:
        image_id = str(row.get("image_id", ""))
        errors = {"missed": int(row.get("fn", 0) or 0)}
        typed = by_image.get(image_id, {})
        counts = {"background": typed.get("background", 0), "duplicate": typed.get("duplicate", 0), "classification": typed.get("classification", 0), "localization": typed.get("localization", 0) + typed.get("classification_localization", 0)}
        total = errors["missed"] + sum(counts.values())
        rows.append({"image_id": image_id, "error_count": total, "gt_count": int(row.get("gt_count", 0) or 0), "missed_count": errors["missed"], "background_count": counts["background"], "duplicate_count": counts["duplicate"], "classification_count": counts["classification"], "localization_count": counts["localization"], "error_rate": (total / int(row.get("gt_count", 0)) if int(row.get("gt_count", 0) or 0) else None), "count_absolute_error": row.get("count_absolute_error")})
    return sorted((row for row in rows if int(row["error_count"]) > 0), key=lambda row: (-int(row["error_count"]), str(row["image_id"])))


def render_bad_cases(dataset: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path, top_k: int = 50, score_threshold: float = 0.25, base_dir: Path | None = None) -> list[Path]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError as exc:
        raise RuntimeError("output.images 需要 Pillow，请安装 pillow") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in list(rows)[: max(0, int(top_k))]:
        image_id = str(row.get("image_id", ""))
        info = dataset.images.get(image_id)
        extra = getattr(info, "extra", {}) if info is not None else {}
        source = extra.get("source_path") if isinstance(extra, Mapping) else None
        source = source or (getattr(info, "file_name", None) if info is not None else None)
        source_path = Path(str(source)) if source else Path()
        if source and not source_path.is_absolute() and base_dir is not None:
            source_path = (base_dir / source_path).resolve()
        if not source or not source_path.is_file():
            continue
        with Image.open(str(source_path)).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for gt in dataset.gt.get(image_id, []):
                draw.rectangle(tuple(gt.bbox), outline=(0, 200, 0), width=3)
            for pred in dataset.predictions.get(image_id, []):
                if float(getattr(pred, "score", 0.0) or 0.0) < score_threshold:
                    continue
                draw.rectangle(tuple(pred.bbox), outline=(220, 40, 40), width=2)
            draw.text((8, 8), f"{image_id}  errors={row.get('error_count', 0)}", fill=(255, 160, 0))
            target = output_dir / f"{_safe_image_name(image_id)}.jpg"
            image.save(target, quality=92)
            paths.append(target)
    return paths


def _safe_image_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value).strip("._") or "image"


def write_bad_case_html(path: Path, rows: Sequence[Mapping[str, Any]], image_paths: Sequence[Path]) -> None:
    by_id = {item.stem: item for item in image_paths}
    lines = ["<!doctype html><meta charset='utf-8'><title>差图索引</title><h1>差图索引</h1><table border='1'><tr><th>图片</th><th>错误数</th><th>漏检</th><th>多余框</th><th>预览</th></tr>"]
    for row in rows:
        image_id = str(row.get("image_id", ""))
        preview = by_id.get(_safe_image_name(image_id))
        preview_link = f"images/{preview.name}" if preview else ""
        preview_html = f"<a href='{html.escape(preview_link)}'><img src='{html.escape(preview_link)}' width='320'></a>" if preview else ""
        lines.append(f"<tr><td>{html.escape(image_id)}</td><td>{row.get('error_count', 0)}</td><td>{row.get('missed_count', 0)}</td><td>{row.get('background_count', 0)}</td><td>{preview_html}</td></tr>")
    lines.append("</table>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
