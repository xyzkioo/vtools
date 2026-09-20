#!/usr/bin/env python3
"""Read-only quality audit for local Ultralytics detection datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BBOX_BOUNDARY_TOLERANCE = 1e-6
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_diagnostics.diagnostics.engine import (  # noqa: E402
    _dataset_names,
    _expand_ultralytics_source,
    _read_yaml_mapping,
    _resolve_dataset_root,
    _ultralytics_label_path,
)
from model_diagnostics.run_model_diagnostics import _next_run_dir  # noqa: E402


def _issue(items: list[dict[str, Any]], severity: str, code: str, split: str,
           image: Path, label: Path | None = None, line: int | None = None,
           detail: str = "") -> None:
    items.append({"severity": severity, "code": code, "split": split,
                  "image": str(image), "label": str(label) if label else "",
                  "line": line if line is not None else "", "detail": detail})


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import io

    output = io.StringIO()
    fields = ["severity", "code", "split", "image", "label", "line", "detail"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, output.getvalue())


def _read_boxes(label: Path, image: Path, split: str, classes: dict[int, str],
                issues: list[dict[str, Any]]) -> list[tuple[int, float, float, float, float]]:
    if not label.exists():
        _issue(issues, "warning", "missing_label", split, image, label)
        return []
    try:
        lines = label.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _issue(issues, "error", "unreadable_label", split, image, label, detail=str(exc))
        return []
    if not any(line.strip() for line in lines):
        _issue(issues, "info", "empty_label", split, image, label)
    boxes: list[tuple[int, float, float, float, float]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for number, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            _issue(issues, "error", "invalid_label_columns", split, image, label, number,
                   "检测框需要 class xc yc w h 共 5 列")
            continue
        try:
            raw_class = float(parts[0])
            values = tuple(float(part) for part in parts[1:])
        except ValueError:
            _issue(issues, "error", "invalid_label_number", split, image, label, number)
            continue
        if not all(math.isfinite(value) for value in (raw_class, *values)):
            _issue(issues, "error", "nonfinite_label", split, image, label, number)
            continue
        if not raw_class.is_integer() or int(raw_class) not in classes:
            _issue(issues, "error", "invalid_class", split, image, label, number,
                   f"class={parts[0]}; 有效类别={sorted(classes)}")
            continue
        xc, yc, width, height = values
        if width <= 0 or height <= 0:
            _issue(issues, "error", "nonpositive_box", split, image, label, number)
            continue
        left, top = xc - width / 2, yc - height / 2
        right, bottom = xc + width / 2, yc + height / 2
        # YOLO labels are commonly rounded to six decimal places. A box that
        # touches an image edge can therefore be off by a few ulps after the
        # arithmetic above. Keep that harmless rounding out of the error
        # report, while still rejecting genuinely invalid annotations.
        if min(left, top) < -BBOX_BOUNDARY_TOLERANCE or max(right, bottom) > 1 + BBOX_BOUNDARY_TOLERANCE:
            _issue(issues, "error", "out_of_bounds_box", split, image, label, number)
            continue
        if min(left, top) < 0 or max(right, bottom) > 1:
            xc = min(1.0, max(0.0, xc))
            yc = min(1.0, max(0.0, yc))
            width = min(width, 2 * min(xc, 1 - xc))
            height = min(height, 2 * min(yc, 1 - yc))
        box = (int(raw_class), xc, yc, width, height)
        if box in seen:
            _issue(issues, "warning", "duplicate_label", split, image, label, number)
            continue
        seen.add(box)
        boxes.append(box)
    return boxes


def _preview(image: Path, boxes: list[tuple[int, float, float, float, float]],
             classes: dict[int, str], output: Path) -> None:
    from PIL import Image, ImageDraw  # type: ignore

    with Image.open(image) as source:
        canvas = source.convert("RGB")
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    for class_id, xc, yc, bw, bh in boxes:
        left, top = (xc - bw / 2) * width, (yc - bh / 2) * height
        right, bottom = (xc + bw / 2) * width, (yc + bh / 2) * height
        draw.rectangle((left, top, right, bottom), outline="red", width=2)
        draw.text((left, max(0, top - 12)), classes[class_id], fill="red")
    canvas.thumbnail((1280, 1280))
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.stem}.",
                                     suffix=".png", delete=False) as handle:
        temp = Path(handle.name)
    try:
        canvas.save(temp, format="PNG")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def _overlap_pairs(boxes: list[tuple[int, float, float, float, float]]) -> int:
    """Count box pairs with IoU >= 0.5 as an annotation review clue."""
    count = 0
    for index, (_, ax, ay, aw, ah) in enumerate(boxes):
        for _, bx, by, bw, bh in boxes[index + 1:]:
            left = max(ax - aw / 2, bx - bw / 2)
            right = min(ax + aw / 2, bx + bw / 2)
            top = max(ay - ah / 2, by - bh / 2)
            bottom = min(ay + ah / 2, by + bh / 2)
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            union = aw * ah + bw * bh - intersection
            if union > 0 and intersection / union >= 0.5:
                count += 1
    return count


def audit(data_yaml: Path, run_dir: Path, sample_count: int = 8) -> dict[str, Any]:
    """Scan train/val/test once and write a reviewable report into run_dir."""
    from PIL import Image  # type: ignore

    data_yaml = data_yaml.expanduser().resolve()
    data = _read_yaml_mapping(data_yaml)
    task = str(data.get("task", "detect")).lower()
    if task not in {"", "detect", "detection"}:
        raise ValueError(f"仅支持检测数据集；task={task}")
    classes = _dataset_names(data.get("names"), data.get("nc"))
    if not classes or sorted(classes) != list(range(len(classes))):
        raise ValueError("data.yaml 需要连续、从 0 开始的 names 或正整数 nc")
    root = _resolve_dataset_root(data_yaml, data.get("path"), data_yaml.parent)
    sources = {split: data.get(split) for split in ("train", "val", "test") if data.get(split)}
    if "val" not in sources and data.get("validation"):
        sources["val"] = data["validation"]
    if not sources:
        raise ValueError("data.yaml 缺少 train/val/test 图片路径")
    paths = {split: _expand_ultralytics_source(source, root, data_yaml)
             for split, source in sources.items()}
    if any(not images for images in paths.values()):
        raise ValueError("至少一个已配置的数据划分没有找到图片")

    issues: list[dict[str, Any]] = []
    image_splits: dict[Path, set[str]] = defaultdict(set)
    hash_images: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()
    image_sizes: Counter[str] = Counter()
    box_sizes: Counter[str] = Counter()
    aspect_ratios: Counter[str] = Counter()
    high_overlap_pairs = 0
    samples: list[str] = []
    preview_dir = run_dir / "samples"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for split, images in paths.items():
        made = 0
        if sample_count:
            positions = {round(index * (len(images) - 1) / max(sample_count - 1, 1))
                         for index in range(min(sample_count, len(images)))}
        else:
            positions = set()
        for position, image in enumerate(images):
            image_splits[image].add(split)
            label = _ultralytics_label_path(image, root)
            try:
                if image.stat().st_size == 0:
                    raise ValueError("文件大小为 0")
                with Image.open(image) as source:
                    source.verify()
                with Image.open(image) as source:
                    width, height = source.size
                    extrema = source.convert("RGB").getextrema()
                if width <= 0 or height <= 0:
                    raise ValueError("图像尺寸为 0")
                digest = _digest(image)
            except (OSError, ValueError, SyntaxError) as exc:
                _issue(issues, "error", "corrupt_image", split, image, label, detail=str(exc))
                continue
            hash_images[digest].append((split, image))
            if all(low == high for low, high in extrema):
                _issue(issues, "warning", "uniform_image", split, image, label)
            short = min(width, height)
            image_sizes["<640" if short < 640 else "640-1279" if short < 1280 else ">=1280"] += 1
            boxes = _read_boxes(label, image, split, classes, issues)
            high_overlap_pairs += _overlap_pairs(boxes)
            for class_id, _, _, bw, bh in boxes:
                class_counts[class_id] += 1
                box_short = min(bw * width, bh * height)
                box_sizes["<16px" if box_short < 16 else "16-31px" if box_short < 32 else ">=32px"] += 1
                ratio = max(bw * width / (bh * height), bh * height / (bw * width))
                aspect_ratios[">=5:1" if ratio >= 5 else "2-5:1" if ratio >= 2 else "<2:1"] += 1
            if position in positions:
                destination = preview_dir / f"{split}_{made + 1:03d}.png"
                try:
                    _preview(image, boxes, classes, destination)
                except (OSError, ValueError) as exc:
                    _issue(issues, "warning", "preview_failed", split, image, label, detail=str(exc))
                else:
                    samples.append(str(destination.relative_to(run_dir)))
                    made += 1
    for image, splits in image_splits.items():
        if len(splits) > 1:
            _issue(issues, "error", "split_leak_same_path", ",".join(sorted(splits)), image)
    for entries in hash_images.values():
        if len(entries) > 1:
            first_split, first_image = entries[0]
            for split, image in entries[1:]:
                if image == first_image:
                    continue  # split_leak_same_path already describes this case
                code = "split_leak_duplicate" if split != first_split else "duplicate_image"
                _issue(issues, "error" if split != first_split else "warning", code,
                       split, image, detail=f"与 {first_split}:{first_image} 内容相同")
    counts = Counter(item["code"] for item in issues)
    severities = Counter(item["severity"] for item in issues)
    summary = {
        "status": "issues_found" if issues else "passed",
        "dataset_yaml": str(data_yaml), "dataset_root": str(root),
        "splits": {split: len(images) for split, images in paths.items()},
        "classes": {str(key): value for key, value in classes.items()},
        "class_counts": {str(key): class_counts[key] for key in classes},
        "image_short_side": dict(image_sizes), "box_short_side": dict(box_sizes),
        "box_aspect_ratio": dict(aspect_ratios),
        "high_overlap_box_pairs_iou_0_5": high_overlap_pairs,
        "issue_counts": dict(counts), "severity_counts": dict(severities),
        "sample_images": samples,
        "thresholds": {"small_box_short_side_px": 16, "extreme_aspect_ratio": 5},
    }
    _write_csv(run_dir / "issues.csv", issues)
    _atomic_text(run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report = ["# 数据集质量检查", "", f"数据集：`{data_yaml}`", "",
              f"状态：**{summary['status']}**", "",
              "## 图片数", "", *[f"- {key}: {value}" for key, value in summary["splits"].items()],
              "", "## 问题数", "", *([f"- {key}: {value}" for key, value in sorted(counts.items())] or ["- 无"]),
              "", "## 类别标注数", "",
              *[f"- {classes[key]} ({key}): {class_counts[key]}" for key in classes],
              "", "## 分布", "", f"图片短边：{dict(image_sizes)}", "",
              f"标注框短边：{dict(box_sizes)}", "", f"框宽高比：{dict(aspect_ratios)}", "",
              f"IoU≥0.5 的标注框对：{high_overlap_pairs}", "",
              "逐项路径和标签行号见 `issues.csv`。框重叠不能直接证明真实遮挡。", ""]
    if samples:
        report.extend(["## 标注抽样", ""])
        report.extend(f"![{path}]({path})" for path in samples)
        report.append("")
    _atomic_text(run_dir / "report.md", "\n".join(report))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读检查 Ultralytics YOLO 检测数据集质量")
    parser.add_argument("--data", required=True, type=Path, help="data.yaml；相对当前工作目录解析")
    parser.add_argument("--output-root", type=Path, default=ROOT / "model_diagnostics" / "runs" / "dataset_quality",
                        help="运行根目录，自动新建 runN；相对当前工作目录解析")
    parser.add_argument("--sample-count", type=int, default=8, help="每个划分生成的标注抽样图数量")
    args = parser.parse_args(argv)
    if args.sample_count < 0:
        parser.error("--sample-count 不能为负数")
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        parser.error(f"数据集 YAML 不存在：{data_yaml}")
    run_dir: Path | None = None
    try:
        run_dir = _next_run_dir({"run": {"root": str(args.output_root.expanduser().resolve())}}, ROOT)
        summary = audit(data_yaml, run_dir, args.sample_count)
    except Exception as exc:
        if run_dir is not None:
            _atomic_text(run_dir / "summary.json", json.dumps({"status": "failed", "error": str(exc),
                                                                "dataset_yaml": str(data_yaml)}, ensure_ascii=False,
                                                               indent=2) + "\n")
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2
    print(f"运行目录：{run_dir.resolve()}")
    print(f"状态：{summary['status']}；问题 {sum(summary['issue_counts'].values())} 项")
    return 1 if summary["severity_counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
