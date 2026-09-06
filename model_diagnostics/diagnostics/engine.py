#!/usr/bin/env python3
"""
Framework-independent diagnostics for object detection models.

The script evaluates predictions against ground-truth boxes.  It does not
import Ultralytics (or any other model framework), and it does not estimate
runtime cost.  The PyCharm entry point separates model use into three modes;
this engine remains the framework-independent evaluation layer:

1. Mode A passes an existing prediction file.
2. Mode B uses the built-in Ultralytics ``.pt`` loader.
3. Mode C runs a vtools-style adapter.

All three paths produce the same canonical format before this engine is
called.  The command-line engine can also be used directly by pointing
``--gt`` at an Ultralytics ``data.yaml`` (or canonical/COCO/YOLO GT) and
passing predictions with ``--pred``.

The output directory contains machine-readable CSV/JSON files and a Markdown
report.  Optional raw/candidate predictions can be supplied with ``--raw-pred``
to quantify losses introduced by a model's filtering or post-processing stages.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Number = float
BBox = Tuple[float, float, float, float]


@dataclass
class ImageInfo:
    image_id: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Instance:
    image_id: str
    bbox: BBox
    class_id: int
    score: Optional[float] = None
    instance_id: str = ""
    ignore: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    images: Dict[str, ImageInfo] = field(default_factory=dict)
    gt: Dict[str, List[Instance]] = field(default_factory=lambda: defaultdict(list))
    predictions: Dict[str, List[Instance]] = field(default_factory=lambda: defaultdict(list))
    warnings: List[str] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)


DEFAULT_CONFIG: Dict[str, Any] = {
    "score_threshold": 0.25,
    "candidate_threshold": 0.001,
    "match_iou": 0.50,
    "candidate_iou_thresholds": [0.30, 0.50, 0.75],
    "ap_iou_thresholds": [round(0.50 + 0.05 * i, 2) for i in range(10)],
    "score_sweep": [0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
    "fp_budgets_per_image": [0.1, 0.5, 1.0, 2.0, 5.0],
    "localization_iou_floor": 0.10,
    "short_side_bins": [0, 8, 16, 32, 64, "inf"],
    "density_bins": [0, 1, 4, 8, "inf"],
    "bootstrap_images": 0,
    "seed": 0,
}


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _image_id(value: Any) -> str:
    return str(value)


def _read_jsonish(path: Path) -> Any:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"无法解析 {path} 第 {line_no} 行: {exc}") from exc
        return rows
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bbox(value: Any, fmt: str = "xyxy") -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox 必须是长度为 4 的数组，收到: {value!r}")
    a, b, c, d = [float(x) for x in value]
    if fmt.lower() in {"xywh", "coco"}:
        return (a, b, a + c, b + d)
    x1, y1, x2, y2 = a, b, c, d
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _image_from_object(obj: Mapping[str, Any], fallback_id: Any = None) -> ImageInfo:
    image_id = _image_id(obj.get("image_id", obj.get("id", fallback_id)))
    if image_id in {"None", ""}:
        raise ValueError(f"缺少 image_id/id: {obj!r}")
    width = obj.get("width", obj.get("image_width"))
    height = obj.get("height", obj.get("image_height"))
    return ImageInfo(
        image_id=image_id,
        width=_as_int(width, 0) or None,
        height=_as_int(height, 0) or None,
        file_name=obj.get("file_name", obj.get("path")),
        extra={k: v for k, v in obj.items() if k not in {"id", "image_id", "width", "height", "image_width", "image_height", "file_name", "path"}},
    )


def _instance_from_object(
    obj: Mapping[str, Any],
    image_id: str,
    default_bbox_format: str,
    index: int,
    is_prediction: bool,
) -> Instance:
    raw_box = obj.get("bbox", obj.get("box"))
    if raw_box is None:
        raise ValueError(f"实例缺少 bbox/box: {obj!r}")
    fmt = str(obj.get("bbox_format", default_bbox_format))
    class_id = obj.get("class_id", obj.get("category_id", obj.get("class", obj.get("label", 0))))
    score = obj.get("score", obj.get("confidence", obj.get("conf"))) if is_prediction else None
    return Instance(
        image_id=image_id,
        bbox=_bbox(raw_box, fmt),
        class_id=_as_int(class_id),
        score=_as_float(score, 0.0) if is_prediction else None,
        instance_id=str(obj.get("instance_id", obj.get("id", f"{image_id}:{index}"))),
        ignore=bool(obj.get("ignore", obj.get("iscrowd", False))),
        extra={k: v for k, v in obj.items() if k not in {"bbox", "box", "bbox_format", "class_id", "category_id", "class", "label", "score", "confidence", "conf", "instance_id", "id", "ignore", "iscrowd"}},
    )


def _record_list(data: Any) -> List[Mapping[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, Mapping)]
    if isinstance(data, Mapping):
        for key in ("records", "samples", "items", "data"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, Mapping)]
    return []


def _load_canonical_ground_truth(data: Any, ds: Dataset) -> None:
    records = _record_list(data)
    if isinstance(data, Mapping) and isinstance(data.get("images"), list):
        for image in data["images"]:
            if isinstance(image, Mapping):
                info = _image_from_object(image)
                ds.images[info.image_id] = info
    if isinstance(data, Mapping) and isinstance(data.get("annotations"), list):
        for index, ann in enumerate(data["annotations"]):
            if not isinstance(ann, Mapping):
                continue
            image_id = _image_id(ann.get("image_id"))
            ds.images.setdefault(image_id, ImageInfo(image_id=image_id))
            ds.gt[image_id].append(_instance_from_object(ann, image_id, "xyxy", index, False))
    for record_index, record in enumerate(records):
        if "ground_truth" not in record and "gt" not in record and "annotations" not in record:
            continue
        info = _image_from_object(record, record_index)
        ds.images[info.image_id] = info
        objects = record.get("ground_truth", record.get("gt", record.get("annotations", [])))
        if isinstance(objects, Mapping):
            objects = objects.get("objects", [])
        for index, obj in enumerate(objects or []):
            if isinstance(obj, Mapping):
                ds.gt[info.image_id].append(_instance_from_object(obj, info.image_id, "xyxy", index, False))
    if isinstance(data, Mapping) and isinstance(data.get("class_names"), Mapping):
        ds.class_names.update({_as_int(k): str(v) for k, v in data["class_names"].items()})


def _load_canonical_predictions(data: Any, images: Dict[str, ImageInfo], warnings: List[str]) -> Dict[str, List[Instance]]:
    predictions: Dict[str, List[Instance]] = defaultdict(list)
    records = _record_list(data)
    if isinstance(data, Mapping) and isinstance(data.get("predictions"), list):
        records = data["predictions"]
    # A top-level list of {image_id, bbox, score, class_id} is also accepted.
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        image_id = record.get("image_id", record.get("id"))
        if image_id is None:
            continue
        image_id = _image_id(image_id)
        if "bbox" in record or "box" in record:
            images.setdefault(image_id, ImageInfo(image_id=image_id))
            predictions[image_id].append(_instance_from_object(record, image_id, "xyxy", record_index, True))
            continue
        info = _image_from_object(record, record_index)
        images.setdefault(info.image_id, info)
        objects = record.get("predictions", record.get("detections", record.get("objects", [])))
        for index, obj in enumerate(objects or []):
            if isinstance(obj, Mapping):
                predictions[info.image_id].append(_instance_from_object(obj, info.image_id, "xyxy", index, True))
    for image_id, values in predictions.items():
        for pred in values:
            if pred.score is None:
                warnings.append(f"预测 {pred.instance_id} 缺少 score，已按 0 处理")
    return predictions


def _load_coco_ground_truth(data: Mapping[str, Any], ds: Dataset) -> None:
    for image in data.get("images", []):
        if isinstance(image, Mapping):
            info = _image_from_object(image)
            ds.images[info.image_id] = info
    for category in data.get("categories", []):
        if isinstance(category, Mapping) and "id" in category:
            ds.class_names[_as_int(category["id"])] = str(category.get("name", category["id"]))
    for index, ann in enumerate(data.get("annotations", [])):
        if not isinstance(ann, Mapping):
            continue
        image_id = _image_id(ann.get("image_id"))
        ds.images.setdefault(image_id, ImageInfo(image_id=image_id))
        ds.gt[image_id].append(_instance_from_object(ann, image_id, "xywh", index, False))


def _load_coco_predictions(data: Any, images: Dict[str, ImageInfo], warnings: List[str]) -> Dict[str, List[Instance]]:
    if isinstance(data, Mapping):
        values = data.get("predictions", data.get("annotations", []))
    else:
        values = data
    predictions: Dict[str, List[Instance]] = defaultdict(list)
    for index, obj in enumerate(values or []):
        if not isinstance(obj, Mapping):
            continue
        image_id = _image_id(obj.get("image_id"))
        images.setdefault(image_id, ImageInfo(image_id=image_id))
        predictions[image_id].append(_instance_from_object(obj, image_id, "xywh", index, True))
    return predictions


def _load_yolo_ground_truth(path: Path, images_dir: Optional[Path], ds: Dataset) -> None:
    if images_dir is None:
        raise ValueError("YOLO 标签需要 --images-dir 才能把归一化坐标还原到像素坐标")
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("读取 YOLO 标签需要 Pillow；请安装 pillow，或改用 canonical/COCO 格式") from exc
    for label_path in sorted(path.glob("*.txt")):
        image_id = label_path.stem
        image_path = _find_image(images_dir, image_id)
        if image_path is None:
            ds.warnings.append(f"找不到 {image_id} 对应图片，已跳过")
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        ds.images[image_id] = ImageInfo(image_id=image_id, width=width, height=height, file_name=str(image_path))
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if len(fields) < 5:
                ds.warnings.append(f"{label_path}:{index + 1} 字段不足，已跳过")
                continue
            class_id, xc, yc, bw, bh = [_as_float(x, 0.0) for x in fields[:5]]
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            ds.gt[image_id].append(Instance(image_id, (x1, y1, x2, y2), int(class_id or 0), instance_id=f"{image_id}:{index}"))


_IMAGE_SUFFIXES = {".avif", ".bmp", ".dng", ".heic", ".heif", ".jp2", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Read a YAML mapping without importing Ultralytics itself."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("读取 Ultralytics data.yaml 需要 PyYAML，请执行: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"数据集 YAML 必须是对象：{path}")
    return value


def _dataset_names(value: Any, nc: Any = None) -> Dict[int, str]:
    """Normalize Ultralytics ``names`` list/dict into the engine's class map."""
    if isinstance(value, Mapping):
        names = {_as_int(key): str(name) for key, name in value.items()}
    elif isinstance(value, (list, tuple)):
        names = {index: str(name) for index, name in enumerate(value)}
    else:
        count = _as_int(nc, 0)
        names = {index: f"class_{index}" for index in range(max(count, 0))}
    return names


def _resolve_dataset_root(data_yaml: Path, raw_path: Any, base_dir: Optional[Path]) -> Path:
    """Resolve the same common ``path`` spellings used in Ultralytics data YAMLs.

    Ultralytics first treats an absolute path as-is, then commonly resolves a
    relative path under its datasets directory. For a standalone diagnostics
    project, resolving relative to the YAML file and project root is the useful
    portable behavior, so both are tried before falling back to the YAML parent.
    """
    if raw_path is None or str(raw_path).strip() == "":
        return data_yaml.parent.resolve()
    raw = Path(str(raw_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [data_yaml.parent / raw]
    if base_dir is not None:
        candidates.append(base_dir / raw)
    candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _expand_ultralytics_source(source: Any, root: Path, data_yaml: Path) -> List[Path]:
    """Expand a data YAML split entry (directory, image, txt list, or list)."""
    values = source if isinstance(source, (list, tuple)) else [source]
    result: List[Path] = []
    for item in values:
        if item is None:
            continue
        raw = Path(str(item)).expanduser()
        candidate = raw if raw.is_absolute() else root / raw
        if not candidate.exists() and not raw.is_absolute():
            fallback = data_yaml.parent / raw
            if fallback.exists():
                candidate = fallback
        if candidate.is_dir():
            result.extend(sorted(path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES))
        elif candidate.is_file() and candidate.suffix.lower() in {".txt", ".list"}:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                listed = Path(line).expanduser()
                if not listed.is_absolute():
                    listed_candidates = [candidate.parent / listed, root / listed, data_yaml.parent / listed]
                    listed = next((item for item in listed_candidates if item.exists()), listed_candidates[0])
                if listed.is_file() and listed.suffix.lower() in _IMAGE_SUFFIXES:
                    result.append(listed)
                elif listed.is_dir():
                    result.extend(sorted(path for path in listed.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES))
        elif candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
            result.append(candidate)
        else:
            raise FileNotFoundError(f"Ultralytics 数据集 split 路径不存在或不是图片：{candidate}")
    # Keep order while removing duplicates, matching the deterministic split order
    # expected by a validation report.
    unique: List[Path] = []
    seen = set()
    for path in result:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _ultralytics_label_path(image_path: Path, root: Path) -> Path:
    """Derive ``labels/...txt`` from the standard ``images/...`` path layout."""
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        relative = Path(image_path.name)
    return (root / "labels" / relative).with_suffix(".txt")


def _ultralytics_image_id(image_path: Path, root: Path, used: set[str]) -> str:
    """Use the filename stem when unique, otherwise a relative path without suffix."""
    stem = image_path.stem
    if stem not in used:
        used.add(stem)
        return stem
    try:
        relative = image_path.relative_to(root).with_suffix("")
        image_id = relative.as_posix()
    except ValueError:
        image_id = image_path.with_suffix("").as_posix()
    used.add(image_id)
    return image_id


def _load_ultralytics_ground_truth(
    data_yaml: Path,
    ds: Dataset,
    split: str = "val",
    base_dir: Optional[Path] = None,
) -> None:
    """Load a detection split from an Ultralytics ``data.yaml``.

    Supported fields intentionally mirror Ultralytics' ``check_det_dataset``:
    ``path``, ``train``/``val``/``test`` and ``names`` (or ``nc``). The actual
    labels remain the standard YOLO ``labels/<split>/*.txt`` files, while the
    diagnostics engine converts their normalized ``xywh`` coordinates to
    original-image ``xyxy`` pixels.
    """
    data = _read_yaml_mapping(data_yaml)
    task = str(data.get("task", "detect")).lower()
    if task not in {"", "detect", "detection"}:
        raise ValueError(f"当前目标检测诊断只支持 task: detect；数据集 YAML 是 task: {task}")
    split = str(split or "val").lower()
    if split == "validation" and not data.get("val"):
        split = "val"
    source = data.get(split)
    if source is None and split == "val":
        source = data.get("validation")
    if source is None:
        raise ValueError(f"{data_yaml} 缺少 {split}: 图片路径；Ultralytics data.yaml 至少需要 train 和 val")

    root = _resolve_dataset_root(data_yaml, data.get("path"), base_dir)
    image_paths = _expand_ultralytics_source(source, root, data_yaml)
    if not image_paths:
        raise ValueError(f"{data_yaml} 的 {split}: 没有找到图片")
    ds.class_names.update(_dataset_names(data.get("names"), data.get("nc")))
    used_ids: set[str] = set()
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("读取 Ultralytics YOLO 标签需要 Pillow；请执行: pip install pillow") from exc

    for image_path in image_paths:
        image_id = _ultralytics_image_id(image_path, root, used_ids)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            ds.warnings.append(f"无法读取图片尺寸，已跳过 {image_path}: {exc}")
            continue
        try:
            relative_name = image_path.relative_to(root).as_posix()
        except ValueError:
            relative_name = image_path.name
        info = ImageInfo(
            image_id=image_id,
            width=width,
            height=height,
            file_name=relative_name,
            extra={"source_path": str(image_path), "dataset_yaml": str(data_yaml.resolve()), "split": split},
        )
        ds.images[image_id] = info
        label_path = _ultralytics_label_path(image_path, root)
        if not label_path.exists():
            ds.warnings.append(f"{image_path} 缺少标签文件 {label_path}，按背景图处理")
            continue
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                ds.warnings.append(
                    f"{label_path}:{index + 1} 不是目标检测标签（需要 5 列 class xc yc w h），已跳过；"
                    "实例分割 polygon 标签暂不支持"
                )
                continue
            try:
                class_id = int(float(fields[0]))
                xc, yc, bw, bh = [float(value) for value in fields[1:5]]
            except ValueError:
                ds.warnings.append(f"{label_path}:{index + 1} 含有非数字字段，已跳过")
                continue
            if not all(-0.01 <= value <= 1.01 for value in (xc, yc, bw, bh)):
                ds.warnings.append(f"{label_path}:{index + 1} 坐标不是归一化值，已跳过")
                continue
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            ds.gt[image_id].append(Instance(image_id, (x1, y1, x2, y2), class_id, instance_id=f"{image_id}:{index}"))


def _convert_ultralytics_ndjson(path: Path) -> Path:
    """Use Ultralytics' official NDJSON compatibility converter when requested."""
    try:
        from ultralytics.data.utils import convert_ndjson_to_yolo_if_needed
    except ImportError as exc:
        raise RuntimeError(
            "读取 Ultralytics NDJSON 需要安装 ultralytics；也可以先用其转换器生成 data.yaml"
        ) from exc
    converted = convert_ndjson_to_yolo_if_needed(str(path))
    converted_path = Path(str(converted))
    if not converted_path.exists():
        raise FileNotFoundError(f"Ultralytics NDJSON 转换后未找到 data.yaml：{converted_path}")
    return converted_path


def _find_image(images_dir: Path, image_id: str, file_name: Optional[str] = None) -> Optional[Path]:
    if file_name:
        candidate = Path(file_name)
        if not candidate.is_absolute():
            candidate = images_dir / candidate
        if candidate.exists():
            return candidate
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_ground_truth(
    path: Path,
    fmt: str = "auto",
    images_dir: Optional[Path] = None,
    split: str = "val",
    base_dir: Optional[Path] = None,
) -> Dataset:
    ds = Dataset()
    normalized_fmt = str(fmt or "auto").lower()
    if normalized_fmt == "ndjson" or path.suffix.lower() == ".ndjson":
        converted = _convert_ultralytics_ndjson(path)
        return load_ground_truth(converted, "ultralytics", images_dir, split=split, base_dir=base_dir)
    if normalized_fmt in {"ultralytics", "yolo_yaml", "data_yaml"} or (
        normalized_fmt == "auto" and path.suffix.lower() in {".yaml", ".yml"}
    ):
        _load_ultralytics_ground_truth(path, ds, split=split, base_dir=base_dir)
        return ds
    if normalized_fmt == "yolo" or path.is_dir():
        _load_yolo_ground_truth(path, images_dir, ds)
        return ds
    data = _read_jsonish(path)
    if normalized_fmt == "coco" or (isinstance(data, Mapping) and "images" in data and "annotations" in data):
        _load_coco_ground_truth(data, ds)
    else:
        _load_canonical_ground_truth(data, ds)
    return ds


def load_predictions(path: Path, images: Dict[str, ImageInfo], fmt: str = "auto") -> Tuple[Dict[str, List[Instance]], List[str]]:
    warnings: List[str] = []
    if path.is_dir():
        raise ValueError("预测目录不是通用格式；请导出 canonical JSON/JSONL、COCO results，或通过 --predictor 生成")
    data = _read_jsonish(path)
    if fmt == "coco" or (isinstance(data, Mapping) and "annotations" in data and "images" not in data):
        return _load_coco_predictions(data, images, warnings), warnings
    if isinstance(data, list) and data and all(isinstance(x, Mapping) and "bbox" in x and "image_id" in x for x in data):
        return _load_coco_predictions(data, images, warnings), warnings
    return _load_canonical_predictions(data, images, warnings), warnings


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("读取 YAML 配置需要 PyYAML，请执行: pip install pyyaml") from exc
        with path.open("r", encoding="utf-8") as handle:
            user = yaml.safe_load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            user = json.load(handle)
    if not isinstance(user, Mapping):
        raise ValueError("配置文件必须是 YAML/JSON 对象")
    # vtools-style run files keep detection thresholds under ``diagnostics``.
    # Flatten that section into the same keys used by the standalone CLI while
    # retaining all original sections in config_used.json.
    if isinstance(user.get("diagnostics"), Mapping):
        merged = dict(user)
        merged.update(dict(user["diagnostics"]))
        user = merged
    config.update(user)
    return config


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _finite_values(values: Iterable[float]) -> List[float]:
    return [float(x) for x in values if x is not None and math.isfinite(float(x))]


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = _finite_values(values)
    return sum(vals) / len(vals) if vals else None


def _median(values: Iterable[float]) -> Optional[float]:
    vals = sorted(_finite_values(values))
    return statistics.median(vals) if vals else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(_finite_values(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _label_bin(value: float, edges: Sequence[Any], prefix: str = "bin") -> str:
    clean: List[float] = []
    for edge in edges:
        if isinstance(edge, str) and edge.lower() in {"inf", "infinity"}:
            clean.append(float("inf"))
        else:
            clean.append(float(edge))
    if not clean:
        return f"{prefix}_unknown"
    for index in range(len(clean) - 1):
        lo, hi = clean[index], clean[index + 1]
        if lo <= value < hi:
            return f"{prefix}_{lo:g}_{hi:g}"
    if value >= clean[-1]:
        return f"{prefix}_{clean[-1]:g}_plus"
    return f"{prefix}_below_{clean[0]:g}"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {"precision": precision, "recall": recall, "f1": f1}


def _pred_score(pred: Instance) -> float:
    return float(pred.score if pred.score is not None else 0.0)


def _active_gt(gts: Sequence[Instance]) -> List[Instance]:
    return [gt for gt in gts if not gt.ignore]


def greedy_match(
    predictions: Sequence[Instance],
    gts: Sequence[Instance],
    score_threshold: float,
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Class-aware one-to-one greedy matching for one image."""
    active = _active_gt(gts)
    ordered = sorted([(idx, p) for idx, p in enumerate(predictions) if _pred_score(p) >= score_threshold], key=lambda x: (-_pred_score(x[1]), x[0]))
    used = set()
    matches: List[Tuple[int, int, float]] = []
    unmatched_pred: List[int] = []
    for pred_index, pred in ordered:
        candidates = [(box_iou(pred.bbox, gt.bbox), gt_index) for gt_index, gt in enumerate(active) if gt_index not in used and gt.class_id == pred.class_id]
        if candidates:
            best_iou, best_gt = max(candidates, key=lambda x: x[0])
        else:
            best_iou, best_gt = 0.0, -1
        if best_iou >= iou_threshold:
            used.add(best_gt)
            matches.append((pred_index, best_gt, best_iou))
        else:
            unmatched_pred.append(pred_index)
    unmatched_gt = [idx for idx in range(len(active)) if idx not in used]
    return matches, unmatched_pred, unmatched_gt


def _max_iou(predictions: Sequence[Instance], gt: Instance, score_threshold: float, same_class: bool) -> Tuple[float, Optional[Instance]]:
    candidates = [p for p in predictions if _pred_score(p) >= score_threshold and (not same_class or p.class_id == gt.class_id)]
    if not candidates:
        return 0.0, None
    pred = max(candidates, key=lambda p: box_iou(p.bbox, gt.bbox))
    return box_iou(pred.bbox, gt.bbox), pred


def candidate_coverage(
    predictions: Mapping[str, Sequence[Instance]],
    gts: Mapping[str, Sequence[Instance]],
    score_threshold: float,
    iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    best_same: List[float] = []
    best_any: List[float] = []
    for image_id, image_gts in gts.items():
        image_predictions = predictions.get(image_id, [])
        for gt in _active_gt(image_gts):
            same, _ = _max_iou(image_predictions, gt, score_threshold, True)
            any_iou, _ = _max_iou(image_predictions, gt, score_threshold, False)
            best_same.append(same)
            best_any.append(any_iou)
    result: Dict[str, Any] = {
        "gt_count": len(best_same),
        "mean_best_same_class_iou": _mean(best_same),
        "median_best_same_class_iou": _median(best_same),
        "mean_best_any_class_iou": _mean(best_any),
    }
    for threshold in iou_thresholds:
        result[f"candidate_recall_iou_{threshold:g}"] = _safe_div(sum(x >= threshold for x in best_same), len(best_same))
        result[f"any_class_candidate_recall_iou_{threshold:g}"] = _safe_div(sum(x >= threshold for x in best_any), len(best_any))
    return result


def _average_precision(tp_flags: Sequence[bool], scores: Sequence[float], total_gt: int) -> Optional[float]:
    if total_gt <= 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    tp_cum, fp_cum = 0, 0
    recalls: List[float] = []
    precisions: List[float] = []
    for index in order:
        if tp_flags[index]:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / total_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))
    # 101-point interpolated AP is explicit and reproducible across frameworks.
    samples = []
    for step in range(101):
        recall_level = step / 100.0
        candidates = [p for r, p in zip(recalls, precisions) if r >= recall_level]
        samples.append(max(candidates) if candidates else 0.0)
    return sum(samples) / len(samples)


def average_precision_for_iou(ds: Dataset, iou_threshold: float) -> Tuple[Optional[float], Dict[int, Optional[float]]]:
    classes = sorted({gt.class_id for values in ds.gt.values() for gt in _active_gt(values)})
    per_class: Dict[int, Optional[float]] = {}
    for class_id in classes:
        all_predictions: List[Instance] = []
        total_gt = 0
        matched: Dict[str, set] = {}
        for image_id, gts in ds.gt.items():
            valid = [gt for gt in _active_gt(gts) if gt.class_id == class_id]
            total_gt += len(valid)
            matched[image_id] = set()
            all_predictions.extend([p for p in ds.predictions.get(image_id, []) if p.class_id == class_id])
        # Store TP flags in the same order as all_predictions.  Matching itself is
        # performed in descending confidence order within each image.
        scores = [_pred_score(p) for p in all_predictions]
        flags = [False] * len(all_predictions)
        by_image: Dict[str, List[Tuple[int, Instance]]] = defaultdict(list)
        for idx, pred in enumerate(all_predictions):
            by_image[pred.image_id].append((idx, pred))
        for image_id, image_preds in by_image.items():
            gts = [gt for gt in _active_gt(ds.gt.get(image_id, [])) if gt.class_id == class_id]
            used = set()
            for pred_idx, pred in sorted(image_preds, key=lambda x: (-_pred_score(x[1]), x[0])):
                candidates = [(box_iou(pred.bbox, gt.bbox), gi) for gi, gt in enumerate(gts) if gi not in used]
                best_iou, best_gt = max(candidates, key=lambda x: x[0]) if candidates else (0.0, -1)
                if best_iou >= iou_threshold:
                    flags[pred_idx] = True
                    used.add(best_gt)
        per_class[class_id] = _average_precision(flags, scores, total_gt)
    values = [x for x in per_class.values() if x is not None]
    return (_mean(values), per_class) if values else (None, per_class)


def compute_ap_metrics(ds: Dataset, iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    per_iou = []
    for threshold in iou_thresholds:
        ap, per_class = average_precision_for_iou(ds, threshold)
        result[f"AP{int(round(threshold * 100)):02d}"] = ap
        result[f"AP{int(round(threshold * 100)):02d}_per_class"] = {str(k): v for k, v in per_class.items()}
        if ap is not None:
            per_iou.append(ap)
    result["mAP50_95"] = _mean(per_iou)
    return result


def _target_attributes(gt: Instance, info: Optional[ImageInfo], image_gts: Sequence[Instance], area_quantiles: Tuple[Optional[float], Optional[float], Optional[float]], config: Mapping[str, Any]) -> Dict[str, Any]:
    width = info.width if info and info.width else None
    height = info.height if info and info.height else None
    w, h = max(0.0, gt.bbox[2] - gt.bbox[0]), max(0.0, gt.bbox[3] - gt.bbox[1])
    short_side = min(w, h)
    area = w * h
    relative_area = area / (width * height) if width and height else None
    other_ious = [box_iou(gt.bbox, other.bbox) for other in _active_gt(image_gts) if other.instance_id != gt.instance_id]
    max_overlap = max(other_ious) if other_ious else 0.0
    q1, q2, q3 = area_quantiles
    if q1 is None:
        area_group = "area_unknown"
    elif area < q1:
        area_group = "area_q1"
    elif area < (q2 if q2 is not None else q1):
        area_group = "area_q2"
    elif area < (q3 if q3 is not None else (q2 if q2 is not None else q1)):
        area_group = "area_q3"
    else:
        area_group = "area_q4"
    return {
        "width_px": w,
        "height_px": h,
        "short_side_px": short_side,
        "area_px2": area,
        "relative_area": relative_area,
        "size_short_side_group": _label_bin(short_side, config.get("short_side_bins", DEFAULT_CONFIG["short_side_bins"]), "short_side"),
        "size_area_quantile_group": area_group,
        "max_gt_overlap": max_overlap,
        "overlap_group": "crowded" if max_overlap >= 0.30 else ("nearby" if max_overlap >= 0.10 else "isolated"),
        "density_group": _label_bin(len(_active_gt(image_gts)), config.get("density_bins", DEFAULT_CONFIG["density_bins"]), "gt_per_image"),
    }


def evaluate_operating(ds: Dataset, config: Mapping[str, Any]) -> Dict[str, Any]:
    score_threshold = float(config.get("score_threshold", 0.25))
    candidate_threshold = float(config.get("candidate_threshold", 0.001))
    match_iou = float(config.get("match_iou", 0.5))
    localization_floor = float(config.get("localization_iou_floor", 0.1))
    candidate_iou_thresholds = [float(x) for x in config.get("candidate_iou_thresholds", [0.3, 0.5, 0.75])]
    tp = fp = fn = 0
    per_image: List[Dict[str, Any]] = []
    per_gt: List[Dict[str, Any]] = []
    per_prediction: List[Dict[str, Any]] = []
    all_areas = [box_area(gt.bbox) for values in ds.gt.values() for gt in _active_gt(values)]
    area_quantiles = (_quantile(all_areas, 0.25), _quantile(all_areas, 0.50), _quantile(all_areas, 0.75))

    for image_id in sorted(set(ds.images) | set(ds.gt) | set(ds.predictions)):
        image_info = ds.images.get(image_id, ImageInfo(image_id=image_id))
        gts = _active_gt(ds.gt.get(image_id, []))
        predictions = list(ds.predictions.get(image_id, []))
        matches, unmatched_pred, unmatched_gt = greedy_match(predictions, gts, score_threshold, match_iou)
        matched_pred = {p for p, _, _ in matches}
        matched_gt = {g for _, g, _ in matches}
        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)
        for pred_idx, gt_idx, match_iou_value in matches:
            pred = predictions[pred_idx]
            gt = gts[gt_idx]
            per_prediction.append({"image_id": image_id, "prediction_id": pred.instance_id, "class_id": pred.class_id, "score": _pred_score(pred), "bbox": list(pred.bbox), "error_type": "TP", "matched_gt_id": gt.instance_id, "iou": match_iou_value, "operating_included": True})
        for pred_idx, pred in enumerate(predictions):
            if pred_idx in matched_pred or _pred_score(pred) < score_threshold:
                if pred_idx not in matched_pred and _pred_score(pred) < score_threshold:
                    # A below-threshold prediction is retained in the table so
                    # that low-confidence correct detections can be audited.
                    nearest = max(enumerate(gts), key=lambda x: box_iou(pred.bbox, x[1].bbox), default=(-1, None))
                    nearest_iou = box_iou(pred.bbox, nearest[1].bbox) if nearest[1] else 0.0
                    same_nearest = max((box_iou(pred.bbox, gt.bbox) for gt in gts if gt.class_id == pred.class_id), default=0.0)
                    any_nearest = nearest_iou
                    if same_nearest >= match_iou:
                        low_type = "below_threshold_correct"
                    elif any_nearest >= match_iou:
                        low_type = "below_threshold_classification"
                    elif same_nearest >= localization_floor:
                        low_type = "below_threshold_localization"
                    else:
                        low_type = "below_threshold"
                    per_prediction.append({"image_id": image_id, "prediction_id": pred.instance_id, "class_id": pred.class_id, "score": _pred_score(pred), "bbox": list(pred.bbox), "error_type": low_type, "matched_gt_id": nearest[1].instance_id if nearest[1] else "", "iou": nearest_iou, "operating_included": False})
                continue
            same_ious = [box_iou(pred.bbox, gt.bbox) for gt in gts if gt.class_id == pred.class_id]
            any_ious = [box_iou(pred.bbox, gt.bbox) for gt in gts]
            best_same = max(same_ious) if same_ious else 0.0
            best_any = max(any_ious) if any_ious else 0.0
            if best_same >= match_iou:
                error_type = "duplicate"
            elif best_any >= match_iou:
                error_type = "classification"
            elif best_same >= localization_floor:
                error_type = "localization"
            elif best_any >= localization_floor:
                error_type = "classification_localization"
            else:
                error_type = "background"
            nearest = max(enumerate(gts), key=lambda x: box_iou(pred.bbox, x[1].bbox), default=(-1, None))
            per_prediction.append({"image_id": image_id, "prediction_id": pred.instance_id, "class_id": pred.class_id, "score": _pred_score(pred), "bbox": list(pred.bbox), "error_type": error_type, "matched_gt_id": nearest[1].instance_id if nearest[1] else "", "iou": box_iou(pred.bbox, nearest[1].bbox) if nearest[1] else 0.0, "operating_included": True})
        for gt_idx, gt in enumerate(gts):
            work_same, work_same_pred = _max_iou(predictions, gt, score_threshold, True)
            work_any, work_any_pred = _max_iou(predictions, gt, score_threshold, False)
            candidate_same, candidate_same_pred = _max_iou(predictions, gt, candidate_threshold, True)
            candidate_any, candidate_any_pred = _max_iou(predictions, gt, candidate_threshold, False)
            matched = gt_idx in matched_gt
            if matched:
                match = next(x for x in matches if x[1] == gt_idx)
                matched_pred_instance = predictions[match[0]]
                reason = "TP"
                score = _pred_score(matched_pred_instance)
            else:
                score = _pred_score(candidate_same_pred) if candidate_same_pred else 0.0
                if candidate_same >= match_iou and work_same < match_iou:
                    reason = "miss_low_confidence"
                elif work_same >= match_iou:
                    reason = "miss_duplicate_competition"
                elif candidate_same >= localization_floor:
                    reason = "miss_localization"
                elif candidate_any >= match_iou:
                    reason = "miss_classification"
                elif candidate_any >= localization_floor:
                    reason = "miss_classification_localization"
                else:
                    reason = "miss_no_candidate"
            row = {"image_id": image_id, "gt_id": gt.instance_id, "class_id": gt.class_id, "matched": matched, "match_score": score, "best_same_class_iou": candidate_same, "best_any_class_iou": candidate_any, "best_same_class_iou_at_work_threshold": work_same, "best_any_class_iou_at_work_threshold": work_any, "best_same_class_score_at_candidate_threshold": _pred_score(candidate_same_pred) if candidate_same_pred else 0.0, "miss_reason": reason}
            row.update(_target_attributes(gt, image_info, gts, area_quantiles, config))
            per_gt.append(row)
        image_metrics = _prf(len(matches), len(unmatched_pred), len(unmatched_gt))
        candidates = candidate_coverage({image_id: predictions}, {image_id: gts}, candidate_threshold, candidate_iou_thresholds)
        image_metrics.update({"image_id": image_id, "gt_count": len(gts), "prediction_count_at_threshold": sum(_pred_score(p) >= score_threshold for p in predictions), "tp": len(matches), "fp": len(unmatched_pred), "fn": len(unmatched_gt), "count_error": sum(_pred_score(p) >= score_threshold for p in predictions) - len(gts), "count_absolute_error": abs(sum(_pred_score(p) >= score_threshold for p in predictions) - len(gts)), "count_exact": sum(_pred_score(p) >= score_threshold for p in predictions) == len(gts)})
        for threshold in candidate_iou_thresholds:
            candidate_matches, _, _ = greedy_match(predictions, gts, candidate_threshold, threshold)
            image_metrics[f"recall_at_iou_{threshold:g}"] = _safe_div(len(candidate_matches), len(gts))
        image_metrics.update({k: v for k, v in candidates.items() if k.startswith("candidate_recall")})
        per_image.append(image_metrics)

    aggregate = {"tp": tp, "fp": fp, "fn": fn, "image_count": len(per_image), "gt_count": sum(row["gt_count"] for row in per_image), "prediction_count_at_threshold": sum(row["prediction_count_at_threshold"] for row in per_image)}
    aggregate.update(_prf(tp, fp, fn))
    aggregate["count_mae"] = _mean(row["count_absolute_error"] for row in per_image)
    aggregate["count_bias_pred_minus_gt"] = _mean(row["count_error"] for row in per_image)
    aggregate["count_exact_rate"] = _mean(1.0 if row["count_exact"] else 0.0 for row in per_image)
    aggregate["mean_fp_per_image"] = _mean(row["fp"] for row in per_image)
    aggregate["mean_fn_per_image"] = _mean(row["fn"] for row in per_image)
    aggregate["mean_matched_score"] = _mean(row["match_score"] for row in per_gt if row["matched"])
    aggregate["median_matched_score"] = _median(row["match_score"] for row in per_gt if row["matched"])
    global_coverage = candidate_coverage(ds.predictions, ds.gt, candidate_threshold, candidate_iou_thresholds)
    for threshold in candidate_iou_thresholds:
        candidate_matches = sum(len(greedy_match(ds.predictions.get(image_id, []), _active_gt(ds.gt.get(image_id, [])), candidate_threshold, threshold)[0]) for image_id in set(ds.images) | set(ds.gt) | set(ds.predictions))
        aggregate[f"recall_at_iou_{threshold:g}"] = _safe_div(candidate_matches, aggregate["gt_count"])
        aggregate[f"candidate_coverage_iou_{threshold:g}"] = global_coverage.get(f"candidate_recall_iou_{threshold:g}")
    aggregate["below_threshold_prediction_count"] = sum(1 for row in per_prediction if not row.get("operating_included", True))
    aggregate["low_confidence_recoverable_gt_count"] = sum(row.get("miss_reason") == "miss_low_confidence" for row in per_gt)
    aggregate["low_confidence_recoverable_gt_rate"] = _safe_div(aggregate["low_confidence_recoverable_gt_count"], aggregate["gt_count"])
    operating_rows = [row for row in per_prediction if row.get("operating_included", True)]
    aggregate["error_counts"] = {name: sum(1 for row in operating_rows if row["error_type"] == name) for name in sorted({row["error_type"] for row in operating_rows})}
    return {"aggregate": aggregate, "per_image": per_image, "per_gt": per_gt, "per_prediction": per_prediction, "area_quantiles": {"q1": area_quantiles[0], "q2": area_quantiles[1], "q3": area_quantiles[2]}}


def group_metrics(per_gt: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    group_fields = ["class_id", "size_short_side_group", "size_area_quantile_group", "overlap_group", "density_group"]
    for row in per_gt:
        for field_name in group_fields:
            groups[(field_name, str(row.get(field_name, "unknown")))].append(row)
    output = []
    for (group_type, group_name), rows in sorted(groups.items()):
        matched = sum(bool(row.get("matched")) for row in rows)
        output.append({"group_type": group_type, "group": group_name, "gt_count": len(rows), "matched": matched, "recall": _safe_div(matched, len(rows)), "mean_best_same_class_iou": _mean(row.get("best_same_class_iou", 0.0) for row in rows), "mean_match_score": _mean(row.get("match_score", 0.0) for row in rows if row.get("matched")), "miss_no_candidate": sum(row.get("miss_reason") == "miss_no_candidate" for row in rows), "miss_localization": sum(row.get("miss_reason") == "miss_localization" for row in rows), "miss_classification": sum(row.get("miss_reason") == "miss_classification" for row in rows)})
    return output


def threshold_sweep(ds: Dataset, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for threshold in config.get("score_sweep", DEFAULT_CONFIG["score_sweep"]):
        local = dict(config)
        local["score_threshold"] = float(threshold)
        result = evaluate_operating(ds, local)["aggregate"]
        rows.append({"score_threshold": float(threshold), **{key: result.get(key) for key in ("tp", "fp", "fn", "precision", "recall", "f1", "mean_fp_per_image", "count_mae")}})
    return rows


def fp_budget_rows(sweep: Sequence[Mapping[str, Any]], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    rows = []
    for budget in budgets:
        if not sweep:
            continue
        selected = min(sweep, key=lambda row: abs(float(row.get("mean_fp_per_image") or 0.0) - float(budget)))
        rows.append({"target_fp_per_image": float(budget), "selected_threshold": selected.get("score_threshold"), "actual_fp_per_image": selected.get("mean_fp_per_image"), "recall": selected.get("recall"), "precision": selected.get("precision"), "f1": selected.get("f1")})
    return rows


def stage_comparison(gt_ds: Dataset, final_predictions: Mapping[str, Sequence[Instance]], raw_predictions: Mapping[str, Sequence[Instance]], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage_name, predictions in (("raw_or_candidate", raw_predictions), ("final", final_predictions)):
        coverage = candidate_coverage(predictions, gt_ds.gt, float(config.get("candidate_threshold", 0.001)), config.get("candidate_iou_thresholds", [0.3, 0.5, 0.75]))
        temp = Dataset(images=gt_ds.images, gt=gt_ds.gt, predictions=defaultdict(list, {k: list(v) for k, v in predictions.items()}))
        operating = evaluate_operating(temp, config)["aggregate"]
        rows.append({"stage": stage_name, "prediction_count": sum(len(v) for v in predictions.values()), "candidate_gt_count": coverage.get("gt_count"), "candidate_mean_best_same_class_iou": coverage.get("mean_best_same_class_iou"), "candidate_recall_iou_0.3": coverage.get("candidate_recall_iou_0.3"), "candidate_recall_iou_0.5": coverage.get("candidate_recall_iou_0.5"), "candidate_recall_iou_0.75": coverage.get("candidate_recall_iou_0.75"), "operating_recall": operating.get("recall"), "operating_precision": operating.get("precision"), "operating_fp_per_image": operating.get("mean_fp_per_image")})
    if len(rows) == 2:
        raw, final = rows
        rows.append({"stage": "final_minus_raw", **{key: (final.get(key) - raw.get(key)) if isinstance(final.get(key), (int, float)) and isinstance(raw.get(key), (int, float)) else None for key in final if key not in {"stage"}}})
    return rows


def _bootstrap_ci(ds: Dataset, config: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(config.get("bootstrap_images", 0) or 0)
    if count <= 0:
        return {}
    image_ids = sorted(set(ds.images) | set(ds.gt) | set(ds.predictions))
    if not image_ids:
        return {}
    rng = random.Random(int(config.get("seed", 0)))
    samples: Dict[str, List[float]] = defaultdict(list)
    for _ in range(count):
        chosen = [rng.choice(image_ids) for _ in image_ids]
        boot = Dataset()
        for new_index, image_id in enumerate(chosen):
            new_id = f"bootstrap:{new_index}:{image_id}"
            info = ds.images.get(image_id, ImageInfo(image_id=image_id))
            boot.images[new_id] = ImageInfo(new_id, info.width, info.height, info.file_name, dict(info.extra))
            boot.gt[new_id] = [Instance(new_id, gt.bbox, gt.class_id, gt.score, f"{new_id}:{gt.instance_id}", gt.ignore, dict(gt.extra)) for gt in ds.gt.get(image_id, [])]
            boot.predictions[new_id] = [Instance(new_id, pred.bbox, pred.class_id, pred.score, f"{new_id}:{pred.instance_id}", pred.ignore, dict(pred.extra)) for pred in ds.predictions.get(image_id, [])]
        agg = evaluate_operating(boot, config)["aggregate"]
        for key in ("precision", "recall", "f1", "count_mae"):
            if agg.get(key) is not None:
                samples[key].append(float(agg[key]))
    result = {}
    for key, values in samples.items():
        result[f"{key}_ci95_low"] = _quantile(values, 0.025)
        result[f"{key}_ci95_high"] = _quantile(values, 0.975)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                normalized[key] = value
            writer.writerow(normalized)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(output_dir: Path, config: Mapping[str, Any], summary: Mapping[str, Any], ap: Mapping[str, Any], stage_rows: Sequence[Mapping[str, Any]], warnings: Sequence[str]) -> None:
    agg = summary["aggregate"]
    lines = ["# 通用目标检测诊断报告", "", "本报告只评估检测质量与错误来源，不包含参数量、FLOPs、显存或延迟。模型框架通过统一预测格式或 predictor adapter 接入。", "", "## 运行配置", "", f"- 工作分数阈值：`{config.get('score_threshold')}`", f"- 候选保留阈值：`{config.get('candidate_threshold')}`", f"- 正式匹配 IoU：`{config.get('match_iou')}`", f"- 图像数：`{agg.get('image_count')}`；GT 数：`{agg.get('gt_count')}`", "", "## 主要结果", "", "| 指标 | 值 |", "|---|---:|", f"| Precision | {_format_value(agg.get('precision'))} |", f"| Recall | {_format_value(agg.get('recall'))} |", f"| F1 | {_format_value(agg.get('f1'))} |", f"| MAE（每图计数） | {_format_value(agg.get('count_mae'))} |", f"| 计数偏差（预测−GT） | {_format_value(agg.get('count_bias_pred_minus_gt'))} |", f"| 每图平均 FP | {_format_value(agg.get('mean_fp_per_image'))} |", f"| 每图平均 FN | {_format_value(agg.get('mean_fn_per_image'))} |", f"| 已匹配目标平均分数 | {_format_value(agg.get('mean_matched_score'))} |", "", "## AP", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("AP50", "AP75", "mAP50_95"):
        lines.append(f"| {key} | {_format_value(ap.get(key))} |")
    lines += ["", "## 低阈值候选诊断", "", "| 指标 | 值 |", "|---|---:|"]
    for threshold in config.get("candidate_iou_thresholds", [0.3, 0.5, 0.75]):
        threshold = float(threshold)
        lines.append(f"| Recall@IoU={threshold:g}（一对一匹配） | {_format_value(agg.get(f'recall_at_iou_{threshold:g}'))} |")
        lines.append(f"| 候选覆盖@IoU={threshold:g}（GT 最佳候选） | {_format_value(agg.get(f'candidate_coverage_iou_{threshold:g}'))} |")
    lines += [f"| 低置信度可找回 GT 数 | {_format_value(agg.get('low_confidence_recoverable_gt_count'))} |", f"| 低置信度可找回 GT 占比 | {_format_value(agg.get('low_confidence_recoverable_gt_rate'))} |"]
    lines += ["", "## 错误计数", "", "| 类型 | 数量 |", "|---|---:|"]
    for key, value in sorted((agg.get("error_counts") or {}).items()):
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 如何确认原因", "", "1. 先看 `per_gt.csv`：区分没有候选、定位不足和类别错误。", "2. 再看 `group_metrics.csv`：比较小目标、拥挤目标、目标密度和类别分组。", "3. 看 `threshold_sweep.csv` 与 `fp_budget.csv`：确认是否只是阈值导致的低召回。", "4. 传入 `--raw-pred` 后查看 `stage_comparison.csv`：只有 raw 候选明显好于 final，才支持后处理或输出筛选造成损失。", "5. 任何结构原因都需要进一步做分辨率、标签复核、消融或特征探针实验；本报告本身只证明可观察到的错误现象。"]
    if stage_rows:
        lines += ["", "## 候选/后处理阶段对照", "", "详见 `stage_comparison.csv`。raw 与 final 的差值用于定位候选筛选造成的损失，不代表某个特定算法（例如 NMS），除非适配器明确提供了该阶段。"]
    if warnings:
        lines += ["", "## 数据警告", ""]
        lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_prediction_dataset(gt_ds: Dataset, predictions: Mapping[str, Sequence[Instance]], warnings: Sequence[str]) -> Dataset:
    return Dataset(images=gt_ds.images, gt=gt_ds.gt, predictions=defaultdict(list, {key: list(value) for key, value in predictions.items()}), warnings=list(warnings), class_names=gt_ds.class_names)


def _load_predictor(spec: str):
    if ":" not in spec:
        raise ValueError("--predictor 使用 module:function 格式")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec} 不是可调用函数")
    return function


def generate_predictions_from_adapter(gt_ds: Dataset, images_dir: Path, spec: str) -> Dict[str, List[Instance]]:
    predictor = _load_predictor(spec)
    predictions: Dict[str, List[Instance]] = defaultdict(list)
    for image_id, info in sorted(gt_ds.images.items()):
        source_path = info.extra.get("source_path") if isinstance(info.extra, Mapping) else None
        image_path = Path(str(source_path)) if source_path else (_find_image(images_dir, image_id, info.file_name) if images_dir is not None else None)
        if image_path is None:
            raise FileNotFoundError(f"找不到 image_id={image_id} 对应图片；请检查 data.yaml、--images-dir 或 file_name")
        result = predictor(str(image_path), {"image_id": image_id, "width": info.width, "height": info.height, **info.extra})
        if result is None:
            result = []
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            raise TypeError(f"predictor 对 {image_id} 的返回值必须是预测字典列表")
        for index, obj in enumerate(result):
            if not isinstance(obj, Mapping):
                raise TypeError(f"predictor 对 {image_id} 返回了非字典实例: {obj!r}")
            predictions[image_id].append(_instance_from_object(obj, image_id, "xyxy", index, True))
    return predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Framework-independent object detection diagnostics")
    parser.add_argument("--gt", required=True, type=Path, help="GT：Ultralytics data.yaml/.ndjson、canonical JSON/JSONL、COCO JSON 或 YOLO 标签目录")
    parser.add_argument("--pred", type=Path, help="预测文件：canonical JSON/JSONL 或 COCO results；使用 --predictor 时可省略")
    parser.add_argument("--raw-pred", type=Path, help="可选 raw/candidate 预测，用于阶段损失对照")
    parser.add_argument("--predictor", help="可选 module:function，生成 canonical 预测")
    parser.add_argument("--images-dir", type=Path, help="旧式 YOLO GT 或 --predictor 使用的图片目录；data.yaml 通常不需要")
    parser.add_argument("--gt-format", default="auto", choices=["auto", "canonical", "coco", "yolo", "ultralytics", "yolo_yaml", "ndjson"])
    parser.add_argument("--pred-format", default="auto", choices=["auto", "canonical", "coco"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Ultralytics data.yaml 使用的 split")
    parser.add_argument("--config", type=Path, help="YAML 或 JSON 配置文件；未提供时使用内置默认值")
    parser.add_argument("--output", type=Path, default=Path("runs/model-diagnostics"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.pred is None and args.predictor is None:
        raise SystemExit("必须提供 --pred 或 --predictor 之一")
    if args.predictor and args.images_dir is None:
        raise SystemExit("使用 --predictor 时必须提供 --images-dir")
    config = load_config(args.config)
    gt_ds = load_ground_truth(args.gt, args.gt_format, args.images_dir, split=args.split)
    warnings = list(gt_ds.warnings)
    if args.pred is not None:
        predictions, pred_warnings = load_predictions(args.pred, gt_ds.images, args.pred_format)
        warnings.extend(pred_warnings)
    else:
        predictions = generate_predictions_from_adapter(gt_ds, args.images_dir, args.predictor)
    ds = _merge_prediction_dataset(gt_ds, predictions, warnings)
    ds.warnings.extend(_validate_dataset(ds))
    args.output.mkdir(parents=True, exist_ok=True)

    summary = evaluate_operating(ds, config)
    ap = compute_ap_metrics(ds, config.get("ap_iou_thresholds", DEFAULT_CONFIG["ap_iou_thresholds"]))
    sweep = threshold_sweep(ds, config)
    budget = fp_budget_rows(sweep, config.get("fp_budgets_per_image", DEFAULT_CONFIG["fp_budgets_per_image"]))
    groups = group_metrics(summary["per_gt"])
    stage_rows: List[Dict[str, Any]] = []
    if args.raw_pred:
        raw_predictions, raw_warnings = load_predictions(args.raw_pred, gt_ds.images, args.pred_format)
        ds.warnings.extend(raw_warnings)
        stage_rows = stage_comparison(gt_ds, predictions, raw_predictions, config)
    ci = _bootstrap_ci(ds, config)
    summary["aggregate"].update(ci)

    _write_json(args.output / "summary.json", {"config": config, "aggregate": summary["aggregate"], "area_quantiles": summary["area_quantiles"], "ap": ap, "warnings": ds.warnings})
    _write_csv(args.output / "per_image.csv", summary["per_image"])
    _write_csv(args.output / "per_gt.csv", summary["per_gt"])
    _write_csv(args.output / "per_prediction.csv", summary["per_prediction"])
    _write_csv(args.output / "group_metrics.csv", groups)
    _write_csv(args.output / "threshold_sweep.csv", sweep)
    _write_csv(args.output / "fp_budget.csv", budget)
    if stage_rows:
        _write_csv(args.output / "stage_comparison.csv", stage_rows)
    _write_json(args.output / "config_used.json", config)
    write_report(args.output, config, summary, ap, stage_rows, ds.warnings)
    print(f"诊断完成：{args.output.resolve()}")
    print(f"Precision={_format_value(summary['aggregate'].get('precision'))} Recall={_format_value(summary['aggregate'].get('recall'))} F1={_format_value(summary['aggregate'].get('f1'))} mAP50={_format_value(ap.get('AP50'))} mAP50-95={_format_value(ap.get('mAP50_95'))}")
    return 0


def _validate_dataset(ds: Dataset) -> List[str]:
    warnings = []
    missing_dimensions = [image_id for image_id, info in ds.images.items() if not info.width or not info.height]
    if missing_dimensions:
        warnings.append(f"{len(missing_dimensions)} 张图片缺少 width/height；相对面积指标可能为 N/A")
    for image_id, predictions in ds.predictions.items():
        for pred in predictions:
            if not 0.0 <= _pred_score(pred) <= 1.0:
                warnings.append(f"{image_id}/{pred.instance_id} 的 score={pred.score} 不在 [0,1]，请确认模型输出是否已经过 sigmoid/softmax")
            if box_area(pred.bbox) <= 0:
                warnings.append(f"{image_id}/{pred.instance_id} 的预测框面积 <= 0")
    for image_id, gts in ds.gt.items():
        for gt in gts:
            if box_area(gt.bbox) <= 0:
                warnings.append(f"{image_id}/{gt.instance_id} 的 GT 框面积 <= 0")
    return sorted(set(warnings))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TypeError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
