#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化工具的配置、输入变换、运行目录和文件输出。"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def write_image(path: Path, image: np.ndarray) -> Path:
    """Encode completely before replacing an existing image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"图像编码失败：{path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def safe_name(value: Any, fallback: str = "item") -> str:
    text = "".join(char if (char.isalnum() or char in "_.-") else "_" for char in str(value or fallback)).strip("._")
    return text or fallback


def image_output_name(image_id: str) -> str:
    """Create a readable, collision-resistant directory name for one image ID."""
    digest = hashlib.sha1(str(image_id).encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(image_id)}_{digest}"


def resolve_device(value: Any):
    import torch

    text = str(value or "auto").strip().lower()
    if text in {"auto", ""}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"配置要求 {text}，但当前 PyTorch 没有可用 CUDA")
    return torch.device(text)


def parse_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("input.size 必须是 [height, width]")
        size = int(value[0]), int(value[1])
        if min(size) <= 0:
            raise ValueError("input.size 的高和宽必须大于 0")
        return size
    text = str("832" if value is None or value == "" else value).lower().replace(" ", "")
    if "x" in text:
        h, w = text.split("x", 1)
        size = int(h), int(w)
        if min(size) <= 0:
            raise ValueError("input.size 的高和宽必须大于 0")
        return size
    number = int(text)
    if number <= 0:
        raise ValueError("input.size 必须大于 0")
    return number, number


def _resolve(value: Any, base: Path) -> Optional[Path]:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path | None = None, *, allow_missing_source: bool = False) -> dict[str, Any]:
    import yaml

    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到可视化配置：{config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("可视化配置顶层必须是 YAML 对象")
    for section in ("features", "cam", "stage_trace"):
        if isinstance(raw.get(section), Mapping) and "enabled" in raw[section]:
            raise ValueError(f"已移除旧版配置字段：{section}.enabled；请使用 modules.visualization.{section}")
    if not isinstance(raw.get("modules"), Mapping):
        raise ValueError("可视化配置必须包含 modules: module_id: true/false")
    config: dict[str, Any] = json.loads(json.dumps(raw))
    config["_config_path"] = str(config_path)
    project = dict(config.get("project") or {})
    root = _resolve(project.get("root", config_path.parents[1]), config_path.parent) or config_path.parents[1]
    project["root"] = str(root)
    if project.get("ultralytics_repo"):
        project["ultralytics_repo"] = str(_resolve(project["ultralytics_repo"], root) or "")
    config["project"] = project

    model = dict(config.get("model") or {})
    for key in ("weights", "adapter"):
        if model.get(key) and (key == "weights" or str(model[key]).endswith(".py") or "/" in str(model[key])):
            resolved = _resolve(model[key], root)
            if resolved is not None:
                model[key] = str(resolved)
    model.setdefault("adapter", "ultralytics")
    model.setdefault("task", "detect")
    model.setdefault("device", "auto")
    model.setdefault("precision", "fp32")
    model.setdefault("fuse", False)
    config["model"] = model

    input_values = dict(config.get("input") or {})
    source = _resolve(input_values.get("source"), root)
    if source is None and not allow_missing_source and str(config.get("mode", "visualize")).lower() != "list_layers":
        raise ValueError("请在 input.source 中填写图片或图片目录")
    dataset_root = _resolve(input_values.get("dataset_root"), root)
    input_values["dataset_root"] = str(dataset_root) if dataset_root is not None else None
    input_values["source"] = str(source) if source is not None else None
    input_values["size"] = list(parse_size(input_values.get("size", [832, 832])))
    if input_values.get("max_images") is not None:
        input_values["max_images"] = int(input_values["max_images"])
    config["input"] = input_values

    detection = dict(config.get("detection") or {})
    detection.setdefault("conf", 0.001)
    detection.setdefault("iou", 0.70)
    detection.setdefault("max_det", 300)
    detection["conf"] = float(detection["conf"])
    detection["iou"] = float(detection["iou"])
    detection["max_det"] = int(detection["max_det"])
    if not math.isfinite(detection["conf"]) or not 0.0 <= detection["conf"] <= 1.0:
        raise ValueError("detection.conf 必须是 0 到 1 之间的有限数值")
    if not math.isfinite(detection["iou"]) or not 0.0 <= detection["iou"] <= 1.0:
        raise ValueError("detection.iou 必须是 0 到 1 之间的有限数值")
    if detection["max_det"] <= 0:
        raise ValueError("detection.max_det 必须大于 0")
    config["detection"] = detection

    run = dict(config.get("run") or {})
    run_root = _resolve(run.get("root", "model_visualization/runs"), root) or root / "model_visualization" / "runs"
    run["root"] = str(run_root)
    run.setdefault("name", "auto")
    config["run"] = run
    return config


def next_run_dir(config: Mapping[str, Any]) -> Path:
    values = dict(config.get("run") or {})
    root = Path(str(values.get("root"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = str(values.get("name", "auto") or "auto").strip()
    if name.lower() != "auto":
        target = root / safe_name(name)
        target.mkdir(parents=True, exist_ok=False)
        return target
    index = 1
    while True:
        target = root / f"run{index}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            index += 1


def list_images(source: str | Path, max_images: int | None = None) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        images = [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    elif path.is_dir():
        images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f"图片输入不存在：{path}")
    if not images:
        raise FileNotFoundError(f"没有在输入中找到图片：{path}")
    return images if max_images is None or max_images <= 0 else images[:max_images]


@dataclass
class TransformMeta:
    source_path: str
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    ratio: float
    padding_value: int = 114
    color_order: str = "BGR_to_RGB"
    normalization: str = "divide_by_255"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def letterbox(image: np.ndarray, size: tuple[int, int], source_path: str) -> tuple[np.ndarray, TransformMeta]:
    target_h, target_w = size
    orig_h, orig_w = image.shape[:2]
    ratio = min(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    resized_w = max(1, int(round(orig_w * ratio)))
    resized_h = max(1, int(round(orig_h * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_left = max(0, (target_w - resized_w) // 2)
    pad_top = max(0, (target_h - resized_h) // 2)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    meta = TransformMeta(
        source_path=str(Path(source_path).resolve()),
        original_height=orig_h,
        original_width=orig_w,
        input_height=target_h,
        input_width=target_w,
        resized_height=resized_h,
        resized_width=resized_w,
        pad_top=pad_top,
        pad_left=pad_left,
        ratio=ratio,
    )
    return canvas, meta


def image_tensor(image_bgr: np.ndarray, device: Any, precision: str):
    import torch

    rgb = image_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    target_device = torch.device(device)
    precision_text = str(precision).lower()
    if precision_text in {"bf16", "bfloat16"} and target_device.type == "cuda":
        target_dtype = torch.bfloat16
    elif precision_text in {"fp16", "half"} and target_device.type == "cuda":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32
    return tensor.to(device=target_device, dtype=target_dtype)


def restore_map(input_map: np.ndarray, meta: TransformMeta) -> np.ndarray:
    resized = cv2.resize(input_map, (meta.input_width, meta.input_height), interpolation=cv2.INTER_LINEAR)
    y1, x1 = meta.pad_top, meta.pad_left
    y2, x2 = y1 + meta.resized_height, x1 + meta.resized_width
    cropped = resized[y1:y2, x1:x2]
    if cropped.size == 0:
        return cv2.resize(resized, (meta.original_width, meta.original_height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(cropped, (meta.original_width, meta.original_height), interpolation=cv2.INTER_LINEAR)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    result: dict[str, Any] = {
        "elements": int(values.size),
        "finite": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.any():
        valid = values[finite]
        result.update(min=float(valid.min()), max=float(valid.max()), mean=float(valid.mean()), std=float(valid.std()))
    return result


def normalize_map(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    cleaned = np.where(finite, data, 0.0)
    lo, hi = float(cleaned[finite].min()), float(cleaned[finite].max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip((cleaned - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def colorize(values: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_map(values), cv2.COLORMAP_JET)


def blend_map(image_bgr: np.ndarray, values: np.ndarray, meta: TransformMeta, alpha: float = 0.5) -> np.ndarray:
    restored = restore_map(normalize_map(values), meta)
    heat = cv2.applyColorMap(restored, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat, alpha, 0.0)


def write_runtime_state(path: Path, requested: Mapping[str, Any], effective: Mapping[str, Any], source: Mapping[str, Any], status: str = "validated") -> None:
    write_json(path, {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested": dict(requested),
        "effective": dict(effective),
        "source": dict(source),
        "status": status,
    })


def append_html_index(run_dir: Path, records: list[Mapping[str, Any]]) -> None:
    """写不依赖外部资源的简单离线索引。"""
    cards = []
    for record in records:
        image = html.escape(str(record.get("image", "")))
        links = " ".join(
            f'<a href="{html.escape(str(x), quote=True)}">{html.escape(str(x))}</a>'
            for x in record.get("links", [])
        )
        cards.append(f"<article><h2>{image}</h2><p>{links}</p></article>")
    html_text = "<!doctype html><meta charset='utf-8'><title>vtools 可视化报告</title><style>body{font-family:sans-serif}article{border-bottom:1px solid #ddd;padding:1em}a{margin-right:1em}</style>" + "".join(cards)
    (run_dir / "index.html").write_text(html_text, encoding="utf-8")


__all__ = [
    "TransformMeta",
    "append_html_index",
    "blend_map",
    "colorize",
    "finite_stats",
    "image_tensor",
    "letterbox",
    "list_images",
    "load_config",
    "next_run_dir",
    "normalize_map",
    "parse_size",
    "resolve_device",
    "restore_map",
    "safe_name",
    "image_output_name",
    "write_csv",
    "write_image",
    "write_json",
    "write_runtime_state",
]
