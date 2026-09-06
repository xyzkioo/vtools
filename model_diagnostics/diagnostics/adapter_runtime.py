"""Run a vtools-style detection adapter and normalize its predictions."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

from core.vision_benchmark_common import (
    BenchmarkConfig,
    InputBundle,
    dtype_for_precision,
    normalize_precision,
    resolve_device,
)


def _load_module(spec: str, project_root: Path) -> ModuleType:
    path = Path(spec).expanduser()
    if not path.is_absolute():
        candidate = project_root / path
        if candidate.is_file():
            path = candidate
    if path.is_file():
        name = f"_diagnostics_adapter_{abs(hash(path.resolve()))}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"无法加载 adapter 文件：{path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def _call(function: Any, values: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**values)
    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return function(**values)
    accepted = {
        name: value
        for name, value in values.items()
        if name in parameters and parameters[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**accepted)


def _device(value: Any) -> Any:
    return resolve_device(value)


def _input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("benchmark.input_size 必须是 [height, width]")
        return int(value[0]), int(value[1])
    text = str(value or "832").lower().replace(" ", "")
    if "x" in text:
        h, w = text.split("x", 1)
        return int(h), int(w)
    return int(text), int(text)


def make_benchmark_config(benchmark: Mapping[str, Any]) -> BenchmarkConfig:
    height, width = _input_size(benchmark.get("input_size", [832, 832]))
    return BenchmarkConfig(
        device=_device(benchmark.get("device", "auto")),
        batch_size=int(benchmark.get("batch_size", 1)),
        height=height,
        width=width,
        precision=normalize_precision(benchmark.get("precision", "fp32")),
        warmup=int(benchmark.get("warmup", 0)),
        repeats=int(benchmark.get("repeats", 1)),
        fuse=bool(benchmark.get("fuse", benchmark.get("fuse_model", False))),
        conf=float(benchmark.get("conf", 0.001)),
        iou=float(benchmark.get("iou", 0.70)),
        max_det=int(benchmark.get("max_det", 3000)),
    )


def _as_detection_list(value: Any) -> Optional[list[dict[str, Any]]]:
    """Accept canonical dicts and common list-like detection outputs."""
    if isinstance(value, Mapping):
        for key in ("predictions", "detections", "objects"):
            if key in value:
                return _as_detection_list(value[key])
        if "bbox" in value or "box" in value:
            return [dict(value)]
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, Mapping) and ("bbox" in item or "box" in item) for item in value):
            return [dict(item) for item in value]
    return None


def _normalize_detections(value: Any, image_id: str) -> list[dict[str, Any]]:
    rows = _as_detection_list(value)
    if rows is None:
        raise TypeError(
            "adapter 输出不是检测列表。请实现 postprocess_detections(outputs, source, image_info, config)，"
            "或让 predict() 直接返回 [{bbox, score, class_id}, ...]。"
        )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        bbox = item.get("bbox", item.get("box"))
        score = item.get("score", item.get("confidence", item.get("conf")))
        class_id = item.get("class_id", item.get("category_id", item.get("class", item.get("label", 0))))
        if bbox is None or score is None:
            raise ValueError(f"{image_id} 的第 {index} 个检测缺少 bbox 或 score")
        normalized.append({
            "instance_id": str(item.get("instance_id", item.get("id", f"{image_id}:{index}"))),
            "bbox": [float(x) for x in bbox],
            "score": float(score),
            "class_id": int(class_id),
        })
    return normalized


class VToolsDetectionAdapter:
    """Adapter loader using the same hook names as vtools/speed_test."""

    def __init__(self, spec: str, project_root: Path, task: str = "detect") -> None:
        self.spec = spec
        self.project_root = project_root
        self.task = task
        self.module: Optional[ModuleType] = None
        self.model: Any = None
        self.config: Optional[BenchmarkConfig] = None

    def load(self, weights: Path, config: BenchmarkConfig) -> None:
        self.config = config
        if self.spec.lower() == "ultralytics":
            self.module = None
            self.model = self._load_ultralytics(weights, config)
            return
        if self.spec.lower() == "checkpoint":
            self.module = None
            self.model = self._load_checkpoint(weights, config)
            return
        self.module = _load_module(self.spec, self.project_root)
        function = getattr(self.module, "build_model", None)
        if not callable(function):
            raise AttributeError(f"adapter {self.spec} 缺少 vtools 约定的 build_model(weights, device)")
        self.model = _call(function, {"weights": weights, "weights_path": str(weights), "device": config.device, "device_str": str(config.device)})
        prepare = getattr(self.module, "prepare_model", None)
        if callable(prepare):
            prepared = _call(prepare, {"model": self.model, "device": config.device, "precision": config.precision, "fuse": config.fuse})
            if prepared is not None:
                self.model = prepared
        else:
            self._default_prepare()

    @staticmethod
    def _load_checkpoint(weights: Path, config: BenchmarkConfig) -> Any:
        """Load the common full-module checkpoint forms without a framework import."""
        import torch

        try:
            checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6 has no weights_only parameter.
            checkpoint = torch.load(str(weights), map_location="cpu")
        model = checkpoint if hasattr(checkpoint, "forward") else None
        if model is None and isinstance(checkpoint, Mapping):
            for key in ("model", "ema"):
                candidate = checkpoint.get(key)
                if hasattr(candidate, "forward"):
                    model = candidate
                    break
        if model is None:
            raise RuntimeError(
                "adapter: checkpoint 只能直接加载完整 nn.Module，"
                "state_dict 请复制 adapters/vision_adapter_template.py 后实现 build_model()"
            )
        return model

    def _default_prepare(self) -> None:
        """Apply the same basic eval/device/dtype preparation as vtools."""
        if self.config is None:
            return
        target = self.model
        if hasattr(target, "model") and hasattr(target.model, "to"):
            target = target.model
        if hasattr(target, "to"):
            target.to(self.config.device)
        if hasattr(target, "eval"):
            target.eval()
        if self.config.precision == "fp16" and hasattr(target, "half"):
            target.half()
        elif self.config.precision == "bf16" and hasattr(target, "bfloat16"):
            target.bfloat16()

    @staticmethod
    def _load_ultralytics(weights: Path, config: BenchmarkConfig) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("使用 adapter: ultralytics 前请安装 ultralytics") from exc
        return YOLO(str(weights), task="detect")

    def _predict_ultralytics(self, source: Any, image_id: str) -> list[dict[str, Any]]:
        results = self.model.predict(source=source, imgsz=self.config.image_size, conf=self.config.conf, iou=self.config.iou, max_det=self.config.max_det, device=str(self.config.device), save=False, verbose=False)
        rows: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            rows.extend({"instance_id": f"{image_id}:{i}", "bbox": box, "score": score, "class_id": int(cls)} for i, (box, score, cls) in enumerate(zip(xyxy, scores, classes)))
        return rows

    def predict(self, source: Any, image_id: str, image_info: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.spec.lower() == "ultralytics":
            return self._predict_ultralytics(source, image_id)
        if self.module is None or self.config is None:
            raise RuntimeError("adapter 尚未加载")
        predict = getattr(self.module, "predict", None)
        values = {"model": self.model, "source": source, "config": self.config, "image_info": dict(image_info), "device": self.config.device, "precision": self.config.precision, "batch_size": self.config.batch_size, "image_size": self.config.image_size}
        if callable(predict):
            outputs = _call(predict, values)
        else:
            make_inputs = getattr(self.module, "make_inputs", None)
            if not callable(make_inputs):
                raise AttributeError(f"adapter {self.spec} 既没有 predict，也没有 make_inputs")
            dtype = dtype_for_precision(self.config.precision, self.config.device)
            bundle = InputBundle.from_value(_call(make_inputs, {"batch_size": 1, "batch": 1, "image_size": self.config.image_size, "imgsz": self.config.image_size, "device": self.config.device, "dtype": dtype, "source": source, "for_export": False}), batch_size=1)
            outputs = self.model(*bundle.args, **bundle.kwargs)
        if outputs is None:
            return []
        direct = _as_detection_list(outputs)
        if direct is not None:
            return _normalize_detections(direct, image_id)
        postprocess = getattr(self.module, "postprocess_detections", getattr(self.module, "to_detections", None))
        if not callable(postprocess):
            raise TypeError(f"adapter {self.spec} 的 predict 输出需要 postprocess_detections() 转成检测列表")
        detections = _call(postprocess, {"outputs": outputs, "source": source, "image_info": dict(image_info), "config": self.config, "model": self.model})
        if detections is None:
            return []
        return _normalize_detections(detections, image_id)


def create_adapter(spec: str, project_root: Path, task: str = "detect") -> VToolsDetectionAdapter:
    return VToolsDetectionAdapter(spec, project_root, task)
