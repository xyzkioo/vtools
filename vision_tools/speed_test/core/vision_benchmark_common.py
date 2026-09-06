#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 PyTorch 视觉模型测速核心。

这个文件只负责测速、计时、统计和适配器加载，不假定模型一定是 YOLO、
检测模型或具有某种固定输出格式。模型的构建、输入和可选后处理由
``BaseAdapter`` 或用户提供的 adapter 模块完成。

用户 adapter 模块至少可以实现：

    build_model(weights, device)
    make_inputs(batch_size, image_size, device, dtype, source=None)

可选实现：run、predict、prepare_model、count_outputs、export_onnx。
详见 ``adapters/vision_adapter_template.py``。
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
from torch import nn

try:
    import cv2
except ImportError:  # 纯模型调用不需要 OpenCV，只有 --source 图片时才需要它
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelSpec:
    """一个待测试模型。"""

    name: str
    weights: Path


@dataclass
class InputBundle:
    """统一表示 ``model(*args, **kwargs)`` 的输入。

    不同 PyTorch 视觉模型的输入签名差异很大：分类模型通常接收一个
    ``Tensor``，torchvision 检测模型可能接收 ``list[Tensor]``，多模态模型
    还可能需要多个张量。用这个对象可以避免测速核心猜测输入结构。
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    description: str = "custom"

    @classmethod
    def from_value(
        cls,
        value: Any,
        batch_size: int = 1,
        description: str = "custom",
    ) -> "InputBundle":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "args" in value:
            raw_args = value.get("args", ())
            if isinstance(raw_args, tuple):
                args = raw_args
            elif isinstance(raw_args, list):
                args = (raw_args,)
            else:
                args = (raw_args,)
            raw_kwargs = value.get("kwargs", {})
            return cls(
                args=args,
                kwargs=dict(raw_kwargs),
                batch_size=int(value.get("batch_size", batch_size)),
                description=str(value.get("description", description)),
            )
        if isinstance(value, torch.Tensor):
            return cls((value,), {}, batch_size, description)
        # tuple 通常表示多个位置参数；list 通常是检测模型的一整个 images 参数。
        if isinstance(value, tuple):
            return cls(value, {}, batch_size, description)
        return cls((value,), {}, batch_size, description)


@dataclass
class BenchmarkConfig:
    device: torch.device
    batch_size: int
    height: int
    width: int
    precision: str = "fp32"
    warmup: int = 30
    repeats: int = 100
    fuse: bool = False
    conf: float = 0.25
    iou: float = 0.70
    max_det: int = 300

    @property
    def image_size(self) -> tuple[int, int]:
        return self.height, self.width


def resolve_device(value: str) -> torch.device:
    """解析 ``auto``、``cuda:0``、``0``、``cpu`` 等设备写法。"""
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 没有可用的 CUDA 设备")
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA 设备 {index} 不存在；当前设备数为 {torch.cuda.device_count()}"
            )
    return device


def normalize_precision(value: str) -> str:
    aliases = {
        "32": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"不支持的精度：{value}，可选 fp32、fp16、bf16")
    return aliases[key]


def dtype_for_precision(precision: str, device: torch.device) -> torch.dtype:
    precision = normalize_precision(precision)
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 测速默认只在 CUDA 上启用；CPU 请使用 fp32 或 bf16")
        return torch.float16
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 CUDA 设备不支持 BF16")
        return torch.bfloat16
    return torch.float32


def statistics(values: list[float]) -> dict[str, float]:
    """计算稳定的延迟统计量，单位为毫秒。"""
    if not values:
        raise ValueError("没有可用于统计的测速结果")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "std_ms": float(array.std(ddof=0)),
    }


def _call_user_function(function: Callable[..., Any], values: dict[str, Any]) -> Any:
    """向用户函数传递它声明过的参数，兼容不同 adapter 风格。"""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**values)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return function(**values)

    filtered = {
        name: value
        for name, value in values.items()
        if name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**filtered)


def _module_from_spec(spec: str) -> ModuleType:
    """加载 ``/path/adapter.py`` 或 ``package.module``。"""
    path = Path(spec).expanduser()
    if path.is_file():
        module_name = f"_vision_adapter_{abs(hash(path.resolve()))}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"无法加载 adapter 文件：{path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def _move_to_device(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    """递归移动用户 adapter 返回的输入。"""
    if isinstance(value, torch.Tensor):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device, dtype) for key, item in value.items()}
    return value


def _module_parameter_count(model: Any) -> Optional[int]:
    candidate = model.model if hasattr(model, "model") and isinstance(model.model, nn.Module) else model
    if not isinstance(candidate, nn.Module):
        return None
    return int(sum(parameter.numel() for parameter in candidate.parameters()))


class BaseAdapter:
    """模型适配器基类。

    普通模型只需继承它并实现 ``load_model``、``make_inputs``；也可以使用
    用户模块适配器，让脚本通过反射调用同名函数。
    """

    adapter_name = "generic"
    supports_real_predict = False

    def load_model(self, weights: Path, device: torch.device) -> Any:
        raise NotImplementedError

    def forward_model(self, model: Any) -> Any:
        return model

    def prepare_model(self, model: Any, device: torch.device, precision: str, fuse: bool = False) -> Any:
        target = self.forward_model(model)
        if isinstance(target, nn.Module):
            target.to(device=device).eval()
            if fuse:
                fuse_method = getattr(target, "fuse", None)
                if callable(fuse_method):
                    try:
                        fuse_method()
                    except TypeError:
                        fuse_method(verbose=False)
            if precision == "fp16":
                target.half()
            elif precision == "bf16":
                target.bfloat16()
        return model

    def make_inputs(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        source: Any = None,
        for_export: bool = False,
    ) -> InputBundle:
        del source, for_export
        generator = torch.Generator(device="cpu").manual_seed(0)
        tensor = torch.rand(
            (batch_size, 3, image_size[0], image_size[1]),
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        return InputBundle((tensor,), batch_size=batch_size, description="NCHW random tensor")

    def run(self, model: Any, inputs: InputBundle) -> Any:
        target = self.forward_model(model)
        return target(*inputs.args, **inputs.kwargs)

    def predict(self, model: Any, source: Any, config: BenchmarkConfig) -> Any:
        """默认 pipeline：adapter 输入构造 + forward + 可选后处理。"""
        dtype = dtype_for_precision(config.precision, config.device)
        inputs = self.make_inputs(
            config.batch_size,
            config.image_size,
            config.device,
            dtype,
            source=source,
        )
        return self.postprocess(self.run(model, inputs))

    def postprocess(self, outputs: Any) -> Any:
        return outputs

    def count_outputs(self, outputs: Any) -> Optional[int]:
        del outputs
        return None

    def parameter_count(self, model: Any) -> Optional[int]:
        return _module_parameter_count(self.forward_model(model))

    def export_onnx(
        self,
        model: Any,
        output_path: Path,
        inputs: InputBundle,
        opset: int,
        dynamic: bool,
    ) -> Path:
        """默认的 ONNX 导出；复杂模型可在 adapter 中覆盖。"""
        del dynamic
        export_model = self.forward_model(model)
        if not isinstance(export_model, nn.Module):
            raise TypeError("当前 adapter 没有提供可导出的 nn.Module")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = _TupleOutputWrapper(export_model)
        # wrapper 本身默认处于 train 模式；显式 eval，避免 exporter 误报并保证
        # BatchNorm/Dropout 等层与测速时的推理状态一致。
        wrapper.eval()
        input_names = [f"input_{index}" for index in range(len(inputs.args))]
        try:
            torch.onnx.export(
                wrapper,
                args=inputs.args,
                kwargs=inputs.kwargs,
                f=str(output_path),
                input_names=input_names,
                opset_version=opset,
                dynamo=True,
                training=torch.onnx.TrainingMode.EVAL,
            )
        except (TypeError, RuntimeError) as error:
            # 老版本 PyTorch 不支持 dynamo 参数时回退到传统导出器。
            if "dynamo" not in str(error).lower() and "unexpected keyword" not in str(error).lower():
                raise
            torch.onnx.export(
                wrapper,
                inputs.args,
                str(output_path),
                input_names=input_names,
                opset_version=opset,
                training=torch.onnx.TrainingMode.EVAL,
            )
        return output_path


class _TupleOutputWrapper(nn.Module):
    """把 tensor/list/dict 输出扁平化成 ONNX 可导出的 tuple。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        outputs = self.model(*args, **kwargs)
        tensors: list[torch.Tensor] = []

        def visit(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(outputs)
        if not tensors:
            raise TypeError("模型输出中没有 Tensor，无法导出 ONNX")
        return tuple(tensors)


class CheckpointAdapter(BaseAdapter):
    """加载保存为完整 ``nn.Module`` 的 PyTorch checkpoint。"""

    adapter_name = "checkpoint"

    def load_model(self, weights: Path, device: torch.device) -> Any:
        try:
            checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容 PyTorch 2.5 及更早版本。
            checkpoint = torch.load(str(weights), map_location="cpu")
        if isinstance(checkpoint, nn.Module):
            return checkpoint.to(device)
        if isinstance(checkpoint, dict):
            # 不同训练框架可能把可推理模型放在 model 或 ema 字段；
            # 两者都支持，仍然不把 state_dict 当作完整模型。
            for key in ("model", "ema"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, nn.Module):
                    return candidate.to(device)
        raise RuntimeError(
            "默认 checkpoint adapter 只能直接加载完整 nn.Module。\n"
            "如果文件是 state_dict，请提供自定义 adapter.py，在 build_model() 中先构建网络再 load_state_dict。"
        )


class ModuleAdapter(BaseAdapter):
    """把用户提供的 adapter 模块映射为 BaseAdapter。"""

    adapter_name = "module"

    def __init__(self, module: ModuleType, spec: str) -> None:
        self.module = module
        self.spec = spec
        self.supports_real_predict = callable(getattr(module, "predict", None))

    def _function(self, name: str) -> Optional[Callable[..., Any]]:
        function = getattr(self.module, name, None)
        return function if callable(function) else None

    def load_model(self, weights: Path, device: torch.device) -> Any:
        function = self._function("build_model")
        if function is None:
            return CheckpointAdapter().load_model(weights, device)
        return _call_user_function(
            function,
            {
                "weights": weights,
                "weights_path": str(weights),
                "device": device,
                "device_str": str(device),
            },
        )

    def forward_model(self, model: Any) -> Any:
        function = self._function("forward_model")
        if function is not None:
            return _call_user_function(function, {"model": model})
        return super().forward_model(model)

    def prepare_model(self, model: Any, device: torch.device, precision: str, fuse: bool = False) -> Any:
        function = self._function("prepare_model")
        if function is not None:
            prepared = _call_user_function(
                function,
                {
                    "model": model,
                    "device": device,
                    "precision": precision,
                    "fuse": fuse,
                },
            )
            return model if prepared is None else prepared
        return super().prepare_model(model, device, precision, fuse)

    def make_inputs(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        source: Any = None,
        for_export: bool = False,
    ) -> InputBundle:
        function = self._function("make_inputs")
        if function is None:
            return super().make_inputs(batch_size, image_size, device, dtype, source, for_export)
        value = _call_user_function(
            function,
            {
                "batch_size": batch_size,
                "batch": batch_size,
                "image_size": image_size,
                "imgsz": image_size,
                "height": image_size[0],
                "width": image_size[1],
                "device": device,
                "dtype": dtype,
                "source": source,
                "for_export": for_export,
            },
        )
        bundle = InputBundle.from_value(value, batch_size=batch_size)
        bundle.args = tuple(_move_to_device(bundle.args, device, dtype))
        bundle.kwargs = _move_to_device(bundle.kwargs, device, dtype)
        return bundle

    def run(self, model: Any, inputs: InputBundle) -> Any:
        function = self._function("run")
        if function is None:
            return super().run(model, inputs)
        return _call_user_function(
            function,
            {
                "model": model,
                "inputs": inputs,
                "input": inputs,
                "x": inputs.args[0] if len(inputs.args) == 1 else inputs.args,
                "args": inputs.args,
                "kwargs": inputs.kwargs,
            },
        )

    def predict(self, model: Any, source: Any, config: BenchmarkConfig) -> Any:
        function = self._function("predict")
        if function is None:
            return super().predict(model, source, config)
        return _call_user_function(
            function,
            {
                "model": model,
                "source": source,
                "config": config,
                "device": config.device,
                "precision": config.precision,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
            },
        )

    def postprocess(self, outputs: Any) -> Any:
        function = self._function("postprocess")
        return outputs if function is None else function(outputs)

    def count_outputs(self, outputs: Any) -> Optional[int]:
        function = self._function("count_outputs")
        return None if function is None else function(outputs)

    def export_onnx(
        self,
        model: Any,
        output_path: Path,
        inputs: InputBundle,
        opset: int,
        dynamic: bool,
    ) -> Path:
        function = self._function("export_onnx")
        if function is None:
            return super().export_onnx(model, output_path, inputs, opset, dynamic)
        result = _call_user_function(
            function,
            {
                "model": model,
                "output_path": output_path,
                "inputs": inputs,
                "opset": opset,
                "dynamic": dynamic,
            },
        )
        return Path(str(result or output_path))


class UltralyticsAdapter(BaseAdapter):
    """兼容 Ultralytics YOLO/RT-DETR 的适配器，作为内置快捷入口。"""

    adapter_name = "ultralytics"
    supports_real_predict = True

    def __init__(self, task: str = "detect", conf: float = 0.25, iou: float = 0.70, max_det: int = 300) -> None:
        self.task = task
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def load_model(self, weights: Path, device: torch.device) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("使用 ultralytics adapter 前请安装 ultralytics") from error
        return YOLO(str(weights), task=self.task)

    def forward_model(self, model: Any) -> Any:
        return model.model if hasattr(model, "model") else model

    def prepare_model(self, model: Any, device: torch.device, precision: str, fuse: bool = False) -> Any:
        target = self.forward_model(model)
        if not isinstance(target, nn.Module):
            raise TypeError("Ultralytics 模型没有可用的 PyTorch nn.Module")
        target.to(device=device).eval()
        if fuse and callable(getattr(target, "fuse", None)):
            try:
                target.fuse(verbose=False)
            except TypeError:
                target.fuse()
        if precision == "fp16":
            target.half()
        elif precision == "bf16":
            target.bfloat16()
        return model

    def predict(self, model: Any, source: Any, config: BenchmarkConfig) -> Any:
        predict_source = source
        # 单张内存图不会因为传入 batch 参数就自动复制成 batch；显式构造
        # 图片列表，保证 CSV 中的 batch 与实际执行数量一致。
        if config.batch_size > 1:
            if isinstance(source, np.ndarray) and source.ndim in {2, 3}:
                predict_source = [source] * config.batch_size
            elif isinstance(source, (str, Path)) and Path(source).is_file():
                predict_source = [str(source)] * config.batch_size
        arguments: dict[str, Any] = {
            "source": predict_source,
            "imgsz": config.image_size,
            "batch": config.batch_size,
            "device": str(config.device),
            "rect": False,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "save": False,
            "verbose": False,
        }
        if config.precision == "fp16":
            arguments["quantize"] = 16
        elif config.precision == "bf16":
            arguments["quantize"] = 32
        try:
            return model.predict(**arguments)
        except TypeError as error:
            # 兼容旧版本 Ultralytics 的 half 参数。
            if "quantize" not in str(error).lower():
                raise
            arguments.pop("quantize", None)
            if config.precision == "fp16":
                arguments["half"] = True
            return model.predict(**arguments)

    def count_outputs(self, outputs: Any) -> Optional[int]:
        results = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        counts: list[int] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                try:
                    counts.append(int(len(boxes)))
                    continue
                except TypeError:
                    pass
            masks = getattr(result, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None:
                counts.append(int(masks.data.shape[0]))
        return int(round(float(np.mean(counts)))) if counts else None


def create_adapter(spec: str, task: str = "detect", conf: float = 0.25, iou: float = 0.70, max_det: int = 300) -> BaseAdapter:
    """根据 ``ultralytics``、``checkpoint`` 或 adapter 模块创建适配器。"""
    key = spec.strip().lower()
    if key in {"ultralytics", "yolo", "rtdetr"}:
        return UltralyticsAdapter(task=task, conf=conf, iou=iou, max_det=max_det)
    if key in {"checkpoint", "torch", "pytorch"}:
        return CheckpointAdapter()
    return ModuleAdapter(_module_from_spec(spec), spec)


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for value in values:
        if "=" not in value:
            path = Path(value).expanduser()
            specs.append(ModelSpec(path.stem, path))
            continue
        name, raw_path = value.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"模型参数格式应为 NAME=PATH，收到：{value}")
        specs.append(ModelSpec(name.strip(), Path(raw_path).expanduser()))
    if not specs:
        raise ValueError("至少提供一个 --model NAME=PATH")
    return specs


def load_image_or_source(source: Optional[str]) -> Any:
    """读取单张图片到内存；视频、目录和非图片路径原样交给 adapter。"""
    if source is None:
        return None
    path = Path(source).expanduser()
    if path.is_file():
        if cv2 is None:
            raise ImportError("读取图片需要 opencv-python；也可以让自定义 adapter 自己处理 source")
        image = cv2.imread(str(path))
        if image is not None:
            return image
    return str(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str) -> Iterator[None]:
    # 模型本身已经由 adapter 转换 dtype；这里主要覆盖自定义 pipeline 中
    # 临时创建的浮点算子，且不影响 FP32 测试。
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def _run_once(adapter: BaseAdapter, model: Any, inputs: InputBundle, config: BenchmarkConfig) -> Any:
    with autocast_context(config.device, config.precision):
        return adapter.run(model, inputs)


@torch.inference_mode()
def benchmark_forward(adapter: BaseAdapter, model: Any, config: BenchmarkConfig) -> dict[str, Any]:
    """测量已准备输入后的模型调用时间，不包含预处理和后处理。"""
    dtype = dtype_for_precision(config.precision, config.device)
    inputs = adapter.make_inputs(
        config.batch_size,
        config.image_size,
        config.device,
        dtype,
        for_export=False,
    )
    for _ in range(config.warmup):
        output = _run_once(adapter, model, inputs, config)
        del output
    synchronize(config.device)

    if config.device.type == "cuda":
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        with torch.cuda.device(config.device):
            for _ in range(config.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = _run_once(adapter, model, inputs, config)
                end.record()
                events.append((start, end))
                del output
        synchronize(config.device)
        times = [float(start.elapsed_time(end)) for start, end in events]
    else:
        times = []
        for _ in range(config.repeats):
            start = time.perf_counter_ns()
            output = _run_once(adapter, model, inputs, config)
            del output
            times.append((time.perf_counter_ns() - start) / 1_000_000)

    result = statistics(times)
    result.update(
        {
            "batch": int(inputs.batch_size),
            "params": adapter.parameter_count(model),
            "peak_delta_mb": None,
            "output_count": None,
        }
    )
    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)
        before = torch.cuda.memory_allocated(config.device)
        output = _run_once(adapter, model, inputs, config)
        synchronize(config.device)
        result["peak_delta_mb"] = float(
            (torch.cuda.max_memory_allocated(config.device) - before) / 1024**2
        )
        result["output_count"] = adapter.count_outputs(output)
        del output
    else:
        output = _run_once(adapter, model, inputs, config)
        result["output_count"] = adapter.count_outputs(output)
        del output
    return result


@torch.inference_mode()
def benchmark_pipeline(
    adapter: BaseAdapter,
    model: Any,
    config: BenchmarkConfig,
    source: Any = None,
) -> dict[str, Any]:
    """测量 adapter 的完整 pipeline。

    对 Ultralytics 或自定义 adapter 的 ``predict``，计时边界是调用前后；
    GPU 操作前后均同步。没有自定义 predict 时，默认 pipeline 是
    ``make_inputs + run``，不会假装包含未知的图像预处理。
    """
    for _ in range(config.warmup):
        output = adapter.predict(model, source, config)
        del output
    synchronize(config.device)

    import time

    times: list[float] = []
    output_counts: list[int] = []
    for _ in range(config.repeats):
        synchronize(config.device)
        start = time.perf_counter_ns()
        output = adapter.predict(model, source, config)
        synchronize(config.device)
        times.append((time.perf_counter_ns() - start) / 1_000_000)
        count = adapter.count_outputs(output)
        if count is not None:
            output_counts.append(int(count))
        del output

    result = statistics(times)
    result["batch"] = config.batch_size
    result["params"] = adapter.parameter_count(model)
    result["output_count"] = (
        int(round(float(np.mean(output_counts)))) if output_counts else None
    )
    return result


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    return tensors
