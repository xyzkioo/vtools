#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 PyTorch / TensorRT 视觉模型速度对比。

这个版本不再调用 ``YOLO(engine)``，而是使用 TensorRT 原生 Python API
绑定输入输出张量，因此只要模型能导出为 ONNX，就不要求输出是 YOLO 的
``result.boxes``。ONNX 导出和模型输入由 adapter 决定；固定 batch/固定
空间尺寸是为了保证不同后端的公平比较。

需要：
    pip install onnx onnxruntime-gpu   # 仅用于验证/可选
    TensorRT Python bindings
    trtexec                         # 可选；builder: python 不需要

示例：

    python benchmark_tensorrt.py \
        --adapter ultralytics \
        --model baseline=/path/to/best.pt \
        --input-size 832 --precision fp16 \
        --engine-dir runs-profile/engines

任意自定义模型请提供 adapter.py，参考 ``adapters/vision_adapter_template.py``。
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from core.vision_benchmark_common import (
    BenchmarkConfig,
    InputBundle,
    ModelSpec,
    benchmark_forward,
    benchmark_pipeline,
    cleanup_model,
    create_adapter,
    dtype_for_precision,
    flatten_tensors,
    load_image_or_source,
    normalize_precision,
    resolve_device,
    statistics,
    synchronize,
)
from core.vision_benchmark_config import apply_python_paths, get_model_entries, load_config
from core.vision_checkpoint_inspector import inspect_entries
from core.vision_run_manager import prepare_run_directory, resolve_cli_path
from backends.tensorrt_engine_builder import build_engine_from_onnx


def parse_input_size(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" in text:
        height_text, width_text = text.split("x", 1)
        height, width = int(height_text), int(width_text)
    else:
        height = width = int(text)
    if height <= 0 or width <= 0:
        raise ValueError("输入尺寸必须是正整数")
    return height, width


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用 PyTorch/TensorRT 视觉模型测速")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 配置文件；省略时读取同目录 benchmark_config.yaml",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="复用指定的 runN 目录；省略时自动创建 runs-profile/runN",
    )
    parser.add_argument("--model", action="append", required=False, metavar="NAME=PATH")
    parser.add_argument("--adapter", default=None, help="checkpoint、ultralytics 或 adapter.py/模块名")
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-size", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--source", help="可选图片/视频/目录；只用于 PyTorch adapter pipeline")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--fuse", action="store_true", default=None)
    parser.add_argument("--engine-dir", type=Path, default=None)
    parser.add_argument("--onnx-dir", type=Path, default=None)
    parser.add_argument("--rebuild-engine", action="store_true")
    parser.add_argument(
        "--builder",
        choices=["python", "trtexec", "auto"],
        default=None,
        help="engine 构建方式；python 使用 TensorRT Python API，默认 python",
    )
    parser.add_argument("--trtexec", default=None, help="trtexec 可执行文件路径")
    parser.add_argument("--workspace-mb", type=int, default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def torch_dtype_from_trt(trt: Any, dtype: Any) -> torch.dtype:
    mapping: dict[Any, torch.dtype] = {}
    for name, torch_dtype in (
        ("FLOAT", torch.float32),
        ("HALF", torch.float16),
        ("BF16", torch.bfloat16),
        ("INT8", torch.int8),
        ("INT32", torch.int32),
        ("INT64", torch.int64),
        ("UINT8", torch.uint8),
        ("BOOL", torch.bool),
    ):
        trt_dtype = getattr(trt.DataType, name, None)
        if trt_dtype is not None:
            mapping[trt_dtype] = torch_dtype
    if dtype not in mapping:
        raise TypeError(f"TensorRT dtype 暂不支持：{dtype}")
    return mapping[dtype]


class TensorRTRunner:
    """用 torch CUDA tensor 直接绑定 TensorRT engine 的 I/O。"""

    def __init__(
        self, engine_path: Path, device: torch.device,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("TensorRT 测速需要 CUDA 设备")
        try:
            import tensorrt as trt
        except ImportError as error:
            raise ImportError("未找到 TensorRT Python bindings，无法运行 engine") from error

        self.trt = trt
        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        self.logger = logger  # 与 runtime/engine 保持相同生命周期。
        with engine_path.open("rb") as file:
            serialized = file.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"TensorRT 无法反序列化 engine：{engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT 无法创建执行上下文")
        self._input_hold: list[torch.Tensor] = []
        self.legacy_api = not hasattr(self.engine, "num_io_tensors")
        if self.legacy_api:
            self.binding_names = [self.engine.get_binding_name(index) for index in range(self.engine.num_bindings)]
            self.input_binding_indices = [
                index for index in range(self.engine.num_bindings) if self.engine.binding_is_input(index)
            ]
            self.output_binding_indices = [
                index for index in range(self.engine.num_bindings) if not self.engine.binding_is_input(index)
            ]
            self.input_names = [self.binding_names[index] for index in self.input_binding_indices]
            self.output_names = [self.binding_names[index] for index in self.output_binding_indices]
        else:
            self.input_names = [
                self.engine.get_tensor_name(index)
                for index in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                self.engine.get_tensor_name(index)
                for index in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
            ]
        if not self.input_names:
            raise RuntimeError("TensorRT engine 没有输入张量")
        # 校验时按 ONNX 图的名字和顺序绑定，不能假定 engine 的枚举顺序相同。
        for requested, current, label in (
            (input_names, self.input_names, "输入"),
            (output_names, self.output_names, "输出"),
        ):
            if requested is not None and (
                len(requested) != len(set(requested)) or set(requested) != set(current)
            ):
                raise ValueError(f"ONNX/engine {label}名称不一致：{requested} != {current}")
        if input_names is not None:
            self.input_names = list(input_names)
        if output_names is not None:
            self.output_names = list(output_names)
        if self.legacy_api:
            self.input_binding_indices = [self.engine.get_binding_index(n) for n in self.input_names]
            self.output_binding_indices = [self.engine.get_binding_index(n) for n in self.output_names]
        if input_names is not None or output_names is not None:
            for name in self.input_names + self.output_names:
                index = self.engine.get_binding_index(name) if self.legacy_api else None
                location = self.engine.get_location(index) if self.legacy_api else self.engine.get_tensor_location(name)
                layout = self.engine.get_binding_format(index) if self.legacy_api else self.engine.get_tensor_format(name)
                if location != trt.TensorLocation.DEVICE or layout != trt.TensorFormat.LINEAR:
                    raise ValueError(f"一致性检查暂只支持 DEVICE/LINEAR I/O：{name}；特殊 shape 输入或布局需专用 runner")

    def input_dtype(self, name: str) -> torch.dtype:
        dtype = (
            self.engine.get_binding_dtype(self.engine.get_binding_index(name))
            if self.legacy_api else self.engine.get_tensor_dtype(name)
        )
        return torch_dtype_from_trt(self.trt, dtype)

    def _input_tensors(self, bundle: InputBundle) -> list[torch.Tensor]:
        tensors = flatten_tensors(bundle.args)
        tensors.extend(flatten_tensors(bundle.kwargs))
        if len(tensors) != len(self.input_names):
            raise ValueError(
                f"adapter 提供了 {len(tensors)} 个 Tensor 输入，但 engine 需要 {len(self.input_names)} 个；"
                "复杂多输入模型请在 adapter 中自定义 export_onnx 和输入映射。"
            )
        return tensors

    def run(self, bundle: InputBundle) -> tuple[torch.Tensor, ...]:
        tensors = self._input_tensors(bundle)
        addresses: dict[str, int] = {}
        converted_inputs: list[torch.Tensor] = []
        legacy_bindings: list[int] = [0] * getattr(self.engine, "num_bindings", 0)
        with torch.cuda.device(self.device):
            for input_position, (name, tensor) in enumerate(zip(self.input_names, tensors)):
                if self.legacy_api:
                    binding_index = self.input_binding_indices[input_position]
                    trt_dtype = self.engine.get_binding_dtype(binding_index)
                else:
                    binding_index = None
                    trt_dtype = self.engine.get_tensor_dtype(name)
                expected_dtype = torch_dtype_from_trt(self.trt, trt_dtype)
                tensor = tensor.to(device=self.device, dtype=expected_dtype).contiguous()
                converted_inputs.append(tensor)
                shape = tuple(int(item) for item in tensor.shape)
                if self.legacy_api:
                    if self.context.set_binding_shape(binding_index, shape) is False:
                        raise ValueError(f"engine 不接受输入 {name} 的形状 {shape}")
                    legacy_bindings[binding_index] = tensor.data_ptr()
                else:
                    if self.context.set_input_shape(name, shape) is False:
                        raise ValueError(f"engine 不接受输入 {name} 的形状 {shape}")
                addresses[name] = tensor.data_ptr()

            outputs: list[torch.Tensor] = []
            for output_position, name in enumerate(self.output_names):
                if self.legacy_api:
                    binding_index = self.output_binding_indices[output_position]
                    shape = tuple(int(item) for item in self.context.get_binding_shape(binding_index))
                    trt_dtype = self.engine.get_binding_dtype(binding_index)
                else:
                    binding_index = None
                    shape = tuple(int(item) for item in self.context.get_tensor_shape(name))
                    trt_dtype = self.engine.get_tensor_dtype(name)
                if any(item < 0 for item in shape):
                    raise RuntimeError(f"无法解析 engine 输出 {name} 的动态形状：{shape}")
                dtype = torch_dtype_from_trt(self.trt, trt_dtype)
                output = torch.empty(shape, dtype=dtype, device=self.device)
                outputs.append(output)
                if self.legacy_api:
                    legacy_bindings[binding_index] = output.data_ptr()
                else:
                    addresses[name] = output.data_ptr()

            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self.legacy_api:
                ok = self.context.execute_async_v2(legacy_bindings, stream)
            else:
                for name, address in addresses.items():
                    self.context.set_tensor_address(name, address)
                ok = self.context.execute_async_v3(stream)
            if ok is False:
                raise RuntimeError("TensorRT execute_async_v3 返回失败")
        # 保持可能由 dtype/contiguous 转换产生的临时输入，直到下一次调用；
        # CUDA allocator 会在同一 stream 上安全处理其生命周期。
        self._input_hold = converted_inputs
        return tuple(outputs)


@torch.inference_mode()
def benchmark_trt(runner: TensorRTRunner, inputs: InputBundle, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        output = runner.run(inputs)
        del output
    synchronize(runner.device)

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.cuda.device(runner.device):
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = runner.run(inputs)
            end.record()
            events.append((start, end))
            del output
    synchronize(runner.device)
    times = [float(start.elapsed_time(end)) for start, end in events]
    result = statistics(times)
    result.update({"batch": inputs.batch_size, "params": None, "peak_delta_mb": None, "output_count": None})
    return result


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "weights",
        "adapter",
        "backend",
        "precision",
        "height",
        "width",
        "batch",
        "params",
        "scope",
        "scope_note",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "fps_per_sample",
        "peak_delta_mb",
        "output_count",
        "onnx",
        "engine",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_row(
    spec: ModelSpec,
    adapter_name: str,
    config: BenchmarkConfig,
    backend: str,
    scope: str,
    scope_note: str,
    result: dict[str, Any],
    onnx_path: Optional[Path] = None,
    engine_path: Optional[Path] = None,
    status: str = "succeeded",
    error: str = "",
) -> dict[str, Any]:
    median = result.get("median_ms")
    return {
        "model": spec.name,
        "weights": str(spec.weights),
        "adapter": adapter_name,
        "backend": backend,
        "precision": config.precision,
        "height": config.height,
        "width": config.width,
        "batch": result.get("batch", config.batch_size),
        "params": result.get("params"),
        "scope": scope,
        "scope_note": scope_note,
        "mean_ms": result.get("mean_ms"),
        "median_ms": median,
        "p90_ms": result.get("p90_ms"),
        "min_ms": result.get("min_ms"),
        "max_ms": result.get("max_ms"),
        "std_ms": result.get("std_ms"),
        "fps_per_sample": None if not median else 1000.0 * result.get("batch", config.batch_size) / median,
        "peak_delta_mb": result.get("peak_delta_mb"),
        "output_count": result.get("output_count"),
        "onnx": str(onnx_path) if onnx_path else "",
        "engine": str(engine_path) if engine_path else "",
        "status": status,
        "error": error,
    }


def main() -> list[dict[str, Any]]:
    args = build_parser().parse_args()
    config_file = load_config(args.config)
    run_dir = prepare_run_directory(config_file, args.run_dir)
    apply_python_paths(config_file)
    benchmark_values = config_file.get("benchmark", {})
    tensorrt_values = config_file.get("tensorrt", {})

    args.device = args.device or benchmark_values.get("device", "auto")
    args.input_size = args.input_size or benchmark_values.get("input_size", "832")
    args.batch = args.batch if args.batch is not None else int(benchmark_values.get("batch_size", 1))
    args.warmup = args.warmup if args.warmup is not None else int(benchmark_values.get("warmup", 30))
    args.repeats = args.repeats if args.repeats is not None else int(benchmark_values.get("repeats", 100))
    args.fuse = bool(args.fuse if args.fuse is not None else benchmark_values.get("fuse_model", False))
    args.precision = args.precision or tensorrt_values.get("precision", "fp16")
    args.engine_dir = resolve_cli_path(
        args.engine_dir or tensorrt_values.get("engine_dir") or "runs-profile/engines",
        run_dir,
        config_file["project"]["root"],
    )
    args.onnx_dir = resolve_cli_path(
        args.onnx_dir or tensorrt_values.get("onnx_dir") or "runs-profile/onnx",
        run_dir,
        config_file["project"]["root"],
    )
    if args.engine_dir is None or args.onnx_dir is None:
        raise ValueError("无法确定 TensorRT engine/ONNX 输出目录")
    args.rebuild_engine = bool(args.rebuild_engine or tensorrt_values.get("rebuild_engine", False))
    args.builder = args.builder or tensorrt_values.get("builder", "python")
    args.trtexec = args.trtexec or tensorrt_values.get("trtexec", "trtexec")
    args.workspace_mb = (
        args.workspace_mb
        if args.workspace_mb is not None
        else int(tensorrt_values.get("workspace_mb", 2048))
    )
    args.opset = args.opset if args.opset is not None else int(tensorrt_values.get("onnx_opset", 18))
    args.output = resolve_cli_path(
        args.output or tensorrt_values.get("output") or "runs-profile/vision_tensorrt_v2.csv",
        run_dir,
        config_file["project"]["root"],
    )
    if args.output is None:
        raise ValueError("无法确定 TensorRT 输出路径")
    model_entries = get_model_entries(
        config_file,
        cli_models=args.model,
        adapter_override=args.adapter,
        task_override=args.task,
    )
    if args.source is None:
        args.source = benchmark_values.get("image", benchmark_values.get("source"))

    if bool(benchmark_values.get("inspect_checkpoint", True)):
        inspection_results = inspect_entries(
            model_entries,
            benchmark_values,
            try_adapter_load=bool(benchmark_values.get("verify_model_load", True)),
            print_results=True,
        )
        inspection_failures = [
            item
            for item in inspection_results
            if item.architecture_status in {"missing", "unknown"}
            or item.adapter_load_status == "failed"
        ]
        if inspection_failures and bool(benchmark_values.get("fail_on_checkpoint_inspection", False)):
            names = ", ".join(item.name for item in inspection_failures)
            raise RuntimeError(f"checkpoint 检查失败：{names}")

    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("batch>0、repeats>0，warmup 不能小于 0")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT 对比脚本必须使用 CUDA，例如 --device 0")
    torch.backends.cudnn.benchmark = True
    height, width = parse_input_size(args.input_size)
    precision = normalize_precision(args.precision)
    if precision == "bf16":
        # TensorRT 不同版本的 BF16 支持差异较大，但仍保留统一入口。
        print("[提示] BF16 engine 需要当前 TensorRT Python API 和 GPU 同时支持。")
    source = load_image_or_source(args.source) if args.source else None
    args.engine_dir.mkdir(parents=True, exist_ok=True)
    args.onnx_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        device=device,
        batch_size=args.batch,
        height=height,
        width=width,
        precision=precision,
        warmup=args.warmup,
        repeats=args.repeats,
        fuse=args.fuse,
        conf=float(benchmark_values.get("conf", 0.25)),
        iou=float(benchmark_values.get("iou", 0.70)),
        max_det=int(benchmark_values.get("max_det", 300)),
    )
    print("=" * 92)
    print("通用 PyTorch / TensorRT 视觉模型测速 v2")
    if run_dir is not None:
        print(f"本次运行目录：{run_dir}")
    print(f"Python：{platform.python_version()} | PyTorch：{torch.__version__}")
    print(f"GPU：{torch.cuda.get_device_name(device)} | 输入：{height}x{width} | batch={args.batch}")
    adapters = ", ".join(sorted({str(item["adapter"]) for item in model_entries}))
    print(f"adapters：{adapters} | precision：{precision} | builder：{args.builder}")
    print("=" * 92)

    rows: list[dict[str, Any]] = []
    for model_entry in model_entries:
        spec = ModelSpec(str(model_entry["name"]), Path(str(model_entry["weights"])))
        adapter = create_adapter(
            str(model_entry["adapter"]),
            str(model_entry.get("task", "detect")),
            float(benchmark_values.get("conf", 0.25)),
            float(benchmark_values.get("iou", 0.70)),
            int(benchmark_values.get("max_det", 300)),
        )
        if not spec.weights.is_file():
            raise FileNotFoundError(f"模型文件不存在：{spec.weights}")
        model: Any = None
        onnx_path = args.onnx_dir / f"{safe_name(spec.name)}_{precision}_{height}x{width}_b{args.batch}.onnx"
        engine_path = args.engine_dir / f"{safe_name(spec.name)}_{precision}_{height}x{width}_b{args.batch}.engine"
        print(f"\n[模型] {spec.name}")
        try:
            model = adapter.load_model(spec.weights, device)
            model = adapter.prepare_model(model, device, precision, args.fuse)
            dtype = dtype_for_precision(precision, device)
            inputs = adapter.make_inputs(args.batch, (height, width), device, dtype, for_export=True)

            run_pytorch = bool(tensorrt_values.get("test_pytorch", True)) and not args.skip_pytorch
            run_pipeline = bool(tensorrt_values.get("test_pipeline", True)) and not args.skip_pipeline
            if run_pytorch:
                forward = benchmark_forward(adapter, model, config)
                rows.append(
                    make_row(
                        spec,
                        adapter.adapter_name,
                        config,
                        "pytorch",
                        "model_call",
                        "input already on device; no preprocessing/postprocessing",
                        forward,
                    )
                )
                if run_pipeline and (source is not None or not adapter.supports_real_predict):
                    pipeline = benchmark_pipeline(adapter, model, config, source)
                    rows.append(
                        make_row(
                            spec,
                            adapter.adapter_name,
                            config,
                            "pytorch",
                            "adapter_pipeline",
                            "adapter.predict boundary; exact preprocessing depends on adapter",
                            pipeline,
                        )
                    )

            if args.rebuild_engine or not engine_path.is_file():
                if not onnx_path.is_file() or args.rebuild_engine:
                    onnx_path = adapter.export_onnx(model, onnx_path, inputs, args.opset, dynamic=False)
                build_engine_from_onnx(
                    onnx_path,
                    engine_path,
                    builder=args.builder,
                    trtexec=args.trtexec,
                    precision=precision,
                    workspace_mb=args.workspace_mb,
                )
            elif not onnx_path.is_file():
                print(f"[TensorRT] 复用已有 engine，不重新导出 ONNX：{engine_path}")

            runner = TensorRTRunner(engine_path, device)
            trt_result = benchmark_trt(runner, inputs, args.warmup, args.repeats)
            rows.append(
                make_row(
                    spec,
                    adapter.adapter_name,
                    config,
                    "tensorrt",
                    "engine_call",
                    "固定输入 engine 调用；不包含图像预处理和后处理",
                    trt_result,
                    onnx_path,
                    engine_path,
                )
            )
            print(
                f"TensorRT：median={trt_result['median_ms']:.3f} ms, "
                f"P90={trt_result['p90_ms']:.3f} ms, "
                f"FPS/样本={1000.0 * args.batch / trt_result['median_ms']:.1f}"
            )
            del runner
        except Exception as error:  # 记录单个模型/后端失败，保留其他模型结果
            message = f"{type(error).__name__}: {error}"
            print(f"[失败] {message}")
            rows.append(
                make_row(
                    spec,
                    adapter.adapter_name,
                    config,
                    "tensorrt",
                    "engine_call",
                    "TensorRT 阶段失败",
                    {"batch": args.batch},
                    onnx_path,
                    engine_path,
                    status="failed",
                    error=message,
                )
            )
        finally:
            if model is not None:
                cleanup_model(model)
            gc.collect()

    save_rows(args.output, rows)
    print("=" * 92)
    print(f"结果已保存：{args.output.resolve()}")
    print("TensorRT engine 与 GPU、TensorRT/CUDA 版本相关，请在目标部署机重新构建和测速。")
    # 返回实际明细，供 run_all 判断失败；捕获过异常不代表后端成功。
    return rows


if __name__ == "__main__":
    main()
