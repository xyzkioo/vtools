#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 PyTorch 视觉模型速度基准。

示例：

    # Ultralytics YOLO/RT-DETR
    python benchmark_pytorch.py \
        --adapter ultralytics \
        --model baseline=/path/to/best.pt \
        --source /path/to/test.jpg \
        --input-size 832 --precision fp16

    # 任意自定义 PyTorch 模型
    python benchmark_pytorch.py \
        --adapter /path/to/my_adapter.py \
        --model my_model=/path/to/checkpoint.pth \
        --input-size 640x960 --batch 4

自定义模型的构建和输入放在 adapter 文件中，参考
``adapters/vision_adapter_template.py``。测速核心不读取 result.boxes，也不假定
模型一定是检测任务。
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from core.vision_benchmark_common import (
    BenchmarkConfig,
    ModelSpec,
    benchmark_forward,
    benchmark_pipeline,
    cleanup_model,
    create_adapter,
    normalize_precision,
    resolve_device,
)
from core.vision_benchmark_config import apply_python_paths, get_model_entries, load_config
from core.vision_checkpoint_inspector import inspect_entries
from core.vision_run_manager import prepare_run_directory, resolve_cli_path
from core.module_selection import SPEED_MODULES, resolve_speed_modules


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用 PyTorch 视觉模型测速")
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
    parser.add_argument(
        "--model",
        action="append",
        required=False,
        metavar="NAME=PATH",
        help="临时覆盖 YAML 中的模型，可重复；也可只传 PATH，名称自动取文件名",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="checkpoint、ultralytics、或 adapter.py/可导入模块名",
    )
    parser.add_argument("--task", default=None, help="Ultralytics 任务类型，如 detect/segment/pose/classify")
    parser.add_argument("--device", default=None, help="auto、cpu、cuda:0 或 0")
    parser.add_argument("--input-size", default=None, help="输入尺寸，例如 832 或 640x960")
    parser.add_argument("--batch", type=int, default=None, help="真实测试 batch size")
    parser.add_argument(
        "--precision",
        action="append",
        choices=["fp32", "fp16", "bf16"],
        help="测试精度，可重复；默认 CUDA 测 fp32+fp16，CPU 只测 fp32",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--source", help="可选的单张图片/视频/目录，供 adapter pipeline 使用")
    parser.add_argument("--skip-pipeline", action="store_true", help="只测准备好输入后的模型调用")
    parser.add_argument("--fuse", action="store_true", default=None, help="若模型提供 fuse()，测速前执行融合")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only", action="append", help="只运行指定模块 ID，可重复或用逗号分隔")
    parser.add_argument("--enable", action="append", help="临时启用模块 ID，可重复或用逗号分隔")
    parser.add_argument("--disable", action="append", help="临时关闭模块 ID，可重复或用逗号分隔")
    parser.add_argument("--list-modules", action="store_true", help="列出测速模块后退出")
    return parser


def print_result(model_name: str, precision: str, forward: dict[str, Any], pipeline: dict[str, Any] | None) -> None:
    print("-" * 84)
    print(f"模型：{model_name} | PyTorch | {precision}")
    print(
        f"纯模型调用：median={forward['median_ms']:.3f} ms, "
        f"P90={forward['p90_ms']:.3f} ms, "
        f"FPS/样本={1000.0 * forward['batch'] / forward['median_ms']:.1f}"
    )
    if pipeline is not None:
        print(
            f"adapter pipeline：median={pipeline['median_ms']:.3f} ms, "
            f"P90={pipeline['p90_ms']:.3f} ms, "
            f"FPS/样本={1000.0 * pipeline['batch'] / pipeline['median_ms']:.1f}"
        )
    print(f"参数量：{forward.get('params') or 'unknown'} | batch={forward['batch']}")
    if forward.get("peak_delta_mb") is not None:
        print(f"纯调用峰值显存增量：{forward['peak_delta_mb']:.2f} MB")
    if forward.get("output_count") is not None:
        print(f"输出数量（adapter 定义）：{forward['output_count']}")


def rows_for_result(
    spec: ModelSpec,
    precision: str,
    forward: dict[str, Any],
    pipeline: dict[str, Any] | None,
    config: BenchmarkConfig,
    adapter_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = {
        "model": spec.name,
        "weights": str(spec.weights),
        "backend": "pytorch",
        "adapter": adapter_name,
        "precision": precision,
        "height": config.height,
        "width": config.width,
        "batch": forward["batch"],
        "params": forward.get("params"),
    }
    rows.append(
        {
            **common,
            "scope": "model_call",
            "scope_note": "input already on device; no preprocessing/postprocessing",
            **{key: forward.get(key) for key in ("mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms", "std_ms")},
            "fps_per_sample": 1000.0 * forward["batch"] / forward["median_ms"],
            "peak_delta_mb": forward.get("peak_delta_mb"),
            "output_count": forward.get("output_count"),
        }
    )
    if pipeline is not None:
        rows.append(
            {
                **common,
                "scope": "adapter_pipeline",
                "scope_note": "adapter.predict boundary; exact preprocessing depends on adapter",
                **{key: pipeline.get(key) for key in ("mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms", "std_ms")},
                "fps_per_sample": 1000.0 * pipeline["batch"] / pipeline["median_ms"],
                "peak_delta_mb": None,
                "output_count": pipeline.get("output_count"),
            }
        )
    return rows


def row_for_model_info(
    spec: ModelSpec,
    precision: str,
    config: BenchmarkConfig,
    adapter_name: str,
    params: int | None,
    peak_delta_mb: float | None,
    note: str,
) -> dict[str, Any]:
    return {
        "model": spec.name,
        "weights": str(spec.weights),
        "backend": "pytorch",
        "adapter": adapter_name,
        "precision": precision,
        "height": config.height,
        "width": config.width,
        "batch": config.batch_size,
        "params": params,
        "scope": "model_info",
        "scope_note": note,
        "mean_ms": None,
        "median_ms": None,
        "p90_ms": None,
        "min_ms": None,
        "max_ms": None,
        "std_ms": None,
        "fps_per_sample": None,
        "peak_delta_mb": peak_delta_mb,
        "output_count": None,
    }


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "status", "error", "architecture_status", "adapter_load_status",
        "weights",
        "backend",
        "adapter",
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
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> list[dict[str, Any]]:
    args = build_parser().parse_args()
    if args.list_modules:
        print("\n".join(SPEED_MODULES))
        return []
    config_file = load_config(args.config)
    run_dir = prepare_run_directory(config_file, args.run_dir)
    apply_python_paths(config_file, args.model)
    benchmark_values = config_file.get("benchmark", {})
    pytorch_values = config_file.get("pytorch", {})
    modules = resolve_speed_modules(config_file, only=args.only, enable=args.enable, disable=args.disable)
    has_model_info = any(modules.get(key) for key in ("model.parameters", "model.flops", "memory.pytorch_peak"))
    has_checkpoint_check = any(modules.get(key) for key in ("checkpoint.inspect", "checkpoint.load_check"))
    if not modules.get("speed.pytorch_call") and not modules.get("speed.pytorch_pipeline") and not has_model_info and not has_checkpoint_check:
        raise ValueError("至少启用一个 PyTorch 测速、模型信息、权重检查或显存模块")

    args.device = args.device or benchmark_values.get("device", "auto")
    args.input_size = args.input_size or benchmark_values.get("input_size", "832")
    args.batch = args.batch if args.batch is not None else int(benchmark_values.get("batch_size", 1))
    args.warmup = args.warmup if args.warmup is not None else int(benchmark_values.get("warmup", 30))
    args.repeats = args.repeats if args.repeats is not None else int(benchmark_values.get("repeats", 100))
    args.conf = args.conf if args.conf is not None else float(benchmark_values.get("conf", 0.25))
    args.iou = args.iou if args.iou is not None else float(benchmark_values.get("iou", 0.70))
    args.max_det = args.max_det if args.max_det is not None else int(benchmark_values.get("max_det", 300))
    args.fuse = bool(args.fuse if args.fuse is not None else benchmark_values.get("fuse_model", False))
    args.output = resolve_cli_path(
        args.output or pytorch_values.get("output") or "runs-profile/vision_speed_v2.csv",
        run_dir,
        config_file["project"]["root"],
    )
    if args.output is None:
        raise ValueError("无法确定 PyTorch 输出路径")
    precision_values = args.precision if args.precision is not None else pytorch_values.get("precisions", [])
    model_entries = get_model_entries(
        config_file,
        cli_models=args.model,
        adapter_override=args.adapter,
        task_override=args.task,
    )
    if args.source is None:
        args.source = benchmark_values.get("image", benchmark_values.get("source"))

    if modules.get("checkpoint.inspect") or modules.get("checkpoint.load_check"):
        inspection_results = inspect_entries(
            model_entries,
            benchmark_values,
            try_adapter_load=bool(modules.get("checkpoint.load_check")),
            print_results=True,
        )
        inspection_failures = [
            item
            for item in inspection_results
            if item.architecture_status in {"missing", "unknown"}
            or item.adapter_load_status == "failed"
        ]
        inspection_rows = []
        for item in inspection_results:
            failed = item in inspection_failures
            inspection_rows.append({"model": item.name, "status": "failed" if failed else "succeeded", "error": item.raw_error or item.adapter_error, "architecture_status": item.architecture_status, "adapter_load_status": item.adapter_load_status})
        if has_checkpoint_check and not modules.get("speed.pytorch_call") and not modules.get("speed.pytorch_pipeline") and not has_model_info:
            save_rows(args.output, inspection_rows)
            print(f"权重检查完成；结果已保存：{args.output.resolve()}")
            return inspection_rows

    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("batch>0、repeats>0，warmup 不能小于 0")

    device = resolve_device(args.device)
    if device.type == "cuda":
        # 输入形状固定时允许 cuDNN 选择更快的卷积算法。
        torch.backends.cudnn.benchmark = True
    height, width = parse_input_size(args.input_size)
    precisions = [normalize_precision(item) for item in (precision_values or [])]
    if not precisions:
        precisions = ["fp32", "fp16"] if device.type == "cuda" else ["fp32"]
    if device.type != "cuda":
        precisions = [item for item in precisions if item != "fp16"]
    if not precisions:
        raise ValueError("当前设备没有可测试的精度")
    info_only = not modules.get("speed.pytorch_call") and not modules.get("speed.pytorch_pipeline")
    if info_only:
        # Parameter/FLOPs inspection is precision-independent.  A memory-only
        # run still performs one lightweight forward below, rather than the
        # configured timing loop.
        precisions = precisions[:1]

    source = None
    if args.source:
        from core.vision_benchmark_common import load_image_or_source

        source = load_image_or_source(args.source)

    print("=" * 92)
    print("通用 PyTorch 视觉模型测速 v2")
    if run_dir is not None:
        print(f"本次运行目录：{run_dir}")
    print(f"Python：{platform.python_version()} | PyTorch：{torch.__version__}")
    adapters = ", ".join(sorted({str(item["adapter"]) for item in model_entries}))
    print(f"设备：{device} | 输入：{height}x{width} | adapters：{adapters}")
    print(f"精度：{', '.join(precisions)} | batch={args.batch} | warmup={args.warmup} | repeats={args.repeats}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print("=" * 92)

    rows: list[dict[str, Any]] = []
    for model_entry in model_entries:
        spec = ModelSpec(str(model_entry["name"]), Path(str(model_entry["weights"])))
        adapter = create_adapter(
            str(model_entry["adapter"]),
            str(model_entry.get("task", "detect")),
            args.conf,
            args.iou,
            args.max_det,
        )
        if not spec.weights.is_file():
            raise FileNotFoundError(f"模型文件不存在：{spec.weights}")
        for precision in precisions:
            config = BenchmarkConfig(
                device=device,
                batch_size=args.batch,
                height=height,
                width=width,
                precision=precision,
                warmup=args.warmup,
                repeats=args.repeats,
                fuse=args.fuse,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                measure_parameters=modules.get("model.parameters", False),
                measure_memory=modules.get("memory.pytorch_peak", False),
            )
            measurement_config = replace(config, warmup=0, repeats=1) if info_only and modules.get("memory.pytorch_peak") else config
            print(f"\n[测速] {spec.name} / {precision}")
            model = adapter.load_model(spec.weights, device)
            try:
                model = adapter.prepare_model(model, device, precision, args.fuse)
                forward = benchmark_forward(adapter, model, measurement_config) if (modules.get("speed.pytorch_call") or (info_only and modules.get("memory.pytorch_peak"))) else None
                pipeline = None
                # Ultralytics 的 predict 没有 source 时会采用内部 demo 数据，
                # 这不属于用户想测试的真实输入，因此默认跳过。
                pipeline_enabled = modules.get("speed.pytorch_pipeline", False) and not args.skip_pipeline
                if pipeline_enabled and (source is not None or not adapter.supports_real_predict):
                    pipeline = benchmark_pipeline(adapter, model, config, source)
                if forward is not None and modules.get("speed.pytorch_call"):
                    print_result(spec.name, precision, forward, pipeline)
                    rows.extend(rows_for_result(spec, precision, forward, pipeline, config, adapter.adapter_name))
                elif pipeline is not None:
                    print(f"模型：{spec.name} | PyTorch | {precision} | 仅 adapter pipeline")
                    rows.extend(rows_for_result(spec, precision, {"batch": args.batch, "median_ms": pipeline["median_ms"], "params": pipeline.get("params")}, pipeline, config, adapter.adapter_name)[1:])
                elif info_only:
                    params = adapter.parameter_count(model) if modules.get("model.parameters") else None
                    note = "只执行模型信息检查"
                    if modules.get("model.flops"):
                        note += "；当前 adapter 未提供 FLOPs 统计接口"
                    peak = forward.get("peak_delta_mb") if forward is not None else None
                    if modules.get("memory.pytorch_peak"):
                        note += "；显存模块仅执行一次前向"
                    rows.append(row_for_model_info(spec, precision, config, adapter.adapter_name, params, peak, note))
                    print(f"模型：{spec.name} | 模型信息 | 参数量={params or 'unknown'} | 峰值显存增量={peak if peak is not None else 'N/A'}")
            finally:
                cleanup_model(model)

    save_rows(args.output, rows)
    print("=" * 92)
    print(f"结果已保存：{args.output.resolve()}")
    print("论文中请区分 model_call（纯模型调用）和 adapter_pipeline（适配器完整流程）。")
    return rows


if __name__ == "__main__":
    main()
