#!/usr/bin/env python3
"""跨子项目入口；PyCharm 和终端都可调用。"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vtools 模块化功能入口")
    parser.add_argument("--tool", choices=["diagnostics", "pytorch", "tensorrt", "consistency", "visualization", "compression"], default="diagnostics")
    parser.add_argument("--config", type=Path, help="对应子项目配置；省略时使用默认 PyCharm 配置")
    parser.add_argument("--only", action="append")
    parser.add_argument("--enable", action="append")
    parser.add_argument("--disable", action="append")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--weights", type=Path, help="可选：覆盖诊断、可视化、测速或压缩入口的模型权重")
    parser.add_argument("--source", type=Path, help="可选：覆盖可视化入口的输入来源")
    parser.add_argument("--train", type=Path, help="可选：覆盖模型压缩训练集目录")
    parser.add_argument("--val", type=Path, help="可选：覆盖模型压缩验证集目录")
    parser.add_argument("--predictions", type=Path, help="可选：覆盖诊断入口的预测文件")
    parser.add_argument("--data", type=Path, help="可选：覆盖诊断入口的数据集 YAML")
    parser.add_argument("--task", choices=["detect", "classify"], help="压缩模型任务")
    parser.add_argument("--structured-scale", help="压缩检测模型的结构化目标规模")
    parser.add_argument("--structured-method", choices=["scale", "torch_pruning"], help="压缩检测模型的结构化剪枝方法")
    parser.add_argument("--structured-epochs", type=int, help="结构化缩放后的微调轮数；0 表示跳过微调")
    parser.add_argument("--structured-initial-weights", type=Path, help="结构化目标规模的预训练初始化权重")
    parser.add_argument("--mode", choices=["list_layers", "visualize"], help="可选：覆盖可视化模式")
    parser.add_argument("--device", help="可选：覆盖运行设备 auto、cpu、cuda:0 或 0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # 两个子项目都曾使用过顶层 ``core`` 兼容导入；测速目录必须排在前面。
    for path in (ROOT / "model_diagnostics", ROOT / "speed_test"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    selected_config = args.config.expanduser().resolve() if args.config else None
    if selected_config and selected_config.name == "tools.yaml":
        try:
            import yaml  # type: ignore
            values = yaml.safe_load(selected_config.read_text(encoding="utf-8")) or {}
            key = {"diagnostics": "diagnostics_config", "pytorch": "speed_config", "tensorrt": "speed_config", "consistency": "speed_config", "visualization": "visualization_config", "compression": "compression_config"}[args.tool]
            candidate = values.get(key)
            if candidate:
                selected_config = (ROOT / str(candidate)).resolve()
        except (ImportError, OSError, ValueError):
            pass
    if args.tool == "diagnostics":
        module_name = "model_diagnostics.run_model_diagnostics"
        default_config = ROOT / "model_diagnostics" / "config" / "pycharm_run.yaml"
    elif args.tool == "pytorch":
        module_name = "speed_test.backends.pytorch_vision_speed_benchmark_v2"
        default_config = ROOT / "speed_test" / "benchmark_config.yaml"
    elif args.tool == "tensorrt":
        module_name = "speed_test.backends.pytorch_vision_tensorrt_benchmark_v2"
        default_config = ROOT / "speed_test" / "benchmark_config.yaml"
    elif args.tool == "consistency":
        module_name = "speed_test.checks.vision_consistency"
        default_config = ROOT / "speed_test" / "benchmark_config.yaml"
    elif args.tool == "visualization":
        module_name = "model_visualization.run_visualization"
        default_config = ROOT / "model_visualization" / "config" / "pycharm_run.yaml"
    else:
        module_name = "model_compression.run_model_compression"
        default_config = ROOT / "model_compression" / "config" / "pycharm_run.yaml"
    old_argv = sys.argv[:]
    sys.argv = [module_name, "--config", str(selected_config or default_config)]
    if args.tool == "diagnostics" and args.predictions:
        sys.argv.extend(["--predictions", str(args.predictions)])
    if args.tool == "diagnostics" and args.weights:
        sys.argv.extend(["--weights", str(args.weights)])
    if args.tool == "diagnostics" and args.data:
        sys.argv.extend(["--data", str(args.data)])
    if args.tool == "diagnostics" and args.device:
        sys.argv.extend(["--device", args.device])
    if args.tool == "visualization":
        for flag in ("weights", "source", "mode", "device"):
            value = getattr(args, flag)
            if value:
                sys.argv.extend([f"--{flag}", str(value)])
    if args.tool in {"pytorch", "tensorrt", "consistency"}:
        if args.weights:
            sys.argv.extend(["--model", str(args.weights)])
        if args.source:
            sys.argv.extend(["--source", str(args.source)])
        if args.device:
            sys.argv.extend(["--device", args.device])
        if args.structured_scale is not None:
            sys.argv.extend(["--structured-scale", args.structured_scale])
        if args.structured_epochs is not None:
            sys.argv.extend(["--structured-epochs", str(args.structured_epochs)])
        if args.structured_initial_weights is not None:
            sys.argv.extend(["--structured-initial-weights", str(args.structured_initial_weights)])
    if args.tool == "compression":
        if args.data:
            sys.argv.extend(["--data", str(args.data)])
        if args.task:
            sys.argv.extend(["--task", args.task])
        if args.weights:
            sys.argv.extend(["--weights", str(args.weights)])
        if args.train:
            sys.argv.extend(["--train", str(args.train)])
        if args.val:
            sys.argv.extend(["--val", str(args.val)])
        if args.device:
            sys.argv.extend(["--device", args.device])
        if args.structured_scale is not None:
            sys.argv.extend(["--structured-scale", args.structured_scale])
        if args.structured_method is not None:
            sys.argv.extend(["--structured-method", args.structured_method])
        if args.structured_epochs is not None:
            sys.argv.extend(["--structured-epochs", str(args.structured_epochs)])
        if args.structured_initial_weights is not None:
            sys.argv.extend(["--structured-initial-weights", str(args.structured_initial_weights)])
    if args.list_modules and args.tool != "visualization":
        sys.argv.append("--list-modules")
    if args.list_modules and args.tool == "visualization":
        print("visualization.features\nvisualization.cam\nvisualization.stage_trace")
        return 0
    for flag in ("only", "enable", "disable"):
        for value in getattr(args, flag) or []:
            sys.argv.extend([f"--{flag}", value])
    try:
        result = importlib.import_module(module_name).main()
        return int(result) if isinstance(result, int) else 0
    except (ValueError, RuntimeError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
