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
    parser.add_argument("--tool", choices=["diagnostics", "pytorch", "tensorrt", "consistency", "visualization"], default="diagnostics")
    parser.add_argument("--config", type=Path, help="对应子项目配置；省略时使用默认 PyCharm 配置")
    parser.add_argument("--only", action="append")
    parser.add_argument("--enable", action="append")
    parser.add_argument("--disable", action="append")
    parser.add_argument("--list-modules", action="store_true")
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
            key = {"diagnostics": "diagnostics_config", "pytorch": "speed_config", "tensorrt": "speed_config", "consistency": "speed_config", "visualization": "visualization_config"}[args.tool]
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
    else:
        module_name = "model_visualization.run_visualization"
        default_config = ROOT / "model_visualization" / "config" / "pycharm_run.yaml"
    old_argv = sys.argv[:]
    sys.argv = [module_name, "--config", str(selected_config or default_config)]
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
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
