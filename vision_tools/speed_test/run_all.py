#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行测速和一致性检查；只汇总本次返回结果，不读取旧 CSV。"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any

from core.vision_benchmark_config import load_config
from core.vision_run_manager import prepare_run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键运行通用视觉模型测速")
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
    return parser


def _invoke(
    label: str,
    module_name: str,
    config_path: str,
    run_dir: str | Path | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """连导入异常也记录；以后端实际结果决定状态，而非只检查函数是否返回。"""
    original_argv = sys.argv[:]
    sys.argv = [label, "--config", config_path]
    if run_dir is not None:
        sys.argv.extend(["--run-dir", str(run_dir)])
    try:
        result = importlib.import_module(module_name).main()
        if isinstance(result, dict):
            # 一致性详细数据只写自己的报告，不混进速度表。
            state = result.get("status", "error")
            return {"backend": label, "status": state, "error": result.get("error", ""),
                    "report": result.get("json_output", "")}, []
        if not isinstance(result, list) or not result:
            raise RuntimeError("后端未返回本次有效结果；请使用同一新版压缩包中的全部文件")
        failures = [row for row in result if row.get("status") in {"failed", "error"}]
        return {"backend": label, "status": "failed" if failures else "succeeded",
                "error": " | ".join(str(row.get("error", "failed")) for row in failures)}, result
    except Exception as error:  # noqa: BLE001 - 一键运行需要汇总单阶段失败
        message = f"{type(error).__name__}: {error}"
        print(f"\n[{label}] 阶段失败：{message}")
        return {"backend": label, "status": "failed", "error": message}, []
    finally:
        sys.argv = original_argv


def _merge_results(
    output_path: Path,
    current_rows: list[dict[str, Any]],
    stage_status: list[dict[str, str]],
) -> None:
    """只使用本次内存明细，杜绝后端失败时把上一次的 CSV 当成本次结果。"""
    rows: list[dict[str, Any]] = []
    fields: list[str] = ["record_type", "backend", "status", "error"]
    for original in current_rows:
        row = dict(original)
        row.setdefault("status", "succeeded")
        row.setdefault("error", "")
        row["record_type"] = "measurement"
        rows.append(row)
        for key in row:
            if key not in fields:
                fields.append(key)

    for status in stage_status:
        rows.append({**status, "record_type": "stage_summary"})
        for key in status:
            if key not in fields:
                fields.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_dir = prepare_run_directory(config, args.run_dir)
    config_path = str(config["_config_path"])
    run_values = config.get("run_all", {})
    pytorch_values = config.get("pytorch", {})
    tensorrt_values = config.get("tensorrt", {})

    stage_status: list[dict[str, str]] = []
    current_rows: list[dict[str, Any]] = []
    tensorrt_failed = False

    if bool(run_values.get("run_pytorch", True)) and bool(pytorch_values.get("enabled", True)):
        state, rows = _invoke("pytorch", "backends.pytorch_vision_speed_benchmark_v2", config_path, run_dir)
        stage_status.append(state)
        current_rows.extend(rows)
    else:
        print("[跳过] run_all.run_pytorch 或 pytorch.enabled 为 false")

    if bool(run_values.get("run_tensorrt", True)) and bool(tensorrt_values.get("enabled", True)):
        state, rows = _invoke("tensorrt", "backends.pytorch_vision_tensorrt_benchmark_v2", config_path, run_dir)
        stage_status.append(state)
        current_rows.extend(rows)
        tensorrt_failed = state["status"] != "succeeded"
    else:
        print("[跳过] run_all.run_tensorrt 或 tensorrt.enabled 为 false")

    if bool(run_values.get("run_consistency", True)) and bool(config.get("consistency", {}).get("enabled", False)):
        if tensorrt_failed:
            # 不在本次转换失败之后悄悄验证磁盘里遗留的旧 engine。
            reason = "本次 TensorRT 阶段失败，一致性未执行；修复后重跑，或单独检查指定的已有 engine"
            stage_status.append({"backend": "consistency", "status": "skipped", "error": reason})
            print(f"[跳过] {reason}")
        else:
            state, _ = _invoke("consistency", "checks.vision_consistency", config_path, run_dir)
            stage_status.append(state)
    else:
        print("[跳过] run_all.run_consistency 或 consistency.enabled 为 false")

    summary = Path(
        run_values.get("summary_output", "runs-profile/vision_benchmark_summary.csv")
    )
    _merge_results(summary, current_rows, stage_status)
    if run_dir is not None:
        print(f"本次运行目录：{run_dir}")
    print(f"\n汇总结果已保存：{summary.resolve()}")


if __name__ == "__main__":
    # PyCharm 可能没有继承终端中的 LD_LIBRARY_PATH；在导入 ONNX 前
    # 自动切换到当前 conda 环境的 libstdc++，避免一致性阶段出现 CXXABI 错误。
    from core.vision_runtime import ensure_conda_library_path

    ensure_conda_library_path()
    main()
