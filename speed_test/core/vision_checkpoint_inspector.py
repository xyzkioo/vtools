#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 PyTorch checkpoint 是否包含可恢复的模型架构。

这个工具同时做两层检查：

1. 读取 checkpoint 外层，区分完整 ``nn.Module``、``model/ema`` checkpoint
   和只有 ``state_dict`` 的文件；
2. 可选地用配置中指定的 adapter 实际加载一次，验证当前源码环境能否恢复模型。

``torch.load(weights_only=False)`` 只应对自己信任的本地权重使用，因为完整
模型 checkpoint 依赖 Python pickle。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn

from core.vision_benchmark_common import cleanup_model, create_adapter
from core.vision_benchmark_config import apply_python_paths, get_model_entries, load_config


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin"}


@dataclass
class CheckpointInspection:
    name: str
    path: str
    adapter: str
    task: str
    exists: bool
    suffix: str
    container_type: str = ""
    keys: list[str] = field(default_factory=list)
    architecture_status: str = "unknown"
    model_type: str = ""
    ema_type: str = ""
    state_dict_keys: Optional[int] = None
    raw_error: str = ""
    adapter_load_status: str = "not_run"
    adapter_model_type: str = ""
    adapter_yaml_present: bool = False
    adapter_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_trusted_checkpoint(path: Path) -> Any:
    """兼容不同 PyTorch 版本地读取完整 checkpoint。"""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch 2.5 及更早版本没有 weights_only 参数。
        return torch.load(str(path), map_location="cpu")


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    tensor_values = sum(isinstance(item, torch.Tensor) for item in value.values())
    return tensor_values == len(value)


def _state_dict_size(value: Any) -> Optional[int]:
    return len(value) if isinstance(value, dict) else None


def inspect_checkpoint(
    path: Path,
    name: str,
    adapter_spec: str,
    task: str,
) -> CheckpointInspection:
    """读取 checkpoint 外层，不执行模型推理。"""
    result = CheckpointInspection(
        name=name,
        path=str(path),
        adapter=adapter_spec,
        task=task,
        exists=path.is_file(),
        suffix=path.suffix.lower(),
    )
    if not result.exists:
        result.architecture_status = "missing"
        result.raw_error = f"文件不存在：{path}"
        return result
    if result.suffix not in CHECKPOINT_SUFFIXES:
        result.architecture_status = "not_applicable"
        result.raw_error = "不是 PyTorch checkpoint 后缀，跳过 .pt 结构检查"
        return result

    try:
        checkpoint = _load_trusted_checkpoint(path)
    except Exception as error:  # noqa: BLE001 - 将反序列化错误写入检查报告
        result.architecture_status = "unknown"
        result.raw_error = f"{type(error).__name__}: {error}"
        return result

    result.container_type = type(checkpoint).__name__
    if isinstance(checkpoint, nn.Module):
        result.architecture_status = "full_module"
        result.model_type = f"{type(checkpoint).__module__}.{type(checkpoint).__name__}"
        return result

    if not isinstance(checkpoint, dict):
        result.architecture_status = "unknown"
        return result

    result.keys = [str(key) for key in checkpoint.keys()]
    model_value = checkpoint.get("model")
    ema_value = checkpoint.get("ema")
    if isinstance(model_value, nn.Module):
        result.model_type = f"{type(model_value).__module__}.{type(model_value).__name__}"
    if isinstance(ema_value, nn.Module):
        result.ema_type = f"{type(ema_value).__module__}.{type(ema_value).__name__}"
    if isinstance(model_value, nn.Module) or isinstance(ema_value, nn.Module):
        result.architecture_status = "full_checkpoint"
        return result

    for key in ("state_dict", "model_state_dict", "weights", "params"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            result.architecture_status = "state_dict_only"
            result.state_dict_keys = _state_dict_size(candidate)
            return result
    # 一些训练脚本把参数字典放在 ``model`` 或 ``ema`` 字段中；这些字段
    # 的值不是 nn.Module 时，仍然属于“只有参数”，不能当作完整架构。
    for candidate in (model_value, ema_value):
        if _looks_like_state_dict(candidate):
            result.architecture_status = "state_dict_only"
            result.state_dict_keys = len(candidate)
            return result
    if _looks_like_state_dict(checkpoint):
        result.architecture_status = "state_dict_only"
        result.state_dict_keys = len(checkpoint)
    return result


def verify_adapter_load(
    result: CheckpointInspection,
    conf: float = 0.25,
    iou: float = 0.70,
    max_det: int = 300,
) -> CheckpointInspection:
    """使用指定 adapter 在 CPU 上实际加载一次 checkpoint。"""
    if not result.exists or result.architecture_status == "not_applicable":
        result.adapter_load_status = "skipped"
        return result
    model: Any = None
    try:
        adapter = create_adapter(result.adapter, result.task, conf, iou, max_det)
        model = adapter.load_model(Path(result.path), torch.device("cpu"))
        target = adapter.forward_model(model)
        result.adapter_load_status = "succeeded"
        result.adapter_model_type = f"{type(target).__module__}.{type(target).__name__}"
        result.adapter_yaml_present = getattr(target, "yaml", None) is not None or getattr(model, "yaml", None) is not None
    except Exception as error:  # noqa: BLE001 - 检查工具必须报告具体失败原因
        result.adapter_load_status = "failed"
        result.adapter_error = f"{type(error).__name__}: {error}"
    finally:
        if model is not None:
            cleanup_model(model)
    return result


def conclusion(result: CheckpointInspection) -> str:
    if result.adapter_load_status == "succeeded":
        if result.architecture_status == "state_dict_only":
            return "当前 adapter 已成功重建网络并加载参数"
        return "当前 adapter 可以直接恢复模型"
    if result.adapter_load_status == "failed":
        return "当前 adapter 加载失败；优先检查本地源码、自定义模块和权重匹配"
    if result.architecture_status in {"full_module", "full_checkpoint"}:
        return "包含完整模型对象；加载时仍需要对应 Python 类和源码"
    if result.architecture_status == "state_dict_only":
        return "只有参数，没有完整架构；必须先在 adapter 中实例化网络"
    if result.architecture_status == "missing":
        return "权重文件不存在"
    if result.architecture_status == "not_applicable":
        return "非 PyTorch checkpoint，未执行此项检查"
    return "无法判断；查看 raw_error"


def print_inspection(result: CheckpointInspection) -> None:
    print("-" * 92)
    print(f"[checkpoint 检查] {result.name}")
    print(f"文件：{result.path}")
    print(f"外层类型：{result.container_type or 'unknown'}")
    print(f"架构判断：{result.architecture_status}")
    if result.keys:
        print(f"keys：{', '.join(result.keys[:20])}{' ...' if len(result.keys) > 20 else ''}")
    if result.model_type:
        print(f"model：{result.model_type}")
    if result.ema_type:
        print(f"ema：{result.ema_type}")
    if result.state_dict_keys is not None:
        print(f"参数条目数：{result.state_dict_keys}")
    if result.raw_error:
        print(f"原始检查信息：{result.raw_error}")
    if result.adapter_load_status != "not_run":
        print(f"adapter 实际加载：{result.adapter_load_status}")
        if result.adapter_model_type:
            print(f"恢复后的模型：{result.adapter_model_type}")
        if result.adapter_error:
            print(f"adapter 错误：{result.adapter_error}")
    print(f"结论：{conclusion(result)}")


def inspect_entries(
    entries: Iterable[dict[str, Any]],
    benchmark_values: dict[str, Any],
    *,
    try_adapter_load: bool = True,
    print_results: bool = True,
) -> list[CheckpointInspection]:
    """检查一组配置模型，供独立脚本和测速入口共用。"""
    results: list[CheckpointInspection] = []
    conf = float(benchmark_values.get("conf", 0.25))
    iou = float(benchmark_values.get("iou", 0.70))
    max_det = int(benchmark_values.get("max_det", 300))
    for entry in entries:
        result = inspect_checkpoint(
            Path(str(entry["weights"])),
            str(entry["name"]),
            str(entry.get("adapter", "checkpoint")),
            str(entry.get("task", "detect")),
        )
        if try_adapter_load:
            verify_adapter_load(result, conf, iou, max_det)
        if print_results:
            print_inspection(result)
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 .pt/.pth 是否包含可恢复模型架构")
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置文件")
    parser.add_argument("--model", action="append", metavar="NAME=PATH", help="临时指定模型，可重复")
    parser.add_argument("--adapter", default=None, help="覆盖所有临时模型的 adapter")
    parser.add_argument("--task", default=None, help="覆盖所有临时模型的 task")
    parser.add_argument("--raw-only", action="store_true", help="只检查文件内容，不实际加载模型")
    parser.add_argument("--json-output", type=Path, default=None, help="可选 JSON 报告路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    apply_python_paths(config)
    entries = get_model_entries(
        config,
        cli_models=args.model,
        adapter_override=args.adapter,
        task_override=args.task,
    )
    results = inspect_entries(
        entries,
        config.get("benchmark", {}),
        try_adapter_load=not args.raw_only,
        print_results=True,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([item.as_dict() | {"conclusion": conclusion(item)} for item in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 检查报告已保存：{args.json_output.resolve()}")


if __name__ == "__main__":
    main()
