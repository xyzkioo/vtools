#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取并规范化 ``benchmark_config.yaml``。

脚本本身只消费这里返回的普通 Python 字典，因此用户不需要为了修改
模型、输入或输出路径而编辑测速代码。相对路径统一相对于 ``project.root``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Optional


# 配置文件留在项目根目录；本模块现在位于 ``core/`` 子目录。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "benchmark_config.yaml"


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"配置项 {name} 必须是字典，实际是 {type(value).__name__}")
    return dict(value)


def _resolve_path(value: Any, base: Path, *, keep_plain: bool = False) -> Any:
    """解析配置中的路径；空值保留为 None。"""
    if value is None or value == "":
        return value
    if not isinstance(value, (str, Path)):
        return value
    text = str(value).strip()
    # 可执行文件名（例如 trtexec 或 trtexec.exe）应交给 PATH 查找；
    # 只有包含目录分隔符的值才按 project.root 解析。
    if keep_plain and not any(token in text for token in ("/", "\\")):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _normalize_input_size(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("benchmark.input_size 需要 [height, width]")
        return f"{int(value[0])}x{int(value[1])}"
    return str(value if value is not None else "832")


def _normalize_precisions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError("pytorch.precisions 必须是字符串或列表")
    return [str(item) for item in value]


def load_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    """加载 YAML 并把项目内相对路径转换为绝对路径。"""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}\n"
            "请复制 benchmark_config.yaml，或运行时使用 --config 指定配置文件。"
        )
    try:
        import yaml
    except ImportError as error:
        raise ImportError("读取 YAML 配置需要 PyYAML，请执行：pip install pyyaml") from error

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise TypeError("配置文件顶层必须是字典")

    config = dict(raw)
    # 保留一份未规范化的 YAML 值。运行目录管理器需要据此区分
    # “相对输出路径”（应放到 runN）和明确写在 output root 之外的绝对路径。
    config["_raw_config"] = raw
    project = _as_mapping(config.get("project"), "project")
    project_root_value = project.get("root")
    project_root = Path(project_root_value).expanduser() if project_root_value else config_path.parent
    if not project_root.is_absolute():
        project_root = config_path.parent / project_root
    project_root = project_root.resolve()
    project["root"] = str(project_root)
    if project.get("ultralytics_repo"):
        project["ultralytics_repo"] = _resolve_path(project["ultralytics_repo"], project_root)
    project["python_paths"] = [
        _resolve_path(item, project_root)
        for item in project.get("python_paths", [])
    ]
    config["project"] = project

    run = _as_mapping(config.get("run"), "run")
    run["root"] = _resolve_path(run.get("root", "runs-profile"), project_root)
    run.setdefault("enabled", True)
    run.setdefault("name", "auto")
    config["run"] = run

    benchmark = _as_mapping(config.get("benchmark"), "benchmark")
    benchmark["input_size"] = _normalize_input_size(benchmark.get("input_size", "832"))
    for key in ("image", "source"):
        if benchmark.get(key) is not None:
            benchmark[key] = _resolve_path(benchmark[key], project_root)
    config["benchmark"] = benchmark

    models: list[dict[str, Any]] = []
    raw_models = config.get("models", [])
    if not isinstance(raw_models, list):
        raise TypeError("models 必须是列表")
    for index, item in enumerate(raw_models):
        if isinstance(item, (str, Path)):
            item = {"weights": str(item)}
        if not isinstance(item, dict):
            raise TypeError(f"models[{index}] 必须是字典或权重路径")
        model = dict(item)
        weights = model.get("weights") or model.get("path")
        if not weights:
            raise ValueError(f"models[{index}] 缺少 weights")
        weights_path = Path(str(weights)).expanduser()
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        model["weights"] = str(weights_path.resolve())
        for key in ("engine", "onnx"):
            if model.get(key):
                model[key] = _resolve_path(model[key], project_root)
        model.setdefault("name", weights_path.stem)
        model.setdefault("adapter", benchmark.get("adapter", "checkpoint"))
        model.setdefault("task", benchmark.get("task", "detect"))
        model["enabled"] = bool(model.get("enabled", True))
        adapter = model.get("adapter")
        if isinstance(adapter, str) and (adapter.endswith(".py") or "/" in adapter or "\\" in adapter):
            model["adapter"] = _resolve_path(adapter, project_root)
        models.append(model)
    config["models"] = models

    pytorch = _as_mapping(config.get("pytorch"), "pytorch")
    pytorch["precisions"] = _normalize_precisions(pytorch.get("precisions"))
    if pytorch.get("output") is not None:
        pytorch["output"] = _resolve_path(pytorch["output"], project_root)
    config["pytorch"] = pytorch

    tensorrt = _as_mapping(config.get("tensorrt"), "tensorrt")
    for key in ("engine_dir", "onnx_dir", "output"):
        if tensorrt.get(key) is not None:
            tensorrt[key] = _resolve_path(tensorrt[key], project_root)
    if tensorrt.get("trtexec"):
        tensorrt["trtexec"] = _resolve_path(tensorrt["trtexec"], project_root, keep_plain=True)
    config["tensorrt"] = tensorrt

    consistency = _as_mapping(config.get("consistency"), "consistency")
    for key in ("source", "output", "json_output"):
        if consistency.get(key) is not None:
            consistency[key] = _resolve_path(consistency[key], project_root)
    config["consistency"] = consistency

    run_all = _as_mapping(config.get("run_all"), "run_all")
    if run_all.get("summary_output") is not None:
        run_all["summary_output"] = _resolve_path(run_all["summary_output"], project_root)
    config["run_all"] = run_all
    config["_config_path"] = str(config_path)
    return config


def apply_python_paths(config: dict[str, Any]) -> None:
    """把本地源码仓库和自定义 Python 目录加入导入路径。"""
    project = _as_mapping(config.get("project"), "project")
    candidates: list[Any] = []
    if project.get("root"):
        candidates.append(project["root"])
    if project.get("ultralytics_repo"):
        candidates.append(project["ultralytics_repo"])
    candidates.extend(project.get("python_paths", []))
    for model in config.get("models", []):
        adapter = model.get("adapter") if isinstance(model, dict) else None
        if isinstance(adapter, str) and adapter.endswith(".py"):
            candidates.append(str(Path(adapter).parent))
    for value in candidates:
        path = Path(str(value)).expanduser()
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def get_model_entries(
    config: dict[str, Any],
    cli_models: Optional[Iterable[str]] = None,
    adapter_override: Optional[str] = None,
    task_override: Optional[str] = None,
) -> list[dict[str, Any]]:
    """返回脚本可以直接消费的模型配置列表。"""
    entries: list[dict[str, Any]] = []
    if cli_models:
        for value in cli_models:
            text = str(value)
            if "=" in text:
                name, weights = text.split("=", 1)
                name, weights = name.strip(), weights.strip()
                if not name or not weights:
                    raise ValueError(f"模型参数格式应为 NAME=PATH，收到：{value}")
            else:
                weights = text.strip()
                name = Path(weights).stem
            entries.append({"name": name, "weights": str(Path(weights).expanduser())})
    else:
        entries = [dict(item) for item in config.get("models", []) if item.get("enabled", True)]

    benchmark = _as_mapping(config.get("benchmark"), "benchmark")
    default_adapter = benchmark.get("adapter", "checkpoint")
    default_task = benchmark.get("task", "detect")
    for entry in entries:
        entry["adapter"] = str(adapter_override or entry.get("adapter") or default_adapter)
        entry["task"] = str(task_override or entry.get("task") or default_task)
        entry["weights"] = str(Path(str(entry["weights"])).expanduser())
    if not entries:
        raise ValueError("没有启用的模型，请在 benchmark_config.yaml 的 models 中添加模型")
    return entries


def config_path_value(config: dict[str, Any], section: str, key: str, fallback: Any = None) -> Any:
    """安全读取 section/key，方便入口脚本处理可选配置。"""
    value = config.get(section, {})
    return value.get(key, fallback) if isinstance(value, dict) else fallback
