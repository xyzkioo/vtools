#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为一次测速调用分配独立的 ``run1``、``run2`` ... 目录。

这个模块不依赖 PyTorch，所有入口都可以在真正导入 CUDA/ONNX 依赖前先
分配运行目录。``run_all.py`` 会把已经分配的目录传给子阶段，因而一键
运行的多个阶段仍然属于同一个 run；单独运行入口则自动申请下一个 run。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


RUN_DIR_ENV = "VISION_BENCHMARK_RUN_DIR"


_PATH_DEFAULTS: dict[tuple[str, str], str] = {
    ("pytorch", "output"): "runs-profile/vision_speed_v2.csv",
    ("tensorrt", "output"): "runs-profile/vision_tensorrt_v2.csv",
    ("consistency", "output"): "runs-profile/consistency.csv",
    ("consistency", "json_output"): "runs-profile/consistency.json",
    ("run_all", "summary_output"): "runs-profile/vision_benchmark_summary.csv",
}


def _project_root(config: dict[str, Any]) -> Path:
    project = config.get("project")
    if isinstance(project, dict) and project.get("root"):
        return Path(str(project["root"])).expanduser().resolve()
    config_path = config.get("_config_path")
    return Path(str(config_path)).expanduser().resolve().parent if config_path else Path.cwd().resolve()


def _run_values(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("run")
    return dict(values) if isinstance(values, dict) else {}


def _run_root(config: dict[str, Any]) -> Path:
    values = _run_values(config)
    root = values.get("root") or "runs-profile"
    path = Path(str(root)).expanduser()
    if not path.is_absolute():
        path = _project_root(config) / path
    return path.resolve()


def _raw_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """返回 YAML 中尚未规范化的路径值。

    需要它来区分用户写的相对路径（应放进 runN）和明确写在输出根目录
    外的绝对路径（应保持原位置）。旧的手工构造 config 没有 raw 数据时，
    会回退到当前规范化值。
    """

    raw = config.get("_raw_config")
    if isinstance(raw, dict):
        section_value = raw.get(section)
        if isinstance(section_value, dict) and key in section_value:
            value = section_value.get(key)
            return default if value is None and default is not None else value
    current = config.get(section)
    if isinstance(current, dict) and key in current:
        return current.get(key)
    return default


def _relative_to(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _raw_relative_to_run_root(raw_value: Any, run_root: Path, project_root: Path) -> Optional[Path]:
    """把 ``runs-profile/foo`` 这类旧配置转换为 ``foo``。"""

    if raw_value is None or raw_value == "":
        return None
    raw_path = Path(str(raw_value)).expanduser()
    if raw_path.is_absolute():
        return _relative_to(raw_path, run_root)

    # 相对 project.root 的旧写法（例如 runs-profile/engines）应去掉
    # output root 前缀；其它相对路径则直接成为 runN 下的相对路径。
    run_root_relative = _relative_to(run_root, project_root)
    if run_root_relative is not None and run_root_relative.parts:
        prefix = run_root_relative.parts
        if raw_path.parts[: len(prefix)] == prefix:
            rest = raw_path.parts[len(prefix) :]
            return Path(*rest) if rest else Path(".")
    return raw_path


def _map_output_path(
    config: dict[str, Any],
    raw_value: Any,
    current_value: Any,
    run_dir: Path,
) -> Optional[str]:
    if raw_value is None or raw_value == "":
        if current_value is None or current_value == "":
            return None
        current_path = Path(str(current_value)).expanduser()
        relative = _relative_to(current_path, _run_root(config))
        if relative is None:
            return str(current_path.resolve())
    else:
        relative = _raw_relative_to_run_root(raw_value, _run_root(config), _project_root(config))
        if relative is None:
            # 明确写在 runs-profile 之外的绝对路径不强制搬迁；这是给
            # 需要固定外部日志目录的用户留下的逃生口。
            return str(Path(str(raw_value)).expanduser().resolve())
    return str((run_dir / relative).resolve())


def apply_run_directory(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """把结果报告路径切换到指定的 run 目录。

    ONNX/engine 是可复用的构建缓存，刻意不在这里搬迁。
    """

    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for (section, key), default in _PATH_DEFAULTS.items():
        values = config.setdefault(section, {})
        if not isinstance(values, dict):
            values = {}
            config[section] = values
        raw_value = _raw_value(config, section, key, default)
        current_value = values.get(key, default)
        mapped = _map_output_path(config, raw_value, current_value, target)
        if mapped is not None:
            values[key] = mapped

    run_values = config.setdefault("run", {})
    if isinstance(run_values, dict):
        run_values["dir"] = str(target)
    config["_run_dir"] = str(target)
    return config


def _write_run_info(config: dict[str, Any], run_dir: Path) -> None:
    """写一份很小的元数据，便于从 runN 反查配置文件。"""

    path = run_dir / "run_info.json"
    if path.exists():
        return
    info = {
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config.get("_config_path", "")),
    }
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_run_directory(
    config: dict[str, Any],
    requested: str | Path | None = None,
) -> Optional[Path]:
    """分配或复用一次运行目录，并更新 config 中的产物路径。

    ``requested`` 通常来自 ``--run-dir``。如果没有显式传入，则读取内部
    环境变量（供一键入口的子阶段继承）；再没有才在 ``run.root`` 下原子
    创建下一个 ``runN``。配置中 ``run.enabled: false`` 可恢复旧的固定
    路径行为。
    """

    # 手工构造、没有经过 load_config 的字典不自动产生目录，避免测试或
    # 外部调用意外写入当前目录；正常 YAML 会始终注入 run 默认配置。
    if "run" not in config:
        return None
    values = _run_values(config)
    if not bool(values.get("enabled", True)):
        return None

    project_root = _project_root(config)
    inherited = requested if requested is not None else os.environ.get(RUN_DIR_ENV)
    if inherited:
        target = Path(str(inherited)).expanduser()
        if not target.is_absolute():
            target = project_root / target
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
    else:
        root = _run_root(config)
        root.mkdir(parents=True, exist_ok=True)
        name = str(values.get("name", "auto") or "auto").strip()
        if name.lower() == "auto":
            number = 1
            while True:
                candidate = root / f"run{number}"
                try:
                    # mkdir 本身是原子的；两个终端同时启动也不会拿到
                    # 同一个 run 目录。
                    candidate.mkdir()
                    target = candidate.resolve()
                    break
                except FileExistsError:
                    number += 1
        else:
            candidate = Path(name).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()
            candidate.mkdir(parents=True, exist_ok=False)
            target = candidate

    apply_run_directory(config, target)
    _write_run_info(config, target)
    return target


def resolve_cli_path(value: str | Path | None, run_dir: str | Path | None, project_root: str | Path) -> Optional[Path]:
    """解析命令行产物路径；相对路径也放入当前 run 目录。"""

    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(run_dir).expanduser() if run_dir else Path(project_root).expanduser()
    return (base / path).resolve()


__all__ = [
    "RUN_DIR_ENV",
    "apply_run_directory",
    "prepare_run_directory",
    "resolve_cli_path",
]
