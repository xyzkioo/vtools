"""读取并规范化模型压缩 YAML 配置。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"配置节 {name} 必须是 YAML 对象")
    return dict(value)


def _resolve(value: Any, base: Path, *, keep_plain: bool = False) -> Any:
    if value is None or value == "":
        return value
    if not isinstance(value, (str, Path)):
        return value
    text = str(value).strip()
    if keep_plain and not any(token in text for token in ("/", "\\")):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolve_optional_mapping_paths(values: dict[str, Any], keys: tuple[str, ...], root: Path) -> dict[str, Any]:
    result = dict(values)
    for key in keys:
        if result.get(key) is not None:
            result[key] = _resolve(result[key], root)
    return result


def load_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    """读取 YAML，并将项目内路径统一解析为绝对路径。"""

    config_path = (Path(path).expanduser() if path else DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到模型压缩配置：{config_path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("读取 YAML 配置需要 PyYAML，请执行：pip install pyyaml") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("配置文件顶层必须是 YAML 对象")
    removed = sorted(set(raw) & {"operation", "models"})
    if removed:
        raise ValueError(f"已移除旧版配置字段：{', '.join(removed)}；请使用 model 和 modules")
    config: dict[str, Any] = copy.deepcopy(dict(raw))
    config["_raw_config"] = copy.deepcopy(dict(raw))
    config["_config_path"] = str(config_path)

    project = _mapping(config.get("project"), "project")
    project_root = Path(str(project.get("root", "."))).expanduser()
    if not project_root.is_absolute():
        project_root = config_path.parent / project_root
    project_root = project_root.resolve()
    project["root"] = str(project_root)
    if project.get("ultralytics_repo"):
        project["ultralytics_repo"] = _resolve(project["ultralytics_repo"], project_root)
    if project.get("python_paths") is None:
        project["python_paths"] = []
    if not isinstance(project["python_paths"], (list, tuple)):
        raise TypeError("project.python_paths 必须是路径列表")
    project["python_paths"] = [_resolve(item, project_root) for item in project["python_paths"]]
    config["project"] = project

    run = _mapping(config.get("run"), "run")
    run["root"] = _resolve(run.get("root", "model_compression/runs"), project_root)
    run.setdefault("enabled", True)
    run.setdefault("name", "auto")
    config["run"] = run

    # 运行结果自包含在 runs/runN/ 中；旧 storage/branch 键仅为兼容读取，
    # 不参与运行状态，也不会触发 store/registry 写入。
    config.pop("storage", None)

    model = _mapping(config.get("model"), "model")
    model.setdefault("name", "model")
    model.setdefault("adapter", "torch")
    model.setdefault("task", "classify")
    if model.get("weights") is not None:
        model["weights"] = _resolve(model["weights"], project_root)
    model["adapter"] = str(model.get("adapter") or "torch")
    if model["adapter"].endswith(".py") or "/" in model["adapter"] or "\\" in model["adapter"]:
        model["adapter"] = _resolve(model["adapter"], project_root)
    config["model"] = model

    for section_name in ("teacher", "student"):
        section = _mapping(config.get(section_name), section_name)
        if section.get("weights") is not None:
            section["weights"] = _resolve(section["weights"], project_root)
        if section.get("factory") is not None:
            section["factory"] = str(section["factory"])
        config[section_name] = section

    dataset = _mapping(config.get("dataset"), "dataset")
    dataset = _resolve_optional_mapping_paths(dataset, ("root", "train", "val", "test", "manifest", "data"), project_root)
    config["dataset"] = dataset

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    config["evaluation"] = evaluation
    export = _mapping(config.get("export"), "export")
    if export.get("path") is not None:
        export["path"] = _resolve(export["path"], project_root)
    config["export"] = export
    config["compression"] = _mapping(config.get("compression"), "compression")
    distillation = _mapping(config.get("distillation"), "distillation")
    for key in ("teacher_weights", "student_weights"):
        if distillation.get(key) is not None:
            distillation[key] = _resolve(distillation[key], project_root)
    config["distillation"] = distillation
    structured = _mapping(config["compression"].get("structured"), "compression.structured")
    if structured.get("initial_weights") is not None:
        structured["initial_weights"] = _resolve(structured["initial_weights"], project_root)
    config["compression"]["structured"] = structured
    # 分支和版本指针属于已移除的跨运行注册表，不进入有效配置。
    config.pop("branch", None)
    return config


def apply_python_paths(config: Mapping[str, Any]) -> None:
    """将项目根目录和自定义源码目录加入 ``sys.path``。"""

    from vtools_runtime.ultralytics import add_repo_to_path

    import sys

    project = _mapping(config.get("project"), "project")
    paths = [project.get("root"), project.get("ultralytics_repo"), *project.get("python_paths", [])]
    for value in paths:
        if value and Path(str(value)).is_dir() and str(value) not in sys.path:
            sys.path.insert(0, str(value))
    add_repo_to_path(config, Path(str(project.get("root", "."))))


__all__ = ["DEFAULT_CONFIG_PATH", "apply_python_paths", "load_config"]
