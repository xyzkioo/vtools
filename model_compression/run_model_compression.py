#!/usr/bin/env python3
"""模型压缩工具入口：每次运行自包含，不依赖跨运行注册表。"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    # 直接运行文件时把仓库根目录加入导入路径，保留包限定导入，避免
    # model_compression.modules 与其他子项目的 ``modules`` 重名。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # 支持 ``python model_compression/run_model_compression.py``
    from .core.config import apply_python_paths, load_config
    from .core.model_runtime import (
        build_dataloaders,
        benchmark_model_call,
        evaluate_classification,
        load_model,
        parameter_metrics,
        save_model,
        resolve_device,
    )
    from .core.module_selection import COMPRESSION_MODULES, resolve_compression_modules
    from .core.metrics import compare_to_baseline
    from .core.registry import artifact_info, environment_info
    from .core.run_manager import prepare_run_directory, write_json
    from .modules.distillation import train_distillation
    from .modules.pruning import prune_unstructured
    from .modules.quantization import quantize_dynamic_int8
except ImportError:  # pragma: no cover - compatibility with unusual launchers
    from model_compression.core.config import apply_python_paths, load_config
    from model_compression.core.model_runtime import benchmark_model_call, build_dataloaders, evaluate_classification, load_model, parameter_metrics, save_model, resolve_device
    from model_compression.core.module_selection import COMPRESSION_MODULES, resolve_compression_modules
    from model_compression.core.metrics import compare_to_baseline
    from model_compression.core.registry import artifact_info, environment_info
    from model_compression.core.run_manager import prepare_run_directory, write_json
    from model_compression.modules.distillation import train_distillation
    from model_compression.modules.pruning import prune_unstructured
    from model_compression.modules.quantization import quantize_dynamic_int8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型压缩、实验分支与知识蒸馏")
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置；省略时读取 config/config.yaml")
    parser.add_argument("--run-dir", default=None, help="复用或指定运行目录")
    parser.add_argument("--only", action="append", help="只运行指定模块，可重复或逗号分隔")
    parser.add_argument("--enable", action="append", help="临时启用模块")
    parser.add_argument("--disable", action="append", help="临时关闭模块")
    parser.add_argument("--weights", type=Path, help="临时覆盖 model.weights")
    parser.add_argument("--data", type=Path, help="YOLO 数据集 data.yaml")
    parser.add_argument("--task", choices=["detect", "classify"], help="模型任务")
    parser.add_argument("--train", type=Path, help="临时覆盖 dataset.train")
    parser.add_argument("--val", type=Path, help="临时覆盖 dataset.val")
    parser.add_argument("--device", help="临时覆盖 model.device：auto、cpu、cuda:0 或 0")
    parser.add_argument("--structured-scale", help="临时覆盖结构化目标规模，例如 n、s、m、l 或 x")
    parser.add_argument("--structured-method", choices=["scale", "torch_pruning"], help="结构化剪枝方法")
    parser.add_argument("--structured-epochs", type=int, help="临时覆盖结构化缩放后的微调轮数；0 表示跳过微调")
    parser.add_argument("--structured-initial-weights", type=Path, help="结构化目标规模的预训练初始化权重")
    parser.add_argument("--list-modules", action="store_true", help="列出模块后退出")
    return parser


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _model_config(config: Mapping[str, Any], *, weights: str | None = None, **overrides: Any) -> dict[str, Any]:
    model = _section(config, "model")
    if weights is not None:
        model["weights"] = weights
    model.update(overrides)
    benchmark = _section(config, "benchmark")
    model.setdefault("device", benchmark.get("device", "auto"))
    return model


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(config)
    dataset = _section(config, "dataset")
    distillation = _section(config, "distillation")
    values["input_size"] = dataset.get("input_size", [224, 224])
    values["batch_size"] = distillation.get("batch_size", 1)
    benchmark = _section(config, "benchmark")
    values["device"] = benchmark.get("device", _section(config, "model").get("device", "auto"))
    values["benchmark"] = dict(benchmark)
    values["benchmark"].setdefault("input_size", values["input_size"])
    values["benchmark"].setdefault("batch_size", values["batch_size"])
    values["benchmark"].setdefault("device", values["device"])
    return values


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text) or fallback


def _write_effective_config(config: Mapping[str, Any], run_dir: Path | None) -> None:
    if run_dir is None:
        return
    write_json(run_dir / "effective_config.json", {key: value for key, value in config.items() if not key.startswith("_")})


class _RunRegistry:
    """运行期间的内存状态兼容层。

    旧实现把每次模块调用都写入 ``store/registry.json``。连续模块只需要
    当前模型路径和本次运行内的轻量血缘，因此这里保留旧调用方需要的接口，
    但所有状态仅存在于当前进程，绝不创建跨运行文件。
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "branches": {},
            "versions": {},
            "relations": [],
            "runs": [],
        }
        self._counter = 0

    def save(self) -> None:
        return None

    def ensure_branch(self, name: str, *, from_version: str | None = None, description: str = "") -> dict[str, Any]:
        branch_name = str(name or "main").strip() or "main"
        existing = self.data["branches"].get(branch_name)
        if existing is not None:
            return dict(existing)
        if from_version:
            raise ValueError("已移除跨运行 branch.from_version；请将模型产物路径填写到 model.weights。")
        branch = {"name": branch_name, "description": description, "head_version_id": None}
        self.data["branches"][branch_name] = branch
        return dict(branch)

    def get_branch(self, name: str) -> dict[str, Any]:
        return dict(self.data["branches"].setdefault(str(name or "main"), {
            "name": str(name or "main"), "description": "", "head_version_id": None,
        }))

    def add_version(self, *, name: str, artifact: str | Path, parent_version_id: str | None,
                    run_id: str, method: str, metadata: Mapping[str, Any] | None = None,
                    teacher_version_id: str | None = None,
                    student_init_version_id: str | None = None) -> dict[str, Any]:
        self._counter += 1
        version_id = f"run-{run_id}-{self._counter}"
        record = {
            "id": version_id,
            "name": name,
            "artifact": artifact_info(artifact),
            "parent_version_id": parent_version_id,
            "teacher_version_id": teacher_version_id,
            "student_init_version_id": student_init_version_id,
            "run_id": run_id,
            "method": method,
            "metadata": dict(metadata or {}),
        }
        self.data["versions"][version_id] = record
        if parent_version_id:
            self.data["relations"].append({"source_version_id": parent_version_id, "target_version_id": version_id, "role": "parent"})
        return dict(record)

    def advance_branch(self, branch_name: str, version_id: str) -> dict[str, Any]:
        branch = self.get_branch(branch_name)
        branch["head_version_id"] = version_id
        self.data["branches"][branch["name"]] = branch
        return dict(branch)

    def add_run(self, run_id: str, record: Mapping[str, Any]) -> None:
        self.data["runs"].append({"id": run_id, **dict(record)})

    def resolve_input_version(self, branch_name: str, explicit_weights: str | Path | None) -> tuple[str | None, str | None]:
        branch = self.get_branch(branch_name)
        head = branch.get("head_version_id")
        if head:
            version = self.data["versions"][str(head)]
            return str(head), str(version["artifact"]["path"])
        if explicit_weights:
            return None, str(Path(str(explicit_weights)).expanduser().resolve())
        return None, None

    def versions_for_branch(self, branch_name: str) -> list[dict[str, Any]]:
        current = self.get_branch(branch_name).get("head_version_id")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        while current:
            if current in seen:
                raise ValueError("检测到循环模型血缘")
            seen.add(str(current))
            version = self.data["versions"].get(str(current))
            if version is None:
                break
            rows.append(dict(version))
            current = version.get("parent_version_id")
        return rows


def _ensure_branch(registry: _RunRegistry, config: Mapping[str, Any]) -> dict[str, Any]:
    branch = _section(config, "branch")
    return registry.ensure_branch(
        str(branch.get("name", "main")),
        from_version=branch.get("from_version"),
        description=str(branch.get("description", "")),
    )


def _branch_name(config: Mapping[str, Any]) -> str:
    return str(_section(config, "branch").get("name", "main"))


def _base_version(
    registry: _RunRegistry,
    config: Mapping[str, Any],
    branch: Mapping[str, Any],
    *,
    run_id: str,
) -> str:
    # 前一个模块可能已经推进了分支；不要使用入口时的旧快照。
    current_id, _artifact = registry.resolve_input_version(
        _branch_name(config), _section(config, "model").get("weights")
    )
    if current_id:
        return current_id
    model = _section(config, "model")
    weights = model.get("weights")
    if not weights:
        raise ValueError("当前运行没有模型起点；请填写 model.weights 或 --weights")
    record = registry.add_version(
        name=str(model.get("name", "base_model")),
        artifact=str(weights),
        parent_version_id=None,
        run_id=run_id,
        method="import",
        metadata={"task": model.get("task", "classify"), "adapter": model.get("adapter", "torch"), "environment": environment_info()},
    )
    registry.advance_branch(_branch_name(config), record["id"])
    return str(record["id"])


def _version_artifact(registry: _RunRegistry, version_id: str) -> str:
    version = registry.data["versions"].get(version_id)
    if not version:
        raise KeyError(f"找不到模型版本：{version_id}")
    path = version.get("artifact", {}).get("path")
    if not path:
        raise ValueError(f"模型版本缺少 artifact.path：{version_id}")
    return str(path)


def _version_device(registry: _RunRegistry, version_id: str, config: Mapping[str, Any]) -> Any:
    """动态量化产物固定在 CPU，避免尝试把量化模块迁移到 CUDA。"""

    version = registry.data["versions"].get(version_id, {})
    if version.get("method") == "dynamic_int8":
        return "cpu"
    return _section(config, "model").get("device", "auto")


def _result_path(run_dir: Path, filename: str) -> Path:
    target = run_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _artifact_path(run_dir: Path, relative: str) -> Path:
    """Return an immutable artifact path, avoiding silent overwrite on reruns."""
    target = run_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(2, 10000):
        candidate = target.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"运行目录中的产物过多，无法生成唯一文件名：{target}")


def _run_baseline(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    current_id, current_artifact = registry.resolve_input_version(
        _branch_name(config), _section(config, "model").get("weights")
    )
    if current_id:
        current_path = Path(str(current_artifact)).resolve()
        model_config = _model_config(config, weights=str(current_path))
        model_device = _version_device(registry, str(current_id), config)
    else:
        model_config = _model_config(config)
        model_device = model_config.get("device", "auto")
    model, adapter = load_model(model_config, device=model_device)
    runtime = _runtime_config(config)
    _, val_loader = build_dataloaders(runtime, adapter)
    metrics = parameter_metrics(model)
    evaluation = evaluate_classification(model, val_loader, device=model_device)
    metrics.update({f"evaluation_{key}": value for key, value in evaluation.items()})
    try:
        metrics["model_file_size_bytes"] = artifact_info(model_config["weights"])["size_bytes"]
    except (KeyError, FileNotFoundError):
        metrics["model_file_size_bytes"] = None
    metrics.update({f"benchmark_{key}": value for key, value in benchmark_model_call(model, runtime, device=model_device).items()})
    result = {"module": "baseline.evaluate", "status": "succeeded", "metrics": metrics}
    try:
        result["artifact"] = artifact_info(model_config["weights"])
    except (KeyError, FileNotFoundError):
        pass
    if not registry.get_branch(_branch_name(config)).get("head_version_id"):
        version = registry.add_version(
            name=str(model_config.get("name", "base_model")),
            artifact=str(model_config["weights"]),
            parent_version_id=None,
            run_id=run_id,
            method="baseline",
            metadata=metrics,
        )
        registry.advance_branch(_branch_name(config), version["id"])
        result["version_id"] = version["id"]
    else:
        current_id = str(registry.get_branch(_branch_name(config))["head_version_id"])
        current = registry.data["versions"].get(current_id)
        if isinstance(current, dict):
            current.setdefault("metadata", {}).update(metrics)
            registry.save()
            result["version_id"] = current_id
    return result


def _run_parameters(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    version_id = _base_version(registry, config, branch, run_id=run_id)
    artifact = _version_artifact(registry, version_id)
    model, _ = load_model(_model_config(config, weights=artifact), device=_version_device(registry, version_id, config))
    result = {"module": "model.parameters", "status": "succeeded", "version_id": version_id,
              "artifact": artifact_info(artifact), "metrics": parameter_metrics(model)}
    return result


def _register_transformed(
    *,
    config: Mapping[str, Any],
    registry: _RunRegistry,
    parent_id: str,
    artifact: Path,
    run_id: str,
    method: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    model_name = _section(config, "model").get("name", "model")
    version_metadata = dict(metadata)
    version_metadata.setdefault("environment", environment_info())
    record = registry.add_version(
        name=f"{model_name}-{method}",
        artifact=artifact,
        parent_version_id=parent_id,
        run_id=run_id,
        method=method,
        metadata=version_metadata,
    )
    registry.advance_branch(_branch_name(config), record["id"])
    return record


def _run_quantize(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    model_config = _model_config(config, weights=_version_artifact(registry, parent_id))
    model, adapter = load_model(model_config, device=_version_device(registry, parent_id, config))
    transformed, metadata = quantize_dynamic_int8(model, _section(config, "compression"))
    metadata.update({f"benchmark_{key}": value for key, value in benchmark_model_call(transformed, _runtime_config(config), device="cpu").items()})
    _, val_loader = build_dataloaders(_runtime_config(config), adapter)
    metadata.update({f"evaluation_{key}": value for key, value in evaluate_classification(transformed, val_loader, device="cpu").items()})
    artifact = _artifact_path(run_dir, f"artifacts/quantization/{_safe_name(_section(config, 'model').get('name'), 'model')}-dynamic-int8.pt")
    save_model(transformed, artifact, adapter=None)
    record = _register_transformed(config=config, registry=registry, parent_id=parent_id, artifact=artifact, run_id=run_id, method="dynamic_int8", metadata=metadata)
    result = {"module": "compression.quantize.dynamic_int8", "status": "succeeded", "version_id": record["id"], "parent_version_id": parent_id, "artifact": artifact_info(artifact), "metadata": metadata}
    return result


def _run_prune(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    model_config = _model_config(config, weights=_version_artifact(registry, parent_id))
    model, adapter = load_model(model_config, device=_version_device(registry, parent_id, config))
    transformed, metadata = prune_unstructured(model, _section(config, "compression"))
    metadata.update({f"benchmark_{key}": value for key, value in benchmark_model_call(transformed, _runtime_config(config), device=_version_device(registry, parent_id, config)).items()})
    _, val_loader = build_dataloaders(_runtime_config(config), adapter)
    metadata.update({f"evaluation_{key}": value for key, value in evaluate_classification(transformed, val_loader, device=_version_device(registry, parent_id, config)).items()})
    artifact = _artifact_path(run_dir, f"artifacts/unstructured_pruning/{_safe_name(_section(config, 'model').get('name'), 'model')}-unstructured-l1.pt")
    save_model(transformed, artifact, adapter=adapter)
    record = _register_transformed(config=config, registry=registry, parent_id=parent_id, artifact=artifact, run_id=run_id, method="unstructured_l1", metadata=metadata)
    result = {"module": "compression.prune.unstructured", "status": "succeeded", "version_id": record["id"], "parent_version_id": parent_id, "artifact": artifact_info(artifact), "metadata": metadata}
    return result


def _run_structured_prune(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    """Classification entry point guard; detection is routed separately."""

    raise ValueError("结构化通道裁剪当前只支持 YOLO 检测模型（model.task: detect）。")


def _distillation_model_config(config: Mapping[str, Any], section_name: str, *, fallback_weights: str | None = None) -> dict[str, Any]:
    section = _section(config, section_name)
    distill = _section(config, "distillation")
    if section_name == "teacher":
        weights = section.get("weights", distill.get("teacher_weights"))
        factory = section.get("factory")
    else:
        weights = section.get("weights", distill.get("student_weights"))
        factory = section.get("factory", distill.get("student_factory"))
    values = _model_config(config, weights=weights or fallback_weights)
    if factory:
        values["factory"] = factory
        if section_name == "student":
            values["force_factory"] = True
            if not weights:
                values.pop("weights", None)
    return values


def _source_version(registry: _RunRegistry, path: str | Path, *, run_id: str, role: str) -> str:
    info = artifact_info(path)
    for version_id, version in registry.data.get("versions", {}).items():
        if version.get("artifact", {}).get("sha256") == info.get("sha256"):
            return str(version_id)
    record = registry.add_version(
        name=f"{role}-{Path(path).stem}", artifact=path, parent_version_id=None,
        run_id=run_id, method=f"import_{role}", metadata={"role": role, "environment": environment_info()},
    )
    return str(record["id"])


def _run_distillation(config: Mapping[str, Any], registry: _RunRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    parent_artifact = _version_artifact(registry, parent_id)
    teacher_config = _distillation_model_config(config, "teacher", fallback_weights=parent_artifact)
    student_config = _distillation_model_config(config, "student", fallback_weights=parent_artifact)
    distill_device = _version_device(registry, parent_id, config)
    teacher, teacher_adapter = load_model(teacher_config, device=distill_device)
    student, student_adapter = load_model(student_config, device=distill_device)
    teacher_source = teacher_config.get("weights") or parent_artifact
    student_source = student_config.get("weights")
    teacher_version_id = _source_version(registry, teacher_source, run_id=run_id, role="teacher")
    student_version_id = _source_version(registry, student_source, run_id=run_id, role="student") if student_source else None
    runtime = _runtime_config(config)
    train_loader, val_loader = build_dataloaders(runtime, student_adapter or teacher_adapter)
    output_dir = run_dir / "artifacts" / "distillation"
    if output_dir.exists() and any(output_dir.iterdir()):
        for index in range(2, 10000):
            candidate = run_dir / "artifacts" / "distillation" / f"attempt-{index}"
            if not candidate.exists():
                output_dir = candidate
                break
    training = train_distillation(
        student,
        teacher,
        train_loader,
        val_loader,
        config,
        device=distill_device,
        output_dir=output_dir,
    )
    checkpoint = training.get("best_checkpoint") or training.get("last_checkpoint")
    if not checkpoint:
        raise RuntimeError("蒸馏未生成 checkpoint")
    record = _register_transformed(
        config=config,
        registry=registry,
        parent_id=parent_id,
        artifact=Path(checkpoint),
        run_id=run_id,
        method="distillation_classification",
        metadata={
            "teacher_weights": str(teacher_source),
            "student_weights": str(student_source) if student_source else None,
            "training": training,
        },
    )
    # 保留运行内存中的关系，供同一次运行的 comparison 使用；不会落盘。
    registry.data["versions"][record["id"]]["teacher_version_id"] = teacher_version_id
    registry.data["versions"][record["id"]]["student_init_version_id"] = student_version_id
    relations = [{"source_version_id": teacher_version_id, "target_version_id": record["id"], "role": "teacher"}]
    if student_version_id:
        relations.append({"source_version_id": student_version_id, "target_version_id": record["id"], "role": "student_initialization"})
    registry.data.setdefault("relations", []).extend(relations)
    registry.save()
    result = {
        "module": "distillation.classification",
        "status": "succeeded",
        "version_id": record["id"],
        "parent_version_id": parent_id,
        "artifact": artifact_info(checkpoint),
        "training": training,
    }
    return result


def _run_comparison(config: Mapping[str, Any], registry: _RunRegistry, run_dir: Path) -> dict[str, Any]:
    branch = registry.get_branch(_branch_name(config))
    versions = registry.versions_for_branch(_branch_name(config))
    rows = []
    baseline_row: dict[str, Any] | None = None
    for version in versions:
        metrics = version.get("metadata", {})
        training = metrics.get("training") if isinstance(metrics.get("training"), Mapping) else {}
        row = {"version_id": version["id"], "name": version.get("name"), "method": version.get("method"), "parent_version_id": version.get("parent_version_id"), "map50": metrics.get("evaluation_map50"), "map50_95": metrics.get("evaluation_map50_95"), "accuracy": metrics.get("evaluation_accuracy", metrics.get("best_accuracy", training.get("best_accuracy"))), "size_bytes": version.get("artifact", {}).get("size_bytes"), "actual_sparsity": metrics.get("actual_sparsity"), "latency_ms": metrics.get("benchmark_latency_p50_ms", metrics.get("benchmark_inference_ms_median", metrics.get("after_benchmark_inference_ms_median")))}
        if version.get("method") in {"import", "baseline"}:
            baseline_row = row
        rows.append(row)
    if baseline_row:
        for row in rows:
            row.update(compare_to_baseline(baseline_row, row))
            if row.get("map50_95") is not None and baseline_row.get("map50_95") is not None:
                row["map50_95_delta"] = row["map50_95"] - baseline_row["map50_95"]
    result = {"module": "comparison.report", "status": "succeeded", "branch": branch, "versions": rows}
    return result


def _run_export(config: Mapping[str, Any], registry: _RunRegistry, run_dir: Path) -> dict[str, Any]:
    branch = registry.get_branch(_branch_name(config))
    version_id = branch.get("head_version_id")
    if not version_id:
        raise ValueError("当前分支没有可导出的模型版本")
    source = Path(_version_artifact(registry, str(version_id))).resolve()
    export = _section(config, "export")
    target = Path(str(export.get("path"))).expanduser().resolve() if export.get("path") else _artifact_path(run_dir, f"artifacts/multi_processing/{source.stem}-export{source.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target != source and target.exists():
        target = _artifact_path(target.parent, target.name)
    if source != target:
        shutil.copy2(source, target)
    verification: dict[str, Any] = {"reload": False, "sample_inference": "not_requested"}
    model_config = _model_config(config, weights=str(target))
    loaded, adapter = load_model(model_config, device=_version_device(registry, str(version_id), config))
    verification["reload"] = True
    runtime = _runtime_config(config)
    _, val_loader = build_dataloaders(runtime, adapter)
    if val_loader is not None:
        try:
            import torch  # type: ignore

            batch = next(iter(val_loader))
            inputs = batch["image"] if isinstance(batch, Mapping) else batch[0]
            with torch.no_grad():
                loaded(inputs.to(resolve_device(_version_device(registry, str(version_id), config))))
            verification["sample_inference"] = "succeeded"
        except StopIteration:
            verification["sample_inference"] = "skipped_empty_validation"
    result = {"module": "artifact.export", "status": "succeeded", "version_id": version_id, "artifact": artifact_info(target), "verification": verification}
    return result


_SUMMARY_CATEGORIES = {
    "baseline.evaluate": "baseline",
    "model.parameters": "baseline",
    "compression.prune.unstructured": "unstructured_pruning",
    "compression.prune.structured": "structured_pruning",
    "distillation.classification": "distillation",
    "compression.quantize.dynamic_int8": "quantization",
    "artifact.export": "multi_processing",
}

_PARAMETER_KEYS = {
    "parameter_count", "trainable_parameter_count", "zero_parameter_count",
    "pruned_parameter_count", "requested_sparsity", "actual_sparsity",
    "sparsity", "structured_method", "target_scale", "architecture_yaml",
    "finetune_epochs", "finetune", "temperature", "alpha", "epochs",
    "best_epoch", "teacher_weights", "student_weights", "method",
}


def _relative_path(run_dir: Path, value: Any) -> str:
    path = Path(str(value)).expanduser()
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_summary_value(value: Any, run_dir: Path, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(name): _normalize_summary_value(item, run_dir, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_normalize_summary_value(item, run_dir, key) for item in value]
    if isinstance(value, (str, Path)) and any(token in key.lower() for token in ("path", "checkpoint", "yaml", "artifact")):
        text = str(value)
        candidate = Path(text).expanduser()
        if candidate.is_absolute() and (candidate.exists() or run_dir in candidate.parents):
            return _relative_path(run_dir, candidate)
    return value


def _summary_result(module_id: str, result: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """将模块内部返回值转换成面向用户的去重摘要。"""

    if module_id == "comparison.report":
        rows = []
        for row in result.get("versions", []) if isinstance(result.get("versions"), list) else []:
            if not isinstance(row, Mapping):
                continue
            rows.append({key: value for key, value in row.items()
                         if key not in {"version_id", "parent_version_id"}})
        return {"status": str(result.get("status", "unknown")), "metrics": {"rows": rows}}
    category = _SUMMARY_CATEGORIES.get(module_id)
    if category is None:
        return {"status": str(result.get("status", "unknown")), "module": module_id,
                "reason": result.get("error") or result.get("reason")}
    record: dict[str, Any] = {"status": str(result.get("status", "unknown")), "module": module_id}
    if result.get("method"):
        record["method"] = result["method"]
    raw: dict[str, Any] = {}
    for key in ("metrics", "metadata", "training"):
        value = result.get(key)
        if isinstance(value, Mapping):
            raw.update(value)
    # These nested snapshots duplicate the baseline metrics. Keep deltas and
    # gate decisions, while the baseline category remains the single source.
    raw.pop("before_pruning", None)
    raw.pop("before_structured", None)
    raw.pop("before_unstructured_pruning", None)
    raw.pop("before_structured_pruning", None)
    raw.pop("environment", None)
    for key in ("version_id", "parent_version_id", "branch", "sha256",
                "teacher_version_id", "student_init_version_id"):
        raw.pop(key, None)
    parameters: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for key, value in raw.items():
        text = str(key)
        if (text in _PARAMETER_KEYS or text.endswith("_parameters")):
            parameters[text] = _normalize_summary_value(value, run_dir, text)
        else:
            metrics[text] = _normalize_summary_value(value, run_dir, text)
    if parameters:
        record["parameters"] = parameters
    if metrics:
        record["metrics"] = metrics
    artifact = result.get("artifact")
    if isinstance(artifact, Mapping) and artifact.get("path"):
        record["artifact"] = _relative_path(run_dir, artifact["path"])
        metrics.setdefault("file_size_bytes", artifact.get("size_bytes"))
    elif isinstance(artifact, (str, Path)):
        record["artifact"] = _relative_path(run_dir, artifact)
    if result.get("verification") is not None:
        record["verification"] = result["verification"]
    if result.get("reason"):
        record["reason"] = result["reason"]
    if result.get("error"):
        record["error"] = result["error"]
    record.setdefault("parameters", {})
    record.setdefault("metrics", {})
    return record


def _build_summary(config: Mapping[str, Any], run_dir: Path, run_id: str,
                   selected: Mapping[str, bool], results: list[dict[str, Any]],
                   failures: list[str], started_at: str) -> dict[str, Any]:
    model = _section(config, "model")
    source_weights = model.get("weights")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failures else "succeeded",
        "model": {
            "name": model.get("name", "model"),
            "task": model.get("task", "classify"),
            "adapter": model.get("adapter", "torch"),
            "source_weights": str(source_weights) if source_weights else None,
            "environment": environment_info(),
        },
        "executed_modules": [key for key, enabled in selected.items() if enabled],
        "errors": list(failures),
    }
    category_results: dict[str, list[dict[str, Any]]] = {}
    module_status: dict[str, str] = {}
    for result in results:
        module_id = str(result.get("module", ""))
        module_status[module_id] = str(result.get("status", "unknown"))
        item = _summary_result(module_id, result, run_dir)
        category = _SUMMARY_CATEGORIES.get(module_id)
        if category:
            category_results.setdefault(category, []).append(item)
        elif module_id == "comparison.report":
            summary["comparison"] = item
    summary["module_status"] = module_status
    for category, items in category_results.items():
        if category == "baseline":
            merged: dict[str, Any] = {"status": "succeeded" if all(item.get("status") == "succeeded" for item in items) else "failed",
                                      "module": "baseline.evaluate"}
            for item in items:
                for field in ("parameters", "metrics"):
                    if isinstance(item.get(field), Mapping):
                        merged.setdefault(field, {}).update(item[field])
                if item.get("artifact") and "artifact" not in merged:
                    merged["artifact"] = item["artifact"]
                if item.get("verification") is not None:
                    merged["verification"] = item["verification"]
            summary[category] = merged
        elif len(items) == 1:
            summary[category] = items[0]
        else:
            summary[category] = {"status": "succeeded" if all(item.get("status") == "succeeded" for item in items) else "failed",
                                 "steps": items}

    processing_items = [item for result in results
                        if result.get("module") in _SUMMARY_CATEGORIES
                        and result.get("module") not in {"baseline.evaluate", "model.parameters"}
                        and result.get("status") == "succeeded"]
    if len(processing_items) >= 2:
        final = processing_items[-1]
        final_artifact = final.get("artifact")
        multi: dict[str, Any] = {
            "status": "succeeded",
            "parameters": {"steps": [str(result.get("module")) for result in results
                                        if result.get("module") in _SUMMARY_CATEGORIES
                                        and result.get("module") not in {"baseline.evaluate", "model.parameters"}]},
            "metrics": {"final_stage": final.get("module")},
        }
        if final_artifact:
            source = (run_dir / str(final_artifact)).resolve()
            if source.is_file():
                target = _artifact_path(run_dir, f"artifacts/multi_processing/{source.stem}-final{source.suffix}")
                if source != target:
                    shutil.copy2(source, target)
                multi["artifact"] = _relative_path(run_dir, target)
            else:
                multi["status"] = "failed"
                multi["error"] = f"最终阶段产物不存在：{source}"
        summary["multi_processing"] = multi
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.list_modules:
        print("\n".join(COMPRESSION_MODULES))
        return 0
    config = load_config(args.config)
    if (args.weights or args.train or args.val or args.device or args.data or args.task
            or args.structured_method is not None or args.structured_scale is not None or args.structured_epochs is not None
            or args.structured_initial_weights is not None):
        config = dict(config)
        model = dict(_section(config, "model"))
        dataset = dict(_section(config, "dataset"))
        if args.weights:
            model["weights"] = str(args.weights.expanduser().resolve())
        if args.data:
            dataset["data"] = str(args.data.expanduser().resolve())
        if args.task:
            model["task"] = args.task
        if args.train:
            dataset["train"] = str(args.train.expanduser().resolve())
        if args.val:
            dataset["val"] = str(args.val.expanduser().resolve())
        if args.device:
            model["device"] = args.device
            benchmark = dict(_section(config, "benchmark"))
            benchmark["device"] = args.device
            config["benchmark"] = benchmark
        if (args.structured_method is not None or args.structured_scale is not None or args.structured_epochs is not None
                or args.structured_initial_weights is not None):
            compression = dict(_section(config, "compression"))
            structured = dict(compression.get("structured", {}) if isinstance(compression.get("structured"), Mapping) else {})
            if args.structured_method is not None:
                structured["method"] = args.structured_method
            if args.structured_scale is not None:
                structured["target_scale"] = args.structured_scale
            if args.structured_epochs is not None:
                structured["finetune_epochs"] = args.structured_epochs
            if args.structured_initial_weights is not None:
                structured["initial_weights"] = str(args.structured_initial_weights.expanduser().resolve())
            compression["structured"] = structured
            config["compression"] = compression
        config["model"] = model
        config["dataset"] = dataset
    apply_python_paths(config)
    selected = resolve_compression_modules(config, only=args.only, enable=args.enable, disable=args.disable)
    run_dir = prepare_run_directory(config, args.run_dir)
    if run_dir is None:
        run_dir = Path(_section(config, "project").get("root", Path.cwd())) / "model_compression" / "runs" / "adhoc"
        run_dir.mkdir(parents=True, exist_ok=True)
        for category in ("structured_pruning", "unstructured_pruning", "distillation", "quantization", "multi_processing"):
            (run_dir / "artifacts" / category).mkdir(parents=True, exist_ok=True)
    _write_effective_config(config, run_dir)
    registry = _RunRegistry()
    branch = _ensure_branch(registry, config)
    config = dict(config)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(run_dir.name or uuid.uuid4().hex[:8])
    registry.add_run(run_id, {"status": "running", "branch": _branch_name(config), "modules": selected})
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    failed_dependencies: set[str] = set()
    transform_modules = {
        "compression.quantize.dynamic_int8", "compression.prune.unstructured",
        "compression.prune.structured", "distillation.classification",
        "artifact.export", "comparison.report",
    }
    operations = (
        ("branch.create", lambda: {"module": "branch.create", "status": "skipped", "reason": "运行目录已自包含，不再维护跨运行分支"}),
        ("model.parameters", lambda: _run_parameters(config, registry, branch, run_dir, run_id)),
        ("baseline.evaluate", lambda: _run_baseline(config, registry, branch, run_dir, run_id)),
        ("compression.quantize.dynamic_int8", lambda: _run_quantize(config, registry, branch, run_dir, run_id)),
        ("compression.prune.unstructured", lambda: _run_prune(config, registry, branch, run_dir, run_id)),
        ("compression.prune.structured", lambda: _run_structured_prune(config, registry, branch, run_dir, run_id)),
        ("distillation.classification", lambda: _run_distillation(config, registry, branch, run_dir, run_id)),
        ("comparison.report", lambda: _run_comparison(config, registry, run_dir)),
        ("artifact.export", lambda: _run_export(config, registry, run_dir)),
    )
    for module_id, function in operations:
        if not selected.get(module_id, False):
            continue
        if module_id in transform_modules and ("baseline.evaluate" in failed_dependencies or failed_dependencies & transform_modules):
            reason = "前置模块失败，已跳过依赖阶段"
            results.append({"module": module_id, "status": "skipped", "reason": reason})
            print(f"[{module_id}] 跳过：{reason}")
            continue
        try:
            from model_compression.core.detection import is_detection, run_detection_operation
            if is_detection(config) and module_id not in {"branch.create", "comparison.report"}:
                result = run_detection_operation(module_id, config, registry, run_dir, run_id)
            else:
                result = function()
            results.append(result)
            print(f"[{module_id}] {'跳过' if result.get('status') == 'skipped' else '完成'}")
        except Exception as error:  # noqa: BLE001 - 单个实验失败也要留下可读报告
            message = f"{type(error).__name__}: {error}"
            failures.append(f"{module_id}: {message}")
            failed_dependencies.add(module_id)
            failed_result: dict[str, Any] = {"module": module_id, "status": "failed", "error": message}
            category = _SUMMARY_CATEGORIES.get(module_id)
            if category:
                candidates = sorted((run_dir / "artifacts" / category).glob("*.pt"), key=lambda path: path.stat().st_mtime)
                if candidates:
                    failed_result["artifact"] = artifact_info(candidates[-1])
            results.append(failed_result)
            print(f"[{module_id}] 失败：{message}")
    summary = _build_summary(config, run_dir, run_id, selected, results, failures, started_at)
    write_json(_result_path(run_dir, "summary.json"), summary)
    registry.add_run(run_id, {"status": summary["status"], "summary": str(run_dir / "summary.json")})
    print(f"本次运行目录：{run_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
