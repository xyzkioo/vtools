#!/usr/bin/env python3
"""模型压缩工具入口：基线、分支、量化、剪枝、蒸馏和对比共用一套配置。"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
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
    from .core.registry import ModelRegistry, artifact_info, environment_info
    from .core.run_manager import prepare_run_directory, write_json
    from .modules.distillation import train_distillation
    from .modules.pruning import prune_unstructured
    from .modules.quantization import quantize_dynamic_int8
except ImportError:  # pragma: no cover - compatibility with unusual launchers
    from model_compression.core.config import apply_python_paths, load_config
    from model_compression.core.model_runtime import benchmark_model_call, build_dataloaders, evaluate_classification, load_model, parameter_metrics, save_model, resolve_device
    from model_compression.core.module_selection import COMPRESSION_MODULES, resolve_compression_modules
    from model_compression.core.metrics import compare_to_baseline
    from model_compression.core.registry import ModelRegistry, artifact_info, environment_info
    from model_compression.core.run_manager import prepare_run_directory, write_json
    from model_compression.modules.distillation import train_distillation
    from model_compression.modules.pruning import prune_unstructured
    from model_compression.modules.quantization import quantize_dynamic_int8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型压缩、实验分支与知识蒸馏")
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置；省略时读取 config/pycharm_run.yaml")
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


def _ensure_branch(registry: ModelRegistry, config: Mapping[str, Any]) -> dict[str, Any]:
    branch = _section(config, "branch")
    return registry.ensure_branch(
        str(branch.get("name", "main")),
        from_version=branch.get("from_version"),
        description=str(branch.get("description", "")),
    )


def _branch_name(config: Mapping[str, Any]) -> str:
    return str(_section(config, "branch").get("name", "main"))


def _base_version(
    registry: ModelRegistry,
    config: Mapping[str, Any],
    branch: Mapping[str, Any],
    *,
    run_id: str,
) -> str:
    # 前一个模块可能已经推进了分支；不要使用入口时的旧快照。
    current_branch = registry.get_branch(_branch_name(config))
    if current_branch.get("head_version_id"):
        return str(current_branch["head_version_id"])
    model = _section(config, "model")
    weights = model.get("weights")
    if not weights:
        raise ValueError("当前分支没有起点；请填写 model.weights 或 branch.from_version")
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


def _version_artifact(registry: ModelRegistry, version_id: str) -> str:
    version = registry.data["versions"].get(version_id)
    if not version:
        raise KeyError(f"找不到模型版本：{version_id}")
    path = version.get("artifact", {}).get("path")
    if not path:
        raise ValueError(f"模型版本缺少 artifact.path：{version_id}")
    return str(path)


def _version_device(registry: ModelRegistry, version_id: str, config: Mapping[str, Any]) -> Any:
    """动态量化产物固定在 CPU，避免尝试把量化模块迁移到 CUDA。"""

    version = registry.data["versions"].get(version_id, {})
    if version.get("method") == "dynamic_int8":
        return "cpu"
    return _section(config, "model").get("device", "auto")


def _result_path(run_dir: Path, filename: str) -> Path:
    target = run_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _run_baseline(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    current_branch = registry.get_branch(_branch_name(config))
    current_id = current_branch.get("head_version_id")
    if current_id:
        model_config = _model_config(config, weights=_version_artifact(registry, str(current_id)))
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
    write_json(_result_path(run_dir, "baseline.json"), result)
    return result


def _run_parameters(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    version_id = _base_version(registry, config, branch, run_id=run_id)
    model, _ = load_model(_model_config(config, weights=_version_artifact(registry, version_id)), device=_version_device(registry, version_id, config))
    result = {"module": "model.parameters", "status": "succeeded", "version_id": version_id, "metrics": parameter_metrics(model)}
    write_json(_result_path(run_dir, "model_parameters.json"), result)
    return result


def _register_transformed(
    *,
    config: Mapping[str, Any],
    registry: ModelRegistry,
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


def _run_quantize(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    model_config = _model_config(config, weights=_version_artifact(registry, parent_id))
    model, adapter = load_model(model_config, device=_version_device(registry, parent_id, config))
    transformed, metadata = quantize_dynamic_int8(model, _section(config, "compression"))
    metadata.update({f"benchmark_{key}": value for key, value in benchmark_model_call(transformed, _runtime_config(config), device="cpu").items()})
    _, val_loader = build_dataloaders(_runtime_config(config), adapter)
    metadata.update({f"evaluation_{key}": value for key, value in evaluate_classification(transformed, val_loader, device="cpu").items()})
    artifact = _result_path(run_dir, f"models/{_safe_name(_branch_name(config), 'main')}-dynamic-int8.pt")
    save_model(transformed, artifact, adapter=None)
    record = _register_transformed(config=config, registry=registry, parent_id=parent_id, artifact=artifact, run_id=run_id, method="dynamic_int8", metadata=metadata)
    result = {"module": "compression.quantize.dynamic_int8", "status": "succeeded", "version_id": record["id"], "parent_version_id": parent_id, "artifact": artifact_info(artifact), "metadata": metadata}
    write_json(_result_path(run_dir, "quantization.json"), result)
    return result


def _run_prune(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    model_config = _model_config(config, weights=_version_artifact(registry, parent_id))
    model, adapter = load_model(model_config, device=_version_device(registry, parent_id, config))
    transformed, metadata = prune_unstructured(model, _section(config, "compression"))
    metadata.update({f"benchmark_{key}": value for key, value in benchmark_model_call(transformed, _runtime_config(config), device=_version_device(registry, parent_id, config)).items()})
    _, val_loader = build_dataloaders(_runtime_config(config), adapter)
    metadata.update({f"evaluation_{key}": value for key, value in evaluate_classification(transformed, val_loader, device=_version_device(registry, parent_id, config)).items()})
    artifact = _result_path(run_dir, f"models/{_safe_name(_branch_name(config), 'main')}-pruned.pt")
    save_model(transformed, artifact, adapter=adapter)
    record = _register_transformed(config=config, registry=registry, parent_id=parent_id, artifact=artifact, run_id=run_id, method="unstructured_l1", metadata=metadata)
    result = {"module": "compression.prune.unstructured", "status": "succeeded", "version_id": record["id"], "parent_version_id": parent_id, "artifact": artifact_info(artifact), "metadata": metadata}
    write_json(_result_path(run_dir, "pruning.json"), result)
    return result


def _run_structured_prune(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
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
    return values


def _run_distillation(config: Mapping[str, Any], registry: ModelRegistry, branch: Mapping[str, Any], run_dir: Path, run_id: str) -> dict[str, Any]:
    parent_id = _base_version(registry, config, branch, run_id=run_id)
    parent_artifact = _version_artifact(registry, parent_id)
    teacher_config = _distillation_model_config(config, "teacher", fallback_weights=parent_artifact)
    student_config = _distillation_model_config(config, "student", fallback_weights=parent_artifact)
    distill_device = _version_device(registry, parent_id, config)
    teacher, teacher_adapter = load_model(teacher_config, device=distill_device)
    student, student_adapter = load_model(student_config, device=distill_device)
    runtime = _runtime_config(config)
    train_loader, val_loader = build_dataloaders(runtime, student_adapter or teacher_adapter)
    output_dir = run_dir / "distillation"
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
        metadata={"teacher_version_id": parent_id, "student_init_version_id": parent_id, "training": training},
    )
    # 注册表的关系字段需要单独保留，避免把教师和学生来源混在普通 parent 中。
    registry.data["versions"][record["id"]]["teacher_version_id"] = parent_id
    registry.data["versions"][record["id"]]["student_init_version_id"] = parent_id
    registry.data.setdefault("relations", []).extend([
        {"source_version_id": parent_id, "target_version_id": record["id"], "role": "teacher"},
        {"source_version_id": parent_id, "target_version_id": record["id"], "role": "student_initialization"},
    ])
    registry.save()
    result = {"module": "distillation.classification", "status": "succeeded", "version_id": record["id"], "parent_version_id": parent_id, "training": training}
    write_json(_result_path(run_dir, "distillation.json"), result)
    return result


def _run_comparison(config: Mapping[str, Any], registry: ModelRegistry, run_dir: Path) -> dict[str, Any]:
    branch = registry.get_branch(_branch_name(config))
    versions = registry.versions_for_branch(_branch_name(config))
    rows = []
    baseline_row: dict[str, Any] | None = None
    for version in versions:
        metrics = version.get("metadata", {})
        row = {"version_id": version["id"], "name": version.get("name"), "method": version.get("method"), "parent_version_id": version.get("parent_version_id"), "map50": metrics.get("evaluation_map50"), "map50_95": metrics.get("evaluation_map50_95"), "accuracy": metrics.get("evaluation_accuracy", metrics.get("best_accuracy")), "size_bytes": version.get("artifact", {}).get("size_bytes"), "actual_sparsity": metrics.get("actual_sparsity"), "latency_ms": metrics.get("benchmark_latency_p50_ms")}
        if version.get("method") in {"import", "baseline"}:
            baseline_row = row
        rows.append(row)
    if baseline_row:
        for row in rows:
            row.update(compare_to_baseline(baseline_row, row))
            if row.get("map50_95") is not None and baseline_row.get("map50_95") is not None:
                row["map50_95_delta"] = row["map50_95"] - baseline_row["map50_95"]
    result = {"module": "comparison.report", "status": "succeeded", "branch": branch, "versions": rows}
    write_json(_result_path(run_dir, "comparison.json"), result)
    return result


def _run_export(config: Mapping[str, Any], registry: ModelRegistry, run_dir: Path) -> dict[str, Any]:
    branch = registry.get_branch(_branch_name(config))
    version_id = branch.get("head_version_id")
    if not version_id:
        raise ValueError("当前分支没有可导出的模型版本")
    source = Path(_version_artifact(registry, str(version_id))).resolve()
    export = _section(config, "export")
    target = Path(str(export.get("path"))).expanduser().resolve() if export.get("path") else (run_dir / "export" / source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    manifest = {"version_id": version_id, "source": artifact_info(source), "exported": artifact_info(target), "verification": verification}
    write_json(target.parent / "model_manifest.json", manifest)
    result = {"module": "artifact.export", "status": "succeeded", "version_id": version_id, "artifact": artifact_info(target), "manifest": str(target.parent / "model_manifest.json")}
    write_json(_result_path(run_dir, "export.json"), result)
    return result


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
    operation = str(config.get("operation", "all") or "all").strip().lower()
    operation_modules = {
        "branch": "branch.create",
        "baseline": "baseline.evaluate",
        "parameters": "model.parameters",
        "quantize": "compression.quantize.dynamic_int8",
        "quantization": "compression.quantize.dynamic_int8",
        "dynamic_int8": "compression.quantize.dynamic_int8",
        "prune": "compression.prune.unstructured",
        "pruning": "compression.prune.unstructured",
        "structured_prune": "compression.prune.structured",
        "structured": "compression.prune.structured",
        "distill": "distillation.classification",
        "distillation": "distillation.classification",
        "compare": "comparison.report",
        "export": "artifact.export",
    }
    if not (args.only or args.enable or args.disable) and operation in operation_modules:
        selected = {key: False for key in selected}
        selected[operation_modules[operation]] = True
    run_dir = prepare_run_directory(config, args.run_dir)
    if run_dir is None:
        run_dir = Path(_section(config, "project").get("root", Path.cwd())) / "model_compression" / "runs" / "adhoc"
        run_dir.mkdir(parents=True, exist_ok=True)
    _write_effective_config(config, run_dir)
    storage = _section(config, "storage")
    registry = ModelRegistry(storage.get("registry", Path(run_dir) / "registry.json"))
    branch = _ensure_branch(registry, config)
    run_id = str(run_dir.name or uuid.uuid4().hex[:8])
    registry.add_run(run_id, {"status": "running", "branch": _branch_name(config), "modules": selected})
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    operations = (
        ("branch.create", lambda: {"module": "branch.create", "status": "succeeded", "branch": registry.get_branch(_branch_name(config))}),
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
        try:
            from model_compression.core.detection import is_detection, run_detection_operation
            if is_detection(config) and module_id not in {"branch.create", "comparison.report"}:
                result = run_detection_operation(module_id, config, registry, run_dir, run_id)
            else:
                result = function()
            results.append(result)
            print(f"[{module_id}] 完成")
        except Exception as error:  # noqa: BLE001 - 单个实验失败也要留下可读报告
            message = f"{type(error).__name__}: {error}"
            failures.append(f"{module_id}: {message}")
            results.append({"module": module_id, "status": "failed", "error": message})
            print(f"[{module_id}] 失败：{message}")
    summary = {"run_id": run_id, "status": "failed" if failures else "succeeded", "branch": _branch_name(config), "results": results, "errors": failures}
    write_json(_result_path(run_dir, "summary.json"), summary)
    try:
        write_json(_result_path(run_dir, "lineage.json"), {"branch": registry.get_branch(_branch_name(config)), "versions": registry.versions_for_branch(_branch_name(config))})
    except (KeyError, ValueError):
        # 空分支也要保留 summary；lineage 只在已有版本时生成。
        pass
    registry.add_run(run_id, {"status": summary["status"], "branch": _branch_name(config), "summary": str(run_dir / "summary.json")})
    print(f"本次运行目录：{run_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
