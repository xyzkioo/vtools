#!/usr/bin/env python3
"""CLI backend for generic object-detection diagnostics."""

from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def _read_run_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("读取 YAML 配置需要 PyYAML，请执行: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} 必须是 YAML 对象")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"配置节 {name} 必须是 YAML 对象")
    return value


def _resolve(value: Any, base: Path) -> Optional[Path]:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _project_root(settings: Mapping[str, Any], base: Path | None = None) -> Path:
    project = _mapping(settings.get("project"), "project")
    anchor = (base or PROJECT_ROOT).resolve()
    return _resolve(project.get("root", "."), anchor) or anchor


def _add_python_paths(
    settings: Mapping[str, Any], project_root: Path, input_paths: tuple[Path, ...] = (),
) -> None:
    """Apply the same project/python_paths behavior as vtools config loading."""
    from vtools_runtime.ultralytics import add_repo_to_path

    project = _mapping(settings.get("project"), "project")
    values = [PROJECT_ROOT, project_root]
    if project.get("ultralytics_repo"):
        values.append(project.get("ultralytics_repo"))
    extra_paths = project.get("python_paths") or []
    if isinstance(extra_paths, (str, Path)):
        extra_paths = [extra_paths]
    values.extend(extra_paths)
    for value in values:
        path = _resolve(value, project_root)
        if path is not None and path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    add_repo_to_path(settings, project_root, input_paths)


def _adapter_spec(value: Any, project_root: Path) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # vtools accepts either an adapter.py path or an importable module name.
    if text.endswith(".py") or "/" in text or "\\" in text:
        candidate = _resolve(text, project_root)
        return str(candidate) if candidate is not None else None
    return text


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._")
    return text or fallback


def _next_run_dir(settings: Mapping[str, Any], project_root: Path) -> Path:
    run = _mapping(settings.get("run"), "run")
    root = _resolve(run.get("root", "runs"), project_root) or (project_root / "runs")
    root.mkdir(parents=True, exist_ok=True)
    name = str(run.get("name", "auto") or "auto").strip()
    if name.lower() != "auto":
        target = root / name
        if target.exists():
            raise FileExistsError(f"运行目录已存在，为避免覆盖结果请修改 run.name：{target}")
        target.mkdir(parents=True)
        return target
    index = 1
    while (root / f"run{index}").exists():
        index += 1
    target = root / f"run{index}"
    target.mkdir(parents=True)
    return target


def _image_path(image_info: Any, images_dir: Optional[Path], project_root: Path) -> Optional[Path]:
    """Resolve a canonical/COCO file_name to an actual image path."""
    file_name = getattr(image_info, "file_name", None)
    candidates: list[Path] = []
    extra = getattr(image_info, "extra", {}) or {}
    source_path = extra.get("source_path") if isinstance(extra, Mapping) else None
    if source_path:
        candidates.append(Path(str(source_path)).expanduser())
    if file_name:
        raw = Path(str(file_name)).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            if images_dir is not None:
                candidates.append(images_dir / raw)
            candidates.append(project_root / raw)
    image_id = str(getattr(image_info, "image_id", ""))
    if images_dir is not None:
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            candidates.append(images_dir / f"{image_id}{suffix}")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _write_prediction_file(path: Path, gt_dataset: Any, predictions: Mapping[str, list[dict[str, Any]]]) -> None:
    records = []
    image_ids = sorted(set(gt_dataset.images) | set(predictions))
    for image_id in image_ids:
        info = gt_dataset.images.get(image_id)
        row: dict[str, Any] = {
            "image_id": image_id,
            "predictions": list(predictions.get(image_id, [])),
        }
        if info is not None:
            if info.file_name is not None:
                row["file_name"] = info.file_name
            if info.width is not None:
                row["width"] = info.width
            if info.height is not None:
                row["height"] = info.height
        records.append(row)
    payload: dict[str, Any] = {"records": records}
    if getattr(gt_dataset, "class_names", None):
        payload["class_names"] = {str(k): v for k, v in gt_dataset.class_names.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mode_entry(
    settings: Mapping[str, Any],
    *,
    weights_override: Path | None = None,
    predictions_override: bool = False,
) -> Mapping[str, Any]:
    """Translate the named input mode into one internal model entry."""
    if "mode" not in settings:
        raise ValueError("配置必须填写 mode: predictions_file、ultralytics_model 或 custom_adapter")
    mode = str(settings.get("mode", "")).strip().lower()
    legacy = {"a": "predictions_file", "b": "ultralytics_model", "c": "custom_adapter"}
    if mode in legacy:
        raise ValueError(f"已移除不明确的 mode: {mode.upper()}；请改用 mode: {legacy[mode]}")
    if mode not in {"predictions_file", "ultralytics_model", "custom_adapter"}:
        raise ValueError("mode 必须是 predictions_file、ultralytics_model 或 custom_adapter")

    section_name = mode
    section = _mapping(settings.get(section_name), section_name)
    if mode == "predictions_file":
        predictions = section.get("predictions")
        if predictions is None or not str(predictions).strip():
            raise ValueError("predictions_file 模式必须在 predictions_file.predictions 中填写已有预测文件")
        return {
            "name": section.get("name", "existing_predictions"),
            "mode": mode,
            "predictions": predictions,
            "pred_format": section.get("pred_format", "auto"),
            "task": "detect",
        }

    if mode == "ultralytics_model":
        weights = weights_override or section.get("weights")
        if (weights is None or not str(weights).strip()) and not predictions_override:
            raise ValueError("ultralytics_model 模式必须在 ultralytics_model.weights 中填写 .pt 路径")
        return {
            "name": section.get("name", "ultralytics_model"),
            "mode": mode,
            "weights": weights,
            "adapter": "ultralytics",
            "task": section.get("task", "detect"),
            # Model input modes generate predictions from the model.
            "predictions": None,
        }

    weights = section.get("weights")
    if weights_override:
        weights = weights_override
    adapter = section.get("adapter")
    if (weights is None or not str(weights).strip()) and not predictions_override:
        raise ValueError("custom_adapter 模式必须在 custom_adapter.weights 中填写模型权重路径")
    if adapter is None or not str(adapter).strip():
        raise ValueError("custom_adapter 模式必须在 custom_adapter.adapter 中填写 adapter.py 或模块名")
    return {
        "name": section.get("name", "custom_adapter"),
        "mode": mode,
        "weights": weights,
        "adapter": adapter,
        "task": section.get("task", "detect"),
        "predictions": None,
    }


def _generate_predictions(
    settings: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    gt_dataset: Any,
    project_root: Path,
    images_dir: Optional[Path],
    output_path: Path,
) -> Path:
    from diagnostics.adapter_runtime import create_adapter, make_benchmark_config

    adapter_spec = _adapter_spec(model_entry.get("adapter"), project_root)
    if not adapter_spec:
        raise ValueError("没有配置 adapter；ultralytics_model 使用 ultralytics，custom_adapter 使用 vtools 风格 adapter.py")
    weights = _resolve(model_entry.get("weights"), project_root)
    if weights is None:
        raise ValueError("模型未提供 weights；ultralytics_model/custom_adapter 必须填写权重路径")
    benchmark = _mapping(settings.get("benchmark"), "benchmark")
    config = make_benchmark_config(benchmark)
    adapter = create_adapter(adapter_spec, project_root, str(model_entry.get("task", "detect")))
    adapter.load(weights, config)

    predictions: dict[str, list[dict[str, Any]]] = {}
    for image_id, info in sorted(gt_dataset.images.items()):
        source = _image_path(info, images_dir, project_root)
        if source is None:
            raise FileNotFoundError(
                f"找不到 image_id={image_id} 对应图片；请检查 dataset.data 的 path/split，"
                "或旧式输入的 dataset.images/file_name"
            )
        image_info = {
            "image_id": image_id,
            "width": info.width,
            "height": info.height,
            "file_name": info.file_name,
            **dict(info.extra),
        }
        # vtools' reference adapters receive a string path for ``source``;
        # keep that exact convention so a copied adapter works unchanged.
        predictions[image_id] = adapter.predict(str(source), image_id, image_info)
    _write_prediction_file(output_path, gt_dataset, predictions)
    return output_path


def _run_engine(engine: Any, argv: list[str]) -> int:
    # The engine accepts an explicit argv list, so nested runs do not mutate
    # process-global ``sys.argv`` or interfere with another model in the same
    # parent process.
    return int(engine.main(argv[1:]))


def _run_one_model(
    settings: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    run_dir: Path,
    project_root: Path,
    config_path: Path,
    cli_args: argparse.Namespace,
) -> Path:
    # Import after project_root is known so configured adapter paths work from
    # the repository root or an unrelated working directory.
    try:
        from diagnostics import engine
    except ImportError:
        from model_diagnostics.diagnostics import engine

    dataset = _mapping(settings.get("dataset"), "dataset")
    data_yaml_value = dataset.get("data") or dataset.get("yaml")
    data_yaml_path = _resolve(data_yaml_value, project_root)
    gt_path = data_yaml_path or _resolve(dataset.get("gt"), project_root)
    if gt_path is None:
        raise ValueError("请在 dataset.data 中填写 Ultralytics data.yaml，或在 dataset.gt 中填写 GT 文件/YOLO 标签目录")
    images_dir = _resolve(dataset.get("images"), project_root)
    gt_format = str(dataset.get("gt_format", "ultralytics" if data_yaml_path else "auto"))
    split = str(dataset.get("split", "val"))
    pred_format = str(model_entry.get("pred_format", dataset.get("pred_format", "auto")))
    model_name = _safe_name(model_entry.get("name"), "model")
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Named input modes explicitly control the prediction source.
    if cli_args.predictions is not None:
        prediction_value = str(cli_args.predictions)
    elif model_entry.get("mode") in {"ultralytics_model", "custom_adapter"}:
        prediction_value = None
    else:
        prediction_value = model_entry.get("predictions")
    prediction_path = _resolve(prediction_value, project_root)
    if prediction_path is None:
        try:
            from diagnostics.modules import resolve_modules
        except ImportError:
            from model_diagnostics.diagnostics.modules import resolve_modules
        generation_enabled = bool(resolve_modules(settings, only=cli_args.only, enable=cli_args.enable, disable=cli_args.disable).get("predict.generate", False))
        if not generation_enabled:
            raise ValueError("没有现成 predictions 文件时，必须显式启用 modules.predict.generate")
        prediction_path = model_dir / "raw_data" / "predictions_adapter.json"
        gt_dataset = engine.load_ground_truth(gt_path, gt_format, images_dir, split=split, base_dir=project_root)
        _generate_predictions(settings, model_entry, gt_dataset, project_root, images_dir, prediction_path)
    elif not prediction_path.exists():
        raise FileNotFoundError(f"找不到 predictions 文件：{prediction_path}")

    raw_value = model_entry.get("raw_predictions", dataset.get("raw_predictions"))
    raw_path = _resolve(raw_value, project_root)

    argv = [
        str(PROJECT_ROOT / "diagnostics" / "engine.py"),
        "--gt", str(gt_path),
        "--pred", str(prediction_path),
        "--config", str(config_path),
        "--output", str(model_dir),
        "--gt-format", gt_format,
        "--pred-format", pred_format,
        "--split", split,
    ]
    if raw_path is not None:
        if not raw_path.exists():
            raise FileNotFoundError(f"找不到 raw_predictions 文件：{raw_path}")
        argv.extend(["--raw-pred", str(raw_path)])
    if images_dir is not None:
        argv.extend(["--images-dir", str(images_dir)])
    argv.extend(["--base-dir", str(project_root)])
    for value in cli_args.only or []:
        argv.extend(["--only", value])
    for value in cli_args.enable or []:
        argv.extend(["--enable", value])
    for value in cli_args.disable or []:
        argv.extend(["--disable", value])
    exit_code = _run_engine(engine, argv)
    if exit_code != 0:
        raise RuntimeError(f"诊断引擎失败，退出码={exit_code}")

    run_info_path = model_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else {}
    run_info.update({
        "mode": str(model_entry["mode"]),
        "model": {
            "name": str(model_entry.get("name", model_name)),
            "weights": str(_resolve(model_entry.get("weights"), project_root) or ""),
            "adapter": str(model_entry.get("adapter", "")),
            "task": str(model_entry.get("task", "detect")),
        },
        "config": settings,
        "inputs": {
            **dict(run_info.get("inputs") or {}),
            "gt": str(gt_path),
            "dataset_data": str(data_yaml_path) if data_yaml_path is not None else "",
            "split": split,
            "predictions": str(prediction_path),
        },
    })
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="通用目标检测诊断")
    parser.add_argument("--config", type=Path, default=RUN_CONFIG, help="诊断 YAML/JSON 配置，默认使用 config/config.yaml")
    parser.add_argument("--only", action="append", help="只运行指定模块 ID，可重复或用逗号分隔")
    parser.add_argument("--enable", action="append", help="临时启用模块 ID，可重复或用逗号分隔")
    parser.add_argument("--disable", action="append", help="临时关闭模块 ID，可重复或用逗号分隔")
    parser.add_argument("--predictions", type=Path, help="直接使用已有预测文件，覆盖模型推理")
    parser.add_argument("--run-dir", type=Path, help="指定一个尚不存在的输出目录")
    parser.add_argument("--weights", type=Path, help="直接使用模型权重，覆盖模型配置中的 weights")
    parser.add_argument("--data", type=Path, help="直接使用数据集 YAML，覆盖 dataset.data")
    parser.add_argument("--device", help="临时覆盖 benchmark.device：auto、cpu、cuda:0 或 0")
    parser.add_argument("--list-modules", action="store_true", help="列出可用模块后退出")
    cli_args = parser.parse_args()
    if cli_args.predictions and cli_args.predictions.suffix.lower() in {".pt", ".pth", ".onnx", ".engine", ".kmodel"}:
        raise ValueError("--predictions 需要 JSON/COCO 等预测文件；模型权重请使用 --weights")
    if cli_args.list_modules:
        try:
            from diagnostics.modules import available_modules
        except ImportError:
            from model_diagnostics.diagnostics.modules import available_modules
        print("\n".join(
            module_id for module_id in available_modules()
            if module_id.startswith(("predict.", "validation.", "diagnostics.", "output."))
        ))
        return 0
    config_path = cli_args.config.expanduser().resolve()
    settings = _read_run_config(config_path)
    if "models" in settings:
        raise ValueError("已移除旧版配置字段 models；请使用一个命名 mode")
    dataset_settings = _mapping(settings.get("dataset"), "dataset")
    if "predictions" in dataset_settings:
        raise ValueError("已移除旧版配置字段 dataset.predictions；请使用 predictions_file.predictions")
    if cli_args.predictions is not None and str(settings.get("mode", "")).strip().lower() in {"predictions_file", "a"}:
        predictions_file = dict(settings.get("predictions_file") or {})
        predictions_file["predictions"] = str(cli_args.predictions.expanduser().resolve())
        settings = dict(settings)
        settings["predictions_file"] = predictions_file
    project_root = _project_root(settings, config_path.parent)
    input_paths = tuple(
        path for path in (
            cli_args.data,
            cli_args.weights,
            _resolve(dataset_settings.get("data"), project_root),
            _resolve(_mapping(settings.get("ultralytics_model"), "ultralytics_model").get("weights"), project_root),
        ) if path is not None
    )
    _add_python_paths(settings, project_root, input_paths)

    selected_mode = _mode_entry(
        settings,
        weights_override=cli_args.weights,
        predictions_override=cli_args.predictions is not None,
    )
    models: list[Mapping[str, Any]] = [selected_mode]

    if cli_args.weights:
        if any(str(model.get("mode", "")).lower() == "predictions_file" for model in models):
            raise ValueError("predictions_file 模式使用已有预测文件，不能传 --weights；请使用 --predictions")
        models = [{**dict(model), "weights": str(cli_args.weights)} for model in models]

    if cli_args.data:
        # Keep model and dataset inputs independent. An absolute --data path
        # is especially useful when the YAML file lives outside this repo.
        settings = dict(settings)
        dataset = dict(_mapping(settings.get("dataset"), "dataset"))
        dataset["data"] = str(cli_args.data)
        settings["dataset"] = dataset

    if cli_args.device:
        settings = dict(settings)
        benchmark = dict(_mapping(settings.get("benchmark"), "benchmark"))
        benchmark["device"] = cli_args.device
        settings["benchmark"] = benchmark

    try:
        from diagnostics.modules import resolve_modules
    except ImportError:
        from model_diagnostics.diagnostics.modules import resolve_modules
    selected_modules = resolve_modules(settings, only=cli_args.only, enable=cli_args.enable, disable=cli_args.disable)
    settings = dict(settings)
    settings["modules"] = selected_modules
    settings["cli_overrides"] = {
        "only": list(cli_args.only or []),
        "enable": list(cli_args.enable or []),
        "disable": list(cli_args.disable or []),
        "predictions": str(cli_args.predictions.expanduser().resolve()) if cli_args.predictions is not None else "",
        "weights": str(cli_args.weights.expanduser().resolve()) if cli_args.weights is not None else "",
        "data": str(cli_args.data.expanduser().resolve()) if cli_args.data is not None else "",
        "device": cli_args.device or "",
    }
    if (
        any(model.get("mode") in {"ultralytics_model", "custom_adapter"} for model in models)
        and cli_args.predictions is None
        and not any(model.get("weights") for model in models)
    ):
        raise ValueError("模型模式没有权重；请填写对应 mode 节的 weights 或使用 --weights/--predictions")

    if cli_args.run_dir is not None:
        run_dir = cli_args.run_dir.expanduser().resolve()
        if run_dir.exists():
            raise FileExistsError(f"运行目录已存在，为避免覆盖结果请换一个路径：{run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = _next_run_dir(settings, project_root)
    failures: list[str] = []
    completed: list[Path] = []
    for index, model_entry in enumerate(models):
        name = _safe_name(model_entry.get("name"), f"model_{index + 1}")
        try:
            completed.append(_run_one_model(settings, model_entry, run_dir, project_root, config_path, cli_args))
            print(f"模型 {name} 完成：{completed[-1]}")
        except Exception as exc:  # keep other enabled models running
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            failed_dir = run_dir / name
            if failed_dir.exists():
                info_path = failed_dir / "run_info.json"
                info = {}
                if info_path.exists():
                    try:
                        info = json.loads(info_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        info = {}
                info.update({"status": "failed", "error": failures[-1], "mode": str(model_entry.get("mode", ""))})
                info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"模型 {name} 失败：{failures[-1]}", file=sys.stderr)

    print(f"本次运行目录：{run_dir.resolve()}")
    if failures:
        (run_dir / "errors.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
