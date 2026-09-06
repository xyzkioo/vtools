#!/usr/bin/env python3
"""PyCharm entry point for the generic object-detection diagnostics.

The YAML entry point uses one explicit mode at a time:

* ``A`` reads an existing prediction file and does not load a model.
* ``B`` loads an Ultralytics ``.pt`` through the built-in loader.
* ``C`` loads a model through a vtools-compatible custom adapter.

The diagnostics engine itself remains framework independent and only consumes
canonical ``bbox/score/class_id`` detections.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_CONFIG = PROJECT_ROOT / "config" / "pycharm_run.yaml"


def _read_run_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("读取 PyCharm YAML 配置需要 PyYAML，请执行: pip install pyyaml") from exc
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


def _project_root(settings: Mapping[str, Any]) -> Path:
    project = _mapping(settings.get("project"), "project")
    return _resolve(project.get("root", "."), PROJECT_ROOT) or PROJECT_ROOT


def _add_python_paths(settings: Mapping[str, Any], project_root: Path) -> None:
    """Apply the same project/python_paths behavior as vtools config loading."""
    project = _mapping(settings.get("project"), "project")
    values = [project_root]
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


def _mode_entry(settings: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Translate the simple A/B/C YAML layout into one internal model entry.

    The old ``models: [...]`` list remains supported below for backwards
    compatibility, but new configurations should set ``mode`` and use exactly
    one of ``mode_a``, ``mode_b`` or ``mode_c``.
    """
    if "mode" not in settings:
        return None
    mode = str(settings.get("mode", "")).strip().upper()
    if mode not in {"A", "B", "C"}:
        raise ValueError("mode 必须是 A、B 或 C")

    section_name = f"mode_{mode.lower()}"
    section = _mapping(settings.get(section_name), section_name)
    if mode == "A":
        predictions = section.get("predictions")
        if predictions is None or not str(predictions).strip():
            raise ValueError("模式 A 必须在 mode_a.predictions 中填写已有预测文件")
        return {
            "name": section.get("name", "mode_a_predictions"),
            "mode": mode,
            "predictions": predictions,
            "pred_format": section.get("pred_format", "auto"),
            "task": "detect",
        }

    if mode == "B":
        weights = section.get("weights")
        if weights is None or not str(weights).strip():
            raise ValueError("模式 B 必须在 mode_b.weights 中填写 Ultralytics .pt 路径")
        return {
            "name": section.get("name", "mode_b_ultralytics"),
            "mode": mode,
            "weights": weights,
            "adapter": "ultralytics",
            "task": section.get("task", "detect"),
            # B/C always generate predictions from the model; an old global
            # dataset.predictions value must not silently bypass inference.
            "predictions": None,
        }

    weights = section.get("weights")
    adapter = section.get("adapter")
    if weights is None or not str(weights).strip():
        raise ValueError("模式 C 必须在 mode_c.weights 中填写模型权重路径")
    if adapter is None or not str(adapter).strip():
        raise ValueError("模式 C 必须在 mode_c.adapter 中填写 adapter.py 或模块名")
    return {
        "name": section.get("name", "mode_c_adapter"),
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
        raise ValueError("没有配置 adapter；模式 B 使用 ultralytics，模式 C 使用 vtools 风格 adapter.py")
    weights = _resolve(model_entry.get("weights"), project_root)
    if weights is None:
        raise ValueError("模型未提供 weights；模式 B/C 必须填写权重路径")
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
    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(engine.main())
    finally:
        sys.argv = old_argv


def _run_one_model(
    settings: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    run_dir: Path,
    project_root: Path,
) -> Path:
    # Import after project_root is known so the entry point works from PyCharm,
    # a terminal, or a run configuration with an unrelated working directory.
    from diagnostics import engine

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
    gt_dataset = engine.load_ground_truth(gt_path, gt_format, images_dir, split=split, base_dir=project_root)

    model_name = _safe_name(model_entry.get("name"), "model")
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # New A/B/C entries explicitly control the prediction source.  Preserve
    # the old dataset.predictions fallback only for legacy models entries.
    if model_entry.get("mode") in {"B", "C"}:
        prediction_value = None
    else:
        prediction_value = model_entry.get("predictions", dataset.get("predictions"))
    prediction_path = _resolve(prediction_value, project_root)
    if prediction_path is None:
        prediction_path = model_dir / "predictions_adapter.json"
        _generate_predictions(settings, model_entry, gt_dataset, project_root, images_dir, prediction_path)
    elif not prediction_path.exists():
        raise FileNotFoundError(f"找不到 predictions 文件：{prediction_path}")

    raw_value = model_entry.get("raw_predictions", dataset.get("raw_predictions"))
    raw_path = _resolve(raw_value, project_root)

    argv = [
        str(PROJECT_ROOT / "diagnostics" / "engine.py"),
        "--gt", str(gt_path),
        "--pred", str(prediction_path),
        "--config", str(RUN_CONFIG),
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
    _run_engine(engine, argv)

    metadata = {
        "name": str(model_entry.get("name", model_name)),
        "mode": str(model_entry.get("mode", "legacy")),
        "weights": str(_resolve(model_entry.get("weights"), project_root) or ""),
        "adapter": str(model_entry.get("adapter", "")),
        "task": str(model_entry.get("task", "detect")),
        "gt": str(gt_path),
        "dataset_data": str(data_yaml_path) if data_yaml_path is not None else "",
        "split": split,
        "predictions": str(prediction_path),
    }
    (model_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_dir


def main() -> int:
    settings = _read_run_config(RUN_CONFIG)
    project_root = _project_root(settings)
    _add_python_paths(settings, project_root)

    selected_mode = _mode_entry(settings)
    if selected_mode is not None:
        models: list[Mapping[str, Any]] = [selected_mode]
    else:
        models_value = settings.get("models")
        if models_value is None:
            dataset = _mapping(settings.get("dataset"), "dataset")
            models = [{"name": "provided_predictions", "predictions": dataset.get("predictions")}]
        elif isinstance(models_value, list):
            models = [item for item in models_value if isinstance(item, Mapping) and item.get("enabled", True)]
        else:
            raise ValueError("配置节 models 必须是列表；新配置请使用 mode: A、B 或 C")
    if not models:
        raise ValueError("没有可运行的模型；请检查 mode 或 models 配置")

    run_dir = _next_run_dir(settings, project_root)
    failures: list[str] = []
    completed: list[Path] = []
    for index, model_entry in enumerate(models):
        name = _safe_name(model_entry.get("name"), f"model_{index + 1}")
        try:
            completed.append(_run_one_model(settings, model_entry, run_dir, project_root))
            print(f"模型 {name} 完成：{completed[-1]}")
        except Exception as exc:  # keep other enabled models running
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
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
