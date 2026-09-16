#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI backend for feature maps, CAM and detection stage tracing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_visualization.adapters.ultralytics import UltralyticsAdapter
from model_visualization.core.capture import ActivationCapture, compute_cam, render_features
from model_visualization.core.common import (
    append_html_index,
    blend_map,
    colorize,
    image_output_name,
    list_images,
    load_config,
    next_run_dir,
    safe_name,
    write_csv,
    write_json,
    write_image,
    write_runtime_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vtools 模型特征/CAM/检测阶段可视化")
    parser.add_argument("--config", type=Path, default=None, help="model_visualization/config/config.yaml")
    parser.add_argument("--run-dir", type=Path, default=None, help="指定一个不存在的输出目录")
    parser.add_argument("--mode", choices=["list_layers", "visualize"], default=None)
    parser.add_argument("--source", default=None, help="覆盖 input.source")
    parser.add_argument("--dataset-root", default=None, help="数据集根目录，须与诊断 data.yaml 的 path 一致")
    parser.add_argument("--weights", default=None, help="覆盖 model.weights")
    parser.add_argument("--device", default=None, help="覆盖 model.device：auto、cpu、cuda:0 或 0")
    parser.add_argument("--only", action="append", help="只运行 visualization.features、visualization.cam 或 visualization.stage_trace")
    parser.add_argument("--enable", action="append", help="临时启用可视化模块")
    parser.add_argument("--disable", action="append", help="临时关闭可视化模块")
    parser.add_argument("--list-modules", action="store_true", help="列出可视化模块后退出")
    return parser


def _set_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    module_ids = {"visualization.features", "visualization.cam", "visualization.stage_trace"}
    configured_value = config.get("modules")
    if configured_value is not None and not isinstance(configured_value, Mapping):
        raise ValueError("modules 必须是 module_id: true/false 的对象")
    if not isinstance(configured_value, Mapping):
        raise ValueError("配置必须包含 modules: module_id: true/false")
    unknown_config = set(map(str, configured_value)) - module_ids
    if unknown_config:
        raise ValueError(f"未知可视化模块：{', '.join(sorted(unknown_config))}")
    states = {key: bool(configured_value.get(key, False)) for key in module_ids}
    only = {item.strip() for value in args.only or [] for item in str(value).split(",") if item.strip()}
    enable = {item.strip() for value in args.enable or [] for item in str(value).split(",") if item.strip()}
    disable = {item.strip() for value in args.disable or [] for item in str(value).split(",") if item.strip()}
    if only and (enable or disable):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    unknown = (only | enable | disable) - module_ids
    if unknown:
        raise ValueError(f"未知可视化模块：{', '.join(sorted(unknown))}")
    if only:
        states = {key: key in only for key in module_ids}
    else:
        overlap = enable & disable
        if overlap:
            raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
        for key in enable:
            states[key] = True
        for key in disable:
            states[key] = False
    config["modules"] = states
    if args.mode:
        config["mode"] = args.mode
    project_root = Path(str(config["project"]["root"]))
    if args.source:
        source = Path(args.source).expanduser()
        if not source.is_absolute():
            source = (project_root / source).resolve()
        config.setdefault("input", {})["source"] = str(source)
    if getattr(args, "dataset_root", None):
        root = Path(args.dataset_root).expanduser()
        config.setdefault("input", {})["dataset_root"] = str((project_root / root).resolve())
    if args.weights:
        weights = Path(args.weights).expanduser()
        if not weights.is_absolute():
            weights = (project_root / weights).resolve()
        config.setdefault("model", {})["weights"] = str(weights)
    if args.device:
        config.setdefault("model", {})["device"] = args.device


def _make_run_dir(config: Mapping[str, Any], requested: Path | None) -> Path:
    if requested is None:
        return next_run_dir(config)
    target = requested.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    return target


def _write_config_copy(config: Mapping[str, Any], run_dir: Path) -> None:
    try:
        import yaml

        payload = {key: value for key, value in config.items() if not str(key).startswith("_")}
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except ImportError:
        write_json(run_dir / "config_used.json", config)


def _logical_image_id(image_path: Path, source_root: Path, dataset_root: Path | None = None) -> str:
    """Keep the extension so IDs never depend on the selected image subset."""
    image_path = image_path.resolve()
    if dataset_root is None:
        parts = image_path.parts
        indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
        if indices:
            dataset_root = Path(*parts[:indices[-1]])
        else:
            raise ValueError("非 images 目录布局请设置 input.dataset_root 或 --dataset-root，与诊断数据集根目录一致")
    try:
        return image_path.relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return image_path.as_posix()


def _cam_outputs(
    adapter: UltralyticsAdapter,
    tensor: Any,
    image_bgr: Any,
    transform_meta: Any,
    image_stem: str,
    run_dir: Path,
    config: Mapping[str, Any],
    candidate_index: int | None,
    branch: str | None = None,
    target_rank: int | None = None,
) -> list[str]:
    layers_config = config.get("layers") if isinstance(config.get("layers"), Mapping) else {}
    module_names, preset = adapter.effective_layers(layers_config)
    if str(preset).lower() == "detect_inputs":
        output_module_names = [name for name in module_names if name != adapter.head_name]
    else:
        output_module_names = module_names
    cam_values = config.get("cam") if isinstance(config.get("cam"), Mapping) else {}
    target_values = cam_values.get("target") if isinstance(cam_values.get("target"), Mapping) else {}
    target_spec = dict(target_values)
    if branch:
        target_spec["branch"] = branch
    if target_spec.get("class_id") is not None:
        target_spec["class_id"] = int(target_spec["class_id"])
    cam_root = run_dir / "cams" / safe_name(image_stem)
    if target_rank is not None:
        cam_root = cam_root / f"target_{target_rank}"
    cam_root.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    # A final-detection CAM must point to an actual post-processing result.
    # Do not silently fall back to the highest raw candidate when NMS kept none.
    if str(target_spec.get("kind", "raw_candidate")) == "final_detection" and candidate_index is None:
        metadata_path = cam_root / "cam_meta.json"
        write_json(metadata_path, {
            "status": "unavailable",
            "reason": "no_final_detection",
            "target_kind": "final_detection",
            "candidate_index": None,
            "layers": [],
        })
        return [str(metadata_path.relative_to(run_dir))]
    cam_tensor = tensor.detach().float().requires_grad_(True)
    with ActivationCapture(adapter.model, output_module_names, detach=False) as capture:
        if str(preset).lower() == "detect_inputs" and adapter.head_name:
            capture.add_input_hook(adapter.head_name)
        output = adapter.forward(cam_tensor)
        entries = capture.snapshot()
        maps, metadata = compute_cam(
            adapter.model,
            cam_tensor,
            entries,
            output,
            target_spec=target_spec,
            candidate_index=candidate_index,
            method=str(cam_values.get("method", "layercam")),
        )
    for name, values in maps.items():
        base = safe_name(name)
        color_path = cam_root / f"{base}.png"
        overlay_path = cam_root / f"{base}_overlay.jpg"
        import cv2

        write_image(color_path, colorize(values))
        write_image(overlay_path, blend_map(image_bgr, values, transform_meta))
        links.extend([str(color_path.relative_to(run_dir)), str(overlay_path.relative_to(run_dir))])
    if candidate_index is not None:
        metadata["candidate_index"] = candidate_index
    metadata["layers"] = list(maps)
    metadata_path = cam_root / "cam_meta.json"
    write_json(metadata_path, metadata)
    links.append(str(metadata_path.relative_to(run_dir)))
    return links


def _run_list_layers(adapter: UltralyticsAdapter, config: Mapping[str, Any], run_dir: Path) -> None:
    rows = adapter.list_layers()
    write_csv(run_dir / "layers.csv", rows)
    write_json(run_dir / "runtime_state.json", adapter.runtime_state)
    print(f"模块列表已保存：{run_dir / 'layers.csv'}")
    print(f"检测头：{adapter.head_name or '未识别'}")


def run(config: dict[str, Any], run_dir: Path) -> int:
    import torch

    _write_config_copy(config, run_dir)
    adapter_name = str((config.get("model") or {}).get("adapter", "ultralytics")).lower()
    if adapter_name != "ultralytics":
        raise ValueError("首版可视化只支持 model.adapter: ultralytics；自定义模型请先实现同等 adapter")
    adapter = UltralyticsAdapter(config)
    # Always archive the effective module names alongside visual outputs; this
    # makes a run reproducible even when the user did not first run list_layers.
    write_csv(run_dir / "layers.csv", adapter.list_layers())
    mode = str(config.get("mode", "visualize")).lower()
    if mode == "list_layers":
        _run_list_layers(adapter, config, run_dir)
        return 0
    if mode != "visualize":
        raise ValueError("mode 只支持 list_layers 或 visualize")

    input_values = config.get("input") if isinstance(config.get("input"), Mapping) else {}
    source_root = Path(str(input_values.get("source"))).expanduser().resolve()
    images = list_images(str(source_root), input_values.get("max_images"))
    size_value = input_values.get("size", [832, 832])
    size = (int(size_value[0]), int(size_value[1]))
    layers_config = config.get("layers") if isinstance(config.get("layers"), Mapping) else {}
    module_names, preset = adapter.effective_layers(layers_config)
    if str(preset).lower() == "detect_inputs":
        output_module_names = [name for name in module_names if name != adapter.head_name]
    else:
        output_module_names = module_names
    feature_values = config.get("features") if isinstance(config.get("features"), Mapping) else {}
    cam_values = config.get("cam") if isinstance(config.get("cam"), Mapping) else {}
    stage_values = config.get("stage_trace") if isinstance(config.get("stage_trace"), Mapping) else {}
    module_states = config["modules"]
    records: list[dict[str, Any]] = []
    canonical_raw_records: list[Mapping[str, Any]] = []
    canonical_final_records: list[Mapping[str, Any]] = []
    dataset_root = Path(input_values["dataset_root"]) if input_values.get("dataset_root") else None
    for image_path in images:
        image_id = _logical_image_id(image_path, source_root, dataset_root)
        stem = image_output_name(image_id)
        image_bgr, transform_meta, tensor = adapter.prepare(image_path, size)
        links: list[str] = []
        first_output: Any = None
        with ActivationCapture(adapter.model, output_module_names, detach=True) as capture:
            if str(preset).lower() == "detect_inputs" and adapter.head_name:
                capture.add_input_hook(adapter.head_name)
            with torch.no_grad():
                first_output = adapter.forward(tensor)
            entries = capture.snapshot()
            if module_states["visualization.features"]:
                _stats, feature_links = render_features(entries, image_bgr, transform_meta, run_dir, stem, feature_values)
                links.extend(feature_links)
        stage_info: dict[str, Any] = {}
        if module_states["visualization.stage_trace"]:
            stage_info = adapter.trace_stage(
                tensor,
                config,
                image_bgr=image_bgr,
                transform_meta=transform_meta,
                run_dir=run_dir,
                image_stem=stem,
                image_id=image_id,
            )
            links.extend(stage_info.get("links", []))
            if stage_info.get("status") == "ok":
                if isinstance(stage_info.get("canonical_raw_record"), Mapping):
                    record = dict(stage_info["canonical_raw_record"])
                    record["image_id"] = image_id
                    canonical_raw_records.append(record)
                if isinstance(stage_info.get("canonical_final_record"), Mapping):
                    record = dict(stage_info["canonical_final_record"])
                    record["image_id"] = image_id
                    canonical_final_records.append(record)
        if module_states["visualization.cam"]:
            target_values = cam_values.get("target") if isinstance(cam_values.get("target"), Mapping) else {}
            selection = str(target_values.get("selection", "highest_score")).lower()
            if selection not in {"highest_score", "class_highest", "index"}:
                raise ValueError("cam.target.selection 必须是 highest_score、class_highest 或 index")
            if selection == "class_highest" and target_values.get("class_id") is None:
                raise ValueError("cam.target.selection=class_highest 时必须填写 target.class_id")
            max_targets = int(cam_values.get("max_targets", 1))
            if max_targets <= 0:
                raise ValueError("cam.max_targets 必须大于 0")
            # final_detection needs the exact post-processing index.  If the
            # user enabled CAM without stage_trace, obtain that index in memory
            # but avoid writing a second set of stage files.
            cam_stage_info = stage_info
            if stage_info.get("status") != "ok":
                cam_stage_info = adapter.trace_stage(tensor, config, transform_meta=transform_meta, image_stem=stem, image_id=image_id)
            configured_index = target_values.get("index")
            candidates = [int(configured_index)] if configured_index is not None else []
            if not candidates and cam_stage_info.get("status") == "ok":
                target_kind = str(target_values.get("kind", "raw_candidate"))
                source_rows = cam_stage_info.get("final_rows") if target_kind == "final_detection" else cam_stage_info.get("raw_rows")
                source_rows = list(source_rows or [])
                class_id = target_values.get("class_id")
                if class_id is not None and target_kind == "final_detection":
                    source_rows = [row for row in source_rows if int(row.get("class_id", -1)) == int(class_id)]
                if class_id is not None and target_kind != "final_detection":
                    class_index = int(class_id)
                    source_rows.sort(
                        key=lambda row: float((row.get("class_scores") or [0.0])[class_index] if class_index < len(row.get("class_scores") or []) else 0.0),
                        reverse=True,
                    )
                else:
                    source_rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
                candidates = [int(row["raw_index"]) for row in source_rows[:max_targets]]
                if target_kind != "final_detection" and not candidates and cam_stage_info.get("target_index") is not None:
                    candidates = [int(cam_stage_info["target_index"])]
            if target_values.get("selection", "highest_score") not in {"highest_score", "class_highest", "index"}:
                raise ValueError("cam.target.selection 必须是 highest_score、class_highest 或 index")
            if target_values.get("selection") == "index" and configured_index is None:
                raise ValueError("CAM target.selection=index 时必须填写 target.index")
            if target_values.get("kind", "raw_candidate") == "final_detection" and configured_index is not None and cam_stage_info.get("status") == "ok":
                final_indices = {
                    int(row.get("raw_index"))
                    for row in cam_stage_info.get("final_rows", [])
                    if target_values.get("class_id") is None or int(row.get("class_id", -1)) == int(target_values["class_id"])
                }
                if int(configured_index) not in final_indices:
                    candidates = []
            if not candidates:
                candidates = [None]
            if target_values.get("kind", "raw_candidate") == "final_detection" and cam_stage_info.get("status") != "ok":
                candidates = [None]
            for rank, candidate_index in enumerate(candidates, 1):
                links.extend(_cam_outputs(adapter, tensor, image_bgr, transform_meta, stem, run_dir, config, candidate_index, cam_stage_info.get("branch"), rank if len(candidates) > 1 else None))
        image_meta_path = run_dir / "images" / stem / "image_meta.json"
        write_json(image_meta_path, {
            "image_id": image_id,
            "source": transform_meta.to_dict(),
            "activation_names": list(entries),
            "stage_status": stage_info.get("status") if stage_info else "disabled",
            "model_output_type": type(first_output).__name__,
        })
        links.append(str(image_meta_path.relative_to(run_dir)))
        records.append({"image": str(image_path), "links": sorted(set(links))})
        print(f"[{len(records)}/{len(images)}] {image_path.name} -> {len(links)} 个输出")

    if module_states["visualization.stage_trace"] and bool(stage_values.get("export_canonical", True)):
        canonical_root = run_dir / "canonical"
        raw_canonical_path = canonical_root / "raw_predictions.json"
        final_canonical_path = canonical_root / "final_predictions.json"
        write_json(raw_canonical_path, {
            "records": canonical_raw_records,
            "metadata": {"stage": "raw_candidate", "images": len(canonical_raw_records)},
        })
        write_json(final_canonical_path, {
            "records": canonical_final_records,
            "metadata": {"stage": "final", "images": len(canonical_final_records)},
        })
        if records:
            records[0]["links"].extend([
                str(raw_canonical_path.relative_to(run_dir)),
                str(final_canonical_path.relative_to(run_dir)),
            ])

    append_html_index(run_dir, records)
    write_json(run_dir / "run_metadata.json", {
        "schema_version": 1,
        "mode": mode,
        "images": records,
        "layers": module_names,
        "preset": preset,
        "stage_trace": module_states["visualization.stage_trace"],
        "cam": module_states["visualization.cam"],
    })
    write_runtime_state(
        run_dir / "runtime_state.json",
        requested={
            "device": (config.get("model") or {}).get("device", "auto"),
            "precision": (config.get("model") or {}).get("precision", "fp32"),
            "fuse": bool((config.get("model") or {}).get("fuse", False)),
            "layers": module_names,
            "stage_trace": module_states["visualization.stage_trace"],
            "cam": module_states["visualization.cam"],
        },
        effective={**adapter.runtime_state, "layers": module_names, "preset": preset},
        source={
            "config": config.get("_config_path", ""),
            "project_root": (config.get("project") or {}).get("root", ""),
            "ultralytics_repo": (config.get("project") or {}).get("ultralytics_repo", ""),
        },
    )
    print(f"可视化报告已保存：{run_dir / 'index.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_modules:
        print("visualization.features\nvisualization.cam\nvisualization.stage_trace")
        return 0
    config = load_config(args.config, allow_missing_source=bool(args.source) or args.mode == "list_layers")
    _set_cli_overrides(config, args)
    run_dir = _make_run_dir(config, args.run_dir)
    return run(config, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
