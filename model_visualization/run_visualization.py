#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm/CLI entry point for feature maps, CAM and detection stage tracing."""

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
    list_images,
    load_config,
    next_run_dir,
    safe_name,
    write_csv,
    write_json,
    write_runtime_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vtools 模型特征/CAM/检测阶段可视化")
    parser.add_argument("--config", type=Path, default=None, help="model_visualization/config/pycharm_run.yaml")
    parser.add_argument("--run-dir", type=Path, default=None, help="指定一个不存在的输出目录")
    parser.add_argument("--mode", choices=["list_layers", "visualize"], default=None)
    parser.add_argument("--source", default=None, help="覆盖 input.source")
    parser.add_argument("--weights", default=None, help="覆盖 model.weights")
    return parser


def _set_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.mode:
        config["mode"] = args.mode
    project_root = Path(str(config["project"]["root"]))
    if args.source:
        source = Path(args.source).expanduser()
        if not source.is_absolute():
            source = (project_root / source).resolve()
        config.setdefault("input", {})["source"] = str(source)
    if args.weights:
        weights = Path(args.weights).expanduser()
        if not weights.is_absolute():
            weights = (project_root / weights).resolve()
        config.setdefault("model", {})["weights"] = str(weights)


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
    cam_root = run_dir / "cams" / safe_name(image_stem)
    cam_root.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    for name, values in maps.items():
        base = safe_name(name)
        color_path = cam_root / f"{base}.png"
        overlay_path = cam_root / f"{base}_overlay.jpg"
        import cv2

        cv2.imwrite(str(color_path), colorize(values))
        cv2.imwrite(str(overlay_path), blend_map(image_bgr, values, transform_meta))
        links.extend([str(color_path.relative_to(run_dir)), str(overlay_path.relative_to(run_dir))])
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
    images = list_images(str(input_values.get("source")), input_values.get("max_images"))
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
    records: list[dict[str, Any]] = []
    canonical_raw_records: list[Mapping[str, Any]] = []
    canonical_final_records: list[Mapping[str, Any]] = []
    stem_counts: dict[str, int] = {}
    for image_path in images:
        base_stem = safe_name(image_path.stem)
        stem_counts[base_stem] = stem_counts.get(base_stem, 0) + 1
        stem = base_stem if stem_counts[base_stem] == 1 else f"{base_stem}_{stem_counts[base_stem]}"
        image_bgr, transform_meta, tensor = adapter.prepare(image_path, size)
        links: list[str] = []
        first_output: Any = None
        with ActivationCapture(adapter.model, output_module_names, detach=True) as capture:
            if str(preset).lower() == "detect_inputs" and adapter.head_name:
                capture.add_input_hook(adapter.head_name)
            with torch.no_grad():
                first_output = adapter.forward(tensor)
            entries = capture.snapshot()
            if bool(feature_values.get("enabled", True)):
                _stats, feature_links = render_features(entries, image_bgr, transform_meta, run_dir, stem, feature_values)
                links.extend(feature_links)
        stage_info: dict[str, Any] = {}
        if bool(stage_values.get("enabled", False)):
            stage_info = adapter.trace_stage(
                tensor,
                config,
                image_bgr=image_bgr,
                transform_meta=transform_meta,
                run_dir=run_dir,
                image_stem=stem,
            )
            links.extend(stage_info.get("links", []))
            if stage_info.get("status") == "ok":
                if isinstance(stage_info.get("canonical_raw_record"), Mapping):
                    canonical_raw_records.append(stage_info["canonical_raw_record"])
                if isinstance(stage_info.get("canonical_final_record"), Mapping):
                    canonical_final_records.append(stage_info["canonical_final_record"])
        if bool(cam_values.get("enabled", False)):
            target_values = cam_values.get("target") if isinstance(cam_values.get("target"), Mapping) else {}
            # final_detection needs the exact post-processing index.  If the
            # user enabled CAM without stage_trace, obtain that index in memory
            # but avoid writing a second set of stage files.
            cam_stage_info = stage_info
            if str(target_values.get("kind", "raw_candidate")) == "final_detection" and stage_info.get("status") != "ok":
                cam_stage_info = adapter.trace_stage(tensor, config, transform_meta=transform_meta, image_stem=stem)
            candidate_index = cam_stage_info.get("target_index") if cam_stage_info.get("status") == "ok" else None
            links.extend(_cam_outputs(adapter, tensor, image_bgr, transform_meta, stem, run_dir, config, candidate_index, cam_stage_info.get("branch")))
        image_meta_path = run_dir / "images" / stem / "image_meta.json"
        write_json(image_meta_path, {
            "image_id": stem,
            "source": transform_meta.to_dict(),
            "activation_names": list(entries),
            "stage_status": stage_info.get("status") if stage_info else "disabled",
            "model_output_type": type(first_output).__name__,
        })
        links.append(str(image_meta_path.relative_to(run_dir)))
        records.append({"image": str(image_path), "links": sorted(set(links))})
        print(f"[{len(records)}/{len(images)}] {image_path.name} -> {len(links)} 个输出")

    if bool(stage_values.get("enabled", False)) and bool(stage_values.get("export_canonical", True)):
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
        "stage_trace": bool(stage_values.get("enabled", False)),
        "cam": bool(cam_values.get("enabled", False)),
    })
    write_runtime_state(
        run_dir / "runtime_state.json",
        requested={
            "device": (config.get("model") or {}).get("device", "auto"),
            "precision": (config.get("model") or {}).get("precision", "fp32"),
            "fuse": bool((config.get("model") or {}).get("fuse", False)),
            "layers": module_names,
            "stage_trace": bool(stage_values.get("enabled", False)),
            "cam": bool(cam_values.get("enabled", False)),
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
    config = load_config(args.config)
    _set_cli_overrides(config, args)
    run_dir = _make_run_dir(config, args.run_dir)
    return run(config, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
