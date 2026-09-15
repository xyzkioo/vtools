#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward activation capture, feature-map rendering and gradient CAM helpers.

The module deliberately keeps the model boundary small: an adapter only needs to
expose a normal ``torch.nn.Module``.  Detection-specific target selection lives in
``select_yolo_target`` and understands the raw output format used by YOLO26.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from .common import blend_map, colorize, finite_stats, safe_name, write_csv, write_image


@dataclass
class Activation:
    """One tensor emitted by a selected module."""

    name: str
    tensor: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def _tensor_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield tensors from nested module outputs without losing their paths."""
    try:
        import torch
    except ImportError:
        return
    if torch.is_tensor(value):
        yield prefix or "output", value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_items(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _tensor_items(item, child)


class ActivationCapture:
    """Register hooks for exact module names and expose their last activations."""

    def __init__(self, model: Any, module_names: Iterable[str] = (), detach: bool = True) -> None:
        self.model = model
        self.detach = detach
        self.activations: dict[str, Activation] = {}
        self._handles: list[Any] = []
        modules = dict(model.named_modules())
        requested = [str(name) for name in module_names if str(name)]
        missing = [name for name in requested if name not in modules]
        if missing:
            available = ", ".join(list(modules)[:20])
            raise KeyError(f"找不到可视化模块：{missing}；可用模块示例：{available}")
        for name in requested:
            self._handles.append(modules[name].register_forward_hook(self._make_output_hook(name)))

    def _make_output_hook(self, name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            for path, tensor in _tensor_items(output):
                value = tensor.detach() if self.detach else tensor
                key = name if path == "output" else f"{name}.{path}"
                if not self.detach and getattr(value, "requires_grad", False):
                    value.retain_grad()
                self.activations[key] = Activation(key, value, {"module": name, "output_path": path})

        return hook

    def add_input_hook(self, module_name: str, prefix: str = "input") -> None:
        """Capture tensors entering a module, useful for P-level detection features."""
        modules = dict(self.model.named_modules())
        if module_name not in modules:
            raise KeyError(f"找不到检测头模块：{module_name}")

        def hook(_module: Any, inputs: Any) -> None:
            for path, tensor in _tensor_items(inputs, prefix):
                value = tensor.detach() if self.detach else tensor
                key = f"{module_name}.{path}"
                if not self.detach and getattr(value, "requires_grad", False):
                    value.retain_grad()
                self.activations[key] = Activation(key, value, {"module": module_name, "output_path": path, "kind": "input"})

        # A forward pre-hook receives only inputs and works on all supported
        # PyTorch versions.  ``prepend`` is intentionally avoided for old torch.
        self._handles.append(modules[module_name].register_forward_pre_hook(hook))

    def snapshot(self) -> dict[str, Activation]:
        return dict(self.activations)

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _channel_indices(array: np.ndarray, values: Mapping[str, Any]) -> list[int]:
    channels = dict(values or {})
    mode = str(channels.get("mode", "first")).lower()
    count = int(channels.get("count", 16))
    if count < 1:
        raise ValueError("features.channels.count 必须大于 0")
    total = int(array.shape[0])
    if mode == "explicit":
        indices = [int(index) for index in channels.get("indices", [])]
        return [index for index in indices if 0 <= index < total]
    if mode == "variance":
        scores = np.nan_to_num(array.reshape(total, -1).var(axis=1), nan=-np.inf)
        return np.argsort(-scores)[: min(count, total)].astype(int).tolist()
    if mode != "first":
        raise ValueError("features.channels.mode 必须是 first、variance 或 explicit")
    return list(range(min(count, total)))


def _aggregate(array: np.ndarray, mode: str) -> np.ndarray:
    text = str(mode or "mean_abs").lower()
    if text == "mean":
        return np.nanmean(array, axis=0)
    if text == "max":
        return np.nanmax(array, axis=0)
    if text == "mean_abs":
        return np.nanmean(np.abs(array), axis=0)
    raise ValueError("features.aggregation 必须是 mean_abs、mean 或 max")


def render_features(
    entries: Mapping[str, Activation],
    image_bgr: np.ndarray,
    transform_meta: Any,
    run_dir: Path,
    image_stem: str,
    feature_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Save channel maps, aggregate maps, overlays and a machine-readable stats CSV."""
    import torch

    values = dict(feature_config or {})
    channels_cfg = dict(values.get("channels") or {})
    aggregate_mode = str(values.get("aggregation", "mean_abs"))
    overlay = bool(values.get("overlay", True))
    save_arrays = bool(values.get("save_arrays", False))
    stats_rows: list[dict[str, Any]] = []
    links: list[str] = []
    stats_dir = run_dir / "stats"
    feature_root = run_dir / "features" / safe_name(image_stem)
    activation_root = run_dir / "activations" / safe_name(image_stem)
    array_root = run_dir / "arrays" / safe_name(image_stem)
    for name, activation in entries.items():
        tensor = activation.tensor
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            continue
        array = tensor[0].detach().float().cpu().numpy()
        if array.ndim != 3 or min(array.shape) <= 0:
            continue
        module_name = str(activation.metadata.get("module", name))
        base = safe_name(name)
        selected = _channel_indices(array, channels_cfg)
        for channel in selected:
            channel_map = array[channel]
            path = feature_root / f"{base}_c{channel}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_image(path, colorize(channel_map))
            links.append(str(path.relative_to(run_dir)))
            row = {
                "image": image_stem,
                "module": module_name,
                "activation": name,
                "kind": "channel",
                "channel": channel,
                "shape": list(array.shape),
                "path": str(path.relative_to(run_dir)),
            }
            row.update(finite_stats(channel_map))
            stats_rows.append(row)
        aggregate = _aggregate(array, aggregate_mode)
        aggregate_path = activation_root / f"{base}_aggregate.png"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        write_image(aggregate_path, colorize(aggregate))
        links.append(str(aggregate_path.relative_to(run_dir)))
        aggregate_row = {
            "image": image_stem,
            "module": module_name,
            "activation": name,
            "kind": "aggregate",
            "aggregation": aggregate_mode,
            "shape": list(array.shape),
            "path": str(aggregate_path.relative_to(run_dir)),
        }
        aggregate_row.update(finite_stats(aggregate))
        stats_rows.append(aggregate_row)
        if overlay:
            overlay_path = activation_root / f"{base}_overlay.jpg"
            write_image(overlay_path, blend_map(image_bgr, aggregate, transform_meta))
            links.append(str(overlay_path.relative_to(run_dir)))
        if save_arrays:
            array_path = array_root / f"{base}.npz"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(array_path, activation=array)
            links.append(str(array_path.relative_to(run_dir)))
    if stats_rows and bool(values.get("statistics", True)):
        write_csv(stats_dir / f"{safe_name(image_stem)}_activations.csv", stats_rows)
        links.append(str((stats_dir / f"{safe_name(image_stem)}_activations.csv").relative_to(run_dir)))
    return stats_rows, links


def _output_prediction_dict(output: Any, target_spec: Mapping[str, Any]) -> tuple[Optional[Mapping[str, Any]], dict[str, Any]]:
    """Find the YOLO raw prediction dictionary and selected branch."""
    predictions = None
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, Mapping) and ("one2one" in item or "scores" in item):
                predictions = item
                break
    elif isinstance(output, Mapping):
        predictions = output
    if predictions is None:
        return None, {"reason": "model_output_has_no_prediction_dict"}
    branch = str(target_spec.get("branch", "auto") or "auto").lower()
    if branch == "auto":
        branch = "one2one" if "one2one" in predictions else "one2many" if "one2many" in predictions else "self"
    selected = predictions.get(branch, predictions) if isinstance(predictions, Mapping) else predictions
    if not isinstance(selected, Mapping) or "scores" not in selected:
        return None, {"reason": "prediction_branch_has_no_scores", "branch": branch}
    return selected, {"branch": branch}


def select_yolo_target(
    output: Any,
    target_spec: Mapping[str, Any] | None = None,
    candidate_index: Optional[int] = None,
) -> tuple[Any, dict[str, Any]]:
    """Select a differentiable scalar from YOLO raw class logits.

    ``candidate_index`` is the exact anchor index emitted by stage tracing.  If
    omitted, the highest sigmoid score (optionally restricted by ``class_id``) is
    selected.  The function also has a generic tensor fallback for custom heads.
    """
    import torch

    spec = dict(target_spec or {})
    branch, metadata = _output_prediction_dict(output, spec)
    if branch is not None:
        scores = branch["scores"]
        if scores.ndim != 3:
            raise ValueError(f"YOLO scores 应为 [B,C,N]，实际为 {tuple(scores.shape)}")
        class_id = spec.get("class_id")
        selection = str(spec.get("selection", "highest_score")).lower()
        if selection not in {"highest_score", "class_highest", "index"}:
            raise ValueError("CAM target.selection 必须是 highest_score、class_highest 或 index")
        if selection == "class_highest" and class_id is None:
            raise ValueError("CAM target.selection=class_highest 时必须填写 target.class_id")
        explicit_index = candidate_index if candidate_index is not None else spec.get("index")
        if explicit_index is None and selection == "index":
            raise ValueError("CAM target.selection=index 时必须填写 target.index")
        if explicit_index is None:
            probabilities = scores[0].sigmoid()
            if class_id is None:
                flat_index = int(probabilities.reshape(-1).argmax().item())
                class_index = flat_index // probabilities.shape[1]
                anchor_index = flat_index % probabilities.shape[1]
            else:
                if not 0 <= int(class_id) < probabilities.shape[0]:
                    raise IndexError(f"类别索引越界：{class_id}，类别数={probabilities.shape[0]}")
                class_index = int(class_id)
                anchor_index = int(probabilities[class_index].argmax().item())
        else:
            anchor_index = int(explicit_index)
            if not 0 <= anchor_index < scores.shape[2]:
                raise IndexError(f"候选索引越界：{anchor_index}，候选数={scores.shape[2]}")
            if class_id is None:
                class_index = int(scores[0, :, anchor_index].argmax().item())
            else:
                class_index = int(class_id)
        if not 0 <= class_index < scores.shape[1]:
            raise IndexError(f"类别索引越界：{class_index}，类别数={scores.shape[1]}")
        scalar = scores[0, class_index, anchor_index]
        metadata.update({
            "candidate_index": anchor_index,
            "class_id": class_index,
            "score": float(scalar.detach().sigmoid().item()),
            "target_kind": spec.get("kind", "raw_candidate"),
        })
        return scalar, metadata
    if torch.is_tensor(output):
        tensor = output
        if tensor.ndim == 0:
            return tensor, {"target_kind": "tensor_scalar"}
        if tensor.ndim == 1:
            index = int(spec.get("index", 0) or 0)
            return tensor[index], {"target_kind": "tensor", "index": index}
        flat = tensor[0] if tensor.ndim > 1 else tensor
        index = int(spec.get("index", flat.reshape(-1).argmax().item()) or 0)
        return flat.reshape(-1)[index], {"target_kind": "tensor", "index": index}
    raise TypeError("模型输出不包含可用于 CAM 的 YOLO raw scores 或 Tensor")


def compute_cam(
    model: Any,
    input_tensor: Any,
    entries: Mapping[str, Activation],
    output: Any,
    target_spec: Mapping[str, Any] | None = None,
    candidate_index: Optional[int] = None,
    method: str = "layercam",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute Grad-CAM or LayerCAM maps for captured 4-D activations."""
    import torch
    import torch.nn.functional as F

    selected_entries = [
        (name, entry)
        for name, entry in entries.items()
        if torch.is_tensor(entry.tensor) and entry.tensor.ndim == 4 and getattr(entry.tensor, "requires_grad", False)
    ]
    targets = [entry.tensor for _name, entry in selected_entries]
    if not targets:
        return {}, {"status": "unavailable", "reason": "no_4d_activation"}
    score, target_meta = select_yolo_target(output, target_spec, candidate_index)
    if not getattr(score, "requires_grad", False):
        target_meta.update({"status": "unavailable", "reason": "selected_target_has_no_gradient"})
        return {}, target_meta
    for tensor in targets:
        if getattr(tensor, "requires_grad", False):
            tensor.retain_grad()
    model.zero_grad(set_to_none=True)
    gradients = torch.autograd.grad(score, targets, allow_unused=True, retain_graph=False, create_graph=False)
    maps: dict[str, np.ndarray] = {}
    method_name = str(method or "layercam").lower()
    for (name, entry), gradient in zip(selected_entries, gradients):
        activation = entry.tensor
        if gradient is None or activation.ndim != 4:
            continue
        if method_name == "gradcam":
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        elif method_name == "layercam":
            cam = torch.relu(gradient) * torch.relu(activation)
            cam = cam.sum(dim=1, keepdim=True)
        else:
            raise ValueError("cam.method 只支持 gradcam 或 layercam")
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        maps[name] = cam.detach().float().cpu().numpy()
    status = "ok" if maps else "unavailable"
    if not maps:
        target_meta["reason"] = "autograd_returned_no_gradient"
    target_meta["status"] = status
    target_meta["method"] = method_name
    return maps, target_meta


__all__ = [
    "Activation",
    "ActivationCapture",
    "compute_cam",
    "render_features",
    "select_yolo_target",
]
