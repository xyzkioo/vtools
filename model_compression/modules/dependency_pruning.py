"""依赖感知的 YOLO 结构化通道剪枝。

``torch.nn.utils.prune`` 的结构化掩码不会改变层的形状，不能直接带来
dense CUDA 加速。本模块使用 ``torch-pruning`` 的 DependencyGraph 物理
删除通道，并让依赖的卷积、BatchNorm、Concat 和残差分支同步收缩。

YOLO 的检测头输出通道必须保持不变，因此默认只剪主干和颈部的阶段输出
卷积，不剪 Detect 的分类/回归输出层，也不剪输入 stem。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _structured_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read structured settings from the canonical compression section."""

    compression = config.get("compression", {})
    if isinstance(compression, Mapping):
        section = compression.get("structured")
        if isinstance(section, Mapping):
            return section
    return {}


def _input_size(config: Mapping[str, Any]) -> int:
    section = _structured_section(config)
    section = section if isinstance(section, Mapping) else {}
    value = section.get("pruning_input_size")
    if value is None:
        value = config.get("dataset", {}).get("input_size", 640)
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _stage_candidates(model: Any) -> list[tuple[int, Any]]:
    """Return top-level stage-output convolutions suitable for YOLO pruning."""

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("依赖感知剪枝需要 PyTorch") from exc

    layers = getattr(model, "model", None)
    if layers is None:
        raise ValueError("当前检测模型没有 Sequential 网络，无法建立依赖剪枝图。")
    candidates: list[tuple[int, Any]] = []
    # Top-level Conv blocks are the channel boundaries between YOLO stages.
    # Internal C2f/C3 blocks are deliberately left to the dependency graph as
    # downstream dependants; choosing all of them would over-prune branches.
    for index, block in enumerate(layers):
        conv = getattr(block, "conv", None)
        if not isinstance(conv, torch.nn.Conv2d):
            continue
        if index == 0 or conv.groups != 1 or conv.out_channels < 16:
            continue
        candidates.append((index, conv))
    if not candidates:
        raise ValueError("没有找到可用于依赖感知结构化剪枝的 YOLO 阶段卷积。")
    return candidates


def prune_detector_dependency_aware(
    model: Any,
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Physically prune YOLO channels with ``torch-pruning``.

    The operation is intentionally conservative: it prunes a configurable
    fraction independently from each stage-output convolution, rounds the
    number of removed channels to a hardware-friendly multiple, and skips any
    dependency group that would remove too many channels. The returned model is
    still a normal Ultralytics ``DetectionModel`` and can be fine-tuned/saved
    by the existing detection pipeline.
    """

    try:
        import torch  # type: ignore
        import torch_pruning as tp  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "依赖感知结构化剪枝需要 torch-pruning；请在当前环境安装 torch-pruning。"
        ) from exc

    section = _structured_section(config)
    section = dict(section) if isinstance(section, Mapping) else {}
    ratio = float(section.get("channel_sparsity", section.get("pruning_ratio", 0.10)))
    if not 0.0 <= ratio < 1.0:
        raise ValueError("结构化通道稀疏率必须在 [0, 1) 范围内。")
    round_to = int(section.get("channel_round", 8))
    if round_to < 1:
        raise ValueError("结构化通道对齐数必须大于 0。")
    max_layers = int(section.get("max_pruned_layers", 6))
    if max_layers < 1:
        raise ValueError("max_pruned_layers 必须大于 0。")
    norm = int(section.get("importance_norm", 2))
    if norm < 1:
        raise ValueError("importance_norm 必须大于 0。")

    source_params = _parameter_count(model)
    device_value = config.get("model", {}).get("device", "auto")
    from ..core.model_runtime import resolve_device
    device = resolve_device(device_value)
    model = model.to(device)
    model.eval()
    # Ultralytics checkpoints are loaded in inference mode with gradients off;
    # torch-pruning needs parameter metadata and a traceable graph.
    model.requires_grad_(True)
    size = _input_size(config)
    example_inputs = torch.zeros(1, 3, size, size, device=device)

    # Trace raw predictions from every branch in eval mode. Training mode
    # detaches the one-to-one branch in end-to-end detectors.
    def output_transform(output: Any) -> Any:
        if isinstance(output, tuple) and len(output) == 2:
            return output[1]
        return output

    importance = tp.importance.MagnitudeImportance(p=norm, group_reduction="mean")
    candidates = _stage_candidates(model)
    stage_selection = str(section.get("stage_selection", "all")).strip().lower()
    reverse_selection = stage_selection in {"deepest", "last", "back_to_front"}
    if reverse_selection:
        candidates = list(reversed(candidates))
    elif stage_selection not in {"all", "front_to_back", "first"}:
        raise ValueError(
            "stage_selection 可选 all、front_to_back、first、deepest、last 或 back_to_front。"
        )
    limit = min(max_layers, len(candidates))
    candidates = candidates[:limit]
    # Use deterministic application order; rebuild dependencies after each edit.
    if reverse_selection:
        candidates.sort(key=lambda item: item[0])
    records: list[dict[str, Any]] = []

    for layer_index, layer in candidates:
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(
                model, example_inputs=example_inputs,
                output_transform=output_transform, verbose=False,
            )
        current_channels = int(layer.out_channels)
        # Keep a complete channel-aligned group for the importance calculation.
        group = graph.get_pruning_group(
            layer,
            tp.prune_conv_out_channels,
            list(range(current_channels)),
        )
        scores = importance(group)
        if scores is None or int(scores.numel()) != current_channels:
            records.append({
                "layer_index": layer_index,
                "channels_before": current_channels,
                "channels_pruned": 0,
                "status": "skipped_no_importance",
            })
            continue

        requested = int(current_channels * ratio)
        if requested <= 0:
            records.append({
                "layer_index": layer_index,
                "channels_before": current_channels,
                "channels_pruned": 0,
                "status": "skipped_ratio",
            })
            continue
        if round_to > 1:
            requested = (requested // round_to) * round_to
        # Never remove all channels, and keep at least one alignment block.
        requested = min(requested, current_channels - max(1, round_to))
        if requested <= 0:
            records.append({
                "layer_index": layer_index,
                "channels_before": current_channels,
                "channels_pruned": 0,
                "status": "skipped_min_channels",
            })
            continue
        indices = torch.argsort(scores, descending=False)[:requested].tolist()
        group = graph.get_pruning_group(layer, tp.prune_conv_out_channels, indices)
        if not graph.check_pruning_group(group):
            records.append({
                "layer_index": layer_index,
                "channels_before": current_channels,
                "channels_pruned": 0,
                "status": "skipped_invalid_dependency_group",
            })
            continue
        group.prune()
        records.append({
            "layer_index": layer_index,
            "channels_before": current_channels,
            "channels_pruned": requested,
            "channels_after": int(layer.out_channels),
            "status": "pruned",
        })

    model.requires_grad_(False)
    target_params = _parameter_count(model)
    return model, {
        "method": "torch_pruning_dependency_aware",
        "dependency_pruner": "torch-pruning",
        "pruning_input_size": size,
        "channel_sparsity": ratio,
        "channel_round": round_to,
        "max_pruned_layers": max_layers,
        "importance_norm": norm,
        "stage_selection": stage_selection,
        "source_parameter_count": source_params,
        "structured_parameter_count": target_params,
        "parameter_reduction": 1.0 - target_params / source_params if source_params else None,
        "pruned_layers": records,
        "pruned_layer_count": sum(item["status"] == "pruned" for item in records),
    }


__all__ = ["prune_detector_dependency_aware"]
