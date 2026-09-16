"""结构化模型缩放。

这条路径使用 Ultralytics 模型 YAML 的 width/depth scale 创建一个更小的
检测网络，再把原模型中形状匹配的权重迁移过去。它会真实改变卷积通道数，
因此可以导出为更小的 dense 模型并参与后续微调和测速。
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _matching_state_count(source: Any, target: Any) -> tuple[int, int]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    matched = 0
    total = 0
    for name, value in target_state.items():
        if name not in source_state:
            continue
        total += 1
        if tuple(source_state[name].shape) == tuple(value.shape):
            matched += 1
    return matched, total


def build_structured_detector(
    detector: Any,
    weights: str | Path,
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Build a smaller YOLO architecture and transfer compatible weights.

    The source model must expose Ultralytics' ``model.yaml`` with a ``scales``
    table. The detection head is rebuilt by Ultralytics and therefore keeps the
    original number of classes and output channels.
    """

    try:
        import yaml  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("结构化检测缩放需要 PyYAML 和 ultralytics") from exc

    source_yaml = getattr(getattr(detector, "model", None), "yaml", None)
    if not isinstance(source_yaml, Mapping):
        raise ValueError("当前检测模型没有可复用的 Ultralytics model.yaml，无法进行结构化缩放。")
    scales = source_yaml.get("scales")
    if not isinstance(scales, Mapping) or not scales:
        raise ValueError("当前模型没有 scales 配置；请使用带模型规模定义的 YOLO YAML。")

    section = dict(config.get("structured", {}) or {})
    target_scale = str(section.get("target_scale", section.get("scale", "n"))).strip()
    source_scale = str(source_yaml.get("scale", "")).strip()
    if target_scale not in scales:
        available = ", ".join(sorted(str(key) for key in scales))
        raise ValueError(f"结构化目标规模 {target_scale!r} 不存在，可选：{available}")
    same_scale_control = bool(source_scale and target_scale == source_scale)

    target_yaml = copy.deepcopy(dict(source_yaml))
    target_yaml["scale"] = target_scale
    target_yaml.pop("yaml_file", None)
    # Ultralytics derives the scale from the YAML filename and overwrites the
    # in-file ``scale`` value.  A generic ``structured-s.yaml`` therefore
    # silently falls back to the first scale (usually ``n``).  Preserve the
    # source model family in a filename such as ``yolo26s-vtools.yaml`` so
    # ``s→s`` really builds an s model and ``s→n`` really builds an n model.
    source_yaml_name = Path(str(source_yaml.get("yaml_file", ""))).stem
    family_match = re.match(r"^(.*?)[nslmx]$", source_yaml_name)
    family = family_match.group(1) if family_match else "yolo26"
    target_path = run_dir / f"{family}{target_scale}-vtools.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(target_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    slim = YOLO(str(target_path), task="detect")
    initial_weights = section.get("initial_weights")
    configured_initial = bool(str(initial_weights).strip()) if initial_weights else False
    auto_initial = False
    if configured_initial:
        transfer_weights = Path(str(initial_weights)).expanduser().resolve()
    else:
        # Prefer a target-scale checkpoint when one is already available.  A
        # source ``s`` checkpoint is a poor initializer for an ``n`` network:
        # most of its convolution tensors have incompatible shapes.  Search
        # the repository, the source project, and the source checkpoint's
        # nearby directories without downloading anything implicitly.
        target_checkpoint = f"yolo26{target_scale}.pt"
        source_path = Path(weights).expanduser().resolve()
        candidates = [
            Path.cwd() / "weights" / target_checkpoint,
            Path(__file__).resolve().parents[2] / "weights" / target_checkpoint,
            source_path.parent / target_checkpoint,
        ]
        candidates.extend(parent / "weights" / target_checkpoint for parent in source_path.parents)
        transfer_weights = next((candidate.resolve() for candidate in candidates if candidate.is_file()), source_path)
        auto_initial = transfer_weights != source_path
    if not transfer_weights.is_file():
        raise FileNotFoundError(f"结构化初始化权重不存在：{transfer_weights}")
    transfer_source = YOLO(str(transfer_weights), task="detect")
    matched, comparable = _matching_state_count(transfer_source.model, slim.model)
    slim.load(str(transfer_weights))
    if hasattr(slim.model, "names") and hasattr(detector.model, "names"):
        slim.model.names = copy.deepcopy(detector.model.names)

    source_params = _parameter_count(detector.model)
    target_params = _parameter_count(slim.model)
    return slim, {
        "method": "structured_width_scaling",
        "source_scale": source_scale or None,
        "target_scale": target_scale,
        "same_scale_control": same_scale_control,
        "source_parameter_count": source_params,
        "structured_parameter_count": target_params,
        "parameter_reduction": 1.0 - target_params / source_params if source_params else None,
        "matched_state_items": matched,
        "comparable_state_items": comparable,
        "initial_weights": str(transfer_weights),
        "initialization": (
            "configured_pretrained" if configured_initial
            else "auto_pretrained" if auto_initial
            else "source_shape_match"
        ),
        "architecture_yaml": str(target_path),
    }


def finetune_structured_detector(
    detector: Any,
    config: Mapping[str, Any],
    data: Path,
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Optionally fine-tune the reduced detector and reload its best checkpoint."""

    compression = config.get("compression", config)
    section = dict(compression.get("structured", {}) or {})
    epochs = int(section.get("finetune_epochs", 80))
    if epochs < 0:
        raise ValueError("结构化微调轮数不能小于 0；设为 0 可跳过微调。")
    if epochs == 0:
        return detector, {"finetune_epochs": 0, "finetune": "skipped"}

    dataset = dict(config.get("dataset", {}) or {})
    evaluation = dict(config.get("evaluation", {}) or {})
    input_size = section.get("train_input_size", dataset.get("input_size", 640))
    batch_size = int(section.get("batch_size", evaluation.get("batch_size", 1)))
    if batch_size < 1:
        raise ValueError("结构化微调 batch_size 必须大于 0。")
    device = config.get("model", {}).get("device", "auto")
    if device == "auto":
        device = None
    train_kwargs: dict[str, Any] = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": input_size,
        "batch": batch_size,
        "workers": 0,
        "project": str(run_dir),
        "name": "structured_finetune",
        "exist_ok": True,
        "pretrained": True,
        "resume": False,
        "device": device,
        "lr0": float(section.get("learning_rate", section.get("lr0", 0.001))),
        "plots": False,
    }
    if section.get("optimizer"):
        train_kwargs["optimizer"] = str(section["optimizer"])
    if "patience" in section:
        train_kwargs["patience"] = int(section["patience"])
    if "amp" in section:
        train_kwargs["amp"] = bool(section["amp"])
    # Keep the actual pruned module; rebuilding from model.yaml restores the
    # original channel counts even when pretrained=True.
    preserved = detector.model
    shapes = {k: tuple(v.shape) for k, v in preserved.state_dict().items()}
    base_trainer = detector._smart_load("trainer")

    class PreservedModelTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if int(self.data["nc"]) != int(preserved.model[-1].nc):
                raise ValueError("微调数据集类别数与剪枝模型不一致。")
            self.model = preserved

        def get_model(self, cfg=None, weights=None, verbose=True):
            return preserved

    for key in ("fraction", "warmup_bias_lr", "warmup_epochs"):
        if key in section:
            train_kwargs[key] = section[key]
    detector.train(trainer=PreservedModelTrainer, **train_kwargs)
    trainer = getattr(detector, "trainer", None)
    best = getattr(trainer, "best", None) if trainer is not None else None
    if best and Path(best).is_file():
        from ultralytics import YOLO  # type: ignore

        restored = YOLO(str(best))
        if {k: tuple(v.shape) for k, v in restored.model.state_dict().items()} != shapes:
            raise RuntimeError("微调后网络结构发生变化，拒绝导出。")
        return restored, {"finetune_epochs": epochs, "finetune": "completed", "best_checkpoint": str(best)}
    raise RuntimeError("结构化微调结束但没有找到 best.pt，请检查训练日志和数据集配置。")


__all__ = ["build_structured_detector", "finetune_structured_detector"]
