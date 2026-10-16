"""Train YOLO detectors with the native Ultralytics Fork distillation trainer."""

from __future__ import annotations

import os
import shutil
import math
from pathlib import Path
from typing import Any, Mapping

from .structured import _ultralytics_amp_weights_dir


def _positive_int(section: Mapping[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not str(raw).strip().lstrip('+-').isdigit():
        raise ValueError(f"检测蒸馏 {key} 必须是整数。")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"检测蒸馏 {key} 必须大于等于 {minimum}。")
    return value


def detection_distillation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate training settings before starting an expensive evaluation or training run."""
    section = config.get("distillation") or {}
    if not isinstance(section, Mapping):
        raise ValueError("distillation 必须是配置对象。")
    epochs = _positive_int(section, "epochs", 100)
    batch = _positive_int(section, "batch_size", 4)
    imgsz = section.get("imgsz", config.get("dataset", {}).get("input_size", 640))
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2 or imgsz[0] != imgsz[1]:
            raise ValueError("检测蒸馏 imgsz 必须为单个正整数（方形训练输入）。")
        imgsz = imgsz[0]
    imgsz = _positive_int({"imgsz": imgsz}, "imgsz", 640)
    dis = float(section.get("dis", 6.0))
    if not math.isfinite(dis) or dis <= 0:
        raise ValueError("检测蒸馏 dis 必须为正有限数，0 会关闭蒸馏损失。")
    lr = float(section.get("learning_rate", 0.001))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("检测蒸馏 learning_rate 必须为正有限数。")
    workers = _positive_int(section, "workers", 0, minimum=0)
    optimizer = str(section.get("optimizer", "auto")).strip()
    if optimizer not in {"auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}:
        raise ValueError(f"检测蒸馏 optimizer 不支持：{optimizer}")
    amp = section.get("amp", True)
    if not isinstance(amp, bool):
        raise ValueError("检测蒸馏 amp 必须是布尔值。")
    return {
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "dis": dis,
        "lr0": lr, "seed": _positive_int(section, "seed", 0, minimum=0),
        "workers": workers, "optimizer": optimizer,
        "patience": _positive_int(section, "patience", 100, minimum=0),
        "amp": amp,
    }


def train_detection_distillation(
    *, student_weights: Path, teacher_weights: Path, data: Path,
    config: Mapping[str, Any], output_dir: Path,
) -> dict[str, Any]:
    """Return paths and effective arguments; only the Fork computes distillation loss."""
    settings = detection_distillation_settings(config)
    try:
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.distill_model import DistillationModel  # noqa: F401 - native interface check
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "当前 Ultralytics 未提供 Detect 原生蒸馏，请配置支持 distill_model 的 ultralytics-cn Fork。"
        ) from exc
    if not {"distill_model", "dis"}.issubset(DEFAULT_CFG_DICT):
        raise RuntimeError("当前 Ultralytics 训练配置缺少 distill_model/dis，请切换至支持蒸馏的 Fork。")

    device = config.get("model", {}).get("device", "auto")
    kwargs: dict[str, Any] = {
        "data": str(data), "distill_model": str(teacher_weights), **settings,
        "device": None if str(device).lower() == "auto" else str(device),
        "project": str(output_dir), "name": "training", "exist_ok": False,
        "plots": False, "resume": False,
    }
    student = YOLO(str(student_weights), task="detect")
    with _ultralytics_amp_weights_dir():
        student.train(**kwargs)
    trainer = getattr(student, "trainer", None)
    best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else output_dir / "training" / "weights" / "best.pt"
    last = Path(trainer.last) if trainer is not None and getattr(trainer, "last", None) else output_dir / "training" / "weights" / "last.pt"
    if not best.is_file():
        raise RuntimeError(f"检测蒸馏训练未生成 best.pt：{best}（last.pt={last}）")

    # Copy atomically. The native Fork strips DistillationModel when training ends;
    # the caller must still reload and validate the standalone student.
    target = output_dir / "student-best.pt"
    if target.exists():
        raise FileExistsError(f"蒸馏结果已存在：{target}")
    temporary = output_dir / ".student-best.pt.tmp"
    try:
        shutil.copy2(best, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"best_checkpoint": str(target), "trainer_best": str(best),
            "last_checkpoint": str(last) if last.is_file() else None,
            "train_directory": str(best.parent.parent),
            "epochs": settings["epochs"], "batch_size": settings["batch"],
            "imgsz": settings["imgsz"], "dis": settings["dis"],
            "learning_rate": kwargs["lr0"], "optimizer": kwargs["optimizer"],
            "patience": kwargs["patience"], "amp": kwargs["amp"],
            "workers": kwargs["workers"], "seed": kwargs["seed"]}
