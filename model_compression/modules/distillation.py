"""分类 logits 知识蒸馏。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..core.model_runtime import evaluate_classification, extract_logits, require_torch, resolve_device, save_model


def validate_distillation_config(config: Mapping[str, Any]) -> tuple[float, float]:
    values = config.get("distillation") if isinstance(config.get("distillation"), Mapping) else config
    temperature = float(values.get("temperature", values.get("T", 4.0)))
    alpha = float(values.get("alpha", 0.5))
    if temperature <= 0:
        raise ValueError("distillation.temperature/T 必须大于 0")
    if not 0 <= alpha <= 1:
        raise ValueError("distillation.alpha 必须在 [0, 1] 范围内")
    return temperature, alpha


def distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    """返回 ``(1-alpha) * CE + alpha * T² * KL``。"""

    torch = require_torch()
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必须在 [0, 1] 范围内")
    ce = torch.nn.functional.cross_entropy(student_logits, labels)
    teacher_prob = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_log_prob = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    kl = torch.nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    return (1 - alpha) * ce + alpha * (temperature**2) * kl


def train_distillation(
    student: Any,
    teacher: Any,
    train_loader: Any,
    val_loader: Any = None,
    config: Mapping[str, Any] | None = None,
    *,
    device: Any = "auto",
    output_dir: str | Path | None = None,
    on_epoch: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """执行可恢复的最小分类蒸馏训练，并保存 best/last checkpoint。"""

    torch = require_torch()
    values = dict(config or {})
    section = values.get("distillation") if isinstance(values.get("distillation"), Mapping) else values
    temperature, alpha = validate_distillation_config(values)
    epochs = int(section.get("epochs", 1))
    if epochs < 1:
        raise ValueError("distillation.epochs 必须至少为 1")
    learning_rate = float(section.get("learning_rate", section.get("lr", 1e-3)))
    seed = section.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    if train_loader is None:
        raise ValueError("蒸馏训练需要 dataset.train 或 adapter.build_dataloader")
    target_device = resolve_device(device)
    student = student.to(target_device)
    teacher = teacher.to(target_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((item for item in student.parameters() if item.requires_grad), lr=learning_rate)
    output = Path(output_dir).expanduser().resolve() if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    best_accuracy: float | None = None
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = samples = 0
        for batch in train_loader:
            if isinstance(batch, Mapping):
                inputs, labels = batch["image"], batch["label"]
            else:
                inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.to(target_device), labels.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = extract_logits(teacher(inputs))
            student_logits = extract_logits(student(inputs))
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature=temperature, alpha=alpha)
            loss.backward()
            optimizer.step()
            count = int(labels.shape[0])
            loss_sum += float(loss.item()) * count
            samples += count
        validation = evaluate_classification(student, val_loader, device=target_device) if val_loader is not None else {"status": "skipped"}
        row = {"epoch": epoch, "train_loss": loss_sum / samples if samples else None, "validation": validation}
        history.append(row)
        if on_epoch:
            on_epoch(row)
        accuracy = validation.get("accuracy") if isinstance(validation, Mapping) else None
        if output:
            save_model({"model": student, "epoch": epoch, "history": history}, output / "last.pt")
        if accuracy is not None and (best_accuracy is None or float(accuracy) > best_accuracy):
            best_accuracy, best_epoch = float(accuracy), epoch
            if output:
                save_model({"model": student, "epoch": epoch, "history": history}, output / "best.pt")
    return {
        "status": "succeeded",
        "temperature": temperature,
        "alpha": alpha,
        "epochs": epochs,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "history": history,
        "best_checkpoint": str(output / "best.pt") if output and (output / "best.pt").exists() else None,
        "last_checkpoint": str(output / "last.pt") if output and (output / "last.pt").exists() else None,
    }


__all__ = ["distillation_loss", "train_distillation", "validate_distillation_config"]
