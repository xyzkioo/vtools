# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""单目深度模型的仅尺度校准。

对数深度头预测场景的相对结构（形状）；绝对尺度由独立的双参数对数仿射变换
``d' = exp(a·log d + b)`` 表示，并存储在深度头的 ``cal_a``/``cal_b`` 缓冲区中。
此模块使用少量图像上的真实深度，通过闭式最小二乘拟合 ``(a, b)``，不进行梯度训练，也不修改解码器权重。
它同时支持训练器的自动后训练校准和 ``Model.calibrate()`` API。

自动校准采用“只有有帮助时才校准”的策略（:func:`select_calibration_cv`）：
候选方案在留出的图像上评分，仅当优于未校准输出时才应用，从而修正跨域模型的绝对尺度而不损害域内模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import smart_inference_mode


def _depth_head(model: torch.nn.Module) -> torch.nn.Module | None:
    """返回包含 ``cal_a``/``cal_b`` 缓冲区的 Depth 头模块；不存在时返回 None。"""
    m = model.module if hasattr(model, "module") else model  # unwrap DDP
    seq = getattr(m, "model", None)
    if seq is not None and not isinstance(seq, torch.nn.Sequential):
        seq = getattr(seq, "model", None)
    head = seq[-1] if isinstance(seq, torch.nn.Sequential) else m
    return head if hasattr(head, "cal_a") else None


def _rewind(dataloader) -> None:
    """在部分遍历前将有状态数据加载器回退到 epoch 开始位置。

    训练器的 InfiniteDataLoader 会在多个 ``for`` 循环之间保留同一个迭代器，因此上一次提前结束的遍历
    （例如拟合在达到 ``max_images`` 时停止）会将迭代器留在 epoch 中间，下一次循环会静默地从该位置继续。
    所有需要获取“前 N 个”样本的遍历都必须先回退；尤其是绘图遍历必须看到与 BaseValidator 绘制的
    ``val_batch{ni}.jpg`` 相同的开头批次。普通 DataLoader 和列表测试装置在每次 ``__iter__`` 时都会重启。
    """
    reset = getattr(dataloader, "reset", None)
    if callable(reset):
        reset()


def _delta1_none(log_pred: np.ndarray, log_gt: np.ndarray, a: float, b: float) -> float:
    """应用 ``d' = exp(a·log_pred + b)`` 后，在不进行逐图像对齐时计算 δ1。

    δ1 表示满足 ``max(d'/gt, gt/d') < 1.25`` 的像素比例；在对数空间中，该条件等价于
    ``|a·log_pred + b − log_gt| < log(1.25)``。这是策略优化的部署指标（原始绝对尺度，``align="none"`` 协议）；
    验证评分默认使用尺度不变的 ``align="median"``，无法反映校准效果。
    """
    ld = (a * np.asarray(log_pred, dtype=np.float64) + b) - np.asarray(log_gt, dtype=np.float64)
    return float(np.mean(np.abs(ld) < np.log(1.25)))


def select_calibration(
    lp_fit: np.ndarray,
    lg_fit: np.ndarray,
    lp_score: np.ndarray,
    lg_score: np.ndarray,
) -> dict[str, Any]:
    """选择最能提升留出集原始尺度 δ1 的校准方案，即“只有有帮助时才校准”。

    在 ``*_fit`` 对数像素数组上拟合两个候选方案，并根据 :func:`_delta1_none` 在独立的 ``*_score`` 数组上评分：
    - ``identity``（a=1, b=0）：不校准；
    - ``scale-only``（a=1, b=mean(log_gt − log_pred)）：全局尺度校准。
    选择留出集 δ1 最高的方案，并列时优先选择 identity，因此无法泛化的校准会被拒绝，不会损害自动校准。
    （曾评估过对数斜率仿射候选方案，但其额外参数会在数据集内交叉验证中过拟合，损害跨分布泛化，因此已移除。）

    返回：
        包含获胜方案 ``a``、``b``（浮点数）、``name`` 和 ``scores``（各候选方案 δ1）的字典。
    """
    lp_fit = np.asarray(lp_fit, dtype=np.float64)
    lg_fit = np.asarray(lg_fit, dtype=np.float64)
    candidates = [
        ("identity", 1.0, 0.0),
        ("scale-only", 1.0, float(np.mean(lg_fit - lp_fit))),
    ]
    scored = [(name, a, b, _delta1_none(lp_score, lg_score, a, b)) for name, a, b in candidates]
    best = max(scored, key=lambda s: s[3])  # identity first, so exact ties favor it
    return {"a": best[1], "b": best[2], "name": best[0], "scores": {s[0]: s[3] for s in scored}}


def select_calibration_cv(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    margin: float = 0.0,
    folds: int = 2,
) -> dict[str, Any]:
    """交叉验证“只有有帮助时才校准”：根据 K 折留出集 δ1 选择候选方案。

    ``pairs`` 是逐图像 ``(log_pred, log_gt)`` 数组列表。通过 :func:`select_calibration` 在每一折留出数据上评分各候选类型
    （使用其余数据拟合），再对各类型的留出 δ1 跨折求平均。因此每张图像恰好参与一次评分，
    只在某个噪声划分上获胜的候选方案不会被选中。获胜类型的平均留出 δ1 必须超过 identity 至少 ``margin``（并列时选择更简单的类型），
    最终 ``(a, b)`` 再使用全部 pairs 重新拟合。

    返回：
        包含 ``a``、``b``（浮点数）、``name`` 和 ``cv_scores``（各类型平均留出 δ1）的字典。
    """
    names = ["identity", "scale-only"]
    k = max(2, min(folds, len(pairs)))
    per_fold = {n: [] for n in names}
    for f in range(k):
        fit = [pairs[i] for i in range(len(pairs)) if i % k != f]
        score = [pairs[i] for i in range(len(pairs)) if i % k == f]
        if not fit or not score:
            continue
        s = select_calibration(
            np.concatenate([p[0] for p in fit]),
            np.concatenate([p[1] for p in fit]),
            np.concatenate([p[0] for p in score]),
            np.concatenate([p[1] for p in score]),
        )["scores"]
        for n in names:
            per_fold[n].append(s[n])
    cv = {n: float(np.mean(per_fold[n])) for n in names}
    best = "identity"
    for n in names[1:]:
        if cv[n] > cv[best] + margin:
            best = n
    # 在所有样本对上重新拟合选定类型（交叉验证选择类型，全部数据用于确定参数）。
    if best == "identity":
        a, b = 1.0, 0.0
    else:  # scale-only
        lp = np.concatenate([p[0] for p in pairs])
        lg = np.concatenate([p[1] for p in pairs])
        a, b = 1.0, float(np.mean(lg - lp))
    return {"a": a, "b": b, "name": best, "cv_scores": cv}


@smart_inference_mode()
def _collect_logpairs(
    model: torch.nn.Module, dataloader, device: torch.device | str, max_images: int, max_depth: float = 100.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """运行模型遍历数据加载器，并返回逐图像 ``(log_pred, log_gt)`` 数组列表。

    每张图像对应一项（每项最多采样 20,000 个有效像素），调用方可以将图像划分为独立的拟合集和评分集。
    仅收集 ``(0.001, max_depth)`` 范围内的真实深度，这与 ``DepthMetrics`` 评估的样本范围（Eigen 协议）一致，
    因此无效的远距离真实深度不会影响拟合尺度或选择策略依赖的留出集 δ1。
    运行期间将校准缓冲区重置为单位变换，使拟合使用原始输出，完成后再恢复。
    """
    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    _rewind(dataloader)
    rng = np.random.default_rng(0)
    pairs = []
    seen = 0
    try:
        for batch in dataloader:
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            pred = model(img).float()
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
            for pi, gi in zip(pred, gt):
                valid = (gi > 1e-3) & (gi < max_depth) & (pi > 1e-3) & torch.isfinite(pi)
                if not valid.any():
                    continue
                lp = torch.log(pi[valid]).detach().cpu().numpy()
                lg = torch.log(gi[valid]).detach().cpu().numpy()
                if lp.size > 20_000:
                    idx = rng.choice(lp.size, 20_000, replace=False)
                    lp, lg = lp[idx], lg[idx]
                pairs.append((lp, lg))
            seen += img.shape[0]
            if seen >= max_images:
                break
    finally:
        # 失败的遍历不能清除模型已有的校准；调用方会显式设置选中的值。
        head.cal_a.fill_(a0)
        head.cal_b.fill_(b0)
    return pairs


def fit_calibration_selective(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    max_images: int = 200,
    margin: float = 0.002,
    max_depth: float = 100.0,
) -> dict[str, Any] | None:
    """通过“只有有帮助时才校准”策略选择并应用校准（参见 :func:`select_calibration_cv`）。

    遍历数据加载器收集逐图像 ``(log_pred, log_gt)``，将图像划分为独立的拟合折和评分折（避免数据泄漏），
    依据交叉验证的原始尺度 δ1 选择 identity 或 scale-only，并将获胜方案写入深度头的 ``cal_a``/``cal_b``。

    返回：
        (dict | None): :func:`select_calibration_cv` 的结果字典；没有深度头或有效图像过少时返回 None。
    """
    head = _depth_head(model)
    if head is None:
        LOGGER.warning("校准：未找到包含 cal 缓冲区的深度头，跳过校准。")
        return None
    pairs = _collect_logpairs(model, dataloader, device, max_images, max_depth)
    if len(pairs) < 2:
        LOGGER.warning("校准：有效图像少于 2 张，无法划分拟合集和评分集，跳过校准。")
        return None
    res = select_calibration_cv(pairs, margin=margin)
    res["images"] = len(pairs)
    # 在 CUDA 上缓冲区是推理张量（设备移动在 inference_mode 下执行）；使用重新赋值而不是原地 fill_，
    # 这样写入合法，并确保缓冲区保持普通张量，可由 model.save() 保存。
    head.cal_a = torch.full_like(head.cal_a, res["a"])
    head.cal_b = torch.full_like(head.cal_b, res["b"])
    scores = " ".join(f"{n}={v:.4f}" for n, v in res["cv_scores"].items())
    LOGGER.info(
        f"Depth calibration selected '{res['name']}' (a={res['a']:.4f} b={res['b']:.4f}); CV held-out δ1 {scores}"
    )
    return res


def _plot_calibrated_batches(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    a: float,
    b: float,
    name: str,
    plot_dir: str | Path,
    max_batches: int = 3,
    max_images: int = 4,
) -> None:
    """将 ``val_batch{ni}_calibrated.jpg`` 面板（RGB | GT | 原始 | 校准后）写入 ``plot_dir``。

    将校准缓冲区设为单位变换运行模型以获取原始预测；校准列是对原始预测执行确定性仿射变换
    ``exp(a·log(raw) + b)`` 的结果，不需要第二次前向传播。前 ``max_batches`` 个批次与 BaseValidator 绘制的
    ``val_batch{ni}.jpg`` 相同（验证加载器未打乱），因此文件可以直接比较。
    根据“只有有帮助时才校准”策略，选中的 ``name`` 可能是 ``identity``；此时仍会写入面板（raw == calibrated），
    以记录校准未产生实际变化。完成后恢复缓冲区。
    """
    from ultralytics.utils.plotting import plot_depth_panels

    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    titles = ["RGB", "GT", "raw", f"calibrated ({name} x{np.exp(b):.2f})"]
    plot_dir = Path(plot_dir)
    _rewind(dataloader)
    with torch.no_grad():
        # zip 在 range 耗尽时停止，不会从有状态迭代器额外读取一个批次
        for ni, batch in zip(range(max_batches), dataloader):
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            raw = model(img).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
            cal = torch.exp(a * torch.log(raw.clamp(min=1e-3)) + b)
            plot_depth_panels(
                img,
                [raw, cal],
                plot_dir / f"val_batch{ni}_calibrated.jpg",
                gt=gt,
                titles=titles,
                max_images=max_images,
            )
    head.cal_a.fill_(a0)
    head.cal_b.fill_(b0)


def calibrate_checkpoint(
    ckpt_path: str | Path,
    dataloader,
    device: torch.device | str,
    plot_dir: str | Path | None = None,
    *,
    dataset_hash: str | None = None,
    validation_split: str | None = None,
    max_depth: float = 100.0,
) -> dict | None:
    """原地为已保存的检查点拟合校准参数（用于自动后训练校准）。

    加载检查点，在 ``device`` 上使用浮点副本和“只有有帮助时才校准”策略
    （:func:`fit_calibration_selective`）从 ``dataloader`` 选择校准方案，将选中的缓冲区写入已保存模型并重新保存，
    同时保留检查点的其余内容。

    参数：
        ckpt_path (str | Path): 要原地校准的 ``.pt`` 检查点文件路径。
        dataloader (对象): 产生包含 ``img``（uint8，Bx3xHxW）和 ``depth``（BxHxW，单位米）的批次。
        device (str | torch.device): 执行推理的 Torch 设备。
        plot_dir (str | Path, 可选): 设置后，将前几个验证批次的 ``val_batch{ni}_calibrated.jpg`` 对比面板
            （RGB | GT | 原始 | 校准后）写入此目录。
        dataset_hash (str, 可选): 用于校准的不可变数据集清单标识。
        validation_split (str, 可选): 用于收集校准图像、相对于数据集根目录的划分名称。
        max_depth (float): 有效真实深度的最大值，单位为米；超过此值的像素会从拟合和留出集 δ1 评分中排除，
            与验证指标的 Eigen 协议保持一致。
    """
    from copy import deepcopy

    from ultralytics.utils.patches import torch_load

    ckpt = torch_load(ckpt_path, map_location="cpu")
    saved = ckpt.get("ema") or ckpt.get("model")
    if saved is None or _depth_head(saved) is None:
        return
    work = deepcopy(saved).float()
    res = fit_calibration_selective(work, dataloader, device, max_depth=max_depth)
    if res is None:
        return None
    a, b = res["a"], res["b"]
    for key in ("ema", "model"):
        m = ckpt.get(key)
        if m is not None and _depth_head(m) is not None:
            _depth_head(m).cal_a.fill_(a)
            _depth_head(m).cal_b.fill_(b)
    stored_head = _depth_head(ckpt.get("ema") or ckpt.get("model"))
    a, b = float(stored_head.cal_a), float(stored_head.cal_b)
    provenance = {
        "status": "selected",
        "candidate": res["name"],
        "images": res["images"],
        "a": a,
        "b": b,
        "dataset_hash": dataset_hash,
        "validation_split": validation_split,
        "strategy": "two-fold-held-out-delta1",
        "scores": res["cv_scores"],
    }
    ckpt["depth_calibration"] = provenance
    torch.save(ckpt, ckpt_path)
    LOGGER.info(
        f"Auto-calibration written to {getattr(ckpt_path, 'name', ckpt_path)}: '{res['name']}' a={a:.4f} b={b:.4f}"
    )
    if plot_dir is not None:
        try:
            _plot_calibrated_batches(work, dataloader, device, a, b, res["name"], plot_dir)
            LOGGER.info(f"Calibrated val_batch plots written to {plot_dir}")
        except Exception as e:
            LOGGER.warning(f"Calibrated val plots skipped ({type(e).__name__}: {e})")
    return provenance
