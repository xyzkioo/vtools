# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO 模型的深度估计验证器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import DepthMetrics
from ultralytics.utils.plotting import plot_images


class DepthValidator(DetectionValidator):
    """YOLO 深度估计模型的验证器。

    计算标准深度指标：delta1、abs_rel、rmse 和 silog，并使用验证损失作为主要训练信号。
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: str | Path | None = None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        """初始化 DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """根据数据集深度范围初始化 DepthMetrics 累加器。"""
        self.metrics = DepthMetrics(max_depth=self.data.get("max_depth") or 100.0)
        self.metrics.clear_stats()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理批次：将数据移动到设备、归一化图像，并将深度保持为 float32。"""
        batch = super().preprocess(batch)
        batch["depth"] = batch["depth"].float()
        return batch

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """深度任务不需要 NMS，直接返回预测结果。"""
        return preds

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """累加一个批次的深度指标。"""
        gt_depth = batch["depth"]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
        if preds.shape[-2:] != gt_depth.shape[-2:]:
            preds = F.interpolate(preds.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)
        self.metrics.update_stats(preds, gt_depth)

    def get_stats(self) -> dict[str, float]:
        """汇总并返回指标字典。

        不同进程间的指标归约由 gather_stats() 处理（所有 rank 会在此之前调用该方法）；
        当前方法在 rank 0 上使用已经求和的累加器。
        """
        self.metrics.process()
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """将所有 DDP rank 的深度指标累加器求和到 rank 0。

        验证集会被分片（ContiguousDistributedSampler 为每个 rank 分配不同的数据块），
        因此每个 rank 只保存自身分片的统计和。通过全归约汇总这些统计量，使 rank 0 的 get_stats()
        能够根据完整验证集而不是单个分片计算指标。
        此方法覆盖 DetectionValidator.gather_stats()，后者会归约 DepthMetrics 不具备的检测专用统计量和边界框属性。
        """
        if RANK == -1 or not dist.is_initialized():
            return
        totals = self.metrics._totals
        totals = (
            totals.to(self.device) if totals is not None else torch.zeros(6, dtype=torch.float64, device=self.device)
        )
        count = torch.tensor([self.metrics._count], dtype=torch.float64, device=self.device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        self.metrics._totals = totals
        self.metrics._count = float(count.item())

    def print_results(self) -> None:
        """以检测任务的对齐表格格式记录主要深度指标。

        列与 get_desc() 对齐：Class、Images、delta1、abs_rel、rmse、silog。
        使用 "depth_val" 作为行标签（深度任务没有类别，检测任务会在此处打印“所有”）。
        """
        r = self.metrics.results_dict
        n_images = len(self.dataloader.dataset) if self.dataloader is not None else (self.seen or 0)
        pf = "%22s" + "%11i" + "%11.4g" * 4  # label, Images, delta1, abs_rel, rmse, silog
        LOGGER.info(
            pf
            % (
                "depth_val",
                n_images,
                r.get("metrics/delta1", 0.0),
                r.get("metrics/abs_rel", 0.0),
                r.get("metrics/rmse", 0.0),
                r.get("metrics/silog", 0.0),
            )
        )

    def finalize_metrics(self) -> None:
        """设置指标的最终速度信息。"""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_desc(self) -> str:
        """返回进度条描述文本。"""
        return ("%22s" + "%11s" * 5) % ("Class", "Images", "delta1", "abs_rel", "rmse", "silog")

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """将预测深度叠加图保存到 val_batch{ni}_pred.jpg。

        深度任务没有边界框和类别，因此通过共享的 ``plot_images`` 路径使用深度热力图叠加，
        取代检测任务的绘图器，并保持与语义分割可视化一致的风格。
        """
        plot_images(
            labels={"depth": preds},
            images=batch["img"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
