# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np

from ultralytics.data.utils import add_polygon_background
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import colors, plt_settings


class SemanticSegmentationTrainer(DetectionTrainer):
    """用于训练 YOLO 语义分割模型的训练器。

    此训练器负责语义分割训练，包括数据集构建、模型初始化和验证设置。

    示例：
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer
        >>> args = dict(model="yolo26n-sem.yaml", data="cityscapes8.yaml", epochs=3)
        >>> trainer = SemanticSegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化 SemanticSegmentationTrainer。

        参数：
            cfg (dict): 包含默认训练设置的配置字典。
            overrides (dict, 可选): 参数覆盖字典。
            _callbacks (dict, 可选): 回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "semantic"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        """解析数据集 YAML，并在需要时为多边形标签添加背景元数据。"""
        return add_polygon_background(super().get_dataset())

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """返回带有可选预训练主干网络的 SemanticSegmentationModel。

        参数：
            cfg (str, 可选): 模型配置文件路径。
            weights (str | Path, 可选): 模型权重路径。
            verbose (bool): 是否显示模型信息。

        返回：
            (SemanticSegmentationModel): 语义分割模型。
        """
        model = SemanticSegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于模型评估的 SemanticSegmentationValidator。"""
        return yolo.semantic.SemanticSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_class_weights(self):
        """计算基于像素频率的类别权重；二分类任务跳过（nc==1 时损失使用未加权 BCE）。"""
        if self.data["nc"] > 1:
            super().set_class_weights()

    def get_class_counts(self, max_masks=None):
        """返回训练掩码中的逐类别像素计数，可选地采样至 max_masks 个掩码。"""
        nc = self.data["nc"]
        pixel_counts = np.zeros(nc, dtype=np.float32)
        dataset = self.train_loader.dataset
        labels = getattr(dataset, "labels", [])
        if not labels:
            return pixel_counts
        indices = np.arange(len(labels))
        if max_masks and len(indices) > max_masks:
            indices = np.linspace(0, len(labels) - 1, max_masks).astype(int)
        include_class = getattr(dataset, "include_class", None)
        for idx in indices:
            shape = labels[idx].get("shape")
            try:
                mask = dataset.load_mask(idx, image_shape=tuple(shape) if shape is not None else None)
            except Exception:  # noqa: S112
                continue
            if include_class is not None:
                mask[~np.isin(mask, include_class)] = 255
            valid = (mask >= 0) & (mask < nc) & (mask != 255)
            if valid.any():
                classes, counts = np.unique(mask[valid], return_counts=True)
                pixel_counts[classes.astype(int)] += counts
        return pixel_counts

    def compute_class_weights(self, class_counts):
        """计算 ENet inverse-log `(1/ln(1.02 + p))**cls_pw` 权重 (Paszke et al., 2016, arXiv:1606.02147)."""
        p = class_counts / max(class_counts.sum(), 1.0)  # 像素频率，对稀有类别进行限制，与检测任务不同
        return (1.0 / np.log(1.02 + p)) ** self.args.cls_pw

    @plt_settings()
    def plot_training_labels(self):
        """绘制语义分割训练标签的类别分布。

        从训练数据集中最多采样 1000 个掩码文件，累计逐类别像素计数，并绘制类别分布柱状图，保存为 'labels.jpg'。
        """
        import matplotlib.pyplot as plt

        LOGGER.info(f"正在将标签分布绘制到 {self.save_dir / 'labels.jpg'}...")
        nc = self.data["nc"]
        names = self.data["names"]
        pixel_counts = self.get_class_counts(max_masks=1000)
        if not pixel_counts.any():
            LOGGER.warning("未找到语义掩码文件，跳过标签绘图。")
            return

        _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        bars = ax.bar(range(nc), pixel_counts, color=[[c / 255.0 for c in colors(i, False)] for i in range(nc)])
        ax.set_xlabel("类别")
        ax.set_ylabel("像素")
        ax.set_title("训练标签类别分布")
        if 0 < len(names) < 30:
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if self.on_plot:
            self.on_plot(fname)
