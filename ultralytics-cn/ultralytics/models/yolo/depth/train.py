# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO 模型的深度估计训练器。"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plt_settings


class DepthTrainer(DetectionTrainer):
    """YOLO 深度估计模型的训练器。

    多源训练（img_paths 列表）由基础 DetectionTrainer/BaseDataset 透明处理。

    示例：
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26s-depth.yaml", data="nyu-depth.yaml", epochs=100)
        >>> trainer = DepthTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None
    ) -> None:
        """初始化 DepthTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> DepthModel:
        """返回使用给定配置和权重初始化的 DepthModel。"""
        model = DepthModel(
            cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0}
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理批次：归一化图像，并将深度保持为 float32。"""
        batch = super().preprocess_batch(batch)
        batch["depth"] = batch["depth"].float()
        return batch

    def get_validator(self) -> yolo.depth.DepthValidator:
        """返回用于模型验证的 DepthValidator。"""
        return yolo.depth.DepthValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    @plt_settings()
    def plot_training_labels(self) -> None:
        """将训练集 GT 深度分布绘制到 ``labels.jpg``。

        这是检测/语义分割标签图的深度任务对应版本。继承的 DetectionTrainer 实现会拼接每张图像的
        ``bboxes``/``cls``（深度任务中全部为空）并传给 ``plot_labels``，导致归约操作报错。
        因此改为从训练集中采样 GT 深度图，绘制有效（``> 0``）深度值的直方图，并标注基本统计信息。
        """
        import matplotlib.pyplot as plt
        import numpy as np

        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
        dataset = self.train_loader.dataset
        n = len(dataset.im_files)
        if n == 0:
            LOGGER.warning("No depth maps found, skipping label plot.")
            return

        sample_size = min(1000, n)
        indices = np.linspace(0, n - 1, sample_size).astype(int)
        per_map_cap = max(1, 1_000_000 // sample_size)  # bound total memory to ~1M 值
        values = []
        for idx in indices:
            d = dataset._load_depth(idx)
            if d is None:
                continue
            v = d[d > 0].ravel()
            if v.size == 0:
                continue
            if v.size > per_map_cap:  # 均匀步长可保持空间分布无偏
                v = v[np.linspace(0, v.size - 1, per_map_cap).astype(int)]
            values.append(v)

        if not values:
            LOGGER.warning("No valid depth values found, skipping label plot.")
            return

        values = np.concatenate(values)
        vmin, vmax = float(values.min()), float(np.percentile(values, 99.5))
        mean, median, std = float(values.mean()), float(np.median(values)), float(values.std())

        _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        ax.hist(values, bins=100, range=(vmin, max(vmax, vmin + 1e-6)), color="#3b7dd8")
        ax.axvline(mean, color="#d8643b", linestyle="--", linewidth=1.5, label=f"mean {mean:.2f} m")
        ax.axvline(median, color="#3bd86b", linestyle="--", linewidth=1.5, label=f"median {median:.2f} m")
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Pixels")
        ax.set_title("Training Labels Depth Distribution")
        ax.legend(loc="upper right", frameon=False)
        stats = f"images: {sample_size}\nmin: {vmin:.2f} m\nmax: {values.max():.2f} m\nstd: {std:.2f} m"
        ax.text(
            0.98,
            0.7,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if self.on_plot:
            self.on_plot(fname)

    def final_eval(self) -> None:
        """执行标准最终评估，然后校准已保存的检查点。

        训练完成后，在验证集上拟合仅缩放的对数仿射参数（``cal_a``/``cal_b``），并写入 best.pt/last.pt，
        使模型输出经过指标尺度校准的深度。当设置 ``plots`` 时，还会写入
        ``val_batch{ni}_calibrated.jpg``（RGB | GT | 原始 | 校准后）对比面板。
        """
        super().final_eval()
        if RANK not in {-1, 0}:
            return
        try:
            from .calibrate import calibrate_checkpoint

            LOGGER.info("Calibrating depth output scale on the validation set...")
            plot_ckpt = self.best if self.best.exists() else self.last
            for ckpt in (self.best, self.last):
                if ckpt.exists():
                    plot_dir = self.save_dir if self.args.plots and ckpt == plot_ckpt else None
                    validation_path = self.data.get("val") or self.data.get("test")
                    validation_split = None
                    if isinstance(validation_path, (str, Path)):
                        try:
                            validation_split = (
                                Path(validation_path)
                                .resolve()
                                .relative_to(Path(self.data["path"]).resolve())
                                .as_posix()
                            )
                        except ValueError:
                            pass  # 外部验证路径没有可移植的相对数据集根目录标识。
                    provenance = calibrate_checkpoint(
                        ckpt,
                        self.test_loader,
                        self.device,
                        plot_dir=plot_dir,
                        dataset_hash=self.data.get("hash"),
                        validation_split=validation_split,
                        max_depth=self.data.get("max_depth") or 100.0,
                    )
                    if ckpt == plot_ckpt and provenance is not None:
                        self.depth_calibration = provenance
        except Exception as e:
            LOGGER.warning(f"Calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
