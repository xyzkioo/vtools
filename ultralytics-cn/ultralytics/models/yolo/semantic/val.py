# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.dataset import SemanticDataset
from ultralytics.data.utils import add_polygon_background
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ConfusionMatrix, SemanticMetrics
from ultralytics.utils.plotting import plot_images


class SemanticSegmentationValidator(DetectionValidator):
    """用于语义分割模型的验证器。

    此验证器使用 mIoU 和像素准确率指标评估语义分割模型。

    属性：
        metrics (SemanticMetrics): 语义分割指标计算器。

    示例：
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationValidator
        >>> args = dict(model="yolo26n-sem.pt", data="cityscapes8.yaml")
        >>> validator = SemanticSegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """初始化 SemanticSegmentationValidator.

        参数：
            dataloader (DataLoader, 可选): 用于验证的数据加载器。
            save_dir (Path, 可选): 结果保存目录。
            args (dict, 可选): 验证器参数。
            _callbacks (dict, 可选): 回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semantic"
        self.dataset = None
        self.results_dir = None
        self.metrics = SemanticMetrics()
        self.image_shapes = {}
        self._semantic_target_shape = None

    def init_metrics(self, model):
        """使用模型类别名称初始化指标。

        参数：
            model (nn.Module): 待验证的模型。
        """
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemanticMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
        self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
        self.results_dir = None
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)
        cm_nc = self.metrics.cm_nc
        if cm_nc == 2 and len(self.names) == 1:  # 二分类分割，扩展为包含背景类别
            cm_names = {0: "background", 1: next(iter(self.names.values()))}
        else:
            base = list(self.names.values()) + [str(i) for i in range(len(self.names), cm_nc)]
            cm_names = {i: base[i] for i in range(cm_nc)}
        self.confusion_matrix = ConfusionMatrix(names=cm_names, task="semantic")

    def preprocess(self, batch):
        """预处理图像和掩码组成的批次。

        参数：
            batch (dict): 包含图像和掩码的批次数据。

        返回：
            (dict): 预处理后的批次。
        """
        batch = super().preprocess(batch)
        batch["semantic_mask"] = batch["semantic_mask"].to(self.device, dtype=torch.int32)
        self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def postprocess(self, preds):
        """将 logits 或固化的类别图转换为类别预测结果。

        参数：
            preds (torch.Tensor): 模型原始输出 logits [B, nc, H, W] 或固化的类别图 [B, H, W]。

        返回：
            (torch.Tensor): 预测类别 ID，形状为 [B, H, W]。
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.ndim == 3:
            # [B, H, W] 类别图的 argmax 已固化到计算图中，此处只使用最近邻缩放。
            if tuple(preds.shape[-2:]) != self._semantic_target_shape:
                preds = F.interpolate(preds[:, None].float(), size=self._semantic_target_shape, mode="nearest")[:, 0]
            return preds.to(torch.int32)
        pred_hw = preds.shape[2:]
        if pred_hw[0] != self._semantic_target_shape[0] or pred_hw[1] != self._semantic_target_shape[1]:
            preds = F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
        return preds.argmax(dim=1).to(torch.int32) if self.nc > 1 else preds.gt(0).squeeze(1).to(torch.int32)

    def update_metrics(self, preds, batch):
        """使用预测结果和真实标注更新指标。

        参数：
            preds (torch.Tensor): 预测类别 ID，形状为 [B, H, W]。
            batch (dict): 包含 'semantic_mask' 的批次数据。
        """
        if self.args.save_json:
            self.save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, batch["semantic_mask"])
        self.seen += preds.shape[0]

    def gather_stats(self):
        """在 DDP 验证期间将语义混淆矩阵归约到 rank 0。"""
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return
        if self.metrics.matrix is None:
            cm_nc = self.metrics.cm_nc
            self.metrics.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.float32)
        dist.reduce(self.metrics.matrix, dst=0, op=dist.ReduceOp.SUM)
        # 收集所有 rank 的 nt_per_image
        if RANK == 0:
            gathered_nt = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.nt_per_image, gathered_nt, dst=0)
            self.metrics.nt_per_image = np.sum(gathered_nt, axis=0)
        elif RANK > 0:
            dist.gather_object(self.metrics.nt_per_image, None, dst=0)

    def save_pred_masks(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """将语义预测结果保存为单通道 PNG 掩码。"""
        if self.results_dir is None:
            return
        im_files = batch.get("im_file", [])
        if not im_files:
            return
        preds = preds.cpu().numpy()
        if isinstance(self.dataset, SemanticDataset) and self.dataset.label_mapping:
            preds = self.dataset.convert_label(preds, inverse=True)
        preds = preds.astype(np.uint8, copy=False)
        for pred, im_file in zip(preds, im_files):
            orig_shape = self.image_shapes.get(im_file)
            if orig_shape and pred.shape != orig_shape:
                pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            save_path = self.results_dir / Path(im_file).with_suffix(".png").name
            Image.fromarray(pred).save(save_path)

    def get_stats(self):
        """返回验证统计信息。

        返回：
            (dict): 验证指标字典。
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        if self.metrics.matrix is not None:
            # 内部布局为 [gt, pred]；转置为 [pred, gt] 以符合 ConfusionMatrix 导出格式。
            self.confusion_matrix.matrix = self.metrics.matrix.detach().cpu().numpy().T.astype(float)
        return self.metrics.results_dict

    def get_desc(self):
        """返回评估指标的格式化描述字符串。

        返回：
            (str): 包含指标名称的格式化字符串。
        """
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")

    def print_results(self) -> None:
        """打印训练集或验证集的逐类别指标。"""
        super().print_results()
        if self.args.save_json and self.results_dir is not None:
            LOGGER.info(f"语义预测掩码已保存到 {self.results_dir}")

    def get_dataset(self):
        """解析数据集 YAML，并在需要时为多边形标签添加背景元数据。"""
        return add_polygon_background(super().get_dataset())

    def plot_predictions(self, batch, preds, ni):
        """在输入图像上绘制预测语义掩码。"""
        plot_images(
            images=batch["img"],
            labels={"semantic_mask": preds},
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
