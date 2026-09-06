# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模型验证指标。"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings
from ultralytics.utils.plotting import colors

OKS_SIGMA = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)
RLE_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5])
CITYSCAPES_WEIGHT = np.array(
    [
        0.8373,
        0.918,
        0.866,
        1.0345,
        1.0166,
        0.9969,
        0.9754,
        1.0489,
        0.8786,
        1.0023,
        0.9539,
        0.9843,
        1.1116,
        0.9037,
        1.0865,
        1.0955,
        1.0865,
        1.1529,
        1.0507,
    ]
)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """给定 box1 和 box2，计算交集面积与 box2 面积之比。

    参数：
        box1 (np.ndarray): 形状为 (N, 4) 的 NumPy 数组，表示 x1y1x2y2 格式的 N 个边界框。
        box2 (np.ndarray): 形状为 (M, 4) 的 NumPy 数组，表示 x1y1x2y2 格式的 M 个边界框。
        iou (bool, 可选): 为 True 时计算标准 IoU，否则返回 inter_area/box2_area。
        eps (float, 可选): 用于避免除零的小数值。

    返回：
        (np.ndarray): 形状为 (N, M) 的 NumPy 数组，表示交集面积与 box2 面积之比。
    """
    # 获取边界框坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # 交集面积
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # box2 面积
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # 交集面积与 box2 面积之比
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算边界框的交并比（IoU）。

    参数：
        box1 (torch.Tensor): 形状为 (N, 4) 的张量，表示 (x1, y1, x2, y2) 格式的 N 个边界框。
        box2 (torch.Tensor): 形状为 (M, 4) 的张量，表示 (x1, y1, x2, y2) 格式的 M 个边界框。
        eps (float, optional): 用于避免除零的小数值。

    返回：
        (torch.Tensor): 形状为 N×M 的张量，包含 box1 和 box2 中每个边界框对的 IoU 值。

    参考：
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # 注意：需要使用 .float() 才能得到准确的 IoU 值
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """计算边界框之间的交并比（IoU）。

    只要最后一个维度为 4，此函数就支持 `box1` 和 `box2` 的多种形状，例如 (4,)、(N, 4)、(B, N, 4) 或 (B, N, 1, 4)。
    当 `xywh=True` 时，内部将最后一个维度解析为 (x, y, w, h)；当 `xywh=False` 时解析为 (x1, y1, x2, y2)。

    参数：
        box1 (torch.Tensor): 表示一个或多个边界框的张量，最后一个维度为 4。
        box2 (torch.Tensor): 表示一个或多个边界框的张量，最后一个维度为 4。
        xywh (bool, 可选): 为 True 时输入边界框使用 (x, y, w, h) 格式，否则使用 (x1, y1, x2, y2) 格式。
        GIoU (bool, 可选): 是否计算广义 IoU。
        DIoU (bool, 可选): 是否计算距离 IoU。
        CIoU (bool, 可选): 是否计算完全 IoU。
        eps (float, 可选): 用于避免除零的小数值。

    返回：
        (torch.Tensor): 根据指定标志返回 IoU、GIoU、DIoU 或 CIoU 值。
    """
    # 获取边界框坐标
    if xywh:  # 从 xywh 转换为 xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 并集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing 边界框) 宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex 高度
        if CIoU or DIoU:  # Distance IoU 或 Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算掩码 IoU。

    参数：
        mask1 (torch.Tensor): 形状为 (N, n) 的张量，其中 N 是真实目标数量，n 是图像宽度与高度的乘积。
        mask2 (torch.Tensor): 形状为 (M, n) 的张量，其中 M 是预测目标数量，n 是图像宽度与高度的乘积。
        eps (float, 可选): 用于避免除零的小数值。

    返回：
        (torch.Tensor): 形状为 (N, M) 的张量，表示掩码 IoU。
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    """计算目标关键点相似度（OKS）。

    参数：
        kpt1 (torch.Tensor): 形状为 (N, 17, 3) 的张量，表示真实关键点。
        kpt2 (torch.Tensor): 形状为 (M, 17, 3) 的张量，表示预测关键点。
        area (torch.Tensor): 形状为 (N,) 的张量，表示真实目标面积。
        sigma (列表[float]): 包含 17 个关键点尺度值的列表。
        eps (float, 可选): 用于避免除零的小数值。

    返回：
        (torch.Tensor): 形状为 (N, M) 的张量，表示关键点相似度。
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # 来自 cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # 公式形式
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes: torch.Tensor, floor: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据旋转边界框生成协方差矩阵。

    参数：
        boxes (torch.Tensor): 形状为 (N, 5) 的张量，表示 xywhr 格式的旋转边界框。
        floor (float, 可选): 添加到宽度和高度的较小值，用于限制小于步长边界框的梯度。

    返回：
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 协方差矩阵分量 (a, b, c)，矩阵为 [[a, c], [c, b]]，
            每个分量的形状为 (N, 1)。
    """
    # 高斯边界框；忽略中心点（前两列），因为这里不需要它们。
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12 + floor, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7, floor: float = 0.0
) -> torch.Tensor:
    """计算旋转边界框之间的概率 IoU。

    参数：
        obb1 (torch.Tensor): 真实 OBB，形状为 (N, 5)，格式为 xywhr。
        obb2 (torch.Tensor): 预测 OBB，形状为 (N, 5)，格式为 xywhr。
        CIoU (bool, optional): 是否计算 CIoU。
        eps (float, optional): 用于避免除零的小数值。
        floor (float, optional): 传递给 `_get_covariance_matrix` 的小数值，用于限制小于步长边界框的梯度。

    返回：
        (torch.Tensor): OBB 相似度，形状为 (N,)。

    注意：
        OBB 格式：[center_x, center_y, width, height, rotation_angle]。

    参考：
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1, floor)
    a2, b2, c2 = _get_covariance_matrix(obb2, floor)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # 仅包含宽高比部分
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """计算旋转边界框之间的概率 IoU。

    参数：
        obb1 (torch.Tensor | np.ndarray): 形状为 (N, 5) 的张量，表示 xywhr 格式的真实 OBB。
        obb2 (torch.Tensor | np.ndarray): 形状为 (M, 5) 的张量，表示 xywhr 格式的预测 OBB。
        eps (float, optional): 用于避免除零的小数值。

    返回：
        (torch.Tensor): 形状为 (N, M) 的张量，表示 OBB 相似度。

    参考：
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """计算平滑后的正、负二元交叉熵目标值。

    参数：
        eps (float, optional): 标签平滑使用的 epsilon 值。

    返回：
        pos (float): Positive label smoothing BCE target.
        neg (float): Negative label smoothing BCE target.

    参考：
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix(DataExportMixin):
    """计算和更新目标检测、分类等任务混淆矩阵的类。

    属性：
        task (str): 任务类型，可选 'detect'、'classify'、'semantic' 或 'obb'。
        matrix (np.ndarray): 混淆矩阵，维度取决于任务类型。
        nc (int): 类别数量。
        names (dict[int, str]): 类别名称，用作绘图标签。
        matches (dict | None): 包含按 TP、FP 和 FN 分类的真实标注与预测结果索引。
    """

    def __init__(self, names: dict[int, str] | None = None, task: str = "detect", save_matches: bool = False):
        """初始化 ConfusionMatrix 实例。

        参数：
            names (dict[int, str], 可选): 类别名称，用作绘图标签。
            task (str, 可选): 任务类型，可选 'detect'、'classify'、'semantic' 或 'obb'。
            save_matches (bool, 可选): 是否保存 GT、TP、FP 和 FN 的索引以供可视化。
        """
        names = names if names is not None else {}
        self.task = task
        self.nc = len(names)  # 类别数量
        self.matrix = (
            np.zeros((self.nc, self.nc))
            if self.task in {"classify", "semantic"}
            else np.zeros((self.nc + 1, self.nc + 1))
        )
        self.names = names  # 类别名称
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        """将上一批次的匹配结果追加到 TP、FP、FN 或 GT 列表。

        此方法将批次数据追加到 matches 字典中对应的匹配类型（真正例、假正例或假负例）。

        参数：
            mtype (str): 匹配类型标识符（'TP'、'FP'、'FN' 或 'GT'）。
            batch (dict[str, Any]): 包含检测结果的批次数据，键包括 'bboxes'、'cls'、'conf'、'keypoints' 和 'masks'。
            idx (int): 要从批次中追加的具体检测索引。

        注意：
            对掩码同时处理重叠和非重叠情况。当 masks.max() > 1.0 时，表示 overlap_mask=True 且形状为 (1, H, W)，
            否则直接使用索引。
        """
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                # 注意：masks.max() > 1.0 表示 overlap_mask=True，形状为 (1, H, W)
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """更新分类任务的混淆矩阵。

        参数：
            preds (列表[torch.Tensor]): 预测类别标签。
            targets (列表[torch.Tensor]): 真实类别标签。
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """更新目标检测任务的混淆矩阵。

        参数：
            detections (dict[str, torch.Tensor]): 包含检测边界框及其相关信息的字典。应包含 'cls'、'conf' 和 'bboxes' 键，
                其中 'bboxes' 对普通边界框可为 Array[N, 4]，对带角度的 OBB 可为 Array[N, 5]。
            batch (dict[str, Any]): 包含真实数据的批次字典，含有 'bboxes'（Array[M, 4] 或 Array[M, 5]）和 'cls'（Array[M]）键，
                其中 M 是真实目标数量。
            conf (float, 可选): 检测置信度阈值。
            iou_thres (float, 可选): 将检测结果与真实标注匹配时使用的 IoU 阈值。
        """
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:  # 仅在启用可视化时执行
            self.matches = {k: defaultdict(list) for k in ("TP", "FP", "FN", "GT")}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)  # 保存 GT
        is_obb = gt_bboxes.shape[1] == 5  # 检查边界框是否包含 OBB 角度
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:  # 检查标签是否为空
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1  # FP
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)
            return

        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        m0, m1, _ = matches.transpose().astype(int)
        # matches 的两列均已去重，因此每个 gt 和每个检测结果最多出现一次
        gt_match = np.full(len(gt_classes), -1)
        gt_match[m0] = m1
        matched_det = set(m1.tolist())
        for i, gc in enumerate(gt_classes):
            if (di := gt_match[i].item()) >= 0:
                dc = detection_classes[di]
                self.matrix[dc, gc] += 1  # 类别正确时为 TP，否则同时计为 FP 和 FN
                if dc == gc:
                    self._append_matches("TP", detections, di)
                else:
                    self._append_matches("FP", detections, di)
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)

        for i, dc in enumerate(detection_classes):
            if i not in matched_det:
                self.matrix[dc, self.nc] += 1  # FP
                self._append_matches("FP", detections, i)

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """返回真正例和假正例。

        返回：
            tp (np.ndarray): 真正例。
            fp (np.ndarray): 假正例。
        """
        tp = self.matrix.diagonal()  # 真正例
        fp = self.matrix.sum(1) - tp  # 假正例
        # fn = self.matrix.sum(0) - tp  # 假负例（漏检）
        return (tp, fp) if self.task in {"classify", "semantic"} else (tp[:-1], fp[:-1])  # 移除背景行和列

    def plot_matches(
        self, img: torch.Tensor, im_file: str, save_dir: Path, show_labels: bool = True, show_conf: bool = True
    ) -> None:
        """为每张图像绘制 GT、TP、FP、FN 网格。

        参数：
            img (torch.Tensor): 要绘制的图像。
            im_file (str): 用于保存可视化结果的图像文件名。
            save_dir (Path): 可视化结果保存位置。
            show_labels (bool): 是否在可视化结果中显示类别标签。
            show_conf (bool): 是否显示置信度值。
        """
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        # 创建包含 4 组结果的批次（GT、TP、FP、FN）
        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = torch.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
            mbatch["batch_idx"] = torch.ones(len(mbatch["bboxes"]), device=img.device) * i
            for k in mbatch:
                labels[k] += mbatch[k]

        labels = {k: torch.stack(v, 0) if len(v) else torch.empty(0) for k, v in labels.items()}
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=["Ground Truth", "False Positives", "True Positives", "False Negatives"],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
            show_labels=show_labels,
            show_conf=show_conf,
        )

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """使用 matplotlib 绘制混淆矩阵并保存到文件。

        参数：
            normalize (bool, 可选): 是否对混淆矩阵进行归一化。
            save_dir (str, 可选): 绘图保存目录。
            on_plot (callable, 可选): 绘图完成后调用的回调函数，可接收绘图路径和数据。
        """
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # 归一化列
        array[array < 0.005] = np.nan  # 不标注这些值（否则会显示为 0.00）

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:  # 类别数量较大时进行下采样
            k = max(2, self.nc // 60)  # 下采样步长，始终大于 1
            keep_idx = slice(None, None, k)  # 创建切片而不是数组
            names = names[keep_idx]  # 切分类别名称
            array = array[keep_idx, :][:, keep_idx]  # 切分矩阵的行和列
            n = (self.nc + k - 1) // k  # 保留的类别数量
        nc = n if self.task in {"classify", "semantic"} else n + 1  # 必要时为背景调整数量
        ticklabels = "auto"
        if 0 < nc < 99:
            ticklabels = names if self.task in {"classify", "semantic"} else [*names, "background"]
        xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)
        tick_fontsize = max(6, 15 - 0.1 * nc)  # 最小尺寸为 6
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)  # 最小值为 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制空矩阵产生的 RuntimeWarning：All-NaN slice encountered
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:  # 为混淆矩阵的每个单元格添加数值
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))  # text color 阈值
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "混淆矩阵" + "（归一化）" * normalize
        ax.set_xlabel("真实类别", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("预测类别", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in ("left", "right", "bottom", "top", "outline"):
            if s != "outline":
                ax.spines[s].set_visible(False)  # 混淆矩阵图不显示外框
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)  # 调整布局以保持边距一致
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()})

    def print(self):
        """将混淆矩阵打印到控制台。"""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
        """将混淆矩阵汇总为字典列表，并可选择进行归一化。

        该表示形式便于将矩阵导出为 CSV、XML、HTML、JSON 或 SQL 等多种格式。

        参数：
            normalize (bool): 是否对混淆矩阵值进行归一化。
            decimals (int): 输出值保留的小数位数。

        返回：
            (列表[dict[str, float]]): 字典列表，每个字典表示一个预测类别及其对应的所有真实类别值。

        示例：
            >>> results = model.val(data="coco8.yaml", plots=True)
            >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
            >>> print(cm_dict)
        """
        import re

        names = (
            list(self.names.values())
            if self.task in {"classify", "semantic"}
            else [*list(self.names.values()), "background"]
        )
        clean_names, seen = [], set()
        for name in names:
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)).round(decimals)
        return [
            dict({"Predicted": clean_names[i]}, **{clean_names[j]: array[i, j] for j in range(len(clean_names))})
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """使用比例 f 的盒式滤波器平滑数组。"""
    nf = round(len(y) * f * 2) // 2 + 1  # 滤波器元素数量（必须为奇数）
    p = np.ones(nf // 2)  # 全 1 填充
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # 填充后的 y
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # 平滑后的 y


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] | None = None,
    on_plot=None,
):
    """绘制精确率-召回率曲线。

    参数：
        px (np.ndarray): PR 曲线的 X 值。
        py (np.ndarray): PR 曲线的 Y 值。
        ap (np.ndarray): 平均精确率值。
        save_dir (Path, 可选): 绘图保存路径。
        names (dict[int, str], 可选): 类别索引到类别名称的映射字典。
        on_plot (callable, 可选): 绘图保存后调用的回调函数。
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    names = names if names is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # 类别数小于 21 时显示逐类别图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # 传递 PR 曲线数据用于交互式绘图（类别名称保存在模型层级）
        # 转置 py 以匹配其他曲线的 y[类别][点] 格式
        on_plot(save_dir, {"type": "pr_curve", "x": px.tolist(), "y": py.T.tolist(), "ap": ap.tolist()})


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] | None = None,
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """绘制指标-置信度曲线。

    参数：
        px (np.ndarray): 指标-置信度曲线的 X 值。
        py (np.ndarray): 指标-置信度曲线的 Y 值。
        save_dir (Path, 可选): 绘图保存路径。
        names (dict[int, str], 可选): 类别索引到类别名称的映射字典。
        xlabel (str, 可选): X 轴标签。
        ylabel (str, 可选): Y 轴标签。
        on_plot (callable, 可选): 绘图保存后调用的回调函数。
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    names = names if names is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # 类别数小于 21 时显示逐类别图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(置信度, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # plot(置信度, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # 传递指标-置信度曲线数据用于交互式绘图（类别名称保存在模型层级）
        on_plot(save_dir, {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()})


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """根据召回率和精确率曲线计算平均精度（AP）。

    参数：
        recall (列表[float]): 召回率曲线。
        precision (列表[float]): 精确率曲线。

    返回：
        ap (float): 平均精度。
        mpre (np.ndarray): 精确率包络线曲线。
        mrec (np.ndarray): 修改后的召回率曲线，开头和结尾添加了哨兵值。
    """
    # 在开头和结尾追加哨兵值
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # 计算精确率包络线
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 对曲线下方面积进行积分
    method = "interp"  # 方法：'continuous'、'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101 个插值点（COCO）
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz 已弃用
        ap = func(np.interp(x, mrec, mpre), x)  # 积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # x 轴（召回率）发生变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] | None = None,
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """计算目标检测评估中每个类别的平均精度。

    参数：
        tp (np.ndarray): 指示检测是否正确（True 或 False）的二值数组。
        conf (np.ndarray): 检测结果的置信度分数数组。
        pred_cls (np.ndarray): 检测结果的预测类别数组。
        target_cls (np.ndarray): 目标的真实类别数组。
        plot (bool, 可选): 是否绘制 PR 曲线。
        on_plot (callable, 可选): 图表渲染后接收图表路径和数据的回调函数。
        save_dir (Path, 可选): 保存 PR 曲线的目录。
        names (dict[int, str], 可选): 用于绘制 PR 曲线的类别名称字典。
        eps (float, 可选): 用于防止除零的小值。
        prefix (str, 可选): 保存绘图文件时使用的前缀字符串。

    返回：
        tp (np.ndarray): 最大 F1 指标阈值处每个类别的真正例数量。
        fp (np.ndarray): 最大 F1 指标阈值处每个类别的假正例数量。
        p (np.ndarray): 最大 F1 指标阈值处每个类别的精确率。
        r (np.ndarray): 最大 F1 指标阈值处每个类别的召回率。
        f1 (np.ndarray): 最大 F1 指标阈值处每个类别的 F1 分数。
        ap (np.ndarray): 不同 IoU 阈值下每个类别的平均精度。
        unique_classes (np.ndarray): 包含数据的唯一类别数组。
        p_curve (np.ndarray): 每个类别的精确率曲线。
        r_curve (np.ndarray): 每个类别的召回率曲线。
        f1_curve (np.ndarray): 每个类别的 F1 分数曲线。
        x (np.ndarray): 曲线的 x 轴值。
        prec_values (np.ndarray): mAP@0.5 处每个类别的精确率值。
    """
    names = names if names is not None else {}
    # 按目标置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数量、检测数量

    # 创建精确率-召回率曲线，并计算每个类别的 AP
    x, prec_values = np.linspace(0, 1, 1000), []

    # 平均精度、精确率和召回率曲线
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 标签数量
        n_p = i.sum()  # 预测结果数量
        if n_p == 0 or n_l == 0:
            prec_values.append(np.zeros_like(x))  # 每个类别保留一行，与 `ap` 和 `names` 对齐
            continue

        # 累积假正例和真正例
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # 精确率
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # 根据召回率-精确率曲线计算 AP
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # 计算 F1（精确率和召回率的调和平均值）
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # 字典：仅保留有数据的类别
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 索引
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 值
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """用于计算 Ultralytics YOLO 模型评估指标的类。

    属性：
        p (列表): 每个类别的精确率，形状为 (nc,)。
        r (列表): 每个类别的召回率，形状为 (nc,)。
        f1 (列表): 每个类别的 F1 分数，形状为 (nc,)。
        all_ap (列表): 所有类别和所有 IoU 阈值下的 AP 分数，形状为 (nc, 10)。
        ap_class_index (列表): 每个 AP 分数对应的类别索引，形状为 (nc,)。
        nc (int): 类别数量。

    方法：
        ap50: 所有类别在 IoU 阈值 0.5 下的 AP。
        ap: 所有类别在 0.5 到 0.95 IoU 阈值范围内的 AP。
        mp: 所有类别的平均精确率。
        mr: 所有类别的平均召回率。
        map50: 所有类别在 IoU 阈值 0.5 下的平均 AP。
        map75: 所有类别在 IoU 阈值 0.75 下的平均 AP。
        map: 所有类别在 0.5 到 0.95 IoU 阈值范围内的平均 AP。
        mean_results: 结果平均值，返回 mp、mr、map50 和 map。
        class_result: 类别相关结果，返回 p[i]、r[i]、ap50[i] 和 ap[i]。
        maps: 每个类别的 mAP。
        fitness: 指标的加权组合，即模型适应度。
        update: 使用新的评估结果更新指标属性。
        curves: 返回用于访问精确率、召回率和 F1 等特定指标的曲线列表。
        curves_results: 返回用于访问精确率、召回率和 F1 等特定指标的结果列表。
    """

    def __init__(self) -> None:
        """初始化用于计算 YOLO 模型评估指标的 Metric 实例。"""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0
        self.image_metrics = {}

    @property
    def ap50(self) -> np.ndarray | list:
        """返回所有类别在 IoU 阈值 0.5 下的平均精度（AP）。

        返回：
            (np.ndarray | 列表): 形状为 (nc,) 的数组，包含每个类别的 AP50 值；没有数据时返回空列表。
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """返回所有类别在 IoU 阈值 0.5 到 0.95 下的平均精度（AP）。

        返回：
            (np.ndarray | 列表): 形状为 (nc,) 的数组，包含每个类别的 AP50-95 值；没有数据时返回空列表。
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """返回所有类别的平均精确率。

        返回：
            (float): 所有类别的平均精确率。
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """返回所有类别的平均召回率。

        返回：
            (float): 所有类别的平均召回率。
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """返回 IoU 阈值为 0.5 时的平均精度（mAP）。

        返回：
            (float): IoU 阈值为 0.5 时的 mAP。
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """返回 IoU 阈值为 0.75 时的平均精度（mAP）。

        返回：
            (float): IoU 阈值为 0.75 时的 mAP。
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """返回 IoU 阈值 0.5 到 0.95（步长 0.05）范围内的平均精度（mAP）。

        返回：
            (float): IoU 阈值 0.5 到 0.95（步长 0.05）范围内的 mAP。
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        """返回结果、mp、mr、map50 和 map 的平均值。"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回类别相关结果 p[i]、r[i]、ap50[i] 和 ap[i]。"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """返回每个类别的 mAP。"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        """返回指标加权组合作为模型适应度分数。"""
        w = [0.0, 0.0, 0.0, 1.0]  # [P、R、mAP@0.5、mAP@0.5:0.95] 的权重
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    def update(self, results: tuple):
        """使用一组新的结果更新评估指标。

        参数：
            results (tuple): 包含以下评估指标的元组：
                - p (列表)：每个类别的精确率。
                - r (列表)：每个类别的召回率。
                - f1 (列表)：每个类别的 F1 分数。
                - all_ap (列表)：所有类别和所有 IoU 阈值下的 AP 分数。
                - ap_class_index (列表)：每个 AP 分数对应的类别索引。
                - p_curve (列表)：每个类别的精确率曲线。
                - r_curve (列表)：每个类别的召回率曲线。
                - f1_curve (列表)：每个类别的 F1 曲线。
                - px (列表)：曲线的 X 值。
                - prec_values (列表)：每个类别的精确率值。
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    def clear_image_metrics(self) -> None:
        """清除当前验证运行中保存的逐图像指标。"""
        self.image_metrics.clear()

    @property
    def curves(self) -> list:
        """返回用于访问特定指标曲线的曲线列表。"""
        return []

    @property
    def curves_results(self) -> list[list]:
        """返回用于访问特定指标曲线的曲线结果列表。"""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]

    def update_image_metrics(self, tp: np.ndarray, target_cls: np.ndarray, pred_cls: np.ndarray, im_name: str) -> None:
        """更新 IoU 阈值为 0.5 时的逐图像精确率、召回率、F1、TP、FP 和 FN。

        参数：
            tp (np.ndarray): 形状为 (num_preds, num_iou_thresholds) 的真正例数组，使用第一列（IoU >= 0.5）。
            target_cls (np.ndarray): 图像的真实类别标签。
            pred_cls (np.ndarray): 图像的预测类别标签。
            im_name (str): 用作逐图像键的图像文件名。
        """
        # 使用默认 IoU=0.5 列，以匹配验证器的逐图像匹配策略。
        tp = int(tp[:, 0].sum())
        num_preds = pred_cls.shape[0]
        num_targets = target_cls.shape[0]
        fp = num_preds - tp
        fn = num_targets - tp
        if num_preds == 0 and num_targets == 0:
            # 没有预测结果且没有 GT 的图像属于显然正确的情况，因此报告满分，
            # 不使用下方标准 0/0 回退逻辑将 P/R/F1 置零。
            precision = recall = f1 = 1.0
        else:
            precision = tp / num_preds if num_preds else 0.0
            recall = tp / num_targets if num_targets else 0.0
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom else 0.0
        self.image_metrics[im_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }


class DetMetrics(SimpleClass, DataExportMixin):
    """用于计算精确率、召回率和平均精度（mAP）等检测指标的工具类。

    属性：
        names (dict[int, str]): 类别名称字典。
        box (Metric): 保存检测结果的 Metric 实例。
        speed (dict[str, float]): 保存检测流程各部分执行时间的字典。
        stats (dict[str, list]): 包含真正例、置信度分数、预测类别、目标类别和目标图像列表的字典。
        nt_per_class：每个类别的目标数量。
        nt_per_image：每张图像的目标数量。

    方法：
        update_stats：将新值追加到现有统计集合中。
        process：处理目标检测预测结果并更新指标。
        clear_stats：清除已保存的统计信息。
        keys：返回用于访问特定指标的键列表。
        mean_results：计算检测目标的平均结果，并返回精确率、召回率、mAP50 和 mAP50-95。
        class_result：返回目标检测模型在指定类别上的评估结果。
        maps：返回每个类别的平均精度（mAP）分数。
        fitness：返回边界框目标的适应度。
        ap_class_index：返回每个类别的平均精度索引。
        results_dict：返回包含计算后性能指标和统计信息的字典。
        curves：返回用于访问特定指标曲线的曲线列表。
        curves_results：返回计算后的性能指标和统计信息列表。
        summary：将逐类别检测指标汇总为字典列表。
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """使用类别名称初始化 DetMetrics 实例。

        参数：
            names (dict[int, str], 可选): 类别名称字典。
        """
        self.names = names if names is not None else {}
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        """将新值追加到现有统计集合中。

        参数：
            stat (dict[str, Any]): 包含待追加统计值的字典，键应与 self.stats 中的现有键匹配。
        """
        for k in self.stats:
            self.stats[k].append(stat[k])
        self.box.update_image_metrics(stat["tp"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理目标检测预测结果并更新指标。

        参数：
            save_dir (Path): 绘图保存目录，默认为 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线，默认为 False。
            on_plot (callable, 可选): 绘图生成后调用的回调函数，默认为 None。

        返回：
            (dict[str, np.ndarray]): 包含拼接后统计数组的字典。
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # 拼接为 NumPy 数组
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """清除已保存的统计信息。"""
        for v in self.stats.values():
            v.clear()

    def clear_image_metrics(self) -> None:
        """清除已保存的逐图像指标。"""
        self.box.clear_image_metrics()

    @property
    def keys(self) -> list[str]:
        """返回用于访问特定指标的键列表。"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        """计算检测目标的平均结果，并返回精确率、召回率、mAP50 和 mAP50-95。"""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回目标检测模型在指定类别上的评估结果。"""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回每个类别的平均精度（mAP）分数。"""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """返回边界框目标的适应度。"""
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """返回每个类别的平均精度索引。"""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """返回包含计算后性能指标和统计信息的字典。"""
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线列表。"""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self) -> list[list]:
        """返回计算后的性能指标和统计信息列表。"""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """将逐类别检测指标汇总为字典列表。

        对每个类别同时包含共享标量指标（mAP、mAP50、mAP75）、精确率、召回率和 F1 分数。

        参数：
            normalize (bool): 对于 Detect 指标，是否将所有值归一化到 [0-1]，默认为 True。
            decimals (int): 指标值保留的小数位数。

        返回：
            (列表[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应指标值。

        示例：
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {
            "Box-P": self.box.p,
            "Box-R": self.box.r,
            "Box-F1": self.box.f1,
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):
    """计算并汇总给定类别集合上的检测和分割指标。

    属性：
        names (dict[int, str]): 类别名称字典。
        box (Metric): 保存检测结果的 Metric 实例。
        seg (Metric): 用于计算掩码分割指标的 Metric 实例。
        speed (dict[str, float]): 保存检测流程各部分执行时间的字典。
        stats (dict[str, list]): 包含真正例、置信度分数、预测类别、目标类别和目标图像列表的字典。
        nt_per_class：每个类别的目标数量。
        nt_per_image：每张图像的目标数量。

    方法：
        process：处理给定预测结果集合上的检测和分割指标。
        keys：返回用于访问指标的键列表。
        mean_results：返回边界框和分割结果的平均指标。
        class_result：返回指定类别索引的分类结果。
        maps：返回目标检测和分割模型的 mAP 分数。
        fitness：返回分割模型和边界框模型的适应度分数。
        curves：返回用于访问特定指标曲线的曲线列表。
        curves_results：提供计算后的性能指标和统计信息列表。
        summary：将逐类别分割指标汇总为字典列表。
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """使用类别名称初始化 SegmentMetrics 实例。

        参数：
            names (dict[int, str], 可选): 类别名称字典。
        """
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.stats["tp_m"] = []  # 添加掩码的额外统计信息

    def update_stats(self, stat: dict[str, Any]) -> None:
        """将新值追加到现有统计集合中。

        参数：
            stat (dict[str, Any]): 包含待追加统计值的字典，键应与 self.stats 中的现有键匹配。
        """
        super().update_stats(stat)  # 更新边界框统计信息
        self.seg.update_image_metrics(stat["tp_m"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """清除已保存的逐图像指标。"""
        super().clear_image_metrics()
        self.seg.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理给定预测结果集合上的检测和分割指标。

        参数：
            save_dir (Path): 绘图保存目录，默认为 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线，默认为 False。
            on_plot (callable, 可选): 绘图生成后调用的回调函数，默认为 None。

        返回：
            (dict[str, np.ndarray]): 包含拼接后统计数组的字典。
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # 处理边界框统计信息
        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats

    @property
    def keys(self) -> list[str]:
        """返回用于访问指标的键列表。"""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self) -> list[float]:
        """返回边界框和分割结果的平均指标。"""
        return DetMetrics.mean_results(self) + self.seg.mean_results()

    def class_result(self, i: int) -> list[float]:
        """返回指定类别索引的分类结果。"""
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回目标检测和分割模型的 mAP 分数。"""
        return DetMetrics.maps.fget(self) + self.seg.maps

    @property
    def fitness(self) -> float:
        """返回分割模型和边界框模型的适应度分数。"""
        return self.seg.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线名称列表。"""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回计算得到的性能指标和统计信息列表。"""
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每个类别分割指标的摘要字典列表。
        摘要包含边界框和掩码的标量指标（mAP、mAP50、mAP75），以及每个类别的精确率、召回率和 F1 分数。

        参数：
            normalize (bool): 对 Segment 指标，是否默认将所有数值归一化到 [0, 1]。
            decimals (int): 指标数值保留的小数位数。

        返回：
            (列表[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应的指标值。

        示例：
            >>> results = model.val(data="coco8-seg.yaml")
            >>> seg_summary = results.summary(decimals=4)
            >>> print(seg_summary)
        """
        per_class = {
            "Mask-P": self.seg.p,
            "Mask-R": self.seg.r,
            "Mask-F1": self.seg.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # 获取边界框摘要
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class PoseMetrics(DetMetrics):
    """计算并汇总给定类别集合上的检测和姿态指标。

    属性：
        names (dict[int, str]): 类别名称字典。
        pose (Metric): 用于计算姿态指标的 Metric 实例。
        box (Metric): 用于保存检测结果的 Metric 实例。
        speed (dict[str, float]): 保存检测流程各部分执行时间的字典。
        stats (dict[str, list]): 包含真正例、置信度分数、预测类别、目标类别和目标图像列表的字典。
        nt_per_class: 每个类别的目标数量。
        nt_per_image: 每张图像的目标数量。

    方法：
        process: 处理给定预测结果中的检测和姿态指标。
        keys: 返回用于访问指标的键列表。
        mean_results: 返回边界框和姿态的平均结果。
        class_result: 返回指定类别 i 的类别级检测结果。
        maps: 返回每个类别的边界框和姿态检测平均精度（mAP）。
        fitness: 返回姿态和边界框检测的组合适应度分数。
        curves: 返回用于访问特定指标曲线的曲线名称列表。
        curves_results: 返回计算得到的性能指标和统计信息列表。
        summary: 生成每个类别姿态指标的摘要字典列表。
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """使用类别名称初始化 PoseMetrics 实例。

        参数：
            names (dict[int, str], 可选): 类别名称字典。
        """
        super().__init__(names)
        self.pose = Metric()
        self.stats["tp_p"] = []  # 添加姿态任务的额外统计信息

    def update_stats(self, stat: dict[str, Any]) -> None:
        """将新值追加到现有统计数据集合中，以更新统计信息。

        参数：
            stat (dict[str, Any]): 包含待追加统计值的字典。键应与 self.stats 中的现有键一致。
        """
        super().update_stats(stat)  # 更新边界框统计信息
        self.pose.update_image_metrics(stat["tp_p"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """清除已保存的逐图像指标。"""
        super().clear_image_metrics()
        self.pose.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """根据给定的预测结果处理检测和姿态指标。

        参数：
            save_dir (Path): 保存绘图的目录，默认为 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线，默认为 False。
            on_plot (callable, 可选): 绘图生成后调用的函数。

        返回：
            (dict[str, np.ndarray]): 包含拼接统计数组的字典。
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # 处理边界框统计信息
        results_pose = ap_per_class(
            stats["tp_p"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        return stats

    @property
    def keys(self) -> list[str]:
        """返回评估指标键列表。"""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self) -> list[float]:
        """返回边界框和姿态的平均结果。"""
        return DetMetrics.mean_results(self) + self.pose.mean_results()

    def class_result(self, i: int) -> list[float]:
        """返回指定类别 i 的类别级检测结果。"""
        return DetMetrics.class_result(self, i) + self.pose.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回每个类别的边界框和姿态检测平均精度（mAP）。"""
        return DetMetrics.maps.fget(self) + self.pose.maps

    @property
    def fitness(self) -> float:
        """返回姿态和边界框检测的组合适应度分数。"""
        return self.pose.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线名称列表。"""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回计算得到的性能指标和统计信息列表。"""
        return DetMetrics.curves_results.fget(self) + self.pose.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每个类别姿态指标的摘要字典列表。
        摘要包含边界框和姿态的标量指标（mAP、mAP50、mAP75），以及每个类别的精确率、召回率和 F1 分数。

        参数：
            normalize (bool): 对 Pose 指标，是否默认将所有数值归一化到 [0, 1]。
            decimals (int): 指标数值保留的小数位数。

        返回：
            (列表[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应的指标值。

        示例：
            >>> results = model.val(data="coco8-pose.yaml")
            >>> pose_summary = results.summary(decimals=4)
            >>> print(pose_summary)
        """
        per_class = {
            "Pose-P": self.pose.p,
            "Pose-R": self.pose.r,
            "Pose-F1": self.pose.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # 获取边界框摘要
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class ClassifyMetrics(SimpleClass, DataExportMixin):
    """计算分类指标的类，包括 Top-1 和 Top-5 准确率。

    属性：
        top1 (float): Top-1 准确率。
        top5 (float): Top-5 准确率。
        speed (dict[str, float]): 包含流水线各步骤耗时的字典。

    方法：
        process: 处理目标类别和预测类别，并计算指标。
        fitness: 返回 Top-1 和 Top-5 准确率的平均值，作为适应度分数。
        results_dict: 返回包含模型性能指标和适应度分数的字典。
        keys: 返回 results_dict 属性使用的键列表。
        curves: 返回用于访问特定指标曲线的曲线名称列表。
        curves_results: 返回计算得到的性能指标和统计信息列表。
        summary: 生成分类指标的单行摘要（Top-1 和 Top-5 准确率）。
    """

    def __init__(self) -> None:
        """初始化 ClassifyMetrics 实例。"""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """处理目标类别和预测类别，并据此计算指标。

        参数：
            targets (torch.Tensor): 目标类别。
            pred (torch.Tensor): 预测类别。
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # （Top-1、Top-5）准确率
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self) -> float:
        """返回 Top-1 和 Top-5 准确率的平均值，作为适应度分数。"""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        """返回包含模型性能指标和适应度分数的字典。"""
        return dict(zip([*self.keys, "fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self) -> list[str]:
        """返回 results_dict 属性使用的键列表。"""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self) -> list:
        """返回用于访问特定指标曲线的曲线名称列表。"""
        return []

    @property
    def curves_results(self) -> list:
        """返回用于访问特定指标曲线的曲线结果列表。"""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
        """生成分类指标的单行摘要（Top-1 和 Top-5 准确率）。

        参数：
            normalize (bool): 对分类指标，是否默认将所有数值归一化到 [0, 1]。
            decimals (int): 指标数值保留的小数位数。

        返回：
            (列表[dict[str, float]]): 只包含一个字典的列表，该字典记录 Top-1 和 Top-5 分类准确率。

        示例：
            >>> results = model.val(data="imagenet10")
            >>> classify_summary = results.summary(decimals=4)
            >>> print(classify_summary)
        """
        return [{"top1_acc": round(self.top1, decimals), "top5_acc": round(self.top5, decimals)}]


class OBBMetrics(DetMetrics):
    """用于评估旋转边界框（OBB）检测的指标。

    属性：
        names (dict[int, str]): 类别名称字典。
        box (Metric): 保存检测结果的 Metric 实例。
        speed (dict[str, float]): 保存检测流程各部分执行时间的字典。
        stats (dict[str, list]): 包含真正例、置信度分数、预测类别、目标类别和目标图像列表的字典。
        nt_per_class：每个类别的目标数量。
        nt_per_image：每张图像的目标数量。

    参考：
        https://arxiv.org/pdf/2106.06072.pdf
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """使用类别名称初始化 OBBMetrics 实例。

        参数：
            names (dict[int, str], 可选): 类别名称字典。
        """
        DetMetrics.__init__(self, names)


class SemanticMetrics(SimpleClass, DataExportMixin):
    """用于语义分割的指标，包括 mIoU、像素准确率和逐类别 IoU。

    属性：
        names (dict): 类别名称映射。
        nc (int): 类别数量.
        cm_nc (int): 混淆矩阵边长（二分类分割为 2，否则为 nc）。
        device (torch.device | None): 累计混淆矩阵所用的设备。
        matrix (torch.Tensor | None): 累计混淆矩阵，形状为 (cm_nc, cm_nc)。
        speed (dict): 处理速度统计信息。
        nt_per_image (np.ndarray): 包含每个类别图像数量的数组。
        nt_per_class (np.ndarray): 每个类别的像素数量。
        _miou (float): 缓存的平均 IoU。
        _pixel_accuracy (float): 缓存的像素准确率。
        _per_class_iou (np.ndarray): 缓存的逐类别 IoU 值。
        _per_class_pixel_acc (np.ndarray): 缓存的逐类别像素准确率。
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """初始化语义分割指标。

        参数：
            names (dict, 可选): 类别索引到名称的映射字典。
        """
        self.names = names or {}
        self.nc = len(self.names)
        self.cm_nc = 2 if self.nc == 1 else self.nc
        self.matrix = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.nt_per_image = np.zeros(self.nc, dtype=np.int32)
        self._miou = 0.0
        self._pixel_accuracy = 0.0
        self._per_class_iou = np.zeros(self.nc, dtype=np.float32)
        self._per_class_pixel_acc = np.zeros(self.nc, dtype=np.float32)
        self.nt_per_class = np.zeros(self.nc, dtype=np.int32)

    def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """根据预测结果和目标累计混淆矩阵。

        参数：
            preds (torch.Tensor): 预测类别 ID，形状为 [B, H, W]。
            targets (torch.Tensor): 真实类别 ID，形状为 [B, H, W]。
        """
        if self.matrix is None:
            self.matrix = torch.zeros((self.cm_nc, self.cm_nc), device=preds.device, dtype=torch.float32)

        valid = (targets != 255) & (preds >= 0) & (preds < self.cm_nc) & (targets >= 0) & (targets < self.cm_nc)
        hist = torch.bincount(self.cm_nc * targets[valid] + preds[valid], minlength=self.cm_nc**2).reshape(
            self.cm_nc, self.cm_nc
        )
        self.matrix += hist.to(self.matrix.dtype)

        present = torch.zeros((targets.shape[0], self.cm_nc), dtype=torch.bool, device=targets.device)
        batch_idx = torch.arange(targets.shape[0], device=targets.device).view(-1, 1, 1).expand_as(targets)
        present[batch_idx[valid], targets[valid].long()] = True
        if self.nc == 1:
            self.nt_per_image[0] += int(present[:, 1].sum())
        else:
            self.nt_per_image += present[:, : self.nc].sum(0).cpu().numpy()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot: callable | None = None) -> None:
        """根据累计的混淆矩阵计算最终指标。

        参数：
            save_dir (Path): 绘图保存目录，默认为 Path('.')。
            plot (bool): 是否绘制 IoU 柱状图和混淆矩阵，默认为 False。
            on_plot (callable, 可选): 绘图生成后调用的回调函数，默认为 None。
        """
        if self.matrix is None:
            return

        intersection = torch.diagonal(self.matrix)
        union = self.matrix.sum(1) + self.matrix.sum(0) - intersection
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection, dtype=torch.float32))
        row_sum = self.matrix.sum(1)
        pa = intersection / (row_sum + 1e-10)

        if self.nc == 1:
            self._miou = float(iou[1].item())
            self._per_class_iou = iou[1:].cpu().numpy()
            self._per_class_pixel_acc = pa[1:].cpu().numpy()
            self.nt_per_class = np.array([row_sum[1].item()], dtype=np.int32)
        else:
            # 仅对真实标注中出现的类别计算平均 IoU；没有 GT 像素的类别（不在验证集或被 `classes` 过滤掉）会被排除。
            present = row_sum > 0
            self._miou = float(iou[present].mean().item()) if present.any() else 0.0
            self._per_class_iou = iou.cpu().numpy()
            self._per_class_pixel_acc = pa.cpu().numpy()
            self.nt_per_class = row_sum[: self.nc].cpu().numpy().astype(np.int32)

        self._pixel_accuracy = float((intersection.sum() / (self.matrix.sum() + 1e-10)).item())

        if plot:
            self._plot_iou_bars(save_dir, on_plot)

    def clear_stats(self):
        """清除已累积的统计信息。"""
        self.matrix = None
        self.nt_per_image.fill(0)

    @plt_settings()
    def _plot_iou_bars(self, save_dir, on_plot):
        """绘制逐类别 IoU 柱状图。

        参数：
            save_dir (Path | str): 绘图保存目录。
            on_plot (callable, 可选): 绘图保存后调用的回调函数。
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
        names = list(self.names.values()) if self.names else [str(i) for i in range(self.nc)]
        x = np.arange(self.nc)
        bars = ax.bar(x, self._per_class_iou, color=[[c / 255.0 for c in colors(i, False)] for i in range(self.nc)])
        ax.set_xlabel("类别")
        ax.set_ylabel("IoU")
        ax.set_title("逐类别 IoU")
        ax.set_ylim(0, 1)
        if 0 < len(names) < 30:
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=90, fontsize=10)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
        fname = Path(save_dir) / "iou_bar_chart.png"
        plt.savefig(fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(fname)

    @property
    def miou(self):
        """返回平均 IoU（二分类分割仅返回前景 IoU）。"""
        return self._miou

    @property
    def pixel_accuracy(self):
        """返回整体像素准确率。"""
        return self._pixel_accuracy

    @property
    def per_class_iou(self):
        """返回逐类别 IoU 值（二分类分割仅返回前景 IoU）。"""
        return self._per_class_iou

    @property
    def per_class_pixel_accuracy(self):
        """返回逐类别像素准确率（每个类别的对角线值除以行和）。"""
        return self._per_class_pixel_acc

    @property
    def fitness(self):
        """返回模型适应度，即平均 IoU。"""
        return self.miou

    @property
    def keys(self):
        """返回用于日志记录的指标键。"""
        return ["metrics/mIoU", "metrics/pixel_acc"]

    def mean_results(self):
        """返回用于日志记录的平均结果。"""
        return [self.miou, self.pixel_accuracy]

    def class_result(self, i: int) -> list[float]:
        """返回指定类别的性能评估结果。"""
        if self._per_class_iou is None or len(self._per_class_iou) == 0:
            return [0.0, 0.0]
        c = self.ap_class_index[i]
        return [float(self._per_class_iou[c]), float(self._per_class_pixel_acc[c])]

    @property
    def ap_class_index(self):
        """返回真实标注中出现的类别索引，用于逐类别报告。"""
        return [i for i in range(self.nc) if self.nt_per_class[i] > 0]

    @property
    def results_dict(self):
        """返回 结果 字典."""
        return dict(zip([*self.keys, "fitness"], [*self.mean_results(), self.fitness]))

    @property
    def curves(self):
        """返回空列表，因为语义分割没有 PR 曲线。"""
        return []

    @property
    def curves_results(self):
        """返回空列表（没有 PR 曲线结果）。"""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]:
        """生成逐类别语义分割指标汇总，每行包含全局 mIoU 和像素准确率。

        参数：
            normalize (bool): 语义指标值已处于 [0, 1] 范围，此参数仅为保持接口一致。
            decimals (int): 指标值保留的小数位数。

        返回：
            (列表[dict]): 字典列表，每个类别对应一个字典，包含逐类别 IoU 和共享标量指标。
        """
        miou = round(self.miou, decimals)
        pixel_acc = round(self.pixel_accuracy, decimals)
        per_class = self.per_class_iou
        names = self.names or {i: str(i) for i in range(len(per_class))}
        return [
            {
                "Class": names.get(c, str(c)),
                "Images": int(self.nt_per_image[c]),
                "Pixels": int(self.nt_per_class[c]),
                "IoU": round(float(per_class[c]), decimals),
                "mIoU": miou,
                "pixel_acc": pixel_acc,
            }
            for c in self.ap_class_index
        ]


class DepthMetrics(SimpleClass, DataExportMixin):
    """单目深度估计指标：delta1-3、abs_rel、rmse 和 silog。

    指标按图像完成计算，再在验证集上求平均，使每张图像的权重相同，不受有效像素数量影响，
    与 Depth Anything V2 和 Monodepth2 使用的逐样本平均方式一致。有效真实深度像素少于 10 个的图像会被完全跳过，
    这与 Depth Anything V2 对其有效掩码采用的下限相同。非有限预测值会按深度边界计分，而不是从平均值中隐藏该图像。
    逐图像结果在 CPU 上以 float64 累计，因此 DDP 归约仍是简单的求和再 all_reduce。按照标准 Eigen 评估协议，
    gt 超出 (min_depth, max_depth) 的像素会被排除，预测结果会被限制在该范围内。

    属性：
        min_depth (float): 有效深度的最小值，单位为米。
        max_depth (float): 有效深度的最大值，单位为米。
    """

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 100.0,
        align: str = "median",
    ) -> None:
        """初始化深度指标累计器。

        参数：
            min_depth (float): 有效深度最小值，单位为米；gt <= min_depth 的像素会被忽略。
            max_depth (float): 有效深度最大值，单位为米；gt >= max_depth 的像素会被忽略，预测结果会被限制为该值。
            align (str): 按照 Depth Anything 评估协议，在评分前执行逐图像尺度对齐。
                "median" 使用 median(gt)/median(pred) 重新缩放每个预测，使尺度存在歧义的输出可与真实深度比较；
                "none" 禁用对齐，直接使用预测结果的原始输出尺度评分。
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.align = align
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self._totals = None
        self._count = 0.0
        self._results = {}

    def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """累计逐图像指标，并执行逐图像尺度对齐。

        参数：
            preds (torch.Tensor): 预测深度，形状为 (B,1,H,W) 或 (B,H,W)。
            targets (torch.Tensor): 真实深度，单位为米，形状相同。
        """
        p = preds.squeeze(1) if preds.ndim == 4 else preds
        g = targets.squeeze(1) if targets.ndim == 4 else targets
        if p.ndim == 2:  # single 图像 (H,W) -> (1,H,W) so alignment is always per-图像
            p, g = p[None], g[None]
        for pi, gi in zip(p, g):
            # Eigen 协议：仅对 gt 位于 (min_depth, max_depth) 内的像素评分
            mask = (gi > self.min_depth) & (gi < self.max_depth)
            if int(mask.sum()) < 10:  # Depth Anything V2 floor: aligning the median of a few pixels is meaningless
                continue
            pv = pi[mask].float()
            gv = gi[mask].float()
            if self.align == "median":
                finite = torch.isfinite(pv)
                if finite.any():
                    scale = torch.median(gv[finite]) / torch.median(pv[finite].clamp_min(self.min_depth))
                    pv = pv * scale
            pv = torch.nan_to_num(pv, nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth).clamp(
                self.min_depth, self.max_depth
            )
            thresh = torch.maximum(pv / gv, gv / pv)
            log_diff = torch.log(pv) - torch.log(gv)
            # λ=1 方差形式（ZoeDepth/KITTI），按图像完成计算，使 silog 也按样本平均
            silog = (log_diff.pow(2).mean() - log_diff.mean().pow(2)).clamp_min(0.0).sqrt() * 100
            image_metrics = torch.stack(
                [
                    (thresh < 1.25).float().mean(),
                    (thresh < 1.25**2).float().mean(),
                    (thresh < 1.25**3).float().mean(),
                    (torch.abs(pv - gv) / gv).mean(),
                    ((pv - gv) ** 2).mean().sqrt(),
                    silog,
                ]
            )
            if self._totals is None:
                self._totals = torch.zeros(6, dtype=torch.float64)
            self._totals += image_metrics.cpu().double()  # float64 on CPU; MPS 张量 cannot be float64
            self._count += 1.0

    def process(self, *args, **kwargs) -> None:
        """对累计的逐图像结果求平均，完成指标计算。"""
        if self._totals is None or self._count == 0:
            self._results = dict.fromkeys(self.keys, 0.0)
            return
        d1, d2, d3, abs_rel, rmse, silog = (float(x) for x in self._totals / self._count)
        self._results = {
            "metrics/delta1": d1,
            "metrics/delta2": d2,
            "metrics/delta3": d3,
            "metrics/abs_rel": abs_rel,
            "metrics/rmse": rmse,
            "metrics/silog": silog,
        }

    def clear_stats(self) -> None:
        """重置指标累计器。"""
        self._totals = None
        self._count = 0.0
        self._results = {}

    @property
    def keys(self) -> list[str]:
        """用于日志记录的指标键。"""
        return [
            "metrics/delta1",
            "metrics/delta2",
            "metrics/delta3",
            "metrics/abs_rel",
            "metrics/rmse",
            "metrics/silog",
        ]

    def mean_results(self) -> list[float]:
        """按 `keys` 顺序返回指标值。"""
        return [self._results.get(k, 0.0) for k in self.keys]

    @property
    def delta1(self) -> float:
        """逐图像平均像素比例，其中 max(p/g, g/p) < 1.25。"""
        return self._results.get("metrics/delta1", 0.0)

    @property
    def delta2(self) -> float:
        """逐图像平均像素比例，其中 max(p/g, g/p) < 1.25**2。"""
        return self._results.get("metrics/delta2", 0.0)

    @property
    def delta3(self) -> float:
        """逐图像平均像素比例，其中 max(p/g, g/p) < 1.25**3。"""
        return self._results.get("metrics/delta3", 0.0)

    @property
    def abs_rel(self) -> float:
        """逐图像平均绝对相对误差。"""
        return self._results.get("metrics/abs_rel", 0.0)

    @property
    def rmse(self) -> float:
        """逐图像平均均方根误差（单位：米）。"""
        return self._results.get("metrics/rmse", 0.0)

    @property
    def silog(self) -> float:
        """逐图像平均尺度不变对数误差（乘以 100）。"""
        return self._results.get("metrics/silog", 0.0)

    @property
    def fitness(self) -> float:
        """适应度 = delta1（越高越好）。"""
        return self._results.get("metrics/delta1", 0.0)

    @property
    def results_dict(self) -> dict[str, float]:
        """返回包含适应度的结果字典。"""
        return dict(zip([*self.keys, "fitness"], [*self.mean_results(), self.fitness]))

    @property
    def curves(self) -> list:
        """深度任务没有 PR 曲线。"""
        return []

    @property
    def curves_results(self) -> list:
        """深度任务没有 PR 曲线结果。"""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]:
        """全局深度指标的单行汇总。"""
        return [{k.split("/")[-1]: round(v, decimals) for k, v in self._results.items()}]
