# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Zhang 等人提出的 Varifocal Loss。

    该损失函数通过关注难分类样本并平衡正负样本，解决目标检测中的类别不平衡问题。

    属性：
        gamma (float): 控制损失关注难分类样本程度的聚焦参数。
        alpha (float): 用于解决类别不平衡的平衡因子。

    参考：
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """使用聚焦参数和平衡参数初始化 VarifocalLoss。"""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """计算预测结果与真实标签之间的 varifocal 损失。"""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False, device=pred_score.device.type):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """在现有 loss_fcn() 外封装 Focal Loss，例如 criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)。

    该损失函数通过降低简单样本的权重并关注训练期间的困难负样本，解决类别不平衡问题。

    属性：
        gamma (float): 控制损失关注难分类样本程度的聚焦参数。
        alpha (torch.Tensor): 用于解决类别不平衡的平衡因子。
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """使用聚焦参数和平衡参数初始化 FocalLoss。"""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """计算带调制因子的 focal 损失，以解决类别不平衡问题。"""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # 损失 *= self.alpha * (1.000001 - p_t) ** self.gamma  # 使用非零幂以保持梯度稳定

        # TF 实现：https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # 将 logits 转换为概率
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """用于计算分布焦点损失（DFL）的损失准则。"""

    def __init__(self, reg_max: int = 16) -> None:
        """使用回归最大值初始化 DFL 模块。"""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """返回左侧和右侧 DFL 损失的加权和，方法参见 https://arxiv.org/abs/2006.04388。"""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # 目标 left
        tr = tl + 1  # 目标 right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # 只计算一次 log_softmax，然后执行两次 gather；cross_entropy(x, t) = -log_softmax(x).gather(t)
        logp = F.log_softmax(pred_dist, dim=1)
        return -(
            logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """用于计算目标检测训练中边界框损失的损失准则。"""

    def __init__(self, reg_max: int = 16):
        """使用回归最大值和 DFL 设置初始化 BboxLoss 模块。"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算边界框的 IoU 损失和 DFL 损失。"""
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL 损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # 根据图像尺寸归一化 ltrb
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """残差对数似然估计损失。

    属性：
        size_average (bool): 是否按批次大小对损失求平均。
        use_target_weight (bool): 是否使用加权损失。
        residual (bool): 是否加入 L1 损失，让流模型学习残差误差分布。

    参考：
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """使用目标权重和残差选项初始化 RLELoss。

        参数：
            use_target_weight (bool): 是否使用目标权重计算损失。
            size_average (bool): 是否对各元素的损失求平均。
            residual (bool): 是否包含残差对数似然项。
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        参数：
            sigma (torch.Tensor): 输出 sigma，形状为 (N, D)。
            log_phi (torch.Tensor): 输出 log_phi，形状为 (N)。
            error (torch.Tensor): 误差，形状为 (N, D)。
            target_weight (torch.Tensor): 不同关节类型的权重，形状为 (N)。
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """用于计算旋转边界框训练损失的损失准则。"""

    floor = 0.01

    def __init__(self, reg_max: int):
        """使用回归最大值和 DFL 设置初始化 RotatedBboxLoss 模块。"""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算旋转边界框的 IoU 损失和 DFL 损失。"""
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], floor=self.floor)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL 损失
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """用于计算多通道 Dice 损失的损失准则。"""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """使用平滑和归约选项初始化 MultiChannelDiceLoss。

        参数：
            smooth (float): 用于避免除零的平滑因子。
            reduction (str): 归约方式（'mean'、'sum' 或 'none'）。
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算预测结果与目标之间的多通道 Dice 损失。"""
        assert pred.size() == target.size(), "预测结果和目标的尺寸必须相同。"

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """用于计算 BCE 与 Dice 组合损失的损失准则。"""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """使用 BCE 和 Dice 的权重因子初始化 BCEDiceLoss。

        参数：
            weight_bce (float): BCE 损失分量的权重因子。
            weight_dice (float): Dice 损失分量的权重因子。
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算预测结果与目标之间的 BCE 和 Dice 组合损失。"""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # 下采样到与 pred 相同的尺寸
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """用于计算关键点损失的损失准则。"""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """使用关键点 sigma 初始化 KeypointLoss。"""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """计算关键点损失因子和关键点的欧氏距离损失。"""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # 根据公式计算
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # 来自 cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """用于计算 YOLOv8 目标检测训练损失的损失准则。"""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # 模型必须解除并行封装
        """使用模型参数和任务对齐分配设置初始化 v8DetectionLoss。"""
        device = next(model.parameters()).device  # 获取模型设备
        h = model.args  # 超参数

        m = model.model[-1]  # Detect() 模块
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # 模型步幅
        self.nc = m.nc  # 类别数量
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"

        # 用于处理不平衡数据集的类别权重
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """将目标转换为张量格式并缩放坐标。"""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # 图像 索引
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """根据锚框点和分布解码预测的目标边界框坐标。"""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, 锚框, 通道
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """计算边界框、cls 和 dfl 损失之和并乘以批次大小，同时返回前景掩码和目标索引。
        """
        loss = torch.zeros(3, device=self.device)  # 边界框, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # 目标
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 预测边界框
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 分类损失，可选择使用类别权重
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox 损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # 边界框 gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # 损失(边界框, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """解析模型预测结果并提取特征。"""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算边界框、cls 和 dfl 损失之和，并乘以批次大小。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """使用已分配的目标计算检测损失。"""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """用于计算 YOLOv8 分割训练损失的损失准则。"""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # 模型必须解除并行封装
        """使用模型参数和掩码重叠设置初始化 v8SegmentationLoss。"""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "seg_loss", *self.loss_names[1:], "sem_loss")
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算并返回检测和分割的组合损失。"""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # 边界框, seg, cls, dfl, semantic
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        else:
            pred_semantic = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # 注意：当前暂时重新分配索引以保持一致，今后需要移除。
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # 批次大小、掩码数量、掩码高度、掩码宽度
        if fg_mask.sum():
            # Masks 损失
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # 下采样
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semantic is not None:
                sem_idx = batch["sem_masks"].to(self.device).long().unsqueeze(1)  # Nx1xHxW
                if self.overlap:
                    present = masks != 0  # NxHxW
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    present = torch.zeros(batch_size, *masks.shape[-2:], dtype=torch.bool, device=self.device)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i):
                            present[i] = instance_mask_i.sum(dim=0) != 0
                # 未覆盖像素处的 one-hot 目标置零，避免 F.one_hot 产生 int64 类型的 NxHxWxC 中间张量
                sem_masks = torch.zeros(sem_idx.shape[0], self.nc, *sem_idx.shape[2:], device=self.device)
                sem_masks.scatter_(1, sem_idx, present.unsqueeze(1).float())  # NxCxHxW

                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box  # 分割损失增益

        # 警告：下面的代码可防止多 GPU DDP 出现 PyTorch “未使用梯度”错误，请勿删除
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # 无穷大求和可能导致损失变为 nan
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        loss[1] *= self.hyp.box  # 分割损失增益
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # 损失(边界框, seg, cls, dfl, semantic)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """计算单张图像的实例分割损失。

        参数：
            gt_mask (torch.Tensor): 真实掩码，形状为 (N, H, W)，其中 N 为目标数量。
            pred (torch.Tensor): 预测掩码系数，形状为 (N, 32)。
            proto (torch.Tensor): 掩码原型，形状为 (32, H, W)。
            xyxy (torch.Tensor): xyxy 格式的真实边界框，已归一化到 [0, 1]，形状为 (N, 4)。
            area (torch.Tensor): 每个真实边界框的面积，形状为 (N,)。

        返回：
            (torch.Tensor): 单张图像计算得到的掩码损失。

        注意：
            该函数使用公式 pred_mask = torch.einsum('in,nhw->ihw', pred, proto)，
            根据掩码原型和预测掩码系数生成预测掩码。
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """计算实例分割损失。

        参数：
            fg_mask (torch.Tensor): 二值张量，形状为 (BS, N_anchors)，表示哪些锚框为正样本。
            masks (torch.Tensor): 真实掩码；当 `overlap` 为 True 时形状为 (BS, H, W)，否则为
                (N_instances_in_batch, H, W)。
            target_gt_idx (torch.Tensor): 每个锚框对应的真实目标索引，形状为 (BS, N_anchors)。
            target_bboxes (torch.Tensor): 每个锚框对应的真实边界框，形状为 (BS, N_anchors, 4)。
            batch_idx (torch.Tensor): 批次索引，形状为 (N_labels_in_batch, 1)。
            proto (torch.Tensor): 掩码原型，形状为 (BS, 32, H, W)。
            pred_masks (torch.Tensor): 每个锚框的预测掩码，形状为 (BS, N_anchors, 32)。
            imgsz (torch.Tensor): 输入图像尺寸张量，形状为 (2)，即 (H, W)。

        返回：
            (torch.Tensor): 计算得到的实例分割损失。

        注意：
            在允许更高内存占用的情况下，可以按批次计算损失以提高速度。
            例如，pred_mask 可以按如下方式计算：
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # 归一化到 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # 目标边界框的面积
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # 归一化到掩码尺寸
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # 警告：下面的代码可防止多 GPU DDP 出现 PyTorch “未使用梯度”错误，请勿删除
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # 无穷大求和可能导致损失变为 nan

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """用于计算 YOLOv8 姿态估计训练损失的损失准则。"""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10):  # 模型必须解除并行封装
        """使用模型参数和关键点专用损失函数初始化 v8PoseLoss。"""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", *self.loss_names[1:])
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # 关键点数量

        sigmas = getattr(model, "kpt_oks_sigmas", None)
        if sigmas is None:
            sigmas = (
                torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
            )
        else:
            sigmas = torch.as_tensor(sigmas, device=self.device, dtype=torch.float32).flatten()
            if len(sigmas) != nkpt or not torch.all(sigmas > 0):
                raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas.tolist()}")

        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算总损失，并将其分离用于姿态估计。"""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # 边界框, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # 注意：当前暂时重新分配索引以保持一致，今后需要移除。
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # 预测关键点边界框
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # 关键点损失
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # 姿态损失增益
        loss[2] *= self.hyp.kobj  # 关键点目标损失增益

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # 损失(边界框, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """将预测关键点解码为图像坐标。"""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """根据批次索引和目标真实索引，为每个锚框选择目标关键点。

        参数：
            keypoints (torch.Tensor): 真实关键点，形状为 (N_kpts_in_batch, N_kpts_per_object, kpts_dim)。
            batch_idx (torch.Tensor): 关键点的批次索引张量，形状为 (N_kpts_in_batch, 1)。
            target_gt_idx (torch.Tensor): 将锚框映射到真实目标的索引张量，形状为 (BS, N_anchors)。
            masks (torch.Tensor): 表示目标存在性的二值掩码张量，形状为 (BS, N_anchors)。

        返回：
            (torch.Tensor): 选中的关键点张量，形状为 (BS, N_anchors, N_kpts_per_object, kpts_dim)。
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # 查找单张图像中的最大关键点数量
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # 创建用于保存批次关键点的张量
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # 向量化填充：使用累积偏移量计算每个关键点在批次内的位置
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # 扩展 target_gt_idx 的维度，使其与 batched_keypoints 的形状匹配
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # 使用 target_gt_idx_expanded 从 batched_keypoints 中选择关键点
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算模型的关键点损失。

        此函数计算给定批次的关键点损失和关键点目标损失。关键点损失基于预测关键点与真实关键点的差异，
        关键点目标损失是用于判断关键点是否存在的二分类损失。

        参数：
            masks (torch.Tensor): 表示目标存在性的二值掩码张量，形状为 (BS, N_anchors)。
            target_gt_idx (torch.Tensor): 将锚框映射到真实目标的索引张量，形状为 (BS, N_anchors)。
            keypoints (torch.Tensor): 真实关键点，形状为 (N_kpts_in_batch, N_kpts_per_object, kpts_dim)。
            batch_idx (torch.Tensor): 关键点的批次索引张量，形状为 (N_kpts_in_batch, 1)。
            stride_tensor (torch.Tensor): 锚框步幅张量，形状为 (N_anchors, 1)。
            target_bboxes (torch.Tensor): (x1, y1, x2, y2) 格式的真实边界框，形状为 (BS, N_anchors, 4)。
            pred_kpts (torch.Tensor): 预测关键点，形状为 (BS, N_anchors, N_kpts_per_object, kpts_dim)。

        返回：
            kpts_loss (torch.Tensor): 关键点损失。
            kpts_obj_loss (torch.Tensor): 关键点目标损失。
        """
        # 使用辅助方法选择目标关键点
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose 损失

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # 关键点目标损失

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """支持 RLE 损失、用于计算 YOLO26 姿态估计训练损失的损失准则。"""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # 模型必须解除并行封装
        """使用模型参数和关键点专用损失函数（包括 RLE 损失）初始化 PoseLoss26。"""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # 关键点数量
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.loss_names += ("rle_loss",)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算总损失，并将其分离用于姿态估计。"""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # 边界框, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # 注意：当前暂时重新分配索引以保持一致，今后需要移除。
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # 关键点损失
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # 姿态损失增益
        loss[2] *= self.hyp.kobj  # 关键点目标损失增益
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # RLE 损失增益

        # 损失(边界框, kpt_location, kpt_visibility, cls, dfl[, rle])
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """将预测关键点解码为图像坐标。"""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """计算关键点的 RLE（残差对数似然估计）损失。

        参数：
            pred_kpt (torch.Tensor): 带 sigma 的预测关键点，形状为 (N, num_keypoints, kpts_dim)，其中 kpts_dim >= 4。
            gt_kpt (torch.Tensor): 真实关键点，形状为 (N, num_keypoints, kpts_dim)。
            kpt_mask (torch.Tensor): 有效关键点掩码，形状为 (N, num_keypoints)。

        返回：
            (torch.Tensor): RLE 损失。
        """
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()

        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        if not error.numel():
            return pred_kpt[..., :0].sum()

        # 过滤 NaN 和 Inf 值，避免它们传播到损失中
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # 防止数值不稳定
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算模型的关键点损失。

        此函数计算给定批次的关键点损失和关键点目标损失。关键点损失基于预测关键点与真实关键点之间的差异，
        关键点目标损失是用于判断关键点是否存在的二分类损失。

        参数：
            masks (torch.Tensor): 表示目标存在的二值掩码张量，形状为 (BS, N_anchors)。
            target_gt_idx (torch.Tensor): 将锚框映射到真实目标的索引张量，形状为 (BS, N_anchors)。
            keypoints (torch.Tensor): 真实关键点，形状为 (N_kpts_in_batch, N_kpts_per_object, kpts_dim)。
            batch_idx (torch.Tensor): 关键点的批次索引张量，形状为 (N_kpts_in_batch, 1)。
            stride_tensor (torch.Tensor): 锚框步长张量，形状为 (N_anchors, 1)。
            target_bboxes (torch.Tensor): (x1, y1, x2, y2) 格式的真实边界框，形状为 (BS, N_anchors, 4)。
            pred_kpts (torch.Tensor): 预测关键点，形状为 (BS, N_anchors, N_kpts_per_object, kpts_dim)。

        返回：
            kpts_loss (torch.Tensor): 关键点损失。
            kpts_obj_loss (torch.Tensor): 关键点目标损失。
            rle_loss (torch.Tensor): RLE 损失。
        """
        # 使用继承的辅助方法选择目标关键点
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose 损失

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj 损失

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """用于计算分类训练损失的损失准则。"""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算预测结果与真实标签之间的分类损失。"""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, {"loss": loss.detach()}


class v8OBBLoss(v8DetectionLoss):
    """计算旋转 YOLO 模型中的目标检测、分类和边界框分布损失。"""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """使用模型、分配器和旋转边界框损失初始化 v8OBBLoss；模型必须已解除并行封装。"""
        super().__init__(model, tal_topk=tal_topk)
        self.loss_names = (*self.loss_names, "angle_loss")
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """为有向边界框检测预处理目标。"""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # 图像 索引
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算并返回有向边界框检测损失。"""
        loss = torch.zeros(4, device=self.device)  # 边界框, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # 批次大小

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # 目标
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # 过滤尺寸过小的旋转框，使训练更稳定
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb for help."
            ) from e

        # 预测边界框
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # 只需缩放前四个元素
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls 损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # Bbox 损失
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores[fg_mask].sum(-1)
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle 损失
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # 边界框损失增益
        loss[1] *= self.hyp.cls  # 分类损失增益
        loss[2] *= self.hyp.dfl  # DFL 损失增益
        loss[3] *= self.hyp.angle  # 角度损失增益

        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # 损失(边界框, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """根据锚框点和分布解码预测的目标边界框坐标。

        参数：
            anchor_points (torch.Tensor): 锚框点，形状为 (h*w, 2)。
            pred_dist (torch.Tensor): 预测的旋转距离，形状为 (bs, h*w, 4)。
            pred_angle (torch.Tensor): 预测角度，形状为 (bs, h*w, 1)。

        返回：
            (torch.Tensor): 带角度的预测旋转边界框，形状为 (bs, h*w, 5)。
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, 锚框, 通道
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """计算有向边界框的角度损失。

        参数：
            pred_bboxes (torch.Tensor): 预测边界框，形状为 [N, 5]（x、y、w、h、theta）。
            target_bboxes (torch.Tensor): 目标边界框，形状为 [N, 5]（x、y、w、h、theta）。
            fg_mask (torch.Tensor): 表示有效预测结果的前景掩码。
            weight (torch.Tensor): 每个预测结果的损失权重。
            target_scores_sum (torch.Tensor): 用于归一化的目标分数之和。
            lambda_val (int): 控制对宽高比的敏感度。

        返回：
            (torch.Tensor): 计算得到的角度损失。
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class DepthLoss26:
    """用于计算 YOLO 深度估计训练损失的损失准则。

    遵循 Depth Anything 方法，使用尺度不变对数损失（SILog）和梯度匹配损失。
    SILog 处理尺度歧义，梯度损失保留边缘。
    """

    def __init__(self, model: torch.nn.Module):
        """初始化 DepthLoss26."""
        device = next(model.parameters()).device
        self.device = device
        h = model.args  # 超参数
        self.silog_weight = h.dlog
        self.grad_weight = h.dgrad
        self.silog_lambda = h.dlam  # 1.0 表示尺度不变，0.0 表示 log-RMSE
        self.grad_scales = 4
        self.loss_names = "dlog_loss", "dgrad_loss"

    @staticmethod
    def _grad_l1(pred_log: torch.Tensor, gt_log: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor:
        """计算预测值与真实值 log-depth 空间梯度（dx、dy）之间的 L1 损失，并使用有效掩码进行筛选。

        只有在参与计算的两个像素都有效时才保留梯度，因此只在真实值有定义的位置匹配边缘。
        """
        pred_dx = (pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
        gt_dx = (gt_log[:, :, :, 1:] - gt_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
        pred_dy = (pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
        gt_dy = (gt_log[:, :, 1:, :] - gt_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
        return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

    def __call__(
        self, preds: dict[str, torch.Tensor] | torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算 depth estimation 损失.

        参数：
            preds (dict | torch.Tensor): 包含 "depth" 键的字典，或形状为 (B, 1, H, W) 的原始预测深度张量。
            batch (dict): Dict with "depth" key holding (B, H, W) ground truth depth in meters.

        返回：
            loss_sum (torch.Tensor): Total 损失 scaled by 批次大小.
            loss_items (dict[str, torch.Tensor]): Detached silog/gradient 损失 keyed by loss_names.
        """
        loss = torch.zeros(2, device=self.device)
        pred_depth = preds["depth"] if isinstance(preds, dict) else preds
        gt_depth = batch["depth"].to(self.device)

        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)

        if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)

        valid = gt_depth > 0.001
        if valid.sum() < 10:
            # 保持结果与计算图连接，使 BaseTrainer 的无条件 backward() 能够正常工作。
            return pred_depth.sum() * 0.0, dict(zip(self.loss_names, loss.detach()))

        pred_valid = pred_depth[valid]
        gt_valid = gt_depth[valid]

        pred_valid = pred_valid.clamp(min=0.001)

        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        # 使用中心化方差形式：按构造保证非负，并在接近收敛时对 fp16 更稳定。
        m = log_diff.mean()
        silog = torch.sqrt(((log_diff - m) ** 2).mean() + (1.0 - self.silog_lambda) * m**2 + 1e-6)
        loss[0] = silog * self.silog_weight

        # 多尺度梯度匹配损失。
        pred_log = torch.log(pred_depth.clamp(min=0.001))
        gt_log = torch.log(gt_depth.clamp(min=0.001))
        valid_f = valid.float()
        grad_loss = self._grad_l1(pred_log, gt_log, valid_f)
        for _ in range(1, max(self.grad_scales, 1)):
            if pred_log.shape[-1] < 4 or pred_log.shape[-2] < 4:
                break
            vp = F.avg_pool2d(valid_f, 2)
            occupied = vp > 0
            # 当图像中已占用的网格大部分已填满时继续处理：连续填充适合这种情况，而 LiDAR 稀疏散点不适合。
            keep = (vp.sum(dim=(1, 2, 3)) > 0.7 * occupied.sum(dim=(1, 2, 3))).view(-1, 1, 1, 1)
            if not keep.any():
                break
            denom = vp.clamp(min=1e-6)
            pred_log = F.avg_pool2d(pred_log * valid_f, 2) / denom
            gt_log = F.avg_pool2d(gt_log * valid_f, 2) / denom
            valid_f = occupied.float() * keep  # zeroed 图像 cannot re-enter deeper levels
            grad_loss = grad_loss + self._grad_l1(pred_log, gt_log, valid_f)
        loss[1] = grad_loss * self.grad_weight

        return loss * pred_depth.shape[0], dict(zip(self.loss_names, loss.detach()))


class E2EDetectLoss:
    """计算端到端检测训练损失的准则类。"""

    def __init__(self, model: torch.nn.Module):
        """使用给定模型初始化 E2EDetectLoss，并配置一对多和一对一检测损失。"""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算边界框、分类和 DFL 损失之和，并乘以批次大小。"""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], {
            k: loss_one2many[1][k] + loss_one2one[1][k] for k in loss_one2many[1]
        }


class E2ELoss:
    """计算端到端检测训练损失的准则类。"""

    def __init__(self, model: torch.nn.Module, loss_fn=v8DetectionLoss):
        """使用给定模型初始化 E2ELoss，并配置一对多和一对一检测损失。"""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # 初始化增益
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # 最终增益
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算边界框、分类和 DFL 损失之和，并乘以批次大小。"""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """根据衰减计划更新一对多损失和一对一损失的权重。"""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """根据当前更新步数计算一对多损失的衰减权重。"""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """计算文本-视觉提示检测训练损失的准则类。"""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """使用给定模型初始化 TVPDetectLoss，并配置任务提示和视觉提示准则。"""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names)  # 去除 "_loss" 后缀
        # 注意：以下信息会在 __call__ 中变化，因此需要保存
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """解析模型预测结果并提取特征。"""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算文本-视觉提示检测损失。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算文本-视觉提示检测损失。"""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, dict(zip(self.loss_names, loss.detach()))

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        return vp_loss[0][1], dict(zip(self.loss_names, vp_loss[1].values()))

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """从模型输出中提取视觉提示特征。"""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """计算文本-视觉提示分割训练损失的准则类。"""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """使用给定模型初始化 TVPSegmentLoss，并配置任务提示和视觉提示准则。"""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names if k != "sem_loss")  # 去除 "_loss" 后缀
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算文本-视觉提示分割损失。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算文本-视觉提示分割损失。"""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, dict(zip(self.loss_names, loss.detach()))

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        # zip 会丢弃末尾的 "sem_loss" 项，以匹配日志中的列
        return cls_loss, dict(zip(self.loss_names, vp_loss[1].values()))


class SemanticSegmentationLoss(nn.Module):
    """使用交叉熵项和 Dice 项计算语义分割损失。

    属性：
        nc (int): 语义类别数量。
        ce (nn.CrossEntropyLoss): ignore_index=255 的交叉熵损失。
    """

    def __init__(self, model: torch.nn.Module):
        """初始化语义分割损失。

        参数：
            model (torch.nn.Module): 包含 SemanticSegment 头部的模型。
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        weight = getattr(model, "class_weights", None)  # cls_pw 频率权重；否则使用硬编码的 Cityscapes 权重
        if weight is None and self.use_cityscapes_weight:
            weight = torch.from_numpy(CITYSCAPES_WEIGHT)
        weight = None if weight is None else weight.to(device=self.device, dtype=self.dtype)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction="sum")  # 二分类：不支持类别加权
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="sum").to(device=self.device, dtype=self.dtype)
            if weight is not None:
                # 非持久化：weight 是确定性常量，无需序列化到 ckpt 的 state_dict 中。
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """调整掩码尺寸，使其匹配预测结果的空间维度。"""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks, valid):
        """在展平的像素上计算交叉熵，以避免使用 CUDA nll_loss2d 路径。"""
        flat = masks.reshape(-1)
        if self.nc == 1:
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
            denominator = valid.sum()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = flat.long()
            denominator = valid.sum() if self.ce.weight is None else self.ce.weight[target[valid]].sum()
        return self.ce(logits, target) / denominator.clamp_min(1)

    def _dice_loss(self, preds, masks, valid):
        """计算 Dice 损失 excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks, valid)
        flat_target = masks.reshape(-1)
        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks, valid):
        """计算单类别（二分类）分割的 Dice 损失。

        值为 255 的像素会从 Dice 项中排除，以匹配 BCE 的有效像素筛选规则。
        """
        valid = valid.reshape_as(masks).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """计算语义分割损失，并支持可选的辅助损失。

        参数：
            preds (torch.Tensor | tuple): 主 logits [B, nc, H', W']，或 (main, aux) 元组。
            batch (dict): 包含 'semantic_mask' [B, H, W] 的批次字典，其中保存类别 ID（255 表示忽略）。

        返回：
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): 总损失乘以 batch_size，以及包含已分离损失项的字典
                (ce_loss、dice_loss、aux_loss)。
        """
        # 如果存在辅助 logits，则将其解包。
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        valid = masks.reshape(-1) != 255
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # 主交叉熵和 Dice 损失。
        ce_loss = self._ce_loss(preds, masks, valid)
        dice_loss = self._dice_loss(preds, masks, valid)
        total = ce_loss + dice_loss

        # 辅助交叉熵损失。匹配 ce_loss 的数据类型，确保 AMP 下可以加到总损失中。
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks, valid) * 0.4
            total += aux_loss

        loss_items = {"ce_loss": ce_loss.detach(), "dice_loss": dice_loss.detach(), "aux_loss": aux_loss.detach()}
        return total * preds.shape[0], loss_items
