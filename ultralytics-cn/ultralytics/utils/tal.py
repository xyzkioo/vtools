# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """用于目标检测的任务对齐分配器。

    此类根据结合分类和定位信息的任务对齐指标，将真实目标（gt）分配给锚框。

    属性：
        topk (int): 要考虑的候选项数量。
        topk2 (int): 用于额外筛选的第二个 topk 值。
        num_classes (int): 目标类别数量。
        alpha (float): 任务对齐指标中分类部分的 alpha 参数。
        beta (float): 任务对齐指标中定位部分的 beta 参数。
        stride (列表): 不同特征层级的步幅列表。
        stride_val (int): `select_candidates_in_gts` 使用的步幅值。
        eps (float): 防止除零的小数值。
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
    ):
        """使用可自定义超参数初始化 TaskAlignedAssigner 对象。

        参数：
            topk (int, 可选): 要考虑的候选项数量。
            num_classes (int, 可选): 目标类别数量。
            alpha (float, 可选): 任务对齐指标中分类部分的 alpha 参数。
            beta (float, 可选): 任务对齐指标中定位部分的 beta 参数。
            stride (列表, 可选): 不同特征层级的步幅列表。
            eps (float, 可选): 防止除零的小数值。
            topk2 (int, 可选): 用于额外筛选的第二个 topk 值。
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """计算任务对齐分配结果。

        参数：
            pd_scores (torch.Tensor): 预测分类分数，形状为 (bs, num_total_anchors, num_classes)。
            pd_bboxes (torch.Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)。
            anc_points (torch.Tensor): 锚框点，形状为 (num_total_anchors, 2)。
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (bs, n_max_boxes, 1)。

        返回：
            target_labels (torch.Tensor): 目标标签，形状为 (bs, num_total_anchors)。
            target_bboxes (torch.Tensor): 目标边界框，形状为 (bs, num_total_anchors, 4)。
            target_scores (torch.Tensor): 目标分数，形状为 (bs, num_total_anchors, num_classes)。
            fg_mask (torch.Tensor): 前景掩码，形状为 (bs, num_total_anchors)。
            target_gt_idx (torch.Tensor): 目标真实标签索引，形状为 (bs, num_total_anchors)。

        参考：
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # 在 except 块外恢复：退出该块会释放 e.__traceback__，让失败尝试的 GPU 中间结果返回分配器，
        # 从而保证下方的数据复制可以成功
        LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
        result = self._forward(*(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)))
        return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """计算任务对齐分配结果。

        参数：
            pd_scores (torch.Tensor): 预测分类分数，形状为 (bs, num_total_anchors, num_classes)。
            pd_bboxes (torch.Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)。
            anc_points (torch.Tensor): 锚框点，形状为 (num_total_anchors, 2)。
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (bs, n_max_boxes, 1)。

        返回：
            target_labels (torch.Tensor): 目标标签，形状为 (bs, num_total_anchors)。
            target_bboxes (torch.Tensor): 目标边界框，形状为 (bs, num_total_anchors, 4)。
            target_scores (torch.Tensor): 目标分数，形状为 (bs, num_total_anchors, num_classes)。
            fg_mask (torch.Tensor): 前景掩码，形状为 (bs, num_total_anchors)。
            target_gt_idx (torch.Tensor): 目标真实标签索引，形状为 (bs, num_total_anchors)。
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # 分配目标
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 归一化
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """获取每个真实边界框对应的正样本掩码。

        参数：
            pd_scores (torch.Tensor): 预测分类分数，形状为 (bs, num_total_anchors, num_classes)。
            pd_bboxes (torch.Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)。
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)。
            anc_points (torch.Tensor): 锚框点，形状为 (num_total_anchors, 2)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (bs, n_max_boxes, 1)。

        返回：
            mask_pos (torch.Tensor): 正样本掩码，形状为 (bs, max_num_obj, h*w)。
            align_metric (torch.Tensor): 对齐指标，形状为 (bs, max_num_obj, h*w)。
            overlaps (torch.Tensor): 预测边界框与真实边界框之间的重叠度，形状为 (bs, max_num_obj, h*w)。
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # 获取锚框对齐指标，形状为 (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 获取 top-k 指标掩码，形状为 (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # 合并所有掩码，得到最终掩码，形状为 (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """根据预测边界框和真实边界框计算对齐指标。

        参数：
            pd_scores (torch.Tensor): 预测分类分数，形状为 (bs, num_total_anchors, num_classes)。
            pd_bboxes (torch.Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)。
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (bs, n_max_boxes, h*w)。

        返回：
            align_metric (torch.Tensor): 结合分类和定位信息的对齐指标。
            overlaps (torch.Tensor): 预测边界框与真实边界框之间的 IoU 重叠度。
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]  # b, 1
        # 获取每个网格点对应每个真实类别的分数
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算水平边界框的 IoU。

        参数：
            gt_bboxes (torch.Tensor): 真实边界框。
            pd_bboxes (torch.Tensor): 预测边界框。

        返回：
            (torch.Tensor): 每对边界框之间的 IoU 值。
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """根据给定指标选择前 k 个候选项。

        参数：
            metrics (torch.Tensor): 形状为 (b, max_num_obj, h*w) 的指标张量，其中 b 为批次大小，max_num_obj 为最大对象数量，h*w 为锚框点总数。
            topk_mask (torch.Tensor, 可选): 形状为 (b, max_num_obj, topk) 的可选布尔张量；未提供时根据给定指标自动计算 top-k 值。

        返回：
            (torch.Tensor): 包含所选 top-k 候选项的张量，形状为 (b, max_num_obj, h*w)。
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # 统计 top-k 列表为每个锚框选择的次数；scatter_add_ 一次性累加重复索引
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        # 过滤无效边界框
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """为正样本锚框点计算目标标签、目标边界框和目标分数。

        参数：
            gt_labels (torch.Tensor): 真实标签，形状为 (b, max_num_obj, 1)，其中 b 为批次大小，max_num_obj 为最大目标数量。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (b, max_num_obj, 4)。
            target_gt_idx (torch.Tensor): 正样本锚框点对应的真实目标索引，形状为 (b, h*w)，其中 h*w 为锚框点总数。
            fg_mask (torch.Tensor): 布尔张量，形状为 (b, h*w)，指示正样本（前景）锚框点。

        返回：
            target_labels (torch.Tensor): 正样本锚框点的目标标签，形状为 (b, h*w)。
            target_bboxes (torch.Tensor): 正样本锚框点的目标边界框，形状为 (b, h*w, 4)。
            target_scores (torch.Tensor): 正样本锚框点的目标分数，形状为 (b, h*w, num_classes)。
        """
        # Assigned 目标 标签, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned 目标 边界框, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # 已分配目标的分数
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        target_scores = target_scores * (fg_mask[:, :, None] > 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """选择位于真实边界框内的正样本锚框中心。

        参数：
            xy_centers (torch.Tensor): 锚框中心坐标，形状为 (h*w, 2)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (b, n_boxes, 4)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (b, n_boxes, 1)。
            eps (float, 可选): 用于保证数值稳定性的小值。

        返回：
            (torch.Tensor): 正样本锚框的布尔掩码，形状为 (b, n_boxes, h*w)。

        注意：
            - b：批次大小；n_boxes：真实边界框数量；h：高度；w：宽度。
            - 边界框格式：[x_min, y_min, x_max, y_max]。
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride_val  # floor tiny sides so the pool grows monotonically
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
        return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """当锚框被分配给多个真实目标时，选择 IoU 最高的真实边界框。

        参数：
            mask_pos (torch.Tensor): 正样本掩码，形状为 (b, n_max_boxes, h*w)。
            overlaps (torch.Tensor): IoU 重叠度，形状为 (b, n_max_boxes, h*w)。
            n_max_boxes (int): 真实边界框的最大数量。
            align_metric (torch.Tensor): 用于选择最佳匹配的对齐指标。

        返回：
            target_gt_idx (torch.Tensor): 已分配真实目标的索引，形状为 (b, h*w)。
            fg_mask (torch.Tensor): Foreground 掩码, 形状 (b, h*w).
            mask_pos (torch.Tensor): Updated positive 掩码, 形状 (b, n_max_boxes, h*w).
        """
        # 转换 (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # 一个锚框被分配给多个真实边界框
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # 更新对齐指标
            # (b, n_max_boxes, topk2)
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # 查找每个网格点对应的真实目标索引
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """使用任务对齐指标将真实目标分配给旋转边界框。"""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算旋转边界框的 IoU。"""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """为旋转边界框选择 gt 中的正样本锚框中心。

        参数：
            xy_centers (torch.Tensor): 锚框中心坐标，形状为 (h*w, 2)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (b, n_boxes, 5)。
            mask_gt (torch.Tensor): 有效真实边界框掩码，形状为 (b, n_boxes, 1)。

        返回：
            (torch.Tensor): 正样本锚框布尔掩码，形状为 (b, n_boxes, h*w)。
        """
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride_val
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (b, n_boxes, 5) -> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """根据特征生成锚框。"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i in range(len(feats)):  # 使用 len(feats) 避免遍历 strides 张量产生 TracerWarning
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # arange(out=new_*) 可避免非确定性的 CUDA cumsum，同时在跟踪过程中保留运行时设备继承关系。
        sx = torch.arange(w, out=feats[0].new_full((w,), 0, dtype=dtype)) + grid_cell_offset  # x 方向偏移
        sy = torch.arange(h, out=feats[0].new_full((h,), 0, dtype=dtype)) + grid_cell_offset  # y 方向偏移
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将距离（ltrb）转换为边界框（xywh 或 xyxy）。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh 边界框
    return torch.cat((x1y1, x2y2), dim)  # xyxy 边界框


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """将边界框（xyxy）转换为距离（ltrb）。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """根据锚框点和分布解码预测的旋转边界框坐标。

    参数：
        pred_dist (torch.Tensor): 预测旋转距离，形状为 (bs, h*w, 4)。
        pred_angle (torch.Tensor): 预测角度，形状为 (bs, h*w, 1)。
        anchor_points (torch.Tensor): 锚框点，形状为 (h*w, 2)。
        dim (int, 可选): 执行拆分的维度。

    返回：
        (torch.Tensor): 预测旋转边界框，形状为 (bs, h*w, 4)。
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """将旋转边界框（xywh）转换为距离（ltrb），这是 dist2rbox 的逆变换。

    参数：
        target_bboxes (torch.Tensor): 目标旋转边界框，形状为 (bs, h*w, 4)，格式为 [x, y, w, h]。
        anchor_points (torch.Tensor): 锚框点，形状为 (h*w, 2)。
        target_angle (torch.Tensor): 目标角度，形状为 (bs, h*w, 1)。
        dim (int, 可选): 执行拆分的维度。
        reg_max (int, 可选): 用于截断的最大回归值。

    返回：
        (torch.Tensor): 旋转距离，形状为 (bs, h*w, 4)，格式为 [l, t, r, b]。
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
