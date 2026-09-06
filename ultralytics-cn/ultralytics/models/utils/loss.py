# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """用于计算各类损失分量的 DETR（DEtection TRansformer）损失类。

    此类计算分类损失、边界框损失和 GIoU 损失，并可为 DETR 对象检测模型计算辅助损失。

    属性：
        nc (int): 类别数量.
        loss_gain (dict[str, float]): 不同损失分量的权重系数。
        aux_loss (bool): 是否计算辅助损失。
        use_fl (bool): 是否使用 FocalLoss。
        use_vfl (bool): 是否使用 VarifocalLoss。
        use_uni_match (bool): 是否为辅助分支标签分配使用固定层。
        uni_match_ind (int): use_uni_match 为 True 时使用的固定层索引。
        matcher (HungarianMatcher): 用于计算匹配代价和索引的对象。
        fl (FocalLoss | None): use_fl 为 True 时使用的 Focal Loss 对象，否则为 None。
        vfl (VarifocalLoss | None): use_vfl 为 True 时使用的 Varifocal Loss 对象，否则为 None。
        device (torch.device): 保存张量的设备。
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """使用可自定义的损失分量和权重初始化 DETR 损失函数。

        未提供 loss_gain 时使用默认值，使用预设代价权重初始化 HungarianMatcher，并支持辅助损失和多种损失类型。

        参数：
            nc (int): 类别数量.
            loss_gain (dict[str, float], 可选): 不同损失分量的权重系数。
            aux_loss (bool): 是否使用每个解码器层的辅助损失。
            use_fl (bool): 是否使用 FocalLoss。
            use_vfl (bool): 是否使用 VarifocalLoss。
            use_uni_match (bool): 是否为辅助分支标签分配使用固定层。
            uni_match_ind (int): uni_match 使用的固定层索引。
            gamma (float): 聚焦参数，控制损失对难分类样本的关注程度。
            alpha (float): 用于处理类别不平衡的平衡因子。
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """根据预测结果、目标值和真实分数计算分类损失。

        参数：
            pred_scores (torch.Tensor): 预测类别分数，形状为 (B, N, C)。
            targets (torch.Tensor): 目标类别索引，形状为 (B, N)。
            gt_scores (torch.Tensor): 真实置信度分数，形状为 (B, N)。
            num_gts (int): 真实对象数量。
            postfix (str, 可选): 追加到损失名称后的字符串，用于区分多损失场景。

        返回：
            (dict[str, torch.Tensor]): 包含分类损失值的字典。

        注意：
            此函数支持以下分类损失类型：
            - Varifocal Loss（self.vfl 不为 None 且 num_gts > 0 时）
            - Focal Loss（self.fl 不为 None 时）
            - BCE Loss（默认回退方案）
        """
        # Logits： [b, query, num_classes]，gt_class： 列表[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS 损失

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """计算预测边界框和真实边界框之间的边界框损失与 GIoU 损失。

        参数：
            pred_bboxes (torch.Tensor): 预测边界框，形状为 (N, 4)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (N, 4)。
            postfix (str, 可选): 追加到损失名称后的字符串，用于区分多损失场景。

        返回：
            (dict[str, torch.Tensor]): 包含以下内容的字典：
                - loss_bbox{postfix}：预测边界框与真实边界框之间的 L1 损失，并乘以 bbox 损失权重。
                - loss_giou{postfix}：预测边界框与真实边界框之间的 GIoU 损失，并乘以 giou 损失权重。

        注意：
            未提供真实边界框（空列表）时，两个损失都返回零值张量。
        """
        # Boxes: [b, query, 4], gt_bbox: 列表[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # 此函数用于未来的 RT-DETR Segment 模型
    # def _get_loss_mask(self, 掩码, gt_mask, match_indices, postfix=''):
    #     # 掩码: [b, query, h, w], gt_mask: 列表[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     如果 sum(len(a) for a in gt_mask) == 0：
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         返回 损失
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO：torch 尚未提供 `sigmoid_focal_loss`；由于当前未使用掩码分支，因此暂不紧急。
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.张量([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     返回 损失

    # 此函数用于未来的 RT-DETR Segment 模型
    # @staticmethod
    # def _dice_loss(inputs, 目标, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     返回 损失.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """获取中间解码器层的辅助损失。

        参数：
            pred_bboxes (torch.Tensor): 辅助层的预测边界框。
            pred_scores (torch.Tensor): 辅助层的预测分数。
            gt_bboxes (torch.Tensor): 真实边界框。
            gt_cls (torch.Tensor): 真实类别。
            gt_groups (列表[int]): 每张图像中的真实对象数量。
            match_indices (列表[tuple], 可选): 预先计算的匹配索引。
            masks (torch.Tensor, 可选): 使用分割任务时的预测掩码。
            gt_mask (torch.Tensor, 可选): 使用分割任务时的真实掩码。

        返回：
            (dict[str, torch.Tensor]): 辅助损失字典。
        """
        # 注意：损失包括 class、bbox、giou、mask 和 dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if 掩码 is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     损失[3] += loss_[f'loss_mask{postfix}']
            #     损失[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if 掩码 is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """从匹配索引中提取批次索引、源索引和目标索引。

        参数：
            match_indices (列表[tuple]): List of tuples containing matched 索引.

        返回：
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination 索引.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据匹配索引将预测边界框分配给真实边界框。

        参数：
            pred_bboxes (torch.Tensor): Predicted bounding 边界框.
            gt_bboxes (torch.Tensor): Ground truth bounding 边界框.
            match_indices (列表[tuple]): List of tuples containing matched 索引.

        返回：
            pred_assigned (torch.Tensor): Assigned predicted bounding 边界框.
            gt_assigned (torch.Tensor): Assigned ground truth bounding 边界框.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算单个预测层的损失。

        参数：
            pred_bboxes (torch.Tensor): 预测边界框。
            pred_scores (torch.Tensor): 预测类别分数。
            gt_bboxes (torch.Tensor): 真实边界框。
            gt_cls (torch.Tensor): 真实类别。
            gt_groups (列表[int]): 每张图像中的真实对象数量。
            masks (torch.Tensor, 可选): 使用分割任务时的预测掩码。
            gt_mask (torch.Tensor, 可选): 使用分割任务时的真实掩码。
            postfix (str, 可选): 追加到损失名称后的字符串。
            match_indices (列表[tuple], 可选): 预先计算的匹配索引。

        返回：
            (dict[str, torch.Tensor]): 损失字典。
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(掩码, gt_mask, match_indices, postfix) if 掩码 is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """计算预测边界框和分数的损失。

        参数：
            pred_bboxes (torch.Tensor): 预测边界框，形状为 (L, B, N, 4)。
            pred_scores (torch.Tensor): 预测类别分数，形状为 (L, B, N, C)。
            batch (dict[str, Any]): 包含 cls、bboxes 和 gt_groups 的批次信息。
            postfix (str, 可选): 损失名称后缀。
            **kwargs (Any): 其他参数，可能包含 'match_indices'。

        返回：
            (dict[str, torch.Tensor]): 计算得到的损失，包括主损失和辅助损失（如果启用）。

        注意：
            使用 pred_bboxes 和 pred_scores 的最后一个元素计算主损失；self.aux_loss 为 True 时，
            使用其余元素计算辅助损失。
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """继承 DETRLoss 的实时检测 Transformer（RT-DETR）检测损失类。

    此类计算 RT-DETR 模型的检测损失，包括标准检测损失；提供去噪元数据时，还会计算额外的去噪训练损失。
    """

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """执行前向计算，得到检测损失和可选的去噪损失。

        参数：
            preds (tuple[torch.Tensor, torch.Tensor]): 包含预测边界框和分数的元组。
            batch (dict[str, Any]): 包含真实标注信息的批次数据。
            dn_bboxes (torch.Tensor, 可选): 去噪边界框。
            dn_scores (torch.Tensor, 可选): 去噪分数。
            dn_meta (dict[str, Any], 可选): 去噪元数据。

        返回：
            (dict[str, torch.Tensor]): 包含总损失以及适用时的去噪损失的字典。
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # 检查去噪元数据，以计算去噪训练损失
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # 获取去噪匹配索引
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, dn_meta["dn_gt_idx"])

            # 计算去噪训练损失
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # 未提供去噪元数据时，将去噪损失设为零
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, dn_gt_idx: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """获取去噪匹配索引。

        参数：
            dn_pos_idx (列表[torch.Tensor]): 包含去噪正样本索引的张量列表。
            dn_num_group (int): 去噪组数量。
            dn_gt_idx (列表[torch.Tensor]): 每张图像中用于重建去噪查询的真实标注索引。

        返回：
            (列表[tuple[torch.Tensor, torch.Tensor]]): 包含去噪匹配索引的元组列表。
        """
        dn_match_indices = []
        for i, gt_idx in enumerate(dn_gt_idx):
            gt_idx = gt_idx.repeat(dn_num_group)
            assert len(dn_pos_idx[i]) == len(gt_idx), (
                f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
            )
            dn_match_indices.append((dn_pos_idx[i], gt_idx))
        return dn_match_indices
