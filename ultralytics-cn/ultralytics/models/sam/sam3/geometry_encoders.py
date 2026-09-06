# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
from torch import nn

from ultralytics.nn.modules.utils import _get_clones
from ultralytics.utils.ops import xywh2xyxy


def is_right_padded(mask: torch.Tensor):
    """判断填充掩码的填充是否位于右侧（遵循 PyTorch 约定，1 表示填充值）。"""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """将两个右侧填充的序列拼接为连续的右侧填充序列。

    遵循 PyTorch 约定，张量采用序列优先布局，掩码采用批次优先布局，其中 1 表示填充值。

    参数：
        seq1 (torch.Tensor): 形状为 (seq1_length, batch_size, hidden_size) 的张量。
        mask1 (torch.Tensor): 形状为 (batch_size, seq1_length) 的张量。
        seq2 (torch.Tensor): 形状为 (seq2_length, batch_size, hidden_size) 的张量。
        mask2 (torch.Tensor): 形状为 (batch_size, seq2_length) 的张量。
        return_index (bool): 为 True 时，同时返回 seq2 元素在拼接序列中的索引，可据此取回 seq2 元素。

    返回：
        (tuple): return_index 为 False 时返回 (concatenated_sequence, concatenated_mask)，否则返回
            (concatenated_sequence, concatenated_mask, index)。
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert(is_right_padded(mask1), "Mask is not right padded")
    torch._assert(is_right_padded(mask2), "Mask is not right padded")

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros((max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:seq1_length, :, :] = seq1

    # 此时 seq1 的元素已经位于正确位置
    # 只需移动 seq2 的元素

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(0, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """用于处理几何提示的工具类。

    序列遵循 PyTorch 约定，即序列维在前、批次维在后。各张量形状如下：box_embeddings 为 N_boxes x B x C_box，
    box_mask 为 B x N_boxes；如果没有需要屏蔽的内容，box_mask 可以为 None。
    point_embeddings 为 N_points x B x C_point，point_mask 为 B x N_points；如果没有需要屏蔽的内容，point_mask 可以为 None。
    mask_embeddings 为 N_masks x B x 1 x H_mask x W_mask，mask_mask 为 B x N_masks；如果没有需要屏蔽的内容，mask_mask 可以为 None。

    此类还保存正负标签，这些张量同样采用批次优先布局。如果标签为 None，则默认全部为正标签。
    box_labels 为形状 N_boxes x B 的 long 张量，point_labels 为形状 N_points x B 的 long 张量，
    mask_labels 为形状 N_masks x B 的 long 张量。
    """

    def __init__(self, box_embeddings=None, box_mask=None, box_labels=None):
        """初始化 Prompt 对象。"""
        # 检查是否为空提示
        if box_embeddings is None:
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            return

        # 获取序列长度、批次大小和设备
        box_seq_len = box_embeddings.shape[0]
        bs = box_embeddings.shape[1]
        device = box_embeddings.device

        # 未提供时初始化标签和注意力掩码
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)

        # 维度检查
        assert list(box_embeddings.shape[:2]) == [box_seq_len, bs], (
            f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        )
        assert box_embeddings.shape[-1] == 4, (
            f"Expected box embeddings to have 4 coordinates, got {box_embeddings.shape[-1]}"
        )
        assert list(box_mask.shape) == [bs, box_seq_len], (
            f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        )
        assert list(box_labels.shape) == [box_seq_len, bs], (
            f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        )

        # 设备检查
        assert box_embeddings.device == device, (
            f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        )
        assert box_mask.device == device, f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert box_labels.device == device, f"Expected box labels to be on device {device}, got {box_labels.device}"

        self.box_embeddings = box_embeddings
        self.box_mask = box_mask
        self.box_labels = box_labels

    def append_boxes(self, boxes, labels=None, mask=None):
        """将边界框提示追加到现有提示中。

        参数：
            boxes (torch.Tensor): 形状为 (N_new_boxes, B, 4) 的归一化边界框坐标张量。
            labels (torch.Tensor | None): 形状为 (N_new_boxes, B) 的可选正负标签张量。
            mask (torch.Tensor | None): 形状为 (B, N_new_boxes) 的可选注意力掩码张量。
        """
        if self.box_embeddings is None:
            # 第一个边界框：执行初始化
            self.box_embeddings = boxes
            bs = boxes.shape[1]
            box_seq_len = boxes.shape[0]

            if labels is None:
                labels = torch.ones(box_seq_len, bs, device=boxes.device, dtype=torch.long)
            if mask is None:
                mask = torch.zeros(bs, box_seq_len, device=boxes.device, dtype=torch.bool)

            self.box_labels = labels
            self.box_mask = mask
            return

        # 追加到现有边界框
        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == bs, f"Batch size mismatch: expected {bs}, got {boxes.shape[1]}"

        if labels is None:
            labels = torch.ones(boxes.shape[0], bs, device=boxes.device, dtype=torch.long)
        if mask is None:
            mask = torch.zeros(bs, boxes.shape[0], dtype=torch.bool, device=boxes.device)

        assert list(boxes.shape[:2]) == list(labels.shape[:2]), (
            f"Shape mismatch between boxes {boxes.shape} and labels {labels.shape}"
        )

        # 使用辅助函数进行拼接
        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(self.box_embeddings, self.box_mask, boxes, mask)


class SequenceGeometryEncoder(nn.Module):
    """几何边界框提示的编码器，假设输入边界框使用“归一化 CxCyWH”格式。

    边界框可以通过以下三种方式中的任意一种进行编码：
    - 直接投影：将坐标空间线性投影到 d_model
    - 池化：从骨干网络中提取 RoI align 特征
    - 位置编码器：对边界框中心进行位置编码

    这三种方式彼此兼容，选择多种方式时会将结果相加。

    作为替代方案，边界框也可以编码为两个角点（左上角和右下角）。

    编码后的序列还可以通过 Transformer 进一步处理。
    """

    def __init__(
        self,
        encode_boxes_as_points: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        use_act_ckpt: bool = False,
    ):
        """初始化 SequenceGeometryEncoder。"""
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size

        # 标签嵌入：按边界框编码时使用 2 个标签（正/负）
        # 按点编码时使用 6 个标签（普通点、左上角点、右下角点的正/负标签）
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)

        # 用于池化的 CLS 词元
        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)

        # 点编码（encode_boxes_as_points 为 True 时使用）
        if encode_boxes_as_points:
            self.points_direct_project = nn.Linear(2, self.d_model)
            self.points_pool_project = None
            self.points_pos_enc_project = None
        else:
            # 边界框编码模块
            assert boxes_direct_project or boxes_pos_enc or boxes_pool, "错误：至少需要一种边界框编码方式"
            self.points_direct_project = None
            self.points_pool_project = None
            self.points_pos_enc_project = None

            self.boxes_direct_project = None
            self.boxes_pool_project = None
            self.boxes_pos_enc_project = None

            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(self.d_model, self.d_model, self.roi_size)
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)

        self.encode = None
        if num_layers > 0:
            assert add_cls, "使用 Transformer 时强烈建议添加 CLS 词元"
            self.encode = _get_clones(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)

        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        """编码点（将边界框转换为角点时使用）。"""
        # 直接投影坐标
        points_embed = self.points_direct_project(points.to(img_feats.dtype))

        # 添加标签嵌入
        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats: torch.Tensor):
        """使用已配置的编码方式对边界框进行编码。"""
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes.to(img_feats.dtype))
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]

            # 将边界框转换为 xyxy 格式并反归一化
            boxes_xyxy = xywh2xyxy(boxes.to(img_feats.dtype))
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = scale.to(device=boxes_xyxy.device, non_blocking=True)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale

            # RoI 对齐
            # 为提高 ultralytics 导入速度，ROI align 所需的 torchvision 算子按需导入
            from torchvision.ops import roi_align

            sampled = roi_align(img_feats, boxes_xyxy.transpose(0, 1).unbind(0), self.roi_size)
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)

            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc.to(img_feats.dtype))
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        # 添加标签嵌入
        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        """编码几何边界框提示。

        参数：
            geo_prompt (Prompt): 包含边界框嵌入、掩码和标签的提示对象。
            img_feats (列表[torch.Tensor]): 来自骨干网络的图像特征列表。
            img_sizes (列表[tuple[int, int]]): 每个特征层级对应的 (H, W) 元组列表。
            img_pos_embeds (列表[torch.Tensor] | None): 图像特征的可选位置嵌入。

        返回：
            (encoded_embeddings, attention_mask) 元组。
        """
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels

        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(seq_first_img_feats)
        )

        # 如有需要，准备图像特征以进行池化
        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # 重塑为 NxCxHxW
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat

        if self.encode_boxes_as_points:
            # 将边界框转换为角点
            assert boxes is not None and boxes.shape[-1] == 4

            boxes_xyxy = xywh2xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)

            # 调整角点标签（偏移 2 和 4）
            labels_tl = boxes_labels + 2
            labels_br = boxes_labels + 4

            # 拼接左上角和右下角点
            points = torch.cat([top_left, bottom_right], dim=0)
            points_labels = torch.cat([labels_tl, labels_br], dim=0)
            points_mask = torch.cat([boxes_mask, boxes_mask], dim=1)

            final_embeds, final_mask = self._encode_points(
                points=points,
                points_mask=points_mask,
                points_labels=points_labels,
                img_feats=img_feats,
            )
        else:
            # Encode 边界框 directly
            final_embeds, final_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
            )

        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs

        # 如果配置了 CLS 令牌，则添加该令牌
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)

        # 最终投影
        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))

        # Transformer 编码层
        if self.encode is not None:
            for lay in self.encode:
                final_embeds = lay(
                    tgt=final_embeds,
                    memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                )
            final_embeds = self.encode_norm(final_embeds)

        return final_embeds, final_mask
