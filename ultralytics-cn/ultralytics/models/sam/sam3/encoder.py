# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# 基于 https://github.com/IDEA-Research/GroundingDINO
from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules.utils import _get_clones

from .model_misc import get_valid_ratio


class TransformerEncoderLayer(nn.Module):
    """先执行自注意力、再执行交叉注意力的 Transformer 编码器层。

    此层以前称为 TransformerDecoderLayer，后来更名以准确反映其在架构中的作用。
    它先通过自注意力处理输入序列，再与另一个输入（通常是图像特征）执行交叉注意力。

    此层支持 pre-norm 和 post-norm 配置，并支持在注意力机制的不同阶段加入位置编码。
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module = None,
        cross_attention: nn.Module = None,
    ):
        """初始化 Transformer 编码器层。

        参数：
            d_model: 模型维度或隐藏尺寸。
            dim_feedforward: 前馈网络维度。
            dropout: Dropout 概率
            pos_enc_at_attn: Whether to add positional encodings at self-attention
            pos_enc_at_cross_attn_keys: Whether to add positional encodings to keys in cross-attention
            pos_enc_at_cross_attn_queries: Whether to add positional encodings to queries in cross-attention
            pre_norm: Whether to use pre-norm (True) or post-norm (False) architecture
            self_attention: Self-attention 模块
            cross_attention: 用于关注图像特征的交叉注意力模块。
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)
        self.cross_attn_image = cross_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)

        # 实现前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """执行后归一化架构的前向传播。

        在后归一化架构中，归一化操作在注意力和前馈操作之后执行。

        参数：
            tgt (torch.Tensor): 输入张量 to be processed.
            memory (torch.Tensor): Memory 张量 for cross-attention.
            tgt_mask (torch.Tensor): Mask for self-attention.
            memory_mask (torch.Tensor): Mask for cross-attention.
            tgt_key_padding_mask (torch.Tensor): Key 填充 掩码 for self-attention.
            memory_key_padding_mask (torch.Tensor): Key 填充 掩码 for cross-attention.
            pos (torch.Tensor): Positional encoding for memory.
            query_pos (torch.Tensor): Positional encoding for query.
            **kwargs (Any): Additional keyword arguments.

        返回：
            处理后的张量。
        """
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # 自注意力
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 对图像执行交叉注意力
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dac: bool = False,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """执行前归一化架构的前向传播。

        在前归一化架构中，归一化操作在注意力和前馈操作之前执行。

        参数：
            tgt: 输入张量 to be processed
            memory: Memory 张量 for cross-attention
            dac: Whether to use Divide-and-Conquer attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key 填充 掩码 for self-attention
            memory_key_padding_mask: Key 填充 掩码 for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query

        返回：
            处理后的张量。
        """
        if dac:
            # 只对前一半查询执行自注意力
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt).contiguous()
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # 重新组合
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        memory = memory.to(tgt2.dtype).contiguous()
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dac: bool = False,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """执行 Transformer 编码器层的前向传播。

        参数：
            tgt: 要处理的输入张量。
            memory: 用于交叉注意力的记忆张量（例如图像特征）。
            dac: Whether to use Divide-and-Conquer attention (only apply self-attention to first half)
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key 填充 掩码 for self-attention
            memory_key_padding_mask: Key 填充 掩码 for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query

        返回：
            经过自注意力、交叉注意力和前馈网络处理后的张量。
        """
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            # attn_bias=attn_bias,
            # **kwds,
        )


class TransformerEncoder(nn.Module):
    """处理多层级特征的 Transformer 编码器。

    此编码器接收多层级特征（例如来自骨干网络的特征），并通过一组 Transformer 编码器层进行处理。
    它支持来自不同层级（例如不同分辨率）的特征，并可在训练期间使用激活检查点来节省内存。

    参数：
        layer: 要重复堆叠的编码器层。
        num_layers: 要堆叠的编码器层数。
        d_model: 模型维度和隐藏尺寸。
        num_feature_levels: 要处理的特征层级数量。
        frozen: 是否冻结此模块的参数。
        use_act_checkpoint: 是否在训练期间使用激活检查点。
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        """初始化 Transformer 编码器。"""
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.use_act_checkpoint = use_act_checkpoint

        # 为每层分配索引，使部分层能够根据自身层索引决定执行的操作
        # 例如，仅在选定层对记忆库执行交叉注意力
        for layer_idx, encoder_layer in enumerate(self.layers):
            encoder_layer.layer_idx = layer_idx

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        """为 Transformer 编码器准备多层级特征。"""
        assert len(srcs) == self.num_feature_levels, "mismatch between expected and received # of feature levels"

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
                device=src_flatten.device,
                dtype=src_flatten.dtype,
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: list[torch.Tensor],
        src_key_padding_masks: list[torch.Tensor] | None = None,
        pos: list[torch.Tensor] | None = None,
        prompt: torch.Tensor = None,
        prompt_key_padding_mask: torch.Tensor = None,
        encoder_extra_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过 Transformer 编码器处理多层特征。

        参数：
            src: 多层特征列表，每个元素形状为 (batch_size, 通道, 高度, 宽度)。
            src_key_padding_masks: 每个特征层级对应的填充掩码列表，每个元素形状为 (batch_size, 高度, 宽度)。
            pos: 每个特征层级对应的位置嵌入列表，每个元素形状为 (batch_size, 通道, 高度, 宽度)。
            prompt: 可选的文本或提示特征，形状为 (seq_len, batch_size, d_model)。
            prompt_key_padding_mask: 提示特征的可选填充掩码，形状为 (batch_size, seq_len)。
            encoder_extra_kwargs: 可选 additional arguments to pass to 每个 encoder 层

        返回：
            包含以下内容的元组：
            - 输出：形状为 (seq_len, batch_size, d_model) 的处理后特征
            - key_padding_masks_flatten：展平后的填充掩码
            - lvl_pos_embed_flatten：展平后的位置嵌入
            - level_start_index：每个特征层级的起始索引
            - spatial_shapes：每个特征层级的空间维度
            - valid_ratios：每个特征层级的有效比例
        """
        assert len(src) == self.num_feature_levels, "must be equal to num_feature_levels"
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        # 展平多层级特征并添加层级位置嵌入
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if self.training:
                assert self.use_act_checkpoint, "activation ckpt not enabled in encoder"
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = layer(**layer_kwargs)
        # 以序列优先格式返回
        return (
            output.transpose(0, 1),
            (key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """融合文本和图像特征的 Transformer 编码器。

    此编码器扩展 TransformerEncoder 以同时处理文本和图像特征，并能够将池化文本特征添加到图像特征中，
    从而实现更好的跨模态融合。它支持使用 torch.compile 进行性能优化。

    参数：
        layer (nn.Module): 要重复堆叠的编码器层。
        num_layers (int): 要堆叠的编码器层数。
        d_model (int): 模型维度和隐藏尺寸。
        num_feature_levels (int): 要处理的特征层级数量。
        add_pooled_text_to_img_feat (bool): 是否将池化后的文本特征添加到图像特征。
        pool_text_with_mask (bool): 池化文本特征时是否使用掩码。
        compile_mode (str | None): torch.compile 的模式，为 None 时禁用编译。
        **kwargs (Any): 传递给父类的其他参数。
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: str | None = None,
        **kwargs,
    ):
        """初始化带有文本-图像融合功能的 Transformer 编码器。"""
        super().__init__(
            layer,
            num_layers,
            d_model,
            num_feature_levels,
            **kwargs,
        )
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    def forward(
        self,
        src: list[torch.Tensor],
        prompt: torch.Tensor,
        src_key_padding_mask: list[torch.Tensor] | None = None,
        src_pos: list[torch.Tensor] | None = None,
        prompt_key_padding_mask: torch.Tensor = None,
        feat_sizes: list[int] | None = None,
        encoder_extra_kwargs: dict | None = None,
    ):
        """执行带有文本-图像融合功能的 Transformer 编码器前向传播。"""
        # 恢复视觉特征的空间形状
        bs = src[0].shape[1]  # seq first
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            assert all(x.dim() == 4 for x in src), "expected list of (bs, c, h, w) tensors"

        if self.add_pooled_text_to_img_feat:
            # 融合：将平均池化后的文本特征添加到图像特征
            pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, self.pool_text_with_mask)
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]  # prompt is seq first
            src = [x.add_(pooled_text) for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = super().forward(
            src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    """仅对有效词元的提示嵌入执行平均池化。"""
    # prompt 的形状为 (seq, bs, dim)
    if not pool_with_mask:
        return prompt.mean(dim=0)

    # prompt_mask 形状为 (bs, seq)，False 表示有效，True 表示填充
    assert prompt_mask.dim() == 2
    # is_valid 形状为 (seq, bs, 1)，1 表示有效，0 表示填充
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    # num_valid 的形状为 (bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

    # 对所有有效词元执行平均池化
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text
