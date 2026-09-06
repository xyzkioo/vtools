# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""各种实用模型。"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


class DotProductScoring(torch.nn.Module):
    """计算查询特征与池化提示嵌入之间点积得分的模块。"""

    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        """初始化 DotProductScoring 模块。"""
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an 可选 MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    @staticmethod
    def mean_pool_text(prompt, prompt_mask):
        """仅对有效词元的提示嵌入执行平均池化。"""
        # is_valid 的形状为 (seq, bs, 1)，其中 1 表示有效，0 表示填充
        is_valid = (~prompt_mask).to(prompt.dtype).permute(1, 0)[..., None]
        # num_valid 的形状为 (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # 对所有有效词元执行平均池化，pooled_prompt 的形状为 (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        """计算 hs 与 prompt 之间的点积得分。"""
        # hs 的形状为 (num_layer, bs, num_query, d_model)
        # prompt 的形状为 (seq, bs, d_model)
        # prompt_mask 的形状为 (bs, seq)，其中 1 表示填充，0 表示有效
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # 如果指定了 MLP，则对 prompt 应用 MLP
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt.to(hs.dtype))

        # 首先获取 prompt 的平均池化结果
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # 然后将 pooled_prompt 和 hs 投影到 d_proj 维度
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # 最后获取形状为 (num_layer, bs, num_query, 1) 的点积得分
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # 将得分限制在最大值范围内，避免损失计算或匹配器出现数值问题
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    """对层输出执行逐通道缩放的 LayerScale 模块。"""

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        """初始化 LayerScale 模块。"""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """对输入张量应用 LayerScale。"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerWrapper(nn.Module):
    """由编码器和解码器组成的 Transformer 封装器。"""

    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
        pos_enc_at_input_dec=True,
    ):
        """初始化 TransformerWrapper。"""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # 两阶段模式
        assert two_stage_type in ["none"], f"unknown param {two_stage_type} of two_stage_type"
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        """初始化模型参数。"""
        for n, p in self.named_parameters():
            if p.dim() > 1 and "box_embed" not in n and "query_embed" not in n and "reference_points" not in n:
                nn.init.xavier_uniform_(p)


def get_valid_ratio(mask):
    """根据掩码计算有效高度和宽度的比例。"""
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor: torch.Tensor, num_feats: int = 256):
    """为二维或四维坐标张量生成正弦位置嵌入。

    此函数使用不同频率的正弦和余弦函数创建嵌入，形式类似于 Transformer 模型使用的位置编码。
    它同时支持二维位置张量 (x, y) 和用于边界框坐标的四维张量 (x, y, w, h)。

    参数：
        pos_tensor (torch.Tensor): 输入位置张量；二维坐标形状为 (n_query, bs, 2)，四维边界框坐标形状为 (n_query, bs, 4)。
        num_feats (int): 输出嵌入的特征维度，必须为偶数，默认为 256。

    返回：
        (torch.Tensor): 正弦位置嵌入；二维输入的形状为 (n_query, bs, num_feats)，四维输入的形状为
            (n_query, bs, num_feats * 2)。

    异常：
        AssertionError: If num_feats is not even.
        ValueError: 当 pos_tensor.size(-1) 不是 2 或 4 时抛出。

    示例：
        >>> pos_2d = torch.rand(100, 8, 2)  # 100 queries, batch size 8, 2D coordinates
        >>> embeddings_2d = gen_sineembed_for_position(pos_2d, num_feats=256)
        >>> embeddings_2d.shape
        torch.Size([100, 8, 256])
        >>> pos_4d = torch.rand(50, 4, 4)  # 50 queries, batch size 4, 4D coordinates
        >>> embeddings_4d = gen_sineembed_for_position(pos_4d, num_feats=128)
        >>> embeddings_4d.shape
        torch.Size([50, 4, 256])
    """
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos
