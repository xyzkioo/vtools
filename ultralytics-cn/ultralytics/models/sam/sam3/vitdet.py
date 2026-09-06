# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
ViTDet backbone adapted from Detectron2.
This module implements Vision Transformer (ViT) backbone for object detection.

Rope embedding code adopted from:
1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
2. https://github.com/naver-ai/rope-vit
3. https://github.com/lucidrains/rotary-embedding-torch
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint

from ultralytics.models.sam.modules.blocks import PatchEmbed
from ultralytics.models.sam.modules.utils import (
    apply_rotary_enc,
    compute_axial_cis,
    concat_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)
from ultralytics.utils.checks import check_requirements

from .model_misc import LayerScale


class Attention(nn.Module):
    """带相对位置嵌入和二维 RoPE 的多头注意力块。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
        cls_token: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: tuple[int, int] | None = None,
        rope_interp: bool = False,
    ):
        """
        参数：
            dim (int): 输入通道数.
            num_heads (int): 注意力头数量。
            qkv_bias (bool): 为 True 时，为查询、键和值添加可学习偏置。
            use_rel_pos (bool): 为 True 时，将相对位置嵌入添加到注意力图中。
            rel_pos_zero_init (bool): 为 True 时，将相对位置参数初始化为零。
            input_size (tuple[int, int] or None): 用于计算相对位置尺寸或 rope 尺寸的输入分辨率。
            cls_token (bool): 是否存在 cls_token。
            use_rope (bool): 是否使用二维 rope（与 use_rel_pos 独立，也可以同时使用）。
            rope_theta (float): 控制 rope 的频率。
            rope_pt_size (tuple[int, int] or None): Size of rope in previous stage of 训练, needed for interpolation
                or tiling.
            rope_interp (bool): Whether to interpolate (or extrapolate) rope to match 输入 尺寸.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 相对位置嵌入和 RoPE
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        # 初始化相对位置嵌入和 RoPE
        self._setup_rel_pos(rel_pos_zero_init, input_size)
        self._setup_rope_freqs(input_size)

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True, input_size: tuple[int, int] | None = None) -> None:
        """设置相对位置嵌入。"""
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return

        assert input_size is not None
        assert self.cls_token is False, "not supported"
        # 初始化相对位置嵌入
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

        if not rel_pos_zero_init:
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

        # 预先计算相对坐标
        H, W = input_size
        q_coords = torch.arange(H)[:, None]
        k_coords = torch.arange(W)[None, :]
        relative_coords = (q_coords - k_coords) + (H - 1)
        self.relative_coords = relative_coords.long()

    def _setup_rope_freqs(self, input_size: tuple[int, int] | None = None) -> None:
        """设置二维 RoPE 频率。"""
        if not self.use_rope:
            self.freqs_cis = None
            return

        assert input_size is not None
        # 确定 RoPE 输入尺寸
        if self.rope_pt_size is None:
            self.rope_pt_size = input_size

        # 初始化二维 RoPE 频率
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.head_dim,
            theta=self.rope_theta,
        )

        # 对 RoPE 进行插值
        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / input_size[0]
        # 获取缩放后的 freqs_cis
        freqs_cis = self.compute_cis(
            end_x=input_size[0],
            end_y=input_size[1],
            scale_pos=scale_pos,
        )
        if self.cls_token:
            t = torch.zeros(
                self.head_dim // 2,
                dtype=torch.float32,
                device=freqs_cis.device,
            )
            cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)

        self.freqs_cis = freqs_cis

    def _apply_rope(self, q, k) -> tuple[Tensor, Tensor]:
        """对 q 和 k 应用二维 RoPE。"""
        if not self.use_rope:
            return q, k

        assert self.freqs_cis is not None
        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis.to(q.device))

    def forward(self, x: Tensor) -> Tensor:
        """执行注意力块的前向传播。"""
        s = 1 if self.cls_token else 0  # 用于排除 cls_token
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0  # no cls_token
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = math.sqrt(L - s)

        # qkv 形状为 (3, B, nHead, L, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        # q、k、v 形状为 (B, nHead, L, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # 处理 RoPE 和相对位置嵌入
        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = concat_rel_pos(
                q.flatten(0, 1),
                k.flatten(0, 1),
                (H, W),
                x.shape[1:3],
                self.rel_pos_h,
                self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )

        # SDPA 需要 [B, nheads, H*W, C]，因此转置回原顺序
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        if ndim == 4:
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)

        return x


class Block(nn.Module):
    """支持窗口注意力的 Transformer 块。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
        use_rope: bool = False,
        rope_pt_size: tuple[int, int] | None = None,
        rope_interp: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: float | None = None,
    ):
        """
        参数：
            dim (int): 输入通道数.
            num_heads (int): 每个 ViT 块中的注意力头数量。
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): 为 True 时，为查询、键和值添加可学习偏置。
            drop_path (float): Stochastic depth rate.
            norm_layer (Callable): Normalization 层 constructor.
            act_layer (Callable): Activation 层 constructor.
            use_rel_pos (bool): 为 True 时，将相对位置嵌入添加到注意力图中。
            rel_pos_zero_init (bool): 为 True 时，将相对位置参数初始化为零。
            window_size (int): 窗口注意力块的窗口尺寸。为 0 时不使用窗口注意力。
            input_size (tuple[int, int] | None): 用于计算相对位置尺寸的输入分辨率。
            use_rope (bool): 是否使用二维 rope（与 use_rel_pos 独立，也可以同时使用）。
            rope_pt_size (tuple[int, int] | None): 训练前一阶段的 rope 尺寸，插值或平铺时需要。
            rope_interp (bool): 是否插值（或外推）rope 以匹配目标输入尺寸；通常应将源尺寸指定为 rope_pt_size。
            cls_token (bool): 是否存在 cls_token。
            dropout (float): Dropout 比例。
            init_values (float | None): 层缩放初始值；为 None 时不使用层缩放。
        """
        super().__init__()

        check_requirements("timm")
        from timm.layers import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        """执行 Transformer 块的前向传播。"""
        shortcut = x
        x = self.norm1(x)
        # 窗口划分
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))
        # 还原窗口划分
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x


class ViT(nn.Module):
    """此模块实现 :paper:`vitdet` 中的 Vision Transformer（ViT）骨干网络。“Exploring Plain Vision Transformer
    Backbones for Object Detection", https://arxiv.org/abs/2203.16527.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] | str = "LayerNorm",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: tuple[int, ...] | bool = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: tuple[int, ...] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: int | None = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: float | None = None,  # 用于 LayerScale
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = True,
    ):
        """
        参数：
            img_size (int): 输入图像 尺寸. Only relevant for rel pos or rope.
            patch_size (int): Patch 尺寸.
            in_chans (int): 输入图像的通道数。
            embed_dim (int): Patch embedding 维度.
            depth (int): Depth of ViT.
            num_heads (int): 每个 ViT 块中的注意力头数量。
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): 为 True 时，为查询、键和值添加可学习偏置。
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (Callable or str): Normalization 层 constructor or 名称.
            act_layer (Callable): Activation 层 constructor.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            tile_abs_pos (bool): If True, tile absolute positional embeddings instead of interpolation.
            rel_pos_blocks (tuple[int, ...] | bool): Blocks which have rel pos embeddings.
            rel_pos_zero_init (bool): 为 True 时，将相对位置参数初始化为零。
            window_size (int): Window 尺寸 for window attention blocks.
            global_att_blocks (tuple[int, ...]): Indexes for blocks using global attention (other blocks use window
                attention).
            use_rope (bool): Whether to use rope 2d (independent of rel_pos_blocks, as it can be used together).
            rope_pt_size (int | None): Size of rope in previous stage of 训练, needed for interpolation or tiling.
            use_interp_rope (bool): Whether to interpolate (or extrapolate) rope to match 目标 输入 尺寸, expected to
                specify source 尺寸 as rope_pt_size.
            pretrain_img_size (int): 预训练模型使用的输入图像尺寸。
            pretrain_use_cls_token (bool): 为 True 时，预训练模型使用类别令牌。
            retain_cls_token (bool): 是否保留 cls_token。
            dropout (float): Dropout 比例，应用于注意力、MLP 的残差块以及 MLP 内部。
            return_interm_layers (bool): 是否返回中间层（所有全局注意力块）。
            init_values (float | None): 层缩放初始值；为 None 时不使用层缩放。
            ln_pre (bool): 为 True 时，在 Transformer 块前应用层归一化。
            ln_post (bool): 为 True 时，在 Transformer 块后应用层归一化。
            bias_patch_embed (bool): 为 True 时，在图像块嵌入卷积中使用偏置。
            compile_mode (str | None): 前向计算的编译模式；为 None 时禁用编译。
            use_act_checkpoint (bool): 为 True 时，使用激活检查点。
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool) and rel_pos_blocks:
            self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True

        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert len(window_block_indexes) == 0, "windowing not supported with cls token"

            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"

            scale = embed_dim**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-5)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        # 处理绝对位置嵌入
        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos

        if self.use_abs_pos:
            # 使用预训练图像尺寸初始化绝对位置嵌入。
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.patch_size = patch_size
        self.window_size = window_size
        self.blocks = nn.ModuleList()
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=((window_size, window_size) if rope_pt_size is None else (rope_pt_size, rope_pt_size)),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
            )

            if i not in window_block_indexes:
                cur_stage += 1

            self.use_act_checkpoint = use_act_checkpoint

            self.blocks.append(block)

        self.return_interm_layers = return_interm_layers
        self.channel_list = [embed_dim] * len(self.full_attn_ids) if return_interm_layers else [embed_dim]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.ln_pre = norm_layer(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = norm_layer(embed_dim) if ln_post else nn.Identity()

        self.apply(self._init_weights)

        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)
            if self.use_act_checkpoint and self.training:
                torch._dynamo.config.optimize_ddp = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """初始化权重。"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """执行 ViT 前向传播并获取特征图。"""
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            # 如果保留 cls_token，则不执行此操作
            # 保持空间形状
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if self.use_act_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if (i == self.full_attn_ids[-1]) or (self.return_interm_layers and i in self.full_attn_ids):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(feats.shape[0], h, w, feats.shape[-1]).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs

    def set_imgsz(self, imgsz: list[int] | None = None):
        """为新的输入图像尺寸设置相对位置嵌入和 RoPE 频率。"""
        imgsz = imgsz if imgsz is not None else [1008, 1008]
        for block in self.blocks:
            if block.window_size != 0:
                continue
            block.attn._setup_rel_pos(input_size=(imgsz[0] // self.patch_size, imgsz[1] // self.patch_size))
            block.attn._setup_rope_freqs(input_size=(imgsz[0] // self.patch_size, imgsz[1] // self.patch_size))
