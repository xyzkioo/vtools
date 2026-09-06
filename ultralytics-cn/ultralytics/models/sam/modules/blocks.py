# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.nn.modules import MLP, LayerNorm2d, MLPBlock

from .transformer import Attention, TwoWayAttentionBlock, TwoWayTransformer
from .utils import add_decomposed_rel_pos, apply_rotary_enc, compute_axial_cis, window_partition, window_unpartition


class DropPath(nn.Module):
    """在训练期间为神经网络实现随机深度正则化。

    属性：
        drop_prob (float): Probability of dropping a 路径 训练期间.
        scale_by_keep (bool): 是否根据保留概率缩放输出。

    方法：
        forward: 训练期间对输入张量应用随机深度，并可选择进行缩放。

    示例：
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """初始化训练期间使用的随机深度正则化 DropPath 模块。"""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        """在训练期间对输入张量应用随机深度，并支持可选缩放。"""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MaskDownSampler(nn.Module):
    """用于高效处理输入掩码的下采样和嵌入模块。

    此类通过卷积层、层归一化和激活函数逐步降低输入掩码的空间维度，同时扩展其通道维度。

    属性：
        encoder (nn.Sequential): 用于掩码下采样和嵌入的卷积层、层归一化及激活函数序列。

    方法：
        forward: 对输入掩码进行下采样并编码为 embed_dim 个通道。

    示例：
        >>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
        >>> input_mask = torch.randn(1, 1, 256, 256)
        >>> output = mask_downsampler(input_mask)
        >>> print(output.shape)
        torch.Size([1, 256, 16, 16])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: type[nn.Module] = nn.GELU,
        interpol_size: tuple[int, int] | None = None,
    ):
        """初始化用于逐步下采样和通道扩展的掩码下采样模块。"""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple)), (
                f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x: Tensor) -> Tensor:
        """使用卷积层和 LayerNorm2d 将输入掩码下采样并编码为 embed_dim 个通道。"""
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(x.dtype)
        return self.encoder(x)


class CXBlock(nn.Module):
    """用于卷积神经网络高效特征提取的 ConvNeXt 块。

    此模块实现 ConvNeXt 架构的改进版本，可提升特征提取的性能和灵活性。

    属性：
        dwconv (nn.Conv2d): Depthwise or standard 2D convolution 层.
        norm (LayerNorm2d): Layer normalization applied to 通道.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear 层.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear 层.
        gamma (nn.Parameter | None): 用于层缩放的可学习缩放参数。
        drop_path (nn.Module): DropPath 层 for stochastic depth regularization.

    方法：
        forward: Processes 输入 张量 through the ConvNeXt block.

    示例：
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        """初始化用于卷积神经网络高效特征提取的 ConvNeXt 块。

        此块实现了 ConvNeXt 架构的改进版本，在特征提取方面提供更好的性能和灵活性。

        参数：
            dim (int): 输入通道数。
            kernel_size (int): 卷积核大小。
            填充 (int): 卷积的填充尺寸。
            drop_path (float): 随机深度比例。
            layer_scale_init_value (float): Layer Scale 的初始值。
            use_dwconv (bool): 是否使用深度卷积。
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 逐点/1x1 卷积使用线性层实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """对输入张量应用 ConvNeXt 块操作，包括卷积和残差连接。"""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """通过神经网络多层结构融合特征的模块。

    此类将一系列相同层依次应用于输入张量，也可以先对输入进行投影。

    属性：
        proj (nn.Module): 可选的输入投影层；不需要投影时使用 Identity。
        layers (nn.ModuleList): 按顺序应用的相同层列表。

    方法：
        forward: 对输入张量应用特征融合器。

    示例：
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer: nn.Module, num_layers: int, dim: int | None = None, input_projection: bool = False):
        """初始化通过多层结构进行特征融合的 Fuser 模块。

        此模块创建一系列相同的层，并可选择先应用输入投影。

        参数：
            层 (nn.Module): 在融合器中重复使用的层。
            num_layers (int): 重复该层的次数。
            dim (int | None): 输入投影使用的维度。
            input_projection (bool): 是否使用输入投影。
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """将一系列层应用于输入张量，并可选择先进行投影。"""
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SAM2TwoWayAttentionBlock(TwoWayAttentionBlock):
    """在两个方向执行自注意力和交叉注意力的双向注意力块。

    此模块扩展 TwoWayAttentionBlock，包含四个主要部分：稀疏输入自注意力、稀疏到密集交叉注意力、稀疏输入 MLP，以及密集到稀疏交叉注意力。

    属性：
        self_attn (Attention): Self-attention 层 for queries.
        norm1 (nn.LayerNorm): Layer normalization after the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention 层 from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after the second attention block.
        mlp (MLP): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after the MLP block.
        norm4 (nn.LayerNorm): Layer normalization after the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention 层 from keys to queries.
        skip_first_layer_pe (bool): 是否跳过第一层的位置编码。

    方法：
        forward: Processes 输入 through the attention blocks and MLP.

    示例：
        >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """初始化在两个方向执行自注意力和交叉注意力的 SAM2TwoWayAttentionBlock。

        此模块扩展 TwoWayAttentionBlock，包含稀疏输入自注意力、稀疏到密集交叉注意力、稀疏输入 MLP，以及密集到稀疏交叉注意力。

        参数：
            embedding_dim (int): 嵌入的通道维度。
            num_heads (int): 注意力层的头数。
            mlp_dim (int): MLP 模块的隐藏维度。
            activation (type[nn.Module]): MLP 模块的激活函数。
            attention_downsample_rate (int): 注意力计算的下采样倍率。
            skip_first_layer_pe (bool): 是否跳过第一层的位置编码。
        """
        super().__init__(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act=activation)


class SAM2TwoWayTransformer(TwoWayTransformer):
    """用于同时关注图像和查询点的双向 Transformer 模块。

    此类扩展 TwoWayTransformer，实现一种使用带位置嵌入查询关注输入图像的专用 Transformer 解码器，适用于目标检测、图像分割和点云处理。

    属性：
        depth (int): Transformer 的层数。
        embedding_dim (int): 输入嵌入的通道维度。
        num_heads (int): 多头注意力的头数。
        mlp_dim (int): MLP 模块的内部通道维度。
        layers (nn.ModuleList): 组成 Transformer 的 SAM2TwoWayAttentionBlock 层列表。
        final_attn_token_to_image (Attention): 从查询到图像的最终注意力层。
        norm_final_attn (nn.LayerNorm): 应用于最终查询的层归一化。

    方法：
        forward: 使用 Transformer 处理输入图像嵌入和查询嵌入。

    示例：
        >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
        >>> print(output[0].shape, output[1].shape)
        torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """初始化 SAM2TwoWayTransformer 实例。

        此 Transformer 解码器使用带位置嵌入的查询关注输入图像，适用于目标检测、图像分割和点云处理等任务。

        参数：
            depth (int): Transformer 的层数。
            embedding_dim (int): 输入嵌入的通道维度。
            num_heads (int): 多头注意力的头数，必须能整除 embedding_dim。
            mlp_dim (int): MLP 模块的内部通道维度。
            activation (type[nn.Module]): MLP 模块使用的激活函数。
            attention_downsample_rate (int): 注意力计算的下采样倍率。
        """
        super().__init__(depth, embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SAM2TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )


class RoPEAttention(Attention):
    """在 Transformer 架构的注意力机制中实现旋转位置编码。

    此类在基础 Attention 类中加入旋转位置编码（RoPE），以增强注意力机制的位置感知能力。

    属性：
        compute_cis (Callable): 用于…的函数 compute axial complex numbers for rotary encoding.
        freqs_cis (torch.Tensor): Precomputed frequency 张量 for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key 长度 for cross-attention to memories.

    方法：
        forward: 应用旋转位置编码，并计算查询、键和值张量之间的注意力。

    示例：
        >>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(
        self,
        *args,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (32, 32),  # 512 分辨率下步长为 16 的特征尺寸 [w, h]
        **kwargs,
    ):
        """初始化带旋转位置编码的 RoPEAttention，以增强位置感知能力。"""
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat  # 重复 q rope 以匹配 k 长度，交叉注意力访问记忆时需要

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        """应用旋转位置编码，并计算查询、键和值张量之间的注意力。"""
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 拆分为多个注意力头
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 应用旋转位置编码
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # 计算注意力
        out = F.scaled_dot_product_attention(q, k, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """对张量执行池化和可选归一化，并处理空间维度排列。"""
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """实现带可选查询池化的多尺度自注意力，以高效提取特征。

    此类提供灵活的多尺度注意力实现，可通过池化对查询特征进行可选下采样，旨在增强模型在视觉任务中捕获多尺度信息的能力。

    属性：
        dim (int): 特征图的输入维度。
        dim_out (int): 注意力模块的输出维度。
        num_heads (int): 注意力头数。
        scale (float): 点积注意力的缩放因子。
        q_pool (nn.Module | None): 查询特征的可选池化模块。
        qkv (nn.Linear): 查询、键和值的线性投影层。
        proj (nn.Linear): 输出投影层。

    方法：
        forward: 对输入张量应用多尺度注意力。

    示例：
        >>> import torch
        >>> from torch import nn
        >>> x = torch.randn(1, 64, 64, 256)
        >>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
        >>> output = msa(x)
        >>> print(output.shape)
        torch.Size([1, 64, 64, 256])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        """初始化带可选查询池化的多尺度注意力，以高效提取特征。"""
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用带可选查询池化的多尺度注意力，提取多尺度特征。"""
        B, H, W, _ = x.shape
        # qkv 形状为 (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q、k、v 形状为 (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q 池化（用于阶段切换时的下采样）
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # 下采样后的形状
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch 的 SDPA 需要 [B, nheads, H*W, C]，因此进行转置
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # 转置回原排列
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """带窗口划分和查询池化的多尺度注意力块，用于高效视觉 Transformer。

    此类实现带可选窗口划分和下采样的多尺度注意力机制，适用于视觉 Transformer 架构。

    属性：
        dim (int): 模块的输入维度。
        dim_out (int): 模块的输出维度。
        norm1 (nn.Module): 第一个归一化层。
        window_size (int): 窗口划分的尺寸。
        pool (nn.Module | None): 查询下采样的池化层。
        q_stride (tuple[int, int] | None): 查询池化的步幅。
        attn (MultiScaleAttention): 多尺度注意力模块。
        drop_path (nn.Module): 用于正则化的随机深度层。
        norm2 (nn.Module): 第二个归一化层。
        mlp (MLP): 多层感知器模块。
        proj (nn.Linear | None): 用于维度匹配的投影层。

    方法：
        forward: 使用多尺度块处理输入张量。

    示例：
        >>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 512])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module | str = "LayerNorm",
        q_stride: tuple[int, int] | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        window_size: int = 0,
    ):
        """初始化带窗口划分和可选查询池化的多尺度注意力块。"""
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            act=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过多尺度注意力和 MLP 处理输入，并支持可选窗口划分和下采样。"""
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # 跳跃连接
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # 窗口划分
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # 窗口注意力 + Q 池化（阶段切换时）
        x = self.attn(x)
        if self.q_stride:
            # Q 池化改变了张量形状
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # 还原窗口划分
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP 模块
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PositionEmbeddingSine(nn.Module):
    """为图像等二维输入生成正弦位置嵌入的模块。

    此类为二维空间位置实现正弦位置编码，可用于计算机视觉任务中的 Transformer 模型。

    属性：
        num_pos_feats (int): 位置特征数量（嵌入维度的一半）。
        temperature (int): 正弦函数的温度参数。
        normalize (bool): 是否归一化位置嵌入。
        scale (float): normalize 为 True 时嵌入使用的缩放因子。
        cache (dict): 存储预计算嵌入的缓存。

    方法：
        _encode_xy: 使用正弦和余弦函数编码二维位置。
        encode_boxes: 将边界框坐标和尺寸编码为位置嵌入。
        encode_points: 使用正弦位置嵌入编码二维点坐标。
        forward: 为二维输入生成正弦位置嵌入。

    示例：
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 128, 224, 224])
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        """初始化二维图像输入的正弦位置嵌入。"""
        super().__init__()
        assert num_pos_feats % 2 == 0, "模型宽度必须为偶数"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """使用正弦/余弦函数将二维位置编码为 Transformer 位置嵌入。"""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """将边界框坐标和尺寸编码为检测任务的位置嵌入。"""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """使用正弦嵌入编码二维点，并附加标签。"""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tensor:
        """为图像等二维输入生成正弦位置嵌入。"""
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """使用随机空间频率的位置编码。

    此类使用随机空间频率为输入坐标生成位置嵌入，尤其适用于需要位置信息的 Transformer 模型。

    属性：
        positional_encoding_gaussian_matrix (torch.Tensor): 包含编码随机值的缓冲区。

    方法：
        _pe_encoding: 对已归一化到 [0,1] 的点进行位置编码。
        forward: 为指定尺寸的网格生成位置编码。
        forward_with_coords: 对未归一化到 [0,1] 的点进行位置编码。

    示例：
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        """初始化供 Transformer 使用的随机空间频率位置嵌入。"""
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

        # 关闭确定性算法，避免 forward() 报错：cumsum_cuda_kernel 没有确定性实现
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """使用随机空间频率编码归一化到 [0,1] 的坐标。"""
        # 假设 coords 位于 [0, 1]^2 方形区域，形状为 d_1 x ... x d_n x 2
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # 输出形状为 d_1 x ... x d_n x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        """使用随机空间频率为网格生成位置编码。"""
        h, w = size
        grid = torch.ones(
            (h, w),
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=self.positional_encoding_gaussian_matrix.dtype,
        )
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """根据给定图像尺寸归一化输入坐标到 [0,1]，并进行位置编码。"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C


class Block(nn.Module):
    """支持窗口注意力和残差传播的 Transformer 块。

    此类实现一个 Transformer 块，可使用全局或窗口自注意力，随后连接前馈网络；同时支持相对位置嵌入，适用于视觉 Transformer 架构。

    属性：
        norm1 (nn.Module): 第一个归一化层。
        attn (REAttention): 带可选相对位置编码的自注意力层。
        norm2 (nn.Module): 第二个归一化层。
        mlp (MLPBlock): 多层感知器模块。
        window_size (int): 注意力窗口尺寸；为 0 时使用全局注意力。

    方法：
        forward: 使用 Transformer 块处理输入。

    示例：
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """初始化带可选窗口注意力和相对位置嵌入的 Transformer 块。

        此构造函数建立一个 Transformer 块，可使用全局或窗口自注意力，随后连接前馈网络；同时支持相对位置嵌入，适用于视觉 Transformer 架构。

        参数：
            dim (int): 输入通道数.
            num_heads (int): 自注意力层的头数。
            mlp_ratio (float): MLP 隐藏维度与嵌入维度的比值。
            qkv_bias (bool): 为 True 时，为查询、键和值投影添加可学习偏置。
            norm_layer (type[nn.Module]): 使用的归一化层类型。
            act_layer (type[nn.Module]): MLP 模块使用的激活函数类型。
            use_rel_pos (bool): 为 True 时，在注意力中使用相对位置嵌入。
            rel_pos_zero_init (bool): 为 True 时，将相对位置参数初始化为零。
            window_size (int): 注意力窗口尺寸；为 0 时使用全局注意力。
            input_size (tuple[int, int] | None): 用于计算相对位置参数的输入尺寸。
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = REAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 Transformer 块处理输入，并支持窗口自注意力和残差连接。"""
        shortcut = x
        x = self.norm1(x)
        # 窗口划分
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # 还原窗口划分
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class REAttention(nn.Module):
    """用于 Transformer 架构高效自注意力的相对位置注意力模块。

    此类实现带相对位置嵌入的多头注意力机制，适用于视觉 Transformer 模型。

    属性：
        num_heads (int): 注意力头数。
        scale (float): 注意力计算的缩放因子。
        qkv (nn.Linear): 查询、键和值的线性投影层。
        proj (nn.Linear): 输出投影层。
        use_rel_pos (bool): 是否使用相对位置嵌入。
        rel_pos_h (nn.Parameter): 高度维度的相对位置嵌入。
        rel_pos_w (nn.Parameter): 宽度维度的相对位置嵌入。

    方法：
        forward: 对输入张量应用带可选相对位置编码的多头注意力。

    示例：
        >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
        >>> x = torch.randn(1, 32, 32, 256)
        >>> output = attention(x)
        >>> print(output.shape)
        torch.Size([1, 32, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """初始化用于 Transformer 架构的相对位置注意力模块。

        此模块实现带可选相对位置编码的多头注意力，专为 Transformer 模型中的视觉任务设计。

        参数：
            dim (int): 输入通道数.
            num_heads (int): 注意力头数。
            qkv_bias (bool): 为 True 时，为查询、键和值投影添加可学习偏置。
            use_rel_pos (bool): 为 True 时使用相对位置编码。
            rel_pos_zero_init (bool): 为 True 时，将相对位置参数初始化为零。
            input_size (tuple[int, int] | None): 用于计算相对位置参数的输入尺寸；use_rel_pos 为 True 时必须提供。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # 初始化相对位置嵌入
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用带可选相对位置编码的多头注意力。"""
        B, H, W, _ = x.shape
        # qkv 形状为 (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q、k、v 形状为 (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class PatchEmbed(nn.Module):
    """用于视觉 Transformer 架构的图像到图像块嵌入模块。

    此模块使用卷积层将输入图像转换为图像块嵌入序列，通常作为视觉 Transformer 的第一层，将图像数据转换为后续 Transformer 块所需的格式。

    属性：
        proj (nn.Conv2d): 将图像块投影为嵌入的卷积层。

    方法：
        forward: 对输入张量应用图像块嵌入。

    示例：
        >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 14, 14, 768])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        """初始化将图像块转换为嵌入的 PatchEmbed 模块。

        此模块通常作为视觉 Transformer 架构的第一层，将图像数据转换为后续 Transformer 块所需的格式。

        参数：
            kernel_size (tuple[int, int]): 提取图像块所用卷积核的尺寸。
            stride (tuple[int, int]): 卷积操作的步幅。
            padding (tuple[int, int]): 卷积前为输入添加的填充。
            in_chans (int): 输入图像通道数。
            embed_dim (int): 输出图像块嵌入的维度。
            bias (bool): 卷积层是否包含偏置项。
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过卷积并转置结果张量计算图像块嵌入。"""
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C
