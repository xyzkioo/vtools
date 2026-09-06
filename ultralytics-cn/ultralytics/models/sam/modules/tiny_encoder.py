# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# --------------------------------------------------------
# TinyViT 模型架构
# Copyright (c) 2022 Microsoft
# 改编自 LeViT 和 Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# 构建 TinyViT 模型
# --------------------------------------------------------

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import LayerNorm2d
from ultralytics.utils.instance import to_2tuple
from ultralytics.utils.torch_utils import TORCH_1_11

CKPT_KWARGS = {"use_reentrant": False} if TORCH_1_11 else {}  # use_reentrant 在 torch 1.11 中加入，torch 2.9 需要此参数


class Conv2d_BN(torch.nn.Sequential):
    """依次执行二维卷积和批归一化的序列容器。

    此模块将二维卷积层与批归一化结合，为卷积神经网络提供常用构建块。批归一化的权重和偏置会初始化为特定值，以获得最佳训练性能。

    属性：
        c (torch.nn.Conv2d): 二维卷积层。
        bn (torch.nn.BatchNorm2d): 批归一化层。

    示例：
        >>> conv_bn = Conv2d_BN(3, 64, ks=3, stride=1, pad=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = conv_bn(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 64, 224, 224])
    """

    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
    ):
        """初始化依次执行二维卷积和批归一化的序列容器。

        参数：
            a (int): 输入通道数.
            b (int): 输出通道数.
            ks (int, 可选): 卷积核尺寸。
            stride (int, 可选): 卷积步幅。
            pad (int, 可选): 卷积填充。
            dilation (int, 可选): 卷积膨胀系数。
            groups (int, 可选): 卷积分组数。
            bn_weight_init (float, 可选): 批归一化权重的初始值。
        """
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """将图像嵌入为图像块，并投影到指定嵌入维度。

    此模块使用一系列卷积层将输入图像转换为图像块嵌入，在降低空间维度的同时增加通道维度。

    属性：
        patches_resolution (tuple[int, int]): 嵌入后图像块的分辨率。
        num_patches (int): 图像块总数。
        in_chans (int): 输入通道数。
        embed_dim (int): 嵌入维度。
        seq (nn.Sequential): 用于图像块嵌入的卷积层和激活层序列。

    示例：
        >>> import torch
        >>> patch_embed = PatchEmbed(in_chans=3, embed_dim=96, resolution=224, activation=nn.GELU)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 96, 56, 56])
    """

    def __init__(self, in_chans: int, embed_dim: int, resolution: int, activation):
        """初始化使用卷积层进行图像到图像块转换和投影的图像块嵌入模块。

        参数：
            in_chans (int): 输入通道数.
            embed_dim (int): 嵌入维度。
            resolution (int): 输入图像分辨率。
            activation (nn.Module): 卷积层之间使用的激活函数。
        """
        super().__init__()
        img_size: tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过图像块嵌入序列处理输入张量，将图像转换为图像块嵌入。"""
        return self.seq(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv（MBConv）层，是 EfficientNet 架构的一部分。

    此模块实现带扩展、深度卷积和投影阶段的移动倒置瓶颈卷积，并使用残差连接改善梯度流动。

    属性：
        in_chans (int): 输入通道数.
        hidden_chans (int): 扩展后的隐藏通道数。
        out_chans (int): 输出通道数.
        conv1 (Conv2d_BN): 用于通道扩展的第一层卷积。
        act1 (nn.Module): First activation function.
        conv2 (Conv2d_BN): Depthwise convolutional 层.
        act2 (nn.Module): Second activation function.
        conv3 (Conv2d_BN): Final convolutional 层 for projection.
        act3 (nn.Module): Third activation function.
        drop_path (nn.Module): Drop 路径 层 (Identity 用于推理).

    示例：
        >>> in_chans, out_chans = 64, 64
        >>> mbconv = MBConv(in_chans, out_chans, expand_ratio=4, activation=nn.ReLU, drop_path=0.1)
        >>> x = torch.randn(1, in_chans, 56, 56)
        >>> output = mbconv(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, in_chans: int, out_chans: int, expand_ratio: float, activation, drop_path: float):
        """使用指定的输入/输出通道、扩展倍率和激活函数初始化 MBConv 层。

        参数：
            in_chans (int): 输入通道数.
            out_chans (int): 输出通道数.
            expand_ratio (float): 隐藏层的通道扩展倍率。
            activation (nn.Module): 使用的激活函数。
            drop_path (float): 随机深度的丢弃路径概率。
        """
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        # 注意：`DropPath` 仅在训练时需要。
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 MBConv 前向传播，应用卷积和跳跃连接。"""
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        return self.act3(x)


class PatchMerging(nn.Module):
    """合并特征图中的相邻图像块，并投影到新的维度。

    此类使用带批归一化的一系列卷积层执行图像块合并，融合空间信息并调整特征维度，在降低空间分辨率的同时可以增加通道维度。

    属性：
        input_resolution (tuple[int, int]): 特征图的输入分辨率（高度、宽度）。
        dim (int): 特征图的输入维度。
        out_dim (int): The 输出 维度 after merging and projection.
        act (nn.Module): The activation function used between convolutions.
        conv1 (Conv2d_BN): 用于维度投影的第一层卷积。
        conv2 (Conv2d_BN): The second convolutional 层 for spatial merging.
        conv3 (Conv2d_BN): The third convolutional 层 for final projection.

    示例：
        >>> input_resolution = (56, 56)
        >>> patch_merging = PatchMerging(input_resolution, dim=64, out_dim=128, activation=nn.ReLU)
        >>> x = torch.randn(4, 64, 56, 56)
        >>> output = patch_merging(x)
        >>> print(output.shape)
        torch.Size([4, 784, 128])
    """

    def __init__(self, input_resolution: tuple[int, int], dim: int, out_dim: int, activation):
        """初始化用于合并和投影特征图中相邻图像块的 PatchMerging 模块。

        参数：
            input_resolution (tuple[int, int]): 特征图的输入分辨率（高度、宽度）。
            dim (int): 特征图的输入维度。
            out_dim (int): The 输出 维度 after merging and projection.
            activation (nn.Module): The activation function used between convolutions.
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征图应用图像块合并和维度投影。"""
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):
    """包含多个 MobileNetV3 风格倒置瓶颈卷积（MBConv）的卷积层。

    此层可选地对输出执行下采样，并支持在训练期间使用梯度检查点以节省内存。

    属性：
        dim (int): 输入和输出的维度。
        input_resolution (tuple[int, int]): Resolution of 输入 图像.
        depth (int): 此块中的 MBConv 层数。
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
        blocks (nn.ModuleList): List of MBConv 层.
        downsample (nn.Module | None): 用于对输出执行下采样的函数。

    示例：
        >>> input_tensor = torch.randn(1, 64, 56, 56)
        >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
        >>> output = conv_layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        activation,
        drop_path: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ):
        """使用给定维度和配置初始化 ConvLayer。

        此层由多个 MobileNetV3 风格的倒置瓶颈卷积（MBConv）组成，并可选择对输出执行下采样。

        参数：
            dim (int): 输入和输出的维度。
            input_resolution (tuple[int, int]): 输入图像的分辨率。
            depth (int): 此模块中 MBConv 层的数量。
            activation (nn.Module): 每个卷积后应用的激活函数。
            drop_path (float | 列表[float], 可选): DropPath 比例。可以是单个浮点数，也可以是每个 MBConv 对应的浮点数列表。
            downsample (nn.Module | None, 可选): 用于对输出执行下采样的函数。为 None 时跳过下采样。
            use_checkpoint (bool, 可选): 是否使用梯度检查点以节省内存。
            out_dim (int | None, 可选): 输出维度。为 None 时与 `dim` 相同。
            conv_expand_ratio (float, 可选): MBConv 层的扩展比例。
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建模块块
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # 图像块合并层
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过卷积层处理输入，应用 MBConv 块和可选下采样。"""
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, **CKPT_KWARGS) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)


class MLP(nn.Module):
    """用于 Transformer 架构的多层感知器（MLP）模块。

    此模块执行层归一化、带中间激活函数的两个全连接层以及 dropout，通常用于处理 Transformer 架构中的令牌嵌入。

    属性：
        norm (nn.LayerNorm): 应用于输入的层归一化。
        fc1 (nn.Linear): 第一个全连接层。
        fc2 (nn.Linear): 第二个全连接层。
        act (nn.Module): 在第一个全连接层后应用的激活函数。
        drop (nn.Dropout): 在激活函数后应用的 Dropout 层。

    示例：
        >>> import torch
        >>> from torch import nn
        >>> mlp = MLP(in_features=256, hidden_features=512, out_features=256, activation=nn.GELU, drop=0.1)
        >>> x = torch.randn(32, 100, 256)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([32, 100, 256])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation=nn.GELU,
        drop: float = 0.0,
    ):
        """使用可配置的输入、隐藏和输出维度初始化多层感知器。

        参数：
            in_features (int): 输入特征数量。
            hidden_features (int | None, 可选): 隐藏特征数量。
            out_features (int | None, 可选): 输出特征数量。
            activation (nn.Module): 在第一个全连接层后应用的激活函数。
            drop (float, 可选): Dropout 概率。
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = activation()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用 MLP 操作：层归一化、全连接层、激活函数和 dropout。"""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(torch.nn.Module):
    """具有空间感知能力和可训练注意力偏置的多头注意力模块。

    此模块实现支持空间感知的多头注意力机制，根据空间分辨率应用注意力偏置，并为分辨率网格中每个唯一空间偏移提供可训练偏置。

    属性：
        num_heads (int): 注意力头数量。
        scale (float): 注意力分数的缩放因子。
        key_dim (int): 键和值的维度。
        nh_kd (int): num_heads 与 key_dim 的乘积。
        d (int): 值向量的维度。
        dh (int): d 与 num_heads 的乘积。
        attn_ratio (float): 影响值向量维度的注意力比例。
        norm (nn.LayerNorm): 应用于输入的层归一化。
        qkv (nn.Linear): 用于计算查询、键和值投影的线性层。
        proj (nn.Linear): 用于最终投影的线性层。
        attention_biases (nn.Parameter): 可学习的注意力偏置。
        attention_bias_idxs (torch.Tensor): 注意力偏置的索引。
        ab (torch.Tensor): 推理时缓存的注意力偏置，训练期间删除。

    示例：
        >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
        >>> x = torch.randn(1, 196, 256)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 196, 256])
    """

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4,
        resolution: tuple[int, int] = (14, 14),
    ):
        """初始化具有空间感知能力的多头注意力模块。

        此模块实现具有空间感知能力的多头注意力机制，根据空间分辨率应用注意力偏置。
        对于分辨率网格中空间位置之间的每个唯一偏移，都包含可训练的注意力偏置。

        参数：
            dim (int): 输入和输出的维度。
            key_dim (int): 键和查询的维度。
            num_heads (int, 可选): 注意力头数量。
            attn_ratio (float, 可选): 影响值向量维度的注意力比例。
            resolution (tuple[int, int], 可选): 输入特征图的空间分辨率。
        """
        super().__init__()

        assert isinstance(resolution, tuple) and len(resolution) == 2, "'resolution' argument not tuple of length 2"
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode: bool = True):
        """将模块设置为训练模式，并处理缓存注意力偏置的 'ab' 属性。"""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用带空间感知和可训练注意力偏置的多头注意力。"""
        B, N, _ = x.shape  # B, N, C

        # 归一化
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if not self.training:
            self.ab = self.ab.to(self.attention_biases.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(x)


class TinyViTBlock(nn.Module):
    """对输入应用自注意力和局部卷积的 TinyViT 块。

    此块是 TinyViT 架构的关键组件，将自注意力机制与局部卷积结合以高效处理输入特征，支持窗口注意力以提高计算效率，并包含残差连接。

    属性：
        dim (int): 输入和输出的维度。
        input_resolution (tuple[int, int]): 输入特征图的空间分辨率。
        num_heads (int): 注意力头数量。
        window_size (int): 注意力窗口大小。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        drop_path (nn.Module): 随机深度层，推理期间为恒等函数。
        attn (Attention): 自注意力模块。
        mlp (MLP): 多层感知器模块。
        local_conv (Conv2d_BN): Depth-wise local convolution 层.

    示例：
        >>> input_tensor = torch.randn(1, 196, 192)
        >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 196, 192])
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation=nn.GELU,
    ):
        """初始化带自注意力和局部卷积的 TinyViT 块。

        此块是 TinyViT 架构的关键组件，将自注意力机制与局部卷积结合，以高效处理输入特征。

        参数：
            dim (int): 输入和输出特征的维度。
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率（高度、宽度）。
            num_heads (int): 注意力头数量。
            window_size (int, 可选): 注意力窗口大小，必须大于 0。
            mlp_ratio (float, 可选): MLP 隐藏维度与嵌入维度的比例。
            drop (float, 可选): Dropout 比例。
            drop_path (float, 可选): 随机深度比例。
            local_conv_size (int, 可选): 局部卷积的核尺寸。
            activation (nn.Module): MLP 使用的激活函数。
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # 注意：`DropPath` 仅在训练时需要。
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, activation=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用自注意力、局部卷积和 MLP 操作。"""
        h, w = self.input_resolution
        b, hw, c = x.shape  # batch, 高度*宽度, 通道
        assert hw == h * w, "input feature has wrong size"
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # 窗口划分
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)

            # 还原窗口
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, hw, c)

        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """返回 TinyViTBlock 参数的字符串表示。

        此方法返回包含 TinyViTBlock 关键信息的格式化字符串，包括维度、输入分辨率、注意力头数量、窗口尺寸和 MLP 比例。

        返回：
            (str): 包含模块参数的格式化字符串。

        示例：
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0)
            >>> print(block.extra_repr())
            dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """TinyViT 架构中一个阶段的基础层。

    此类表示 TinyViT 模型中的单个层，由多个 TinyViT 块和可选下采样操作组成，负责处理整体架构中特定分辨率和维度的特征。

    属性：
        dim (int): 输入和输出特征的维度。
        input_resolution (tuple[int, int]): 输入特征图的空间分辨率。
        depth (int): 此层中 TinyViT 块的数量。
        use_checkpoint (bool): 是否使用梯度检查点以节省内存。
        blocks (nn.ModuleList): 构成此层的 TinyViT 块列表。
        downsample (nn.Module | None): 层末尾的下采样层（如果指定）。

    示例：
        >>> input_tensor = torch.randn(1, 3136, 192)
        >>> layer = BasicLayer(dim=192, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
        >>> output = layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 3136, 192])
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation=nn.GELU,
        out_dim: int | None = None,
    ):
        """初始化 TinyViT 架构中的 BasicLayer。

        此层由多个 TinyViT 块和可选的下采样操作组成，用于处理 TinyViT 模型中特定分辨率和维度的特征图。

        参数：
            dim (int): 输入和输出特征的维度。
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率（高度、宽度）。
            depth (int): 此层中 TinyViT 块的数量。
            num_heads (int): 每个 TinyViT 块中的注意力头数量。
            window_size (int): 注意力计算使用的局部窗口大小。
            mlp_ratio (float, 可选): MLP 隐藏维度与嵌入维度的比例。
            drop (float, 可选): Dropout 比例。
            drop_path (float | 列表[float], 可选): 随机深度比例。可以是单个浮点数，也可以是每个块对应的浮点数列表。
            downsample (nn.Module | None, 可选): 层末尾的下采样层。为 None 时跳过下采样。
            use_checkpoint (bool, 可选): 是否使用梯度检查点以节省内存。
            local_conv_size (int, 可选): 每个 TinyViT 块中局部卷积的核尺寸。
            activation (nn.Module): MLP 使用的激活函数。
            out_dim (int | None, 可选): 下采样后的输出维度。为 None 时与 dim 相同。
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建 blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # 图像块合并层
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 TinyViT 块和可选下采样处理输入。"""
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, **CKPT_KWARGS) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) -> str:
        """返回用于打印的层参数字符串。"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    """TinyViT：用于高效图像分类和特征提取的紧凑视觉 Transformer 架构。

    此类实现 TinyViT 模型，结合视觉 Transformer 与卷积神经网络，以提升视觉任务的效率和性能。
    模型采用层次化处理结构，包括图像块嵌入、多阶段注意力与卷积模块，以及特征细化颈部。

    属性：
        img_size (int): 输入图像 尺寸.
        num_classes (int): 分类类别数量。
        depths (tuple[int, int, int, int]): 每个阶段的块数量。
        num_layers (int): 网络中的层总数。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        patch_embed (PatchEmbed): 图像块嵌入模块。
        patches_resolution (tuple[int, int]): 嵌入图像块的分辨率。
        layers (nn.ModuleList): 网络层列表。
        norm_head (nn.LayerNorm): 分类头使用的层归一化。
        head (nn.Linear): 用于最终分类的线性层。
        neck (nn.Sequential): 用于特征细化的颈部模块。

    示例：
        >>> model = TinyViT(img_size=224, embed_dims=(64, 128, 160, 320), num_heads=(2, 4, 5, 10))
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = model.forward_features(x)
        >>> print(features.shape)
        torch.Size([1, 256, 14, 14])
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_sizes: tuple[int, int, int, int] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        layer_lr_decay: float = 1.0,
    ):
        """初始化 TinyViT 模型。

        此构造函数搭建 TinyViT 架构，包括图像块嵌入、多层注意力与卷积模块，以及分类头。

        参数：
            img_size (int, 可选): Size of 输入 图像.
            in_chans (int, 可选): 输入通道数.
            num_classes (int, 可选): 类别数量 for classification.
            embed_dims (tuple[int, int, int, int], 可选): Embedding 维度 对于每个 stage.
            depths (tuple[int, int, int, int], 可选): 每个阶段的块数量。
            num_heads (tuple[int, int, int, int], 可选): 每个阶段的注意力头数量。
            window_sizes (tuple[int, int, int, int], 可选): 每个阶段的窗口大小。
            mlp_ratio (float, 可选): MLP 隐藏维度与嵌入维度的比例。
            drop_rate (float, 可选): Dropout 比例。
            drop_path_rate (float, 可选): 随机深度比例。
            use_checkpoint (bool, 可选): 是否使用检查点以节省内存。
            mbconv_expand_ratio (float, 可选): MBConv 层的扩展比例。
            local_conv_size (int, 可选): 局部卷积的核尺寸。
            layer_lr_decay (float, 可选): 分层学习率衰减因子。
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, activation=activation
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建 层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = {
                "dim": embed_dims[i_layer],
                "input_resolution": (
                    patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                "depth": depths[i_layer],
                "drop_path": dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                "downsample": PatchMerging if (i_layer < self.num_layers - 1) else None,
                "use_checkpoint": use_checkpoint,
                "out_dim": embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                "activation": activation,
            }
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # 分类头
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # 初始化权重
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay: float):
        """根据深度为 TinyViT 模型设置逐层学习率衰减。"""
        decay_rate = layer_lr_decay

        # 层 -> 模块块（深度）
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            """根据模块深度为模型中的每一层设置学习率缩放系数。"""
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x, scale=lr_scales[i]: _set_lr_scale(x, scale))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x, scale=lr_scales[i - 1]: _set_lr_scale(x, scale))
        assert i == depth
        for m in {self.norm_head, self.head}:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            """检查模块参数中是否存在学习率缩放属性。"""
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    @staticmethod
    def _init_weights(m):
        """初始化 TinyViT 模型中线性层和归一化层的权重。"""
        if isinstance(m, nn.Linear):
            # 注意：此初始化仅用于训练。
            # trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """返回不应使用权重衰减的参数关键字集合。"""
        return {"attention_biases"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """通过特征提取层处理输入，并返回空间特征。"""
        x = self.patch_embed(x)  # 输入 x 的形状为 (N, C, H, W)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(batch, self.patches_resolution[0] // 4, self.patches_resolution[1] // 4, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 TinyViT 模型前向传播，从输入图像中提取特征。"""
        return self.forward_features(x)

    def set_imgsz(self, imgsz: list[int] | None = None):
        """设置图像尺寸，使模型兼容不同大小的图像。"""
        imgsz = imgsz if imgsz is not None else [1024, 1024]
        imgsz = [s // 4 for s in imgsz]
        self.patches_resolution = imgsz
        for i, layer in enumerate(self.layers):
            input_resolution = (
                imgsz[0] // (2 ** (i - 1 if i == 3 else i)),
                imgsz[1] // (2 ** (i - 1 if i == 3 else i)),
            )
            layer.input_resolution = input_resolution
            if layer.downsample is not None:
                layer.downsample.input_resolution = input_resolution
            if isinstance(layer, BasicLayer):
                for b in layer.blocks:
                    b.input_resolution = input_resolution
