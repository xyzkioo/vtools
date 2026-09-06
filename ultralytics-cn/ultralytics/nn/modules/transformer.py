# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer 模块。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器中的单个层。

    此类实现带多头注意力和前馈网络的标准 Transformer 编码器层，同时支持前归一化和后归一化配置。

    属性：
        ma (nn.MultiheadAttention)：多头注意力模块。
        fc1 (nn.Linear)：前馈网络中的第一个线性层。
        fc2 (nn.Linear)：前馈网络中的第二个线性层。
        norm1 (nn.LayerNorm)：注意力之后的层归一化。
        norm2 (nn.LayerNorm)：前馈网络之后的层归一化。
        dropout (nn.Dropout)：前馈网络使用的 Dropout 层。
        dropout1 (nn.Dropout)：注意力之后的 Dropout 层。
        dropout2 (nn.Dropout)：前馈网络之后的 Dropout 层。
        act (nn.Module)：激活函数。
        normalize_before (bool)：是否在注意力和前馈网络之前进行归一化。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        normalize_before: bool = False,
    ):
        """使用指定参数初始化 TransformerEncoderLayer。

        参数：
            c1 (int)：输入维度。
            cm (int)：前馈网络中的隐藏维度。
            num_heads (int)：注意力头数量。
            dropout (float)：Dropout 概率。
            act (nn.Module)：激活函数。
            normalize_before (bool)：是否在注意力和前馈网络之前进行归一化。
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # 实现前馈网络
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = nn.GELU() if act is None else act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """如果提供位置嵌入，则将其添加到张量。"""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行带后归一化的前向传播。

        参数：
            src (torch.Tensor)：输入张量。
            src_mask (torch.Tensor，可选)：源序列的掩码。
            src_key_padding_mask (torch.Tensor，可选)：每个批次源键的填充掩码。
            pos (torch.Tensor，可选)：位置编码。

        返回：
            (torch.Tensor)：经过注意力和前馈网络后的输出张量。
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行带前归一化的前向传播。

        参数：
            src (torch.Tensor)：输入张量。
            src_mask (torch.Tensor，可选)：源序列的掩码。
            src_key_padding_mask (torch.Tensor，可选)：每个批次源键的填充掩码。
            pos (torch.Tensor，可选)：位置编码。

        返回：
            (torch.Tensor)：经过注意力和前馈网络后的输出张量。
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将输入向前传播通过编码器模块。

        参数：
            src (torch.Tensor)：输入张量。
            src_mask (torch.Tensor，可选)：源序列的掩码。
            src_key_padding_mask (torch.Tensor，可选)：每个批次源键的填充掩码。
            pos (torch.Tensor，可选)：位置编码。

        返回：
            (torch.Tensor)：经过 Transformer 编码器层后的输出张量。
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """用于二维数据并带位置嵌入的 AIFI Transformer 层。

    此类继承 TransformerEncoderLayer，通过添加二维正弦-余弦位置嵌入并正确处理空间维度，使其能够处理二维特征图。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module | None = None,
        normalize_before: bool = False,
    ):
        """使用指定参数初始化 AIFI 实例。

        参数：
            c1 (int)：输入维度。
            cm (int)：前馈网络中的隐藏维度。
            num_heads (int)：注意力头数量。
            dropout (float)：Dropout 概率。
            act (nn.Module)：激活函数。
            normalize_before (bool)：是否在注意力和前馈网络之前进行归一化。
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 AIFI Transformer 层的前向传播。

        参数：
            x (torch.Tensor)：输入张量，形状为 ``[B, C, H, W]``。

        返回：
            (torch.Tensor)：输出张量，形状为 ``[B, C, H, W]``。
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, device=x.device)
        # 将 [B, C, H, W] 展平为 [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0, device=None
    ) -> torch.Tensor:
        """构建二维正弦-余弦位置嵌入。

        参数：
            w (int)：特征图宽度。
            h (int)：特征图高度。
            embed_dim (int)：嵌入维度。
            temperature (float)：正弦和余弦函数使用的温度参数。
            device (torch.device，可选)：构建嵌入网格所使用的设备。

        返回：
            (torch.Tensor)：位置嵌入，形状为 ``[1, h*w, embed_dim]``。
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        # 在输入所在设备上构建网格，避免跟踪图固定使用 CPU 上的 `arange`，从而与 GPU 激活冲突
        # （例如 RT-DETR 的 TorchScript 导出中，跟踪得到的 `arange` 被固定在 CPU 上，会导致 GPU 推理失败）。
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        # 为 CoreML 导出将矩阵乘法固定为 fp32：基于整数位置的 fp16 正弦/余弦计算会累积明显误差。
        out_w = grid_w.flatten()[..., None].float() @ omega[None]
        out_h = grid_h.flatten()[..., None].float() @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer 层，参见 https://arxiv.org/abs/2010.11929（移除 LayerNorm 层以提升性能）。"""

    def __init__(self, c: int, num_heads: int):
        """使用线性变换和多头注意力初始化自注意力机制。

        参数：
            c (int)：输入和输出通道维度。
            num_heads (int)：注意力头数量。
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 x 应用 Transformer 块并返回输出。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：经过 Transformer 层后的输出张量。
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """基于 https://arxiv.org/abs/2010.11929 的视觉 Transformer 块。

    此类实现完整的 Transformer 块，支持使用可选卷积层调整通道数、使用可学习的位置嵌入，并堆叠多个 Transformer 层。

    属性：
        conv (Conv，可选)：输入和输出通道不同时使用的卷积层。
        linear (nn.Linear)：可学习的位置嵌入。
        tr (nn.Sequential)：按顺序排列的 Transformer 层容器。
        c2 (int)：输出通道维度。
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """使用位置嵌入以及指定数量的注意力头和层初始化 Transformer 模块。

        参数：
            c1 (int)：输入通道维度。
            c2 (int)：输出通道维度。
            num_heads (int)：注意力头数量。
            num_layers (int)：Transformer 层数量。
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # 可学习的位置嵌入
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入向前传播通过 Transformer 块。

        参数：
            x (torch.Tensor)：输入张量，形状为 ``[b, c1, h, w]``。

        返回：
            (torch.Tensor)：输出张量，形状为 ``[b, c2, h, w]``。
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(nn.Module):
    """多层感知机中的单个模块。"""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """使用指定的嵌入维度、MLP 维度和激活函数初始化 MLPBlock。

        参数：
            embedding_dim (int)：输入和输出维度。
            mlp_dim (int)：隐藏维度。
            act (type)：激活函数类型。
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 MLPBlock 的前向传播。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：经过 MLPBlock 后的输出张量。
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """简单的多层感知机（也称为 FFN）。

    此类实现可配置的 MLP，包含多个线性层、激活函数以及可选的 Sigmoid 输出激活。

    属性：
        num_layers (int)：MLP 的层数。
        layers (nn.ModuleList)：线性层列表。
        sigmoid (bool)：是否对输出应用 Sigmoid。
        act (nn.Module)：激活函数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: nn.Module = None,
    ):
        """使用指定的输入、隐藏和输出维度以及层数初始化 MLP。

        参数：
            input_dim (int)：输入维度。
            hidden_dim (int)：隐藏维度。
            output_dim (int)：输出维度。
            num_layers (int)：层数。
            act (type)：激活函数类型。
            sigmoid (bool)：是否对输出应用 Sigmoid。
            residual (bool)：是否使用残差连接。
            out_norm (nn.Module，可选)：输出归一化层。
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # 是否对输出应用归一化层
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行整个 MLP 的前向传播。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：经过 MLP 后的输出张量。
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        if getattr(self, "residual", False):
            x = x + orig_x
        x = getattr(self, "out_norm", nn.Identity())(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """二维层归一化模块，参考 Detectron2 和 ConvNeXt 的实现。

    此类对二维特征图执行层归一化：沿通道维度进行归一化，同时保留空间维度。

    属性：
        weight (nn.Parameter)：可学习的缩放参数。
        bias (nn.Parameter)：可学习的偏置参数。
        eps (float)：用于保证数值稳定性的小常数。

    参考：
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """使用给定参数初始化 LayerNorm2d。

        参数：
            num_channels (int)：通道数量。
            eps (float)：用于保证数值稳定性的小常数。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对二维特征图执行层归一化。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：归一化后的输出张量。
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力模块，参考 Deformable-DETR 和 PaddleDetection 的实现。

    此模块实现多尺度可变形注意力，可以在多个尺度的特征上进行注意力计算，并学习采样位置和注意力权重。

    属性：
        im2col_step (int)：im2col 操作的步长。
        d_model (int)：模型维度。
        n_levels (int)：特征层级数量。
        n_heads (int)：注意力头数量。
        n_points (int)：每个特征层级中每个注意力头的采样点数量。
        sampling_offsets (nn.Linear)：生成采样偏移量的线性层。
        attention_weights (nn.Linear)：生成注意力权重的线性层。
        value_proj (nn.Linear)：值投影线性层。
        output_proj (nn.Linear)：输出投影线性层。

    参考：
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """使用给定参数初始化 MSDeformAttn。

        参数：
            d_model (int)：模型维度。
            n_levels (int)：特征层级数量。
            n_heads (int)：注意力头数量。
            n_points (int)：每个特征层级中每个注意力头的采样点数量。
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # 将每个注意力头的维度设置为 2 的幂，通常能提高 CUDA 实现的效率。
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """重置模块参数。"""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行多尺度可变形注意力的前向传播。

        参数：
            query (torch.Tensor)：查询张量，形状为 [bs, query_length, C]。
            refer_bbox (torch.Tensor)：参考框，形状为 [bs, query_length, 1, 2 或 4]，取值范围为 [0, 1]；左上角为
                (0, 0)，右下角为 (1, 1)。尺寸为 1 的轴会沿 n_levels 维度自动广播。
            value (torch.Tensor)：值张量，形状为 [bs, value_length, C]。
            value_shapes (list)：形状为 [n_levels, 2] 的列表，即 [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]。
            value_mask (torch.Tensor，可选)：掩码张量，形状为 [bs, value_length]。True 表示填充元素，False 表示非填充元素。

        返回：
            (torch.Tensor)：输出张量，形状为 [bs, Length_{query}, C]。

        参考：
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        # 将 (n_levels, n_points) 合并到一个轴中，使跟踪得到的每个张量维度数都不超过 5（CoreML 导出所需）；
        # refer_bbox 的形状为 (bs, len_q, 1, 2 或 4)，其中尺寸为 1 的轴会自动广播。
        n_total_points = self.n_levels * self.n_points
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, n_total_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, n_total_points)
        attention_weights = F.softmax(attention_weights, -1)
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            offset_normalizer = offset_normalizer[:, None, :].expand(-1, self.n_points, -1).reshape(n_total_points, 2)
            sampling_locations = refer_bbox[:, :, None, :, :] + sampling_offsets / offset_normalizer
        elif num_points == 4:
            sampling_locations = (
                refer_bbox[:, :, None, :, :2] + sampling_offsets / self.n_points * refer_bbox[:, :, None, :, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """可变形 Transformer 解码器层，参考 PaddleDetection 和 Deformable-DETR 的实现。

    此类实现单个解码器层，包含自注意力、基于多尺度可变形注意力的交叉注意力以及前馈网络。

    属性：
        self_attn (nn.MultiheadAttention)：自注意力模块。
        dropout1 (nn.Dropout)：自注意力之后的 Dropout 层。
        norm1 (nn.LayerNorm)：自注意力之后的层归一化。
        cross_attn (MSDeformAttn)：交叉注意力模块。
        dropout2 (nn.Dropout)：交叉注意力之后的 Dropout 层。
        norm2 (nn.LayerNorm)：交叉注意力之后的层归一化。
        linear1 (nn.Linear)：前馈网络中的第一个线性层。
        act (nn.Module)：激活函数。
        dropout3 (nn.Dropout)：前馈网络中的 Dropout 层。
        linear2 (nn.Linear)：前馈网络中的第二个线性层。
        dropout4 (nn.Dropout)：前馈网络之后的 Dropout 层。
        norm3 (nn.LayerNorm)：前馈网络之后的层归一化。

    参考：
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """使用给定参数初始化 DeformableTransformerDecoderLayer。

        参数：
            d_model (int)：模型维度。
            n_heads (int)：注意力头数量。
            d_ffn (int)：前馈网络维度。
            dropout (float)：Dropout 概率。
            act (nn.Module)：激活函数。
            n_levels (int)：特征层级数量。
            n_points (int)：采样点数量。
        """
        super().__init__()

        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU() if act is None else act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """如果提供位置嵌入，则将其添加到输入张量。"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """执行该层前馈网络部分的前向传播。

        参数：
            tgt (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：经过前馈网络后的输出张量。
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行整个解码器层的前向传播。

        参数：
            embed (torch.Tensor)：输入嵌入。
            refer_bbox (torch.Tensor)：参考框。
            feats (torch.Tensor)：特征图。
            shapes (list)：特征图尺寸列表。
            padding_mask (torch.Tensor，可选)：填充掩码。
            attn_mask (torch.Tensor，可选)：注意力掩码。
            query_pos (torch.Tensor，可选)：查询位置嵌入。

        返回：
            (torch.Tensor)：经过解码器层后的输出张量。
        """
        # 自注意力
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # 交叉注意力
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # 前馈网络
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """基于 PaddleDetection 实现的可变形 Transformer 解码器。

    此类实现完整的可变形 Transformer 解码器，包含多个解码器层，以及用于边界框回归和分类的预测头。

    属性：
        layers (nn.ModuleList)：解码器层列表。
        num_layers (int)：解码器层数量。
        hidden_dim (int)：隐藏维度。
        eval_idx (int)：评估时使用的层索引。

    参考：
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """使用给定参数初始化 DeformableTransformerDecoder。

        参数：
            hidden_dim (int)：隐藏维度。
            decoder_layer (nn.Module)：解码器层模块。
            num_layers (int)：解码器层数量。
            eval_idx (int)：评估时使用的层索引。
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # 解码器嵌入
        refer_bbox: torch.Tensor,  # 参考框
        feats: torch.Tensor,  # 图像特征
        shapes: list,  # 特征图尺寸
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """执行整个解码器的前向传播。

        参数：
            embed (torch.Tensor)：解码器嵌入。
            refer_bbox (torch.Tensor)：参考框。
            feats (torch.Tensor)：图像特征。
            shapes (list)：特征图尺寸列表。
            bbox_head (nn.Module)：边界框预测头。
            score_head (nn.Module)：分数预测头。
            pos_mlp (nn.Module)：位置 MLP。
            attn_mask (torch.Tensor，可选)：注意力掩码。
            padding_mask (torch.Tensor，可选)：填充掩码。

        返回：
            dec_bboxes (torch.Tensor)：解码后的边界框。
            dec_cls (torch.Tensor)：解码后的分类分数。
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
