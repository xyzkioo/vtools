# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import uniform_

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"


def _get_clones(module, n):
    """从给定模块创建包含多个副本的列表。

    参数：
        module (nn.Module)：要复制的模块。
        n (int)：要创建的副本数量。

    返回：
        (nn.ModuleList)：包含 n 个输入模块副本的 ModuleList。

    示例：
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """根据给定的先验概率初始化卷积层或全连接层的偏置值。

    此函数使用逆 Sigmoid（logit）函数，根据先验概率计算偏置初始值。它通常用于目标检测模型，为分类层设置
    指定的正样本预测概率。

    参数：
        prior_prob (float，可选)：用于初始化偏置的先验概率。

    返回：
        (float)：根据先验概率计算得到的偏置初始值。

    示例：
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        偏置初始值：-4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # 返回偏置初始值


def linear_init(module):
    """初始化线性模块的权重和偏置。

    此函数根据输出维度计算均匀分布的边界，并使用该分布初始化线性模块的权重。如果模块包含偏置，也会一并
    初始化。

    参数：
        module (nn.Module)：要初始化的线性模块。

    示例：
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """计算张量的逆 Sigmoid 函数。

    此函数对张量应用 Sigmoid 函数的逆运算，可用于各种神经网络操作，尤其适用于注意力机制和坐标变换。

    参数：
        x (torch.Tensor)：输入张量，取值范围为 [0, 1]。
        eps (float，可选)：用于避免数值不稳定的小 epsilon 值。

    返回：
        (torch.Tensor)：应用逆 Sigmoid 函数后的张量。

    示例：
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        张量([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """在 PyTorch 中实现多尺度可变形注意力。

    此实现将 ``(num_levels, num_points)`` 两个轴折叠为单个 ``num_total_points`` 轴，使跟踪得到的每个张量
    的秩都不超过 5，这是 CoreML MIL 转换器支持的最大秩。在 CUDA 和 CPU 上，它在数值上等价于秩为 6 的参考
    实现。

    参数：
        value (torch.Tensor)：值张量，形状为 ``(bs, num_keys, num_heads, embed_dims)``。
        value_spatial_shapes (list)：每个层级的空间形状，格式为 ``[(H_0, W_0), ..., (H_{L-1}, W_{L-1})]``。
        sampling_locations (torch.Tensor)：采样位置，形状为 ``(bs, num_queries, num_heads, num_levels * num_points, 2)``。
        attention_weights (torch.Tensor)：注意力权重，形状为 ``(bs, num_queries, num_heads, num_levels * num_points)``。

    返回：
        (torch.Tensor)：输出张量，形状为 ``(bs, num_queries, num_heads * embed_dims)``。

    参考：
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_total_points, _ = sampling_locations.shape
    num_points = num_total_points // len(value_spatial_shapes)

    # (bs, num_keys, num_heads, embed_dims) -> 每个层级对应一个 (bs*num_heads, embed_dims, H*W) 元组
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in value_spatial_shapes], dim=-1)
    # 映射到 [-1, 1] 范围的 grid_sample 坐标，并按层级拆分为 (bs*num_heads, num_queries, num_points, 2) 元组
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(num_points, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * num_heads, embed_dims, h, w)
        sampling_value_list.append(
            F.grid_sample(value_l, sampling_grids[level], mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * num_heads, 1, num_queries, num_total_points)
    output = (
        (torch.cat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
