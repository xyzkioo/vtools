# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""卷积模块。"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # 卷积核、填充、膨胀系数
    """计算使输出保持 same 形状所需的填充大小。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际卷积核尺寸
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p


class Conv(nn.Module):
    """带批归一化和激活函数的标准卷积模块。

    属性：
        conv (nn.Conv2d)：卷积层。
        bn (nn.BatchNorm2d)：批归一化层。
        act (nn.Module)：激活函数层。
        default_act (nn.Module)：默认激活函数（SiLU）。
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化 Conv 层。

        参数：
            c1 (int)：输入通道数。
            c2 (int)：输出通道数。
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int，可选)：填充大小。
            g (int)：分组数量。
            d (int)：膨胀系数。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入张量依次应用卷积、批归一化和激活函数。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：输出张量。
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """在不使用批归一化的情况下，对输入应用卷积和激活函数。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：输出张量。
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """支持卷积融合的简化 RepConv 模块。

    属性：
        conv (nn.Conv2d)：主 3x3 卷积层。
        cv2 (nn.Conv2d)：附加的 1x1 卷积层。
        bn (nn.BatchNorm2d)：批归一化层。
        act (nn.Module)：激活函数层。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化 Conv2 层。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int，可选)：填充大小。
            g (int)：分组数量。
            d (int)：膨胀系数。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # 添加 1x1 卷积

    def forward(self, x):
        """对输入张量依次应用卷积、批归一化和激活函数。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """对输入张量应用融合后的卷积、批归一化和激活函数。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：输出张量。
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """融合并行卷积。"""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """由 1x1 卷积和深度卷积组成的轻量卷积模块。

    此实现基于 PaddleDetection 的 HGNetV2 主干网络。

    属性：
        conv1 (Conv)：1x1 卷积层。
        conv2 (DWConv)：深度卷积层。
    """

    def __init__(self, c1, c2, k=1, act=None):
        """使用给定参数初始化 LightConv 层。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：深度卷积的卷积核尺寸。
            act (nn.Module)：激活函数。
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """对输入张量依次应用两个卷积。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：输出张量。
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """深度卷积模块。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """使用给定参数初始化深度卷积。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：卷积核尺寸。
            s (int)：步长。
            d (int)：膨胀系数。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """深度转置卷积模块。"""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """使用给定参数初始化深度转置卷积。

        参数：
            c1 (int)：输入通道数。
            c2 (int)：输出通道数。
            k (int)：卷积核尺寸。
            s (int)：步长。
            p1 (int)：填充。
            p2 (int)：输出填充。
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """带可选批归一化和激活函数的转置卷积模块。

    属性：
        conv_transpose (nn.ConvTranspose2d)：转置卷积层。
        bn (nn.BatchNorm2d | nn.Identity)：批归一化层。
        act (nn.Module)：激活函数层。
        default_act (nn.Module)：默认激活函数（SiLU）。
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """使用给定参数初始化 ConvTranspose 层。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int)：填充大小。
            bn (bool)：是否使用批归一化。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入依次应用转置卷积、批归一化和激活函数。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """对输入应用转置卷积和激活函数。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：输出张量。
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """用于聚合特征信息的 Focus 模块。

    此模块将输入张量切分为 4 个部分，并沿通道维度将它们拼接起来。

    属性：
        conv (Conv)：卷积层。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """使用给定参数初始化 Focus 模块。

        参数：
            c1 (int)：输入通道数。
            c2 (int)：输出通道数。
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int，可选)：填充大小。
            g (int)：分组数量。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """对输入张量执行 Focus 操作和卷积。

        输入形状为 ``(B, C, H, W)``，输出形状为 ``(B, c2, H/2, W/2)``。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # 返回 self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost 卷积模块。

    通过低开销操作用更少的参数生成更多特征。

    属性：
        cv1 (Conv)：主卷积。
        cv2 (Conv)：低开销操作卷积。

    参考：
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """使用给定参数初始化 Ghost 卷积模块。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：卷积核尺寸。
            s (int)：步长。
            g (int)：分组数量。
            act (bool | nn.Module)：激活函数。
        """
        super().__init__()
        c_ = c2 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """对输入张量应用 Ghost 卷积。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor)：拼接后的特征张量。
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """支持训练模式和部署模式的 RepConv 模块。

    此模块用于 RT-DETR，并可在推理期间融合卷积以提高效率。

    属性：
        conv1 (Conv)：3x3 卷积。
        conv2 (Conv)：1x1 卷积。
        bn (nn.BatchNorm2d，可选)：恒等分支的批归一化层。
        act (nn.Module)：激活函数。
        default_act (nn.Module)：默认激活函数（SiLU）。

    参考：
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """使用给定参数初始化 RepConv 模块。

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int)：填充大小。
            g (int)：分组数量。
            d (int)：膨胀系数。
            act (bool | nn.Module)：激活函数。
            bn (bool)：是否在恒等分支中使用批归一化。
            deploy (bool)：是否使用用于推理的部署模式。
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """执行部署模式下的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """执行训练模式下的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """通过融合卷积计算等效卷积核和偏置。

        返回：
            (torch.Tensor)：等效卷积核。
            (torch.Tensor)：等效偏置。
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """将 1x1 卷积核填充为 3x3 尺寸。

        参数：
            kernel1x1 (torch.Tensor)：1x1 卷积核。

        返回：
            (torch.Tensor)：填充后的 3x3 卷积核。
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """将批归一化层与卷积权重融合。

        参数：
            branch (Conv | nn.BatchNorm2d | None)：要融合的分支。

        返回：
            kernel (torch.Tensor)：融合后的卷积核。
            bias (torch.Tensor)：融合后的偏置。
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """创建单个等效卷积以融合卷积，用于推理。"""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """用于特征重校准的通道注意力模块。

    根据全局平均池化结果为各通道应用注意力权重。

    属性：
        pool (nn.AdaptiveAvgPool2d)：全局平均池化。
        fc (nn.Conv2d)：通过 1x1 卷积实现的全连接层。
        act (nn.Sigmoid)：用于生成注意力权重的 Sigmoid 激活函数。

    参考：
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """初始化通道注意力模块。

        参数：
            channels (int)：输入通道数。
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用通道注意力。

        参数：
            x (torch.Tensor)：输入张量。

        返回：
            (torch.Tensor)：应用通道注意力后的输出张量。
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """用于特征重校准的空间注意力模块。

    根据通道统计信息为空间维度应用注意力权重。

    属性：
        cv1 (nn.Conv2d)：空间注意力卷积层。
        act (nn.Sigmoid)：用于生成注意力权重的 Sigmoid 激活函数。
    """

    def __init__(self, kernel_size=7):
        """初始化空间注意力模块。

        参数：
            kernel_size (int)：卷积核尺寸，可取 3 或 7。
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """对输入张量应用空间注意力。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor)：应用空间注意力后的输出张量。
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """卷积块注意力模块（CBAM）。

    结合通道注意力和空间注意力机制，全面细化特征。

    属性：
        channel_attention (ChannelAttention)：通道注意力模块。
        spatial_attention (SpatialAttention)：空间注意力模块。
    """

    def __init__(self, c1, kernel_size=7):
        """使用给定参数初始化 CBAM。

        参数：
            c1 (int): 输入通道数.
            kernel_size (int)：空间注意力卷积核的尺寸。
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """依次对输入张量应用通道注意力和空间注意力。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor)：应用注意力后的输出张量。
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """沿指定维度拼接张量列表。

    属性：
        d (int)：拼接张量所沿的维度。
    """

    def __init__(self, dimension=1):
        """初始化 Concat 模块。

        参数：
            dimension (int)：拼接张量所沿的维度。
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """沿指定维度拼接输入张量。

        参数：
            x (list[torch.Tensor])：输入张量列表。

        返回：
            (torch.Tensor)：拼接后的张量。
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """返回输入中的指定索引项。

    属性：
        index (int)：要从输入中选择的索引。
    """

    def __init__(self, index=0):
        """初始化 Index 模块。

        参数：
            index (int)：要从输入中选择的索引。
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """从输入中选择并返回指定索引项。

        参数：
            x (list[torch.Tensor])：输入张量列表。

        返回：
            (torch.Tensor)：选中的张量。
        """
        return x[self.index]
