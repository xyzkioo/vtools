# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""网络模块。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """分布焦点损失（DFL）的积分模块。

    该模块提出于 Generalized Focal Loss：https://arxiv.org/abs/2006.04388
    """

    def __init__(self, c1: int = 16):
        """使用给定的输入通道数量初始化卷积层。

        参数：
            c1 (int)：输入通道数量。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 DFL 模块应用于输入张量并返回变换后的输出。"""
        b, _, a = x.shape  # 批次、通道、锚框
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # 返回 self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO 分割模型的掩码原型模块。"""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """使用指定的原型数量和掩码数量初始化 Ultralytics YOLO 掩码原型模块。

        参数：
            c1 (int)：输入通道数量。
            c_ (int)：中间通道数量。
            c2 (int)：输出通道数量（原型数量）。
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对上采样后的输入图像执行前向传播。"""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """PPHGNetV2 的 StemBlock，包含 5 个卷积层和一个 maxpool2d。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """初始化 PPHGNetV2 的 StemBlock。

        参数：
            c1 (int)：输入通道数量。
            cm (int)：中间通道数量。
            c2 (int)：输出通道数量。
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 PPHGNetV2 主干网络层的前向传播。"""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """PPHGNetV2 的 HG_Block，包含 2 个卷积层和 LightConv。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module | None = None,
    ):
        """使用指定参数初始化 HGBlock。

        参数：
            c1 (int)：输入通道数量。
            cm (int)：中间通道数量。
            c2 (int)：输出通道数量。
            k (int)：卷积核尺寸。
            n (int)：LightConv 或 Conv 模块数量。
            lightconv (bool)：是否使用 LightConv。
            shortcut (bool)：是否使用残差连接。
            act (nn.Module)：激活函数。
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # 压缩卷积
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # 激励卷积
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 PPHGNetV2 主干网络层的前向传播。"""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """空间金字塔池化（SPP）层，参见 https://arxiv.org/abs/1406.4729。"""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """使用输入/输出通道数和池化卷积核尺寸初始化 SPP 层。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            k (tuple)：最大池化使用的卷积核尺寸。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 SPP 层的前向传播，并完成空间金字塔池化。"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Glenn Jocher 为 YOLOv5 提出的快速空间金字塔池化（SPPF）层。"""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """使用给定的输入/输出通道数和卷积核尺寸初始化 SPPF 层。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            k (int)：卷积核尺寸。
            n (int)：池化迭代次数。
            shortcut (bool)：是否使用残差连接。

        注意：
            此模块等价于 SPP(k=(5, 9, 13))。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入执行连续池化操作，并返回拼接后的特征图。"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """包含 1 个卷积层的 CSP 瓶颈模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """初始化包含 1 个卷积层的 CSP 瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：卷积层数量。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用卷积和残差连接。"""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """包含 2 个卷积层的 CSP 瓶颈模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 2 个卷积层的 CSP 瓶颈模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 模块数量。
            shortcut (bool): 是否使用残差连接。
            g (int): 卷积分组数量。
            e (float): 通道扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # 可选 act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # 或使用 SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行包含 2 个卷积层的 CSP 瓶颈模块前向传播。"""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """包含 2 个卷积层的快速 CSP 瓶颈实现。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """初始化包含 2 个卷积层的 CSP 瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Bottleneck 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 C2f 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """使用 split() 而不是 chunk() 执行前向传播。"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """包含 3 个卷积层的 CSP 瓶颈模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 3 个卷积层的 CSP 瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Bottleneck 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # 可选 act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行包含 3 个卷积层的 CSP 瓶颈模块前向传播。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """使用交叉卷积的 C3 模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """使用交叉卷积初始化 C3 模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Bottleneck 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3 模块。"""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """使用 RepConv 模块初始化 RepC3。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：RepConv 模块数量。
            e (float)：扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RepC3 模块的前向传播。"""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """使用 TransformerBlock() 的 C3 模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """使用 TransformerBlock 初始化 C3 模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Transformer 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """使用 GhostBottleneck() 的 C3 模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """使用 GhostBottleneck 初始化 C3 模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Ghost 瓶颈模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost 瓶颈模块，参见 https://github.com/huawei-noah/Efficient-AI-Backbones。"""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """初始化 Ghost 瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            k (int)：卷积核尺寸。
            s (int)：步长。
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # 逐点卷积
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 深度卷积
            GhostConv(c_, c2, 1, 1, act=False),  # 线性逐点卷积
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用跳跃连接并执行相加。"""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """标准瓶颈模块。"""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """初始化标准瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            k (tuple)：卷积核尺寸。
            e (float)：扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入应用瓶颈模块，并根据配置使用可选的残差连接。"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP 瓶颈模块，参见 https://github.com/WongKinYiu/CrossStagePartialNetworks。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化 CSP 瓶颈模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Bottleneck 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # 应用于 cat(cv2, cv3) 的结果
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入应用包含 4 个卷积层的 CSP 瓶颈模块。"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """使用标准卷积层构成的 ResNet 模块。"""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """初始化 ResNet 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            s (int): 步幅。
            e (int): 通道扩展比例。
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ResNet 模块的前向传播。"""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """由多个 ResNet 模块组成的 ResNet 层。"""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """初始化 ResNet 层。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            s (int): 步幅。
            is_first (bool): 是否为第一层。
            n (int): ResNet 模块数量。
            e (int): 通道扩展比例。
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ResNet 层的前向传播。"""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """最大 Sigmoid 注意力模块。"""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """初始化 MaxSigmoidAttnBlock。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            nh (int): 注意力头数量。
            ec (int): 嵌入通道数。
            gc (int): 引导向量通道数。
            scale (bool): 是否使用可学习的缩放参数。
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """执行 MaxSigmoidAttnBlock 的前向传播。

        参数：
            x (torch.Tensor): 输入张量.
            guide (torch.Tensor): 引导张量。

        返回：
            (torch.Tensor): 经过注意力处理后的输出张量。
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """带有额外注意力模块的 C2f 模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带有注意力机制的 C2f 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 模块数量。
            ec (int): 注意力嵌入通道数。
            nh (int): 注意力头数量。
            gc (int): 注意力引导通道数。
            shortcut (bool): 是否使用残差连接。
            g (int): 卷积分组数量。
            e (float): 通道扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # 可选 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """执行带有注意力机制的 C2f 层前向传播。

        参数：
            x (torch.Tensor): 输入张量.
            guide (torch.Tensor): 注意力机制使用的引导张量。

        返回：
            (torch.Tensor): 处理后的输出张量。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """使用 split() 而不是 chunk() 执行前向传播。

        参数：
            x (torch.Tensor): 输入张量.
            guide (torch.Tensor): 注意力机制使用的引导张量。

        返回：
            (torch.Tensor): 处理后的输出张量。
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn：使用图像相关信息增强文本嵌入。"""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """初始化 ImagePoolingAttn 模块。

        参数：
            ec (int): 嵌入通道数。
            ch (tuple): 各特征图的通道数。
            ct (int): 文本嵌入的通道数。
            nh (int): 注意力头数量。
            k (int): 池化窗口尺寸。
            scale (bool): 是否使用可学习的缩放参数。
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """执行 ImagePoolingAttn 的前向传播。

        参数：
            x (列表[torch.Tensor]): 输入特征图列表。
            text (torch.Tensor): 文本嵌入。

        返回：
            (torch.Tensor): 增强后的文本嵌入。
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """用于视觉语言模型区域与文本相似度计算的对比学习头。"""

    def __init__(self):
        """初始化用于区域与文本相似度计算的 ContrastiveHead。"""
        super().__init__()
        # 注意：使用 -10.0，使初始分类损失与其他损失保持一致
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """执行对比学习的前向计算。

        参数：
            x (torch.Tensor): 图像特征。
            w (torch.Tensor): 文本特征。

        返回：
            (torch.Tensor): 相似度分数。
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """使用批归一化代替 L2 归一化的对比学习头。

    参数：
        embed_dims (int): 文本和图像特征的嵌入维度。
    """

    def __init__(self, embed_dims: int):
        """初始化 BNContrastiveHead。

        参数：
            embed_dims (int): 特征嵌入维度。
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # 注意：使用 -10.0，使初始分类损失与其他损失保持一致
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # 使用 -1.0 更稳定
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """融合 BNContrastiveHead 模块中的批归一化层。"""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """融合后直接返回图像特征，不对其进行修改。"""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """执行带批归一化的对比学习前向计算。

        参数：
            x (torch.Tensor): 图像特征。
            w (torch.Tensor): 文本特征。

        返回：
            (torch.Tensor): 相似度分数。
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """RepBottleneck 瓶颈模块。"""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """初始化 RepBottleneck。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用残差连接。
            g (int): 卷积分组数量。
            k (tuple): 卷积核尺寸。
            e (float): 通道扩展比例。
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """用于高效特征提取的可重复跨阶段局部网络（RepCSP）模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化 RepCSP 层。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): RepBottleneck 模块数量。
            shortcut (bool): 是否使用残差连接。
            g (int): 卷积分组数量。
            e (float): 通道扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN 模块。"""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """初始化 CSP-ELAN 层。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            c4 (int): RepCSP 模块的中间通道数。
            n (int): RepCSP 模块数量。
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RepNCSPELAN4 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """使用 split() 而不是 chunk() 执行前向传播。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """包含 4 个卷积层的 ELAN1 模块。"""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """初始化 ELAN1 层.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            c4 (int): 卷积层的中间通道数。
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv 下采样卷积模块。"""

    def __init__(self, c1: int, c2: int):
        """初始化 AConv 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 AConv 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown 下采样模块。"""

    def __init__(self, c1: int, c2: int):
        """初始化 ADown 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ADown 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN 模块。"""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """初始化 SPP-ELAN 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            k (int): 最大池化窗口尺寸。
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 SPPELAN 层的前向传播。"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear 分支卷积模块。"""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """初始化 CBLinear 模块.

        参数：
            c1 (int): 输入通道数。
            c2s (列表[int]): 各分支的输出通道数列表。
            k (int): 卷积核尺寸。
            s (int): 步幅。
            p (int | None): 填充大小。
            g (int): 分组数量。
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """执行 CBLinear 层的前向传播。"""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse 特征融合模块。"""

    def __init__(self, idx: list[int]):
        """初始化 CBFuse 模块.

        参数：
            idx (列表[int]): 特征选择所使用的索引。
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """执行 CBFuse 层的前向传播。

        参数：
            xs (列表[torch.Tensor]): 输入张量列表。

        返回：
            (torch.Tensor): 融合后的输出张量。
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """使用 3 个卷积层实现的快速 CSP 瓶颈模块。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """初始化包含 3 个卷积层的 CSP 瓶颈层。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 模块数量。
            shortcut (bool): 是否使用残差连接。
            g (int): 卷积分组数量。
            e (float): 通道扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # 可选 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 C3f 层的前向传播。"""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """使用 2 个卷积层实现的快速 CSP 瓶颈模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """初始化 C3k2 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 模块数量。
            c3k (bool): 是否使用 C3k 模块。
            e (float): 通道扩展比例。
            attn (bool): 是否使用注意力模块。
            g (int): 卷积分组数量。
            shortcut (bool): 是否使用残差连接。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k 是一种 CSP 瓶颈模块，可自定义卷积核尺寸以提取神经网络特征。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """初始化 C3k 模块。

        参数：
            c1 (int)：输入通道数量。
            c2 (int)：输出通道数量。
            n (int)：Bottleneck 模块数量。
            shortcut (bool)：是否使用残差连接。
            g (int)：卷积分组数量。
            e (float)：扩展比例。
            k (int): 卷积核尺寸。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW 表示 RepVGG 架构中的深度可分离卷积模块。"""

    def __init__(self, ed: int) -> None:
        """初始化 RepVGGDW 模块.

        参数：
            ed (int): 输入和输出通道数。
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RepVGGDW 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过深度可分离卷积后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """执行融合后的 RepVGGDW 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过深度可分离卷积后的输出张量。
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """融合 RepVGGDW 模块中的卷积层。

        此方法会融合卷积层，并相应更新权重和偏置。
        """
        if not hasattr(self, "conv1"):
            return  # 已完成融合
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """紧凑型倒置模块（CIB）。

    参数：
        c1 (int): 输入通道数.
        c2 (int): 输出通道数.
            shortcut (bool, 可选): 是否添加残差连接，默认为 True。
            e (float, 可选): 隐藏通道的缩放因子，默认为 0.5。
            lk (bool, 可选): 第三个卷积层是否使用 RepVGGDW，默认为 False。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """初始化 CIB 模块。

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用残差连接。
            e (float): 通道扩展比例。
            lk (bool): 是否使用 RepVGGDW。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 CIB 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 输出张量.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """由 C2f 和 CIB 模块组成的卷积模块。

    参数：
        c1 (int): 输入通道数.
        c2 (int): 输出通道数.
        n (int, 可选): 要堆叠的 CIB 模块数量，默认为 1。
        shortcut (bool, 可选): 是否使用残差连接，默认为 False。
        lk (bool, 可选): 是否使用大卷积核，默认为 False。
        g (int, 可选): 分组卷积的分组数量，默认为 1。
        e (float, 可选): CIB 模块的通道扩展比例，默认为 0.5。
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """初始化 C2fCIB 模块.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): CIB 模块数量。
            shortcut (bool): 是否使用残差连接。
            lk (bool): 是否使用大卷积核。
            g (int): 卷积分组数量。
            e (float): 通道扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """对输入张量执行自注意力计算的 Attention 模块。

    参数：
        dim (int): 输入张量的维度。
        num_heads (int): 注意力头数量。
        attn_ratio (float): 注意力键维度与注意力头维度的比例。

    属性：
        num_heads (int): 注意力头数量。
        head_dim (int): 每个注意力头的维度。
        key_dim (int): 注意力键的维度。
        scale (float): 注意力分数的缩放因子。
        qkv (Conv): 用于计算查询、键和值的卷积层。
        proj (Conv): 用于投影注意力结果的卷积层。
        pe (Conv): 用于位置编码的卷积层。
    """

    format = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """初始化多头注意力模块。

        参数：
            dim (int): 输入维度。
            num_heads (int): 注意力头数量。
            attn_ratio (float): 注意力键维度的比例。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 Attention 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 经过自注意力计算后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
            x = x.transpose(-2, -1).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        else:
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """在神经网络中实现位置敏感注意力的 PSABlock 模块。

    此类封装了多头注意力和前馈神经网络层，并支持可选的残差连接。

    属性：
        attn (Attention): 多头注意力模块。
        ffn (nn.Sequential): 前馈神经网络模块。
        add (bool): 是否添加残差连接的标志。

    方法：
        forward：执行 PSABlock 的前向传播，并应用注意力和前馈层。

    示例：
        创建 PSABlock 并执行前向传播
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """初始化 PSABlock。

        参数：
            c (int): 输入和输出通道数。
            attn_ratio (float): 注意力键的维度比例。
            num_heads (int): 注意力头数量。
            shortcut (bool): 是否使用残差连接。
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 PSABlock 的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """在神经网络中实现位置敏感注意力的 PSA 模块。

    此模块将位置敏感注意力和前馈网络应用于输入张量，以增强特征提取和处理能力。

    属性：
        c (int): 初始卷积处理后的隐藏通道数。
        cv1 (Conv): 将输入通道数调整为 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数调整为 c1 的 1x1 卷积层。
        attn (Attention): 用于位置敏感注意力的 Attention 模块。
        ffn (nn.Sequential): 用于进一步处理的前馈网络。

    方法：
        forward：将位置敏感注意力和前馈网络应用于输入张量。

    示例：
        创建 PSA 模块并将其应用于输入张量
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """初始化 PSA 模块.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            e (float): 通道扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 PSA 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """带有注意力机制、用于增强特征提取和处理能力的 C2PSA 模块。

    此模块通过注意力机制增强特征提取和处理能力，并包含一系列用于自注意力和前馈运算的 PSABlock 模块。

    属性：
        c (int): 隐藏通道数。
        cv1 (Conv): 将输入通道数调整为 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数调整为 c1 的 1x1 卷积层。
        m (nn.Sequential): 用于注意力和前馈运算的 PSABlock 模块序列。

    方法：
        forward：执行 C2PSA 模块的前向传播，应用注意力和前馈运算。

    示例：
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    注意：
        此模块本质上与 PSA 模块相同，但经过重构后可以堆叠更多 PSABlock 模块。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2PSA 模块.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块数量。
            e (float): 通道扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过一系列 PSA 模块处理输入张量。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """使用 PSA 模块增强特征提取能力的 C2fPSA 模块。

    此类在 C2f 模块中加入 PSA 模块，以改进注意力机制和特征提取能力。

    属性：
        c (int): 隐藏通道数。
        cv1 (Conv): 将输入通道数调整为 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数调整为 c2 的 1x1 卷积层。
        m (nn.ModuleList): 用于特征提取的 PSABlock 模块列表。

    方法：
        forward：执行 C2fPSA 模块的前向传播。
        forward_split：使用 split() 而不是 chunk() 执行前向传播。

    示例：
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2fPSA 模块.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块数量。
            e (float): 通道扩展比例。
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """使用可分离卷积进行下采样的 SCDown 模块。

    此模块结合逐点卷积和深度卷积执行下采样，在保留通道信息的同时高效降低输入张量的空间维度。

    属性：
        cv1 (Conv): 用于减少通道数的逐点卷积层。
        cv2 (Conv): 用于执行空间下采样的深度卷积层。

    方法：
        forward：将 SCDown 模块应用于输入张量。

    示例：
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """初始化 SCDown 模块.

        参数：
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核尺寸。
            s (int): 步幅。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量执行卷积和下采样。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 下采样后的输出张量。
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """用于加载任意 torchvision 模型的 TorchVision 模块。

    此类可以从 torchvision 库加载模型，可选择加载预训练权重，并通过截断或展开层来自定义模型。

    参数：
        模型 (str): 要加载的 torchvision 模型名称。
        权重 (str, 可选): 要加载的预训练权重，默认为 "DEFAULT"。
        unwrap (bool, 可选): 将模型展开为一个 sequential，其中包含除最后 `truncate` 层以外的所有层。
        truncate (int, 可选): 当 `unwrap` 为 True 时，从末尾截断的层数，默认为 2。
        split (bool, 可选): 是否将中间子模块的输出作为列表返回，默认为 False。

    属性：
        m (nn.Module): 加载的 torchvision 模型，可能已被截断或展开。
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """从 torchvision 加载模型和权重。

        参数：
            模型 (str): 要加载的 torchvision 模型名称。
            权重 (str): 要加载的预训练权重。
            unwrap (bool): 是否展开模型。
            truncate (int): 截断层数。
            split (bool): 是否拆分输出。
        """
        import torchvision  # 仅在此处导入，以加快 `import ultralytics` 的速度

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # EfficientNet、Swin 等部分模型需要展开第二层
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行模型的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor | 列表[torch.Tensor]): 输出张量或张量列表。
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """为 YOLO 模型提供高效注意力机制的区域注意力模块。

    此模块实现基于区域的注意力机制，以具有空间感知能力的方式处理输入特征，
    因此特别适用于目标检测任务。

    属性：
        area (int): 将特征图划分成的区域数量。
        num_heads (int): 注意力机制划分出的注意力头数量。
        head_dim (int): 每个注意力头的维度。
        qkv (Conv): 用于计算查询、键和值张量的卷积层。
        proj (Conv): 投影卷积层。
        pe (Conv): 位置编码卷积层。

    方法：
        forward：将区域注意力应用于输入张量。

    示例：
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """初始化用于 YOLO 模型的区域注意力模块。

        参数：
            dim (int): 隐藏通道数。
            num_heads (int): 注意力机制划分出的注意力头数量。
            area (int): 将特征图划分成的区域数量。
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """为旧检查点补充缺失的 all_head_dim 属性。"""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过区域注意力处理输入张量。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过区域注意力处理后的输出张量。
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q * (self.head_dim**-0.5)).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """用于 YOLO 模型高效特征提取的区域注意力模块。

    此模块将区域注意力机制与前馈网络结合，用于处理特征图。
    它采用新颖的基于区域的注意力方式，在保持有效性的同时比传统自注意力更加高效。

    属性：
        attn (AAttn): 用于处理空间特征的区域注意力模块。
        mlp (nn.Sequential): 用于特征变换的多层感知机。

    方法：
        _init_weights：使用截断正态分布初始化模块权重。
        forward：对输入张量执行区域注意力和前馈处理。

    示例：
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """初始化区域注意力模块。

        参数：
            dim (int): 输入通道数.
            num_heads (int): 注意力机制划分出的注意力头数量。
            mlp_ratio (float): MLP 隐藏维度的扩展比例。
            area (int): 将特征图划分成的区域数量。
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """使用截断正态分布初始化权重。

        参数：
            m (nn.Module): 要初始化的模块。
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 ABlock 的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 经过区域注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """使用区域注意力机制增强特征提取能力的 Area-Attention C2f 模块。

    此模块在 C2f 架构中加入区域注意力和 ABlock 层，以改进特征处理能力。
    它同时支持区域注意力模式和标准卷积模式。

    属性：
        cv1 (Conv): 将输入通道数减少为隐藏通道数的初始 1x1 卷积层。
        cv2 (Conv): 处理拼接特征的最终 1x1 卷积层。
        gamma (nn.Parameter | None): 使用区域注意力时用于残差缩放的可学习参数。
        m (nn.ModuleList): 用于特征处理的 ABlock 或 C3k 模块列表。

    方法：
        forward：通过区域注意力或标准卷积路径处理输入。

    示例：
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """初始化 Area-Attention C2f 模块.

        参数：
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            n (int): 要堆叠的 ABlock 或 C3k 模块数量。
            a2 (bool): 是否使用区域注意力模块；为 False 时改用 C3k 模块。
            area (int): 将特征图划分成的区域数量。
            residual (bool): 是否使用带可学习 gamma 参数的残差连接。
            mlp_ratio (float): MLP 隐藏维度的扩展比例。
            e (float): 隐藏通道的通道扩展比例。
            g (int): 分组卷积的分组数量。
            shortcut (bool): C3k 模块中是否使用残差连接。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        assert c_ % 32 == 0, "ABlock 的维度必须是 32 的倍数。"

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 A2C2f 层的前向传播。

        参数：
            x (torch.Tensor): 输入张量.

        返回：
            (torch.Tensor): 处理后的输出张量。
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """用于基于 Transformer 架构的 SwiGLU 前馈网络。"""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """使用输入维度、输出维度和扩展因子初始化 SwiGLU FFN。

        参数：
            gc (int): 引导通道数。
            ec (int): 嵌入通道数。
            e (int): 扩展因子。
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征执行 SwiGLU 变换。"""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """神经网络模块的残差连接封装器。"""

    def __init__(self, m: nn.Module) -> None:
        """使用被封装模块初始化残差模块。

        参数：
            m (nn.Module): 要使用残差连接封装的模块。
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # 对于使用 l 缩放的模型，请将初始化方式改为
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征应用残差连接。"""
        return x + self.m(x)


class SAVPE(nn.Module):
    """用于增强特征的空间感知视觉提示嵌入模块。"""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """使用通道数、中间通道数和嵌入维度初始化 SAVPE 模块。

        参数：
            ch (列表[int]): 输入通道维度列表。
            c3 (int): 中间通道数。
            embed (int): 嵌入维度。
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """处理输入特征和视觉提示，生成增强后的嵌入。"""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """用于分割模型的 Ultralytics YOLO26 掩码原型模块。"""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """使用指定数量的原型和掩码初始化 Ultralytics YOLO 掩码原型模块。

        参数：
            ch (tuple): 主干网络特征图的通道数元组。
            c_ (int): 中间通道数。
            c2 (int): 输出通道数，即原型数量。
            nc (int): 语义分割的类别数量。
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """融合多尺度特征图并生成掩码原型。"""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            # 固定缩放比例（P4/P5 -> P3）可使上采样在动态形状 CoreML 导出时保持静态
            up_feat = F.interpolate(up_feat, scale_factor=2 ** (i + 1), mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            semantic = self.semseg(feat)
            return (p, semantic)
        return p

    def fuse(self):
        """移除语义分割头，使模型可用于推理。"""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP：一种基于流的生成模型。

    参考：
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """获取单个可逆映射中的缩放模型。"""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """获取单个可逆映射中的平移模型。"""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    def __init__(self):
        super().__init__()

        # loc/cov 已不再读取（log_prob 中的先验是闭式标准正态分布），但仍保留注册，
        # 这样 8.4.126 之前保存的检查点仍可恢复：EMA 状态会以严格模式加载。
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """初始化 模型 权重."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """将数据空间映射到潜空间，并计算雅可比矩阵的对数行列式。
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """计算给定数据空间样本的对数概率。"""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        # 二维闭式 log N(z; 0, I)；使用 fp32 可避免 z**2 在 AMP 下溢出。
        return -0.5 * (z.float() ** 2).sum(-1) - math.log(2 * math.pi) + log_det
