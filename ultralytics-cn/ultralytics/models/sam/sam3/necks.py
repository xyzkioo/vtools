# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""颈部网络是视觉骨干网络与检测模型其余部分之间的接口。"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class Sam3DualViTDetNeck(nn.Module):
    """实现 ViTDet 风格简单 FPN 的颈部，并支持双颈部结构（用于 SAM3 和 SAM2）。"""

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """ViTDet 风格的 SimpleFPN 颈部，基于 detectron2 做了少量调整。

        支持“双颈部”设置：使用两个结构相同但权重不同的颈部，分别用于 SAM3 和 SAM2。

        参数：
            trunk (nn.Module): 骨干网络。
            position_encoding (nn.Module): 要使用的位置编码。
            d_model (int): 模型维度。
            scale_factors (tuple): 每个 FPN 层级的缩放因子。
            add_sam2_neck (bool): 是否为 SAM2 添加第二个颈部。
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # 假设 sam2 neck 只是原始 neck 的克隆
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """从颈部网络获取特征图和位置编码。"""
        xs = self.trunk(tensor_list)
        x = xs[-1]  # simpleFPN
        sam3_out, sam3_pos = self.sam_forward_feature_levels(x, self.convs)
        if self.sam2_convs is None:
            return sam3_out, sam3_pos, None, None
        sam2_out, sam2_pos = self.sam_forward_feature_levels(x, self.sam2_convs)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    def sam_forward_feature_levels(
        self, x: torch.Tensor, convs: nn.ModuleList
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """运行颈部卷积，并计算每个特征层级的位置编码。"""
        outs, poss = [], []
        for conv in convs:
            feat = conv(x)
            outs.append(feat)
            poss.append(self.position_encoding(feat).to(feat.dtype))
        return outs, poss

    def set_imgsz(self, imgsz: list[int] | None = None):
        """为主干骨干网络设置图像尺寸。"""
        imgsz = imgsz if imgsz is not None else [1008, 1008]
        self.trunk.set_imgsz(imgsz)
