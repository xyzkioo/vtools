# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""提供用于组合视觉骨干网络和语言骨干网络的工具。"""

from __future__ import annotations

from copy import copy

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .necks import Sam3DualViTDetNeck


class SAM3VLBackbone(nn.Module):
    """此骨干网络在不进行融合的情况下组合视觉骨干网络和语言骨干网络，因此更接近于同时管理两个骨干网络的便捷封装器。

    它增加了激活检查点和编译功能的支持。
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        """初始化骨干网络组合器。

        参数：
            visual (Sam3DualViTDetNeck): 要使用的视觉骨干网络。
            text (nn.Module): 要使用的文本编码器。
            compile_visual (bool): 是否使用 `torch.compile` 编译视觉骨干网络。
            act_ckpt_whole_vision_backbone (bool): 是否对整个视觉骨干网络的激活值使用检查点。
            act_ckpt_whole_language_backbone (bool): 是否对整个语言骨干网络的激活值使用检查点。
            scalp (int): 从骨干网络输出中丢弃的末尾（最低分辨率）特征层级数量。
        """
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        # 允许对整个视觉和语言骨干网络执行激活检查点
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: torch.Tensor,
        captions: list[str],
        input_boxes: torch.Tensor = None,
        additional_text: list[str] | None = None,
    ):
        """执行骨干网络组合器的前向传播。

        参数：
            samples (torch.Tensor): 输入图像。
            captions (列表[str]): 输入文本描述。
            input_boxes (torch.Tensor, 可选): 文本包含边界框占位符时，包含其空间特征的张量。
            additional_text (列表[str], 可选): 要在同一次骨干网络前向传播中编码的额外文本（不同于 captions）。

        返回：
            (dict): 输出字典，包含以下键：`vision_features`（视觉骨干网络输出）、`language_features`（语言骨干网络输出）、
                `language_mask`（语言骨干网络的注意力掩码）、`vision_pos_enc`（视觉骨干网络的位置编码）；
                提供 `additional_text` 时，还包含 `additional_text_features` 和 `additional_text_mask`（额外文本的语言骨干网络输出和注意力掩码）。
        """
        output = self.forward_image(samples)
        output.update(self.forward_text(captions, input_boxes, additional_text))
        return output

    def forward_image(self, samples: torch.Tensor):
        """执行视觉骨干网络前向传播，并获取 SAM3 和 SAM2 特征。"""
        # 通过骨干网络前向传播
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(samples)
        if self.scalp > 0:
            # 丢弃分辨率最低的特征
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_image_sam2(self, samples: torch.Tensor):
        """执行视觉骨干网络前向传播，仅获取 SAM2 特征。"""
        xs = self.vision_backbone.trunk(samples)
        x = xs[-1]  # simpleFPN

        assert self.vision_backbone.sam2_convs is not None, "SAM2 neck is not available."
        sam2_features, sam2_pos = self.vision_backbone.sam_forward_feature_levels(x, self.vision_backbone.sam2_convs)

        if self.scalp > 0:
            # 丢弃分辨率最低的特征
            sam2_features, sam2_pos = (
                sam2_features[: -self.scalp],
                sam2_pos[: -self.scalp],
            )

        return {
            "vision_features": sam2_features[-1],
            "vision_pos_enc": sam2_pos,
            "backbone_fpn": sam2_features,
        }

    def forward_text(self, captions, input_boxes=None, additional_text=None):
        """执行文本编码器的前向传播。"""
        output = {}

        # 通过文本编码器前向传播
        text_to_encode = copy(captions)
        if additional_text is not None:
            # 如果存在 additional_text，则将其附加到本次前向传播中。
            # 后续用于输出对齐
            text_to_encode += additional_text

        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            text_attention_mask, text_memory, text_embeds = self.language_backbone(text_to_encode, input_boxes)

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[-len(additional_text) :]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds  # 传入编码器前的文本嵌入

        return output

    def set_imgsz(self, imgsz: list[int] | None = None):
        """设置视觉骨干网络的图像尺寸。"""
        imgsz = imgsz if imgsz is not None else [1008, 1008]
        self.vision_backbone.set_imgsz(imgsz)
