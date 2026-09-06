# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

from ultralytics.nn.modules.transformer import MLP


class LinearPresenceHead(nn.Sequential):
    """用于预测图像中类别是否存在的线性存在性头。"""

    def __init__(self, d_model):
        """初始化 LinearPresenceHead。"""
            # 兼容旧检查点的临时处理，使 `LinearPresenceHead` 可以正常加载
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        """执行存在性头的前向传播。"""
        return super().forward(hs)


class MaskPredictor(nn.Module):
    """根据目标查询和像素嵌入预测掩码。"""

    def __init__(self, hidden_dim, mask_dim):
        """初始化 MaskPredictor。"""
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        """根据目标查询和像素嵌入预测掩码。"""
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # 批次大小 was omitted
                mask_preds = torch.einsum("bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed)
        else:
            # 假设包含辅助掩码
            if pixel_embed.ndim == 3:
                # 批次大小 was omitted
                mask_preds = torch.einsum("lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed)

        return mask_preds


class SegmentationHead(nn.Module):
    """根据骨干网络特征和目标查询预测掩码的分割头。"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        """初始化 SegmentationHead。"""
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

            # 用于更新输出字典
        self.instance_keys = ["pred_masks"]

    def _embed_pixels(self, backbone_feats: list[torch.Tensor], encoder_hidden_states) -> torch.Tensor:
        """使用像素解码器生成像素嵌入。"""
        if self.use_encoder_inputs:
            backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
            # 提取视觉嵌入
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(-1, *backbone_feats[-1].shape[1:])

            backbone_visual_feats[-1] = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(self.pixel_decoder, backbone_visual_feats, use_reentrant=False)
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [x for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # 对于 batch_size=1 的训练，可以跳过索引操作以节省内存
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[[0], ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """执行 SegmentationHead 的前向传播。"""
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(backbone_feats=backbone_feats, encoder_hidden_states=encoder_hidden_states)

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    """对骨干网络特征进行上采样的像素解码器模块。"""

    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        """初始化 PixelDecoder。"""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, dynamic=True, fullgraph=True)
            # 这是检查点机制所需的处理。但无法确定模块是否使用检查点，因此默认禁用。
            torch._dynamo.config.optimize_ddp = False

    def forward(self, backbone_feats: list[torch.Tensor]):
        """执行 PixelDecoder 的前向传播。"""
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode)
            if self.shared_conv:
                # 仅使用一个卷积层
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class UniversalSegmentationHead(SegmentationHead):
    """处理语义分割和实例分割的模块。"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
    ):
        """初始化 UniversalSegmentationHead。"""
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer if dot_product_scorer is not None else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, self.d_model, kernel_size=1)

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        prompt: torch.Tensor = None,
        prompt_mask: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """执行 UniversalSegmentationHead 的前向传播。"""
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt.to(tgt2.dtype),
                value=prompt.to(tgt2.dtype),
                key_padding_mask=prompt_mask,
                need_weights=False,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(backbone_feats=backbone_feats, encoder_hidden_states=encoder_hidden_states)

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }
