# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# 此源代码遵循本源代码树根目录 LICENSE 文件中规定的许可证。

from functools import partial

import torch

from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.patches import torch_load

from .modules.decoders import MaskDecoder
from .modules.encoders import FpnNeck, Hiera, ImageEncoder, ImageEncoderViT, MemoryEncoder, PromptEncoder
from .modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modules.sam import SAM2Model, SAMModel
from .modules.tiny_encoder import TinyViT
from .modules.transformer import TwoWayTransformer


def _load_checkpoint(model, checkpoint):
    """从文件路径将检查点加载到模型中。"""
    if checkpoint is None:
        return model

    checkpoint = attempt_download_asset(checkpoint)
    with open(checkpoint, "rb") as f:
        state_dict = torch_load(f)
    # 处理嵌套的 "model" 键
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    return model


def build_sam_vit_h(checkpoint=None):
    """使用指定编码器参数构建并返回 h 尺寸的 Segment Anything Model（SAM）。"""
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint=None):
    """使用指定编码器参数构建并返回 l 尺寸的 Segment Anything Model（SAM）。"""
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    """使用指定编码器参数构建并返回 b 尺寸的 Segment Anything Model（SAM）。"""
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_mobile_sam(checkpoint=None):
    """构建并返回用于高效图像分割的 Mobile Segment Anything Model（Mobile-SAM）。"""
    return _build_sam(
        encoder_embed_dim=[64, 128, 160, 320],
        encoder_depth=[2, 2, 6, 2],
        encoder_num_heads=[2, 4, 5, 10],
        encoder_global_attn_indexes=None,
        mobile_sam=True,
        checkpoint=checkpoint,
    )


def build_sam2_t(checkpoint=None):
    """使用指定架构参数构建并返回 tiny 尺寸的 Segment Anything Model 2（SAM2）。"""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 7, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[5, 7, 9],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )


def build_sam2_s(checkpoint=None):
    """使用指定架构参数构建并返回 small 尺寸的 Segment Anything Model 2（SAM2）。"""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 11, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[7, 10, 13],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )


def build_sam2_b(checkpoint=None):
    """使用指定架构参数构建并返回 base 尺寸的 Segment Anything Model 2（SAM2）。"""
    return _build_sam2(
        encoder_embed_dim=112,
        encoder_stages=[2, 3, 16, 3],
        encoder_num_heads=2,
        encoder_global_att_blocks=[12, 16, 20],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_window_spatial_size=[14, 14],
        encoder_backbone_channel_list=[896, 448, 224, 112],
        checkpoint=checkpoint,
    )


def build_sam2_l(checkpoint=None):
    """使用指定架构参数构建并返回 large 尺寸的 Segment Anything Model 2（SAM2）。"""
    return _build_sam2(
        encoder_embed_dim=144,
        encoder_stages=[2, 6, 36, 4],
        encoder_num_heads=2,
        encoder_global_att_blocks=[23, 33, 43],
        encoder_window_spec=[8, 4, 16, 8],
        encoder_backbone_channel_list=[1152, 576, 288, 144],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    mobile_sam=False,
):
    """使用指定编码器参数构建 Segment Anything Model（SAM）。

    参数：
        encoder_embed_dim (int | 列表[int]): 编码器嵌入维度。
        encoder_depth (int | 列表[int]): 编码器深度。
        encoder_num_heads (int | 列表[int]): 编码器注意力头数量。
        encoder_global_attn_indexes (列表[int] | None): 编码器中的全局注意力索引。
        checkpoint (str | None, 可选): 模型检查点文件路径。
        mobile_sam (bool, 可选): 是否构建 Mobile-SAM 模型。

    返回：
        (SAMModel): 具有指定架构的 Segment Anything Model 实例。

    示例：
        >>> sam = _build_sam(768, 12, 12, [2, 5, 8, 11])
        >>> sam = _build_sam([64, 128, 160, 320], [2, 2, 6, 2], [2, 4, 5, 10], None, mobile_sam=True)
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder = (
        TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=encoder_embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
        if mobile_sam
        else ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    )
    sam = SAMModel(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        sam = _load_checkpoint(sam, checkpoint)
    sam.eval()
    return sam


def _build_sam2(
    encoder_embed_dim=1280,
    encoder_stages=(2, 6, 36, 4),
    encoder_num_heads=2,
    encoder_global_att_blocks=(7, 15, 23, 31),
    encoder_backbone_channel_list=(1152, 576, 288, 144),
    encoder_window_spatial_size=(7, 7),
    encoder_window_spec=(8, 4, 16, 8),
    checkpoint=None,
):
    """使用指定架构参数构建并返回 Segment Anything Model 2（SAM2）。

    参数：
        encoder_embed_dim (int, 可选): 编码器嵌入维度。
        encoder_stages (列表[int], 可选): 编码器每个阶段的块数量。
        encoder_num_heads (int, 可选): 编码器注意力头数量。
        encoder_global_att_blocks (列表[int], 可选): 编码器全局注意力块的索引。
        encoder_backbone_channel_list (列表[int], 可选): 编码器骨干网络每个层级的通道维度。
        encoder_window_spatial_size (列表[int], 可选): 位置嵌入窗口的空间尺寸。
        encoder_window_spec (列表[int], 可选): 编码器每个阶段的窗口规格。
        checkpoint (str | None, 可选): 用于加载预训练权重的检查点文件路径。

    返回：
        (SAM2Model): 已配置并初始化的 SAM2 模型。

    示例：
        >>> sam2_model = _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 7, 2])
        >>> sam2_model.eval()
    """
    image_encoder = ImageEncoder(
        trunk=Hiera(
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            stages=encoder_stages,
            global_att_blocks=encoder_global_att_blocks,
            window_pos_embed_bkg_spatial_size=encoder_window_spatial_size,
            window_spec=encoder_window_spec,
        ),
        neck=FpnNeck(
            d_model=256,
            backbone_channel_list=encoder_backbone_channel_list,
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
        scalp=1,
    )
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, num_layers=4, layer=MemoryAttentionLayer())
    memory_encoder = MemoryEncoder(out_dim=64)

    is_sam2_1 = checkpoint is not None and "sam2.1" in checkpoint
    sam2 = SAM2Model(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        no_obj_embed_spatial=is_sam2_1,
        proj_tpos_enc_in_obj_ptrs=is_sam2_1,
        use_signed_tpos_enc_to_obj_ptrs=is_sam2_1,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )

    if checkpoint is not None:
        sam2 = _load_checkpoint(sam2, checkpoint)
    sam2.eval()
    return sam2


sam_model_map = {
    "sam_h.pt": build_sam_vit_h,
    "sam_l.pt": build_sam_vit_l,
    "sam_b.pt": build_sam_vit_b,
    "mobile_sam.pt": build_mobile_sam,
    "sam2_t.pt": build_sam2_t,
    "sam2_s.pt": build_sam2_s,
    "sam2_b.pt": build_sam2_b,
    "sam2_l.pt": build_sam2_l,
    "sam2.1_t.pt": build_sam2_t,
    "sam2.1_s.pt": build_sam2_s,
    "sam2.1_b.pt": build_sam2_b,
    "sam2.1_l.pt": build_sam2_l,
}


def build_sam(ckpt="sam_b.pt"):
    """根据给定检查点构建并返回 Segment Anything Model（SAM）。

    参数：
        ckpt (str | Path, 可选): 检查点文件路径或预定义 SAM 模型名称。

    返回：
        (SAMModel | SAM2Model): 已配置并初始化的 SAM 或 SAM2 模型实例。

    异常：
        FileNotFoundError：给定检查点不是受支持的 SAM 模型时引发。

    示例：
        >>> sam_model = build_sam("sam_b.pt")
        >>> sam_model = build_sam("path/to/custom_checkpoint.pt")

    注意：
        支持的预定义模型包括：
        - SAM：'sam_h.pt'、'sam_l.pt'、'sam_b.pt'、'mobile_sam.pt'
        - SAM2：'sam2_t.pt'、'sam2_s.pt'、'sam2_b.pt'、'sam2_l.pt'
    """
    model_builder = None
    ckpt = str(ckpt)  # to allow Path ckpt types
    for k in sam_model_map:
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)

    if not model_builder:
        raise FileNotFoundError(f"{ckpt} is not a supported SAM model. Available models are: \n {sam_model_map.keys()}")

    return model_builder(ckpt)
