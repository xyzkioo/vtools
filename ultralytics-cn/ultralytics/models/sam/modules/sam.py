# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from ultralytics.nn.modules import MLP
from ultralytics.utils import LOGGER

from .blocks import SAM2TwoWayTransformer, TwoWayTransformer
from .decoders import MaskDecoder, SAM2MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder
from .utils import get_1d_sine_pe, select_closest_cond_frames

# 用于表示缺失对象的占位分数（较大的负值）
NO_OBJ_SCORE = -1024.0


class SAMModel(nn.Module):
    """用于目标分割任务的 Segment Anything Model（SAM）。

    此类结合图像编码器、提示编码器和掩码解码器，根据图像和输入提示预测对象掩码。

    属性：
        mask_threshold (float): 掩码预测阈值。
        image_encoder (ImageEncoderViT): 将图像编码为嵌入的 backbone。
        prompt_encoder (PromptEncoder): 各种输入提示的编码器。
        mask_decoder (MaskDecoder): 根据图像和提示嵌入预测对象掩码。
        pixel_mean (torch.Tensor): 输入图像像素归一化的均值。
        pixel_std (torch.Tensor): 输入图像像素归一化的标准差。

    方法：
        set_imgsz: 设置图像尺寸，使模型兼容不同大小的图像。

    示例：
        >>> image_encoder = ImageEncoderViT(...)
        >>> prompt_encoder = PromptEncoder(...)
        >>> mask_decoder = MaskDecoder(...)
        >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
        >>> # 后续用法取决于 SAMPredictor 类

    注意：
        所有 forward() 操作都在 SAMPredictor 类中实现。
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: list[float] = (123.675, 116.28, 103.53),
        pixel_std: list[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """初始化 SAMModel 类，根据图像和输入提示预测对象掩码。

        参数：
            image_encoder (ImageEncoderViT): 将图像编码为图像嵌入的 backbone。
            prompt_encoder (PromptEncoder): 编码各种类型的输入提示。
            mask_decoder (MaskDecoder): 根据图像嵌入和编码后的提示预测掩码。
            pixel_mean (列表[float]): 输入图像像素归一化均值。
            pixel_std (列表[float]): 输入图像像素归一化标准差。

        注意：
            所有 forward() 操作都已移至 SAMPredictor。
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def set_imgsz(self, imgsz):
        """设置图像尺寸，使模型兼容不同大小的图像。"""
        if hasattr(self.image_encoder, "set_imgsz"):
            self.image_encoder.set_imgsz(imgsz)
        self.prompt_encoder.input_image_size = imgsz
        self.prompt_encoder.image_embedding_size = [x // 16 for x in imgsz]  # 16 是 ViT 模型固定的 patch 尺寸
        self.image_encoder.img_size = imgsz[0]


class SAM2Model(torch.nn.Module):
    """具有基于内存的视频对象分割能力的 Segment Anything Model 2（SAM2Model）。

    此类扩展 SAM 的功能以处理视频序列，并引入内存机制保持时间一致性、高效跟踪跨帧对象。

    属性：
        mask_threshold (float): 掩码预测的阈值。
        image_encoder (ImageEncoderViT): 用于提取图像特征的视觉编码器。
        memory_attention (nn.Module): 用于关注记忆特征的模块。
        memory_encoder (nn.Module): 用于生成记忆表示的编码器。
        num_maskmem (int): 可访问的记忆帧数量。
        image_size (int): 输入图像尺寸。
        backbone_stride (int): 骨干网络输出的步幅。
        sam_prompt_embed_dim (int): SAM 提示嵌入的维度。
        sam_image_embedding_size (int): SAM 图像嵌入的尺寸。
        sam_prompt_encoder (PromptEncoder): 用于处理输入提示的编码器。
        sam_mask_decoder (SAM2MaskDecoder): 用于生成对象掩码的解码器。
        obj_ptr_proj (nn.Module): 对象指针的投影层。
        obj_ptr_tpos_proj (nn.Module): 对象指针的时间位置编码投影层。
        hidden_dim (int): 模型的隐藏维度。
        mem_dim (int): 编码特征使用的记忆维度。
        use_high_res_features_in_sam (bool): 是否在 SAM 掩码解码器中使用高分辨率特征图。
        use_obj_ptrs_in_encoder (bool): 是否在编码器中关注来自其他帧的对象指针。
        max_obj_ptrs_in_encoder (int): 编码器交叉注意力中来自其他帧的对象指针最大数量。
        add_tpos_enc_to_obj_ptrs (bool): 是否为对象指针添加时间位置编码。
        proj_tpos_enc_in_obj_ptrs (bool): 是否为对象指针的时间位置编码添加额外的线性投影层。
        use_signed_tpos_enc_to_obj_ptrs (bool): 是否在时间位置编码中使用带符号距离。
        only_obj_ptrs_in_the_past_for_eval (bool): 评估期间是否只关注过去的对象指针。
        pred_obj_scores (bool): 是否预测当前帧中是否存在对象。
        pred_obj_scores_mlp (bool): 是否使用 MLP 预测对象分数。
        fixed_no_obj_ptr (bool): 没有对象时是否使用固定的无对象指针。
        soft_no_obj_ptr (bool): 是否以软方式混入无对象指针，以便恢复并减轻错误。
        use_mlp_for_obj_ptr_proj (bool): 是否使用 MLP 进行对象指针投影。
        no_obj_embed_spatial (torch.Tensor | None): 空间帧的无对象嵌入。
        max_cond_frames_in_attn (int): 参与记忆注意力的条件帧最大数量。
        directly_add_no_mem_embed (bool): 是否在第一帧直接将无记忆嵌入添加到图像特征中。
        multimask_output_in_sam (bool): 是否在初始条件帧的第一次点击时输出多个掩码。
        multimask_min_pt_num (int): 在 SAM 中使用多掩码输出所需的最少点击数量。
        multimask_max_pt_num (int): 在 SAM 中使用多掩码输出允许的最多点击数量。
        multimask_output_for_tracking (bool): 是否在跟踪中使用多掩码输出。
        use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为对象指针。
        iou_prediction_use_sigmoid (bool): 是否使用 sigmoid 将 IoU 预测限制在 [0, 1]。
        memory_temporal_stride_for_eval (int): 评估期间记忆库的时间步幅。
        non_overlap_masks_for_mem_enc (bool): 评估期间是否在记忆编码器中对对象掩码应用不重叠约束。
        sigmoid_scale_for_mem_enc (float): 掩码 sigmoid 概率的缩放因子。
        sigmoid_bias_for_mem_enc (float): 掩码 sigmoid 概率的偏置因子。
        binarize_mask_from_pts_for_mem_enc (bool): 评估期间是否将交互帧上的 sigmoid 掩码 logits 二值化。
        use_mask_input_as_output_without_sam (bool): 对包含掩码输入的帧，是否不使用 SAM 提示编码器和掩码解码器，直接输出输入掩码。

    方法：
        forward_image: 通过编码器处理图像批次，提取多层特征。
        track_step: 执行单步跟踪，更新对象掩码和内存特征。
        set_binarize: 为 VideoPredictor 设置二值化选项。
        set_imgsz: 设置图像尺寸，使模型兼容不同大小的图像。

    示例：
        >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
        >>> image_batch = torch.rand(1, 3, 512, 512)
        >>> features = model.forward_image(image_batch)
        >>> track_results = model.track_step(0, True, features, None, None, None, {})
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        """初始化用于基于内存跟踪的视频对象分割 SAM2Model。

        参数：
            image_encoder (nn.Module): 用于提取图像特征的视觉编码器。
            memory_attention (nn.Module): 用于关注记忆特征的模块。
            memory_encoder (nn.Module): 用于生成记忆表示的编码器。
            num_maskmem (int): 可访问的记忆帧数量。
            image_size (int): 输入图像尺寸。
            backbone_stride (int): 图像骨干网络输出的步幅。
            sigmoid_scale_for_mem_enc (float): 掩码 sigmoid 概率的缩放因子。
            sigmoid_bias_for_mem_enc (float): 掩码 sigmoid 概率的偏置因子。
            binarize_mask_from_pts_for_mem_enc (bool): 评估期间是否将包含点击的交互帧上的 sigmoid 掩码 logits 二值化。
            use_mask_input_as_output_without_sam (bool): 对包含掩码输入的帧，是否不使用 SAM 提示编码器和掩码解码器，直接输出输入掩码。
            max_cond_frames_in_attn (int): 参与记忆注意力的条件帧最大数量。
            directly_add_no_mem_embed (bool): 是否在第一帧直接将无记忆嵌入添加到图像特征中。
            use_high_res_features_in_sam (bool): 是否在 SAM 掩码解码器中使用高分辨率特征图。
            multimask_output_in_sam (bool): 是否在初始条件帧的第一次点击时输出多个掩码。
            multimask_min_pt_num (int): 在 SAM 中使用多掩码输出所需的最少点击数量。
            multimask_max_pt_num (int): 在 SAM 中使用多掩码输出允许的最多点击数量。
            multimask_output_for_tracking (bool): 是否在跟踪中使用多掩码输出。
            use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为对象指针。
            iou_prediction_use_sigmoid (bool): 是否使用 sigmoid 将 IoU 预测限制在 [0, 1]。
            memory_temporal_stride_for_eval (int): 评估期间记忆库的时间步幅。
            non_overlap_masks_for_mem_enc (bool): 评估期间是否在记忆编码器中对对象掩码应用不重叠约束。
            use_obj_ptrs_in_encoder (bool): 是否在编码器中关注来自其他帧的对象指针。
            max_obj_ptrs_in_encoder (int): 编码器交叉注意力中来自其他帧的对象指针最大数量。
            add_tpos_enc_to_obj_ptrs (bool): 是否为编码器中的对象指针添加时间位置编码。
            proj_tpos_enc_in_obj_ptrs (bool): 是否为对象指针的时间位置编码添加额外的线性投影层。
            use_signed_tpos_enc_to_obj_ptrs (bool): 是否在对象指针的时间位置编码中使用带符号距离。
            only_obj_ptrs_in_the_past_for_eval (bool): 评估期间是否只关注过去的对象指针。
            pred_obj_scores (bool): 是否预测当前帧中是否存在对象。
            pred_obj_scores_mlp (bool): 是否使用 MLP 预测对象分数。
            fixed_no_obj_ptr (bool): 没有对象时是否使用固定的无对象指针。
            soft_no_obj_ptr (bool): 是否以软方式混入无对象指针，以便恢复并减轻错误。
            use_mlp_for_obj_ptr_proj (bool): 是否使用 MLP 进行对象指针投影。
            no_obj_embed_spatial (bool): 是否为空间帧添加无对象嵌入。
            sam_mask_decoder_extra_args (dict | None): 构建 SAM 掩码解码器时使用的额外参数。
            compile_image_encoder (bool): 是否编译图像编码器以加快推理。
        """
        super().__init__()

        # 第 1 部分：图像 backbone
        self.image_encoder = image_encoder
        # 高分辨率设置使用第 0、1、2 层，默认设置仅使用第 2 层
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # 使用卷积层将掩码提示下采样到步幅 4（与低分辨率 SAM 掩码 logits 的步幅相同），
            # 并将数值范围从 0~1 转换为 SAM logits 范围，以便送入 SAM 掩码解码器生成指针。
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # 这些选项必须同时使用
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # 第 2 部分：使用历史帧的内存（及对象指针）
        # 对当前帧视觉特征进行内存注意力条件化
        self.memory_attention = memory_attention
        self.hidden_dim = memory_attention.d_model

        # 第 3 部分：为上一帧输出提供内存编码器
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            # 如果内存沿通道维度进行了压缩
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # 可访问的内存数量
        # 内存的时间编码
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # 用于表示没有历史帧内存嵌入的单个令牌
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # 将 sigmoid 应用于原始掩码 logits（把范围从 (-inf, +inf) 转换为 (0, 1)），
        # 再送入内存编码器
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # 对于包含掩码输入的帧，是否直接输出输入掩码，
        # 不使用 SAM 提示编码器和掩码解码器
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # 第 4 部分：SAM 风格提示编码器（用于掩码和点输入）
        # 以及用于最终掩码输出的 SAM 风格掩码解码器
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.add_all_frames_to_correct_as_cond = True

        # 模型编译
        if compile_image_encoder:
            # 仅编译前向函数而不是整个模块，以便加载检查点。
            LOGGER.info("Image encoder compilation is enabled. First forward pass will be slow.")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        """返回存储模型参数的设备。"""
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """处理图像和提示输入，在视频序列中生成对象掩码和分数。"""
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference."
            "See notebooks/video_predictor_example.ipynb for an example."
        )

    def _build_sam_heads(self):
        """构建用于图像分割任务的 SAM 风格提示编码器和掩码解码器。"""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # 根据 SAM 构建 PromptEncoder 和 MaskDecoder（`mask_in_chans=16` 等超参数来自 SAM 代码）
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=SAM2TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # 使用线性投影将 SAM 输出令牌转换为对象指针
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # 对对象指针中的时间位置编码使用线性投影，
            # 避免其与空间位置编码产生潜在干扰
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """通过 SAM 提示编码器和掩码头执行前向传播。

        此方法处理图像特征和可选的点/掩码输入，以生成对象掩码和分数。

        参数：
            backbone_features (torch.Tensor): 图像特征，形状为 (B, C, H, W)。
            point_inputs (dict[str, torch.Tensor] | None): 包含点提示的字典，键为 'point_coords'
                （形状为 (B, P, 2) 的 float32 张量，包含 P 个输入点的绝对像素坐标）和 'point_labels'
                （形状为 (B, P) 的 int32 张量，其中 1 表示正点击，0 表示负点击，-1 表示填充）。
            mask_inputs (torch.Tensor | None): 形状为 (B, 1, H*16, W*16) 的浮点或布尔掩码，空间尺寸与图像相同。
            high_res_features (列表[torch.Tensor] | None): 两个特征图的列表，形状分别为 (B, C, 4*H, 4*W) 和
                (B, C, 2*H, 2*W)，作为 SAM 解码器的高分辨率特征图。
            multimask_output (bool): 为 True 时输出 3 个候选掩码及其 IoU 估计值；为 False 时仅输出 1 个掩码及其 IoU 估计值。

        返回：
            low_res_multimasks (torch.Tensor): SAM 输出的低分辨率掩码 logits，形状为 (B, M, H*4, W*4)。
            high_res_multimasks (torch.Tensor): 上采样后的掩码 logits，形状为 (B, M, H*16, W*16)。
            ious (torch.Tensor): 每个输出掩码的 IoU 估计值，形状为 (B, M)。
            low_res_masks (torch.Tensor): 最佳低分辨率掩码，形状为 (B, 1, H*4, W*4)。
            high_res_masks (torch.Tensor): 最佳高分辨率掩码，形状为 (B, 1, H*16, W*16)。
            obj_ptr (torch.Tensor): 输出掩码对应的对象指针向量，形状为 (B, C)。
            object_score_logits (torch.Tensor): 对象分数 logits，形状为 (B, 1)。

        示例：
            >>> backbone_features = torch.rand(1, 256, 32, 32)
            >>> point_inputs = {"point_coords": torch.rand(1, 2, 2), "point_labels": torch.tensor([[1, 0]])}
            >>> mask_inputs = torch.rand(1, 1, 512, 512)
            >>> results = model._forward_sam_heads(backbone_features, point_inputs, mask_inputs)
            >>> (
            ...     low_res_multimasks,
            ...     high_res_multimasks,
            ...     ious,
            ...     low_res_masks,
            ...     high_res_masks,
            ...     obj_ptr,
            ...     object_score_logits,
            ... ) = results
        """
        B = backbone_features.shape[0]
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a）处理点提示
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.shape[0] == B and sam_point_labels.shape[0] == B
        else:
            # 如果未提供点，则使用空点（标签为 -1）进行填充
            sam_point_coords = torch.zeros(B, 1, 2, device=device, dtype=backbone_features.dtype)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b）处理掩码提示
        if mask_inputs is not None:
            # 如果提供 mask_inputs，则在需要时将其缩小为低分辨率掩码输入，
            # 并将其作为密集掩码提示送入 SAM 掩码编码器
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.to(backbone_features.dtype),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # 下采样时使用抗锯齿
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # 否则直接传入 None（SAM 提示编码器会添加学习得到的 `no_mask_embed`，表示没有掩码输入）。
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # 图像已经完成批处理
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # 空间内存掩码在有对象和无对象之间进行硬选择，与实际掩码预测保持一致
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        # 将可能为 bfloat16（或 float16）的掩码转换为 float32
        # （2.1 之前的旧版 PyTorch 不支持对 bf16 使用 `interpolate`）
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # 获取最佳掩码预测（IoU 估计值最高）
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # 从 SAM 输出令牌中提取对象指针（处理遮挡情况）
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # 与掩码不同，允许使用软无对象指针
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.to(obj_ptr.dtype)

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, mask_inputs, backbone_features=None, high_res_features=None):
        """将掩码输入直接作为输出，跳过 SAM 编码器和解码器。"""
        # 对负/正像素使用 -10/+10 作为 logits（经过 sigmoid 后的概率非常接近 0/1）。
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # 下采样时使用抗锯齿
        )
        # 对掩码输入使用全为 1 的虚拟 IoU 预测
        ious = mask_inputs.new_ones(mask_inputs.shape[0], 1).float()
        if not self.use_obj_ptrs_in_encoder or backbone_features is None or high_res_features is None:
            # 使用全零作为虚拟对象指针（形状为 [B, C]）
            obj_ptr = torch.zeros(mask_inputs.shape[0], self.hidden_dim, device=mask_inputs.device)
        else:
            # 使用 SAM 解码器根据掩码输入生成对象指针
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float.to(backbone_features.dtype)),
                high_res_features=high_res_features,
            )
        # 此方法将 mask_input 视为输出，例如直接用它创建空间内存；
        # 以下遵循相同设计原则，使用 mask_input 判断对象是否出现，而不是依赖 SAM 解码器的对象分数。
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """通过编码器处理图像批次，为 SAM 模型提取多层特征。"""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # 预先计算 SAM 解码器中的第 0、1 层投影特征，
            # 避免每次 SAM 点击时重复计算
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out, batch=1):
        """准备并展平图像 backbone 的视觉特征输出，以便进一步处理。"""
        if batch > 1:  # 存在多个提示时扩展特征
            backbone_out = {
                **backbone_out,
                "backbone_fpn": [feat.expand(batch, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
                "vision_pos_enc": [pos.expand(batch, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
            }
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # 将 NxCxHxW 展平为 HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 按反向时间顺序跟踪（用于演示）
    ):
        """将当前帧视觉特征与历史内存融合，准备经过内存条件化的特征。"""
        B = current_vision_feats[-1].size(1)  # 当前帧的批次大小
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) 特征 尺寸
        device = current_vision_feats[-1].device
        # 下方 `self.num_maskmem == 0` 的情况主要用于在图像上复现 SAM。
        # 此时跳过与任何内存的融合。
        if self.num_maskmem == 0:  # 禁用记忆并跳过融合
            return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # 第 1 步：使用历史内存对当前帧视觉特征进行条件化
        if not is_init_cond_frame:
            # 获取由 maskmem backbone 编码的内存
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # 先添加条件帧输出（下方计算时间位置嵌入时，所有条件帧的 t_pos=0）
            assert len(output_dict["cond_frame_outputs"]) > 0
            # 选择时间上最近的最多若干个条件帧用于交叉注意力
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # 为非条件内存添加当前帧之前的最后 (self.num_maskmem - 1) 帧；
            # 最早帧的 t_pos=1，最新帧的 t_pos=self.num_maskmem-1。
            # 也允许不连续选取内存帧（r>1），此时从每隔 r 帧的序列中选取
            # (self.num_maskmem - 2) 帧，并加上最后一帧。
            r = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # 位于当前帧之前的帧数
                if t_rel == 1:
                    # 当 t_rel == 1 时，取最后一帧（与 r 无关）
                    prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                elif not track_in_reverse:
                    # 先在当前帧之前每隔 r 帧的序列中找到最近帧
                    # 当 r=1 时，该帧为 (frame_idx - 2)
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    # 然后继续按每隔 r 帧向前查找
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    # 先在当前帧之后每隔 r 帧的序列中找到最近帧
                    # 当 r=1 时，该帧为 (frame_idx + 2)
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    # 然后继续按每隔 r 帧向后查找
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # 如果未选中的条件帧位于最后 (self.num_maskmem - 1) 帧之中，
                    # 仍将其作为非条件帧参与注意力计算。
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # 跳过填充帧
                # 在演示场景中，"maskmem_features" 可能已卸载到 CPU，
                # 因此将其加载回推理设备（如果已在目标设备上则不执行操作）。
                feats = prev["maskmem_features"].to(device=device, non_blocking=device.type == "cuda")
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # 空间位置编码（评估时可能已卸载到 CPU）
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device=device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # 时间位置编码
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # 构建历史对象指针列表
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # 先添加所选条件帧中的对象指针
                # （评估时可选择仅包含过去帧的对象指针）
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # 时间位置编码表示每个指针与当前帧之间相隔多远
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # 最多添加 (max_obj_ptrs_in_encoder - 1) 个当前帧之前的非条件帧
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # 如果至少存在一个对象指针，则将其加入跨帧注意力
                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # 沿 dim=0 堆叠对象指针，得到形状 [ptr_seq_len, B, C]
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # 根据每个对象指针与当前帧的距离生成时间位置嵌入
                    # （正弦嵌入按最大指针数量归一化）。
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device, dtype=current_vision_feats[-1].dtype)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # 当 self.mem_dim < C 时，将一个指针拆分为 (C // self.mem_dim) 个令牌
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # 对初始条件帧进行编码，不使用任何历史内存
            if self.directly_add_no_mem_embed:
                # 直接添加无内存嵌入（不使用 Transformer 编码器）
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # 第一帧使用虚拟令牌（避免向 Transformer 编码器传入空记忆）
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # 第 2 步：拼接内存，并通过 Transformer 编码器前向传播
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # 将输出 (HW)BC 重塑为 BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """将帧特征和掩码编码为视频分割使用的新内存表示。"""
        B = current_vision_feats[-1].size(1)  # 当前帧的批次大小
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) 特征 尺寸
        # 顶层特征，(HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # 可选地对掩码应用非重叠约束（约束沿批次维度应用，
            # 仅应在评估期间使用，此时所有对象来自同一视频且批次大小为 1）。
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        # 在应用 sigmoid 前使用温度缩放原始掩码 logits
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            # 对原始掩码 logits 应用 sigmoid，将其转换到 (0, 1) 范围
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # 对 sigmoid 概率应用缩放和偏置
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)  # 已经应用 sigmoid
        maskmem_features = maskmem_out["vision_features"]
        # 向空间内存添加无对象嵌入，表示该帧被预测为遮挡
        # （即该帧中没有对象出现）
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[
                ..., None, None
            ].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_out["vision_pos_enc"]

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        """根据当前帧输入执行单步跟踪，更新对象掩码和内存特征。"""
        # SAM 头使用的高分辨率特征图，重塑 (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # 当 use_mask_input_as_output_without_sam=True 时，直接输出掩码输入
            # （将其视为 GT 掩码），不使用 SAM 提示编码器和掩码解码器。
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(mask_inputs, pix_feat, high_res_features)
        else:
            # 将视觉特征与内存库中的历史内存特征融合
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # 应用 SAM 风格分割头
            # 可以将之前预测的低分辨率 SAM 掩码 logits 输入 SAM 掩码解码器，
            # 例如演示中这些 logits 来自早期交互，而不是校正采样。
            # （此时任意 `mask_inputs` 都不应到达这里，因为它们会被送入 _use_mask_as_output。）
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        """对预测掩码运行内存编码器，将其编码为供未来帧使用的新内存特征。"""
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 按时间反向顺序跟踪（用于演示）
        # 是否对预测掩码运行内存编码器。有时可以通过 `run_mem_encoder=False` 跳过内存编码器。
        # 例如，在演示中可能针对每次用户点击多次调用 `track_step`，
        # 仅在用户完成点击后编码内存；在静态图像上的 SAM 训练等消融设置中也不需要内存编码器。
        run_mem_encoder=True,
        # 之前预测的 SAM 掩码 logits（演示中可与新点击一起输入）。
        prev_sam_mask_logits=None,
    ):
        """执行单步跟踪，根据当前帧输入更新目标掩码和记忆特征。"""
        sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs

        current_out = {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
        }
        if not self.training:
            # 仅在推理时添加（避免激活检查点中出现未使用参数；
            # 主要用于演示中使用合并掩码编码空间内存）
            current_out["object_score_logits"] = object_score_logits

        # 对预测掩码运行内存编码器，将其编码为供未来帧使用的新内存特征
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """根据配置和输入确定 SAM 头是否使用多掩码输出。"""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks):
        """对掩码应用非重叠约束，在每个位置保留分数最高的对象。"""
        batch_size = pred_masks.shape[0]
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds"：每个位置分数最高对象的对象索引
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds"：`pred_masks` 中每个对象切片（沿 dim 0）的对象索引
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # 将重叠区域的分数抑制到 -10.0 以下，使前景区域不重叠
        # （此处 sigmoid(-10.0)=4.5398e-05）
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def set_binarize(self, binarize=False):
        """为 VideoPredictor 设置二值化选项。"""
        self.binarize_mask_from_pts_for_mem_enc = binarize

    def set_imgsz(self, imgsz):
        """设置图像尺寸，使模型兼容不同大小的图像。"""
        if hasattr(self.image_encoder, "set_imgsz"):
            self.image_encoder.set_imgsz(imgsz)
        self.image_size = imgsz[0]
        self.sam_prompt_encoder.input_image_size = imgsz
        self.sam_prompt_encoder.image_embedding_size = [
            x // self.backbone_stride for x in imgsz
        ]  # ViT 固定的 16 像素 patch 尺寸
        self.sam_prompt_encoder.mask_input_size = [
            x // self.backbone_stride * 4 for x in imgsz
        ]  # ViT 固定的 16 像素 patch 尺寸
        self.sam_image_embedding_size = self.image_size // self.backbone_stride  # 更新图像嵌入尺寸


class SAM3Model(SAM2Model):
    """具有基于内存的视频对象分割能力的 Segment Anything Model 3（SAM3Model）。"""

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        sigmoid_scale_for_mem_enc=1,
        sigmoid_bias_for_mem_enc=0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        """初始化具有基于内存的视频对象分割能力的 SAM3Model。"""
        super().__init__(
            image_encoder,
            memory_attention,
            memory_encoder,
            num_maskmem,
            image_size,
            backbone_stride,
            sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,
            directly_add_no_mem_embed,
            use_high_res_features_in_sam,
            multimask_output_in_sam,
            multimask_min_pt_num,
            multimask_max_pt_num,
            multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores,
            pred_obj_scores_mlp,
            fixed_no_obj_ptr,
            soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial,
            sam_mask_decoder_extra_args,
            compile_image_encoder,
        )
        self.sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

    def forward_image(self, img_batch: torch.Tensor):
        """通过编码器处理图像批次，为 SAM 模型提取多层特征。"""
        backbone_out = self.image_encoder.forward_image_sam2(img_batch)
        if self.use_high_res_features_in_sam:
            # 预先在 SAM 解码器中计算第 0、1 层投影特征，
            # 避免每次 SAM 点击时重复计算
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def set_imgsz(self, imgsz: tuple[int, int]):
        """设置模型和掩码下采样器的图像尺寸。"""
        super().set_imgsz(imgsz)
        self.memory_encoder.mask_downsampler.interpol_size = [size // 14 * 16 for size in imgsz]

    @staticmethod
    def _suppress_shrinked_masks(pred_masks, new_pred_masks, shrink_threshold=0.3):
        """抑制应用逐像素非重叠约束后面积缩小的掩码。"""
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks_after

    def _suppress_object_pw_area_shrinkage(self, pred_masks):
        """抑制应用逐像素非重叠约束后面积缩小的掩码。

        注意，最终输出仍可能存在重叠。
        """
        # 根据掩码分数应用逐像素非重叠约束
        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(pred_masks)
        # 根据逐像素非重叠约束，完全抑制面积大幅缩小的掩码（可能是噪声）
        # 注意：如果没有掩码大幅缩小，此函数可能不会改变输出。
        pred_masks = self._suppress_shrinked_masks(pred_masks, pixel_level_non_overlapping_masks)
        return pred_masks
