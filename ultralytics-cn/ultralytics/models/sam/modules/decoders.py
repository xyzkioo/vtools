# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import MLP, LayerNorm2d


class MaskDecoder(nn.Module):
    """使用 Transformer 架构生成掩码及其质量分数的解码器模块。

    此类根据图像和提示嵌入预测掩码，使用 Transformer 处理输入，并生成掩码预测结果及其质量分数。

    属性：
        transformer_dim (int): Transformer 模块的通道维度。
        transformer (nn.Module): 用于掩码预测的 Transformer 模块。
        num_multimask_outputs (int): 为消除掩码歧义而预测的掩码数量。
        iou_token (nn.Embedding): IoU 令牌嵌入。
        num_mask_tokens (int): 掩码令牌数量。
        mask_tokens (nn.Embedding): 掩码令牌嵌入。
        output_upscaling (nn.Sequential): 用于放大输出的神经网络序列。
        output_hypernetworks_mlps (nn.ModuleList): 用于生成掩码的超网络 MLP 列表。
        iou_prediction_head (nn.Module): 用于预测掩码质量的 MLP。

    方法：
        forward: 根据图像和提示嵌入预测掩码。
        predict_masks: 掩码预测的内部方法。

    示例：
        >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
        >>> masks, iou_pred = decoder(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True
        ... )
        >>> print(f"Predicted masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """初始化用于生成掩码及其质量分数的 MaskDecoder 模块。

        参数：
            transformer_dim (int): Transformer 模块的通道维度。
            transformer (nn.Module): 用于掩码预测的 Transformer 模块。
            num_multimask_outputs (int): 为消除掩码歧义而预测的掩码数量。
            activation (type[nn.Module]): 放大掩码时使用的激活函数类型。
            iou_head_depth (int): 用于预测掩码质量的 MLP 深度。
            iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏维度。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据图像和提示嵌入预测掩码。

        参数：
            image_embeddings (torch.Tensor): 图像编码器生成的嵌入。
            image_pe (torch.Tensor): 与 image_embeddings 形状相同的位置编码。
            sparse_prompt_embeddings (torch.Tensor): 点和边界框的嵌入。
            dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入。
            multimask_output (bool): 是否返回多个掩码，而不是单个掩码。

        返回：
            masks (torch.Tensor): 批量预测的掩码。
            iou_pred (torch.Tensor): 批量预测的掩码质量。

        示例：
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
            >>> image_emb = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_emb = torch.rand(1, 2, 256)
            >>> dense_emb = torch.rand(1, 256, 64, 64)
            >>> masks, iou_pred = decoder(image_emb, image_pe, sparse_emb, dense_emb, multimask_output=True)
            >>> print(f"Masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择正确的掩码输出
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """通过 Transformer 架构，使用图像和提示嵌入预测掩码及质量分数。"""
        # 拼接输出令牌
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 沿批次维度扩展每张图像的数据，使其对应每个掩码
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行 Transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 放大掩码嵌入，并使用掩码令牌预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测结果
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class SAM2MaskDecoder(nn.Module):
    """基于 Transformer 的解码器，用于根据图像和提示嵌入预测实例分割掩码。

    此类扩展 MaskDecoder 的功能，加入高分辨率特征处理、动态多掩码输出和对象分数预测等特性。

    属性：
        transformer_dim (int): Transformer 的通道维度。
        transformer (nn.Module): 用于预测掩码的 Transformer。
        num_multimask_outputs (int): 消除掩码歧义时要预测的掩码数量。
        iou_token (nn.Embedding): IoU 令牌嵌入。
        num_mask_tokens (int): 掩码令牌总数。
        mask_tokens (nn.Embedding): 掩码令牌嵌入。
        pred_obj_scores (bool): 是否预测对象分数。
        obj_score_token (nn.Embedding): 对象分数令牌嵌入。
        use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为对象指针。
        output_upscaling (nn.Sequential): 用于输出上采样的层。
        use_high_res_features (bool): 是否使用高分辨率特征。
        conv_s0 (nn.Conv2d): 用于高分辨率特征（s0）的卷积层。
        conv_s1 (nn.Conv2d): 用于高分辨率特征（s1）的卷积层。
        output_hypernetworks_mlps (nn.ModuleList): 输出超网络使用的 MLP 列表。
        iou_prediction_head (MLP): 用于 IoU 预测的 MLP。
        pred_obj_score_head (nn.Linear | MLP): 用于对象分数预测的线性层或 MLP。
        dynamic_multimask_via_stability (bool): 是否通过稳定性选择动态多掩码。
        dynamic_multimask_stability_delta (float): 动态多掩码稳定性的 Delta 值。
        dynamic_multimask_stability_thresh (float): 动态多掩码稳定性的阈值。

    方法：
        forward: 根据图像和提示嵌入预测掩码。
        predict_masks: 根据图像和提示嵌入预测实例分割掩码。
        _get_stability_scores：根据两个阈值之间的 IoU 计算掩码稳定性分数。
        _dynamic_multimask_via_stability: 动态选择最稳定的掩码输出。

    示例：
        >>> image_embeddings = torch.rand(1, 256, 64, 64)
        >>> image_pe = torch.rand(1, 256, 64, 64)
        >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
        >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
        >>> decoder = SAM2MaskDecoder(256, transformer)
        >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
        ... )
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """初始化用于预测实例分割掩码的 SAM2MaskDecoder 模块。

        此解码器扩展 MaskDecoder 的功能，加入高分辨率特征处理、动态多掩码输出和对象分数预测等特性。

        参数：
            transformer_dim (int): Transformer 的通道维度。
            transformer (nn.Module): 用于预测掩码的 Transformer。
            num_multimask_outputs (int): 消除掩码歧义时要预测的掩码数量。
            activation (type[nn.Module]): 掩码上采样时使用的激活函数类型。
            iou_head_depth (int): 用于预测掩码质量的 MLP 深度。
            iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏维度。
            use_high_res_features (bool): 是否使用高分辨率特征。
            iou_prediction_use_sigmoid (bool): 是否对 IoU 预测使用 sigmoid。
            dynamic_multimask_via_stability (bool): 是否通过稳定性选择动态多掩码。
            dynamic_multimask_stability_delta (float): 动态多掩码稳定性的 Delta 值。
            dynamic_multimask_stability_thresh (float): 动态多掩码稳定性的阈值。
            pred_obj_scores (bool): 是否预测对象分数。
            pred_obj_scores_mlp (bool): 是否使用 MLP 预测对象分数。
            use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为对象指针。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # 输出单个掩码时，如果单掩码输出令牌的稳定性分数较低，
        # 可以选择动态回退到最佳多掩码输出令牌。
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据图像和提示嵌入预测掩码。

        参数：
            image_embeddings (torch.Tensor): 来自图像编码器的嵌入，形状为 (B, C, H, W)。
            image_pe (torch.Tensor): 与 image_embeddings 形状相同的位置信息编码（B, C, H, W）。
            sparse_prompt_embeddings (torch.Tensor): 点和边界框的嵌入，形状为 (B, N, C)。
            dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入，形状为 (B, C, H, W)。
            multimask_output (bool): 是否返回多个掩码，而不是单个掩码。
            repeat_image (bool): 是否重复图像嵌入。
            high_res_features (列表[torch.Tensor] | None, 可选): 可选的高分辨率特征。

        返回：
            掩码 (torch.Tensor): 批量预测掩码，形状为 (B, N, H, W)。
            iou_pred (torch.Tensor): 批量掩码质量预测结果，形状为 (B, N)。
            sam_tokens_out (torch.Tensor): 批量 SAM 掩码输出令牌，形状为 (B, N, C)。
            object_score_logits (torch.Tensor): 批量对象分数 logits，形状为 (B, 1)。

        示例：
            >>> image_embeddings = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
            >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
            >>> decoder = SAM2MaskDecoder(256, transformer)
            >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
            ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
            ... )
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # 选择正确的掩码输出
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] 形状
        else:
            # 获取掩码输出令牌。这里始终使用单掩码输出令牌。
            # 测试时，即使在一次点击后跟踪（并使用 multimask_output=True），此处仍取单掩码令牌。
            # 原因是训练期间始终在多次点击后进行跟踪，因此训练期间看到的历史令牌始终是单掩码令牌，
            # 并将其作为目标记忆令牌。
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] 形状

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用 Transformer 根据图像和提示嵌入预测实例分割掩码。"""
        # 拼接输出令牌
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 沿批次维度扩展每张图像的数据，使其对应每个掩码
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.shape[0] == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行 Transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # 放大掩码嵌入，并使用掩码令牌预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features or high_res_features is None:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: list[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测结果
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # 对象分数 logits 默认为 10.0，即假设对象存在，此时 sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """根据上下阈值之间的 IoU 计算掩码稳定性分数。"""
        mask_logits = mask_logits.flatten(-2)
        area_i = torch.sum(mask_logits > self.dynamic_multimask_stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -self.dynamic_multimask_stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """根据稳定性分数和 IoU 预测结果动态选择最稳定的掩码输出。

        此方法用于输出单个掩码。当当前单掩码输出（基于输出令牌 0）的稳定性分数低于阈值时，会从多掩码输出（基于输出令牌 1-3）中选择预测 IoU 分数最高的掩码，从而保证点击和跟踪场景都能得到有效掩码。

        参数：
            all_mask_logits (torch.Tensor): 所有预测掩码的 logits，形状为 (B, N, H, W)，其中 B 为批次大小，
                N 为掩码数量（通常为 4），H、W 为掩码尺寸。
            all_iou_scores (torch.Tensor): 所有掩码的预测 IoU 分数，形状为 (B, N)。

        返回：
            mask_logits_out (torch.Tensor): 选中的掩码 logits，形状为 (B, 1, H, W)。
            iou_scores_out (torch.Tensor): 选中的 IoU 分数，形状为 (B, 1)。

        示例：
            >>> decoder = SAM2MaskDecoder(...)
            >>> all_mask_logits = torch.rand(2, 4, 256, 256)  # 2 张图像，每张 4 个掩码
            >>> all_iou_scores = torch.rand(2, 4)
            >>> mask_logits, iou_scores = decoder._dynamic_multimask_via_stability(all_mask_logits, all_iou_scores)
            >>> print(mask_logits.shape, iou_scores.shape)
            torch.Size([2, 1, 256, 256]) torch.Size([2, 1])
        """
        # 从多掩码输出令牌（1~3）中选择最佳掩码
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.shape[0], device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # 获取单掩码输出令牌 0 及其稳定性分数
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # 稳定性分数较低时，动态回退到最佳多掩码输出。
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
