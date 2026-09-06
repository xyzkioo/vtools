# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import LayerNorm2d

from .blocks import (
    Block,
    CXBlock,
    Fuser,
    MaskDownSampler,
    MultiScaleBlock,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
)


class ImageEncoderViT(nn.Module):
    """使用 Vision Transformer（ViT）架构将图像编码到紧凑潜在空间的图像编码器。

    此类将图像划分为图像块，应用 Transformer 块，并通过 neck 模块生成最终编码表示。

    属性：
        img_size (int): 输入图像尺寸，假定为正方形。
        patch_embed (PatchEmbed): 图像块嵌入模块。
        pos_embed (nn.Parameter | None): 图像块的绝对位置嵌入。
        blocks (nn.ModuleList): 用于处理图像块嵌入的 Transformer 块列表。
        neck (nn.Sequential): 用于进一步处理输出的 neck 模块。

    方法：
        forward: 依次通过图像块嵌入、位置嵌入、Transformer 块和 neck 处理输入。

    示例：
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """初始化使用 Vision Transformer 架构编码图像的 ImageEncoderViT 实例。

        参数：
            img_size (int): 输入图像尺寸，假定为正方形。
            patch_size (int): 图像块尺寸。
            in_chans (int): 输入图像通道数。
            embed_dim (int): 图像块嵌入维度。
            depth (int): Transformer 块数量。
            num_heads (int): 每个块中的注意力头数。
            mlp_ratio (float): MLP 隐藏维度与嵌入维度的比值。
            out_chans (int): neck 模块的输出通道数。
            qkv_bias (bool): 为 True 时，为查询、键和值投影添加可学习偏置。
            norm_layer (type[nn.Module]): 使用的归一化层类型。
            act_layer (type[nn.Module]): 使用的激活层类型。
            use_abs_pos (bool): 为 True 时使用绝对位置嵌入。
            use_rel_pos (bool): 为 True 时在注意力图中加入相对位置嵌入。
            rel_pos_zero_init (bool): 为 True 时将相对位置参数初始化为零。
            window_size (int): 窗口注意力块的注意力窗口尺寸。
            global_attn_indexes (tuple[int, ...]): 使用全局注意力的块索引。
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # 使用预训练图像尺寸初始化绝对位置嵌入
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次通过图像块嵌入、位置嵌入、Transformer 块和 neck 模块处理输入。"""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class PromptEncoder(nn.Module):
    """编码输入 SAM 掩码解码器的不同类型提示，生成稀疏嵌入和密集嵌入。

    属性：
        embed_dim (int): 嵌入维度。
        input_image_size (tuple[int, int]): 输入图像尺寸，格式为 (H, W)。
        image_embedding_size (tuple[int, int]): 图像嵌入的空间尺寸，格式为 (H, W)。
        pe_layer (PositionEmbeddingRandom): 随机位置嵌入模块。
        num_point_embeddings (int): 不同类型点的点嵌入数量。
        point_embeddings (nn.ModuleList): 点嵌入列表。
        not_a_point_embed (nn.Embedding): 非任意标签点的嵌入。
        mask_input_size (tuple[int, int]): 输入掩码尺寸。
        mask_downscaling (nn.Sequential): 用于掩码下采样的神经网络。
        no_mask_embed (nn.Embedding): 未提供掩码时使用的嵌入。

    方法：
        get_dense_pe: 返回用于编码点提示的位置编码。
        forward: 编码不同类型的提示，同时返回稀疏嵌入和密集嵌入。

    示例：
        >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
        >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
        >>> boxes = torch.rand(1, 2, 2)
        >>> masks = torch.rand(1, 1, 256, 256)
        >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
        >>> print(sparse_embeddings.shape, dense_embeddings.shape)
        torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        """初始化用于编码各种类型提示的 PromptEncoder 模块。

        参数：
            embed_dim (int): 嵌入维度。
            image_embedding_size (tuple[int, int]): 图像嵌入的空间尺寸，格式为 (H, W)。
            input_image_size (tuple[int, int]): 填充后输入图像的尺寸，格式为 (H, W)。
            mask_in_chans (int): 编码输入掩码所用的隐藏通道数。
            activation (type[nn.Module]): 编码输入掩码时使用的激活函数。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 边界框 corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """返回用于编码点提示的密集位置编码。

        为与图像编码形状匹配的密集点集合生成位置编码，在处理点提示时为模型提供空间信息。

        返回：
            (torch.Tensor): 位置编码张量，形状为 (1, embed_dim, H, W)，其中 H 和 W 分别为图像嵌入尺寸的高度和宽度。

        示例：
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> dense_pe = prompt_encoder.get_dense_pe()
            >>> print(dense_pe.shape)
            torch.Size([1, 256, 64, 64])
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """应用位置编码和标签专属嵌入，编码点提示。"""
        points = points + 0.5  # 移动到像素中心
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype, device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), dtype=labels.dtype, device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """应用位置编码并添加角点嵌入，编码边界框提示。"""
        boxes = boxes + 0.5  # 移动到像素中心
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """通过下采样和卷积层处理输入掩码并生成嵌入。"""
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> int:
        """根据输入提示的批次大小获取输出批次大小。"""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码不同类型的提示，同时返回稀疏嵌入和密集嵌入。

        参数：
            points (tuple[torch.Tensor, torch.Tensor] | None): 要编码的点坐标和标签。第一个张量包含形状为 (B, N, 2) 的坐标，第二个张量包含形状为 (B, N) 的标签。
            boxes (torch.Tensor | None): 要编码的边界框，形状为 (B, M, 2, 2)，其中 M 为边界框数量。
            masks (torch.Tensor | None): 要编码的掩码，形状为 (B, 1, H, W)。

        返回：
            sparse_embeddings (torch.Tensor): 点和边界框的稀疏嵌入，形状为 (B, N, embed_dim)。
            dense_embeddings (torch.Tensor): 掩码的密集嵌入，形状为 (B, embed_dim, embed_H, embed_W)。

        示例：
            >>> encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_emb, dense_emb = encoder(points, boxes, masks)
            >>> print(sparse_emb.shape, dense_emb.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            dtype=self.point_embeddings[0].weight.dtype,
            device=self.point_embeddings[0].weight.device,
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MemoryEncoder(nn.Module):
    """将像素特征和掩码编码为内存表示，以高效执行图像分割。

    此类处理像素级特征和掩码，将其融合为编码后的内存表示，供 SAM（Segment Anything Model）等图像分割模型的下游任务使用。

    属性：
        mask_downsampler (MaskDownSampler): 输入掩码下采样模块。
        pix_feat_proj (nn.Conv2d): 像素特征投影卷积层。
        fuser (Fuser): 像素特征和掩码融合模块。
        position_encoding (PositionEmbeddingSine): 为特征添加位置编码的模块。
        out_proj (nn.Module): 输出投影层，可以是 nn.Identity 或 nn.Conv2d。

    方法：
        forward: 处理输入像素特征和掩码，生成编码后的内存表示。

    示例：
        >>> import torch
        >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
        >>> pix_feat = torch.randn(1, 256, 64, 64)
        >>> masks = torch.randn(1, 1, 1024, 1024)
        >>> out = encoder(pix_feat, masks)
        >>> print(out["vision_features"].shape, out["vision_pos_enc"][0].shape)
        torch.Size([1, 256, 64, 64]) torch.Size([1, 64, 64, 64])
    """

    def __init__(
        self,
        out_dim,
        in_dim=256,  # pix_feats 的输入维度
        interpol_size: tuple[int, int] | None = None,
    ):
        """初始化将像素特征和掩码编码为内存表示的 MemoryEncoder。

        此编码器处理像素级特征和掩码并将其融合，生成适用于 SAM（Segment Anything Model）等图像分割模型下游任务的内存表示。

        参数：
            out_dim (int): 编码特征的输出维度。
            in_dim (int): 像素特征的输入维度。
            interpol_size (tuple[int, int] | None): 掩码插值到的尺寸；为 None 时使用像素特征的尺寸。
        """
        super().__init__()

        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1, interpol_size=interpol_size)

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict:
        """处理像素特征和掩码，生成用于分割的编码内存表示。"""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # 融合像素特征和下采样掩码；如果视觉特征位于 CPU，则将其转换到掩码所在设备
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):
    """使用 trunk-neck 架构编码图像，生成多尺度特征和位置编码。

    此类将用于特征提取的 trunk 网络与用于特征细化和位置编码生成的 neck 网络结合起来，并可选择丢弃最低分辨率特征。

    属性：
        trunk (nn.Module): 用于初始特征提取的 trunk 网络。
        neck (nn.Module): 用于特征细化和位置编码生成的 neck 网络。
        scalp (int): 要丢弃的最低分辨率特征层级数量。

    方法：
        forward: 通过 trunk 和 neck 网络处理输入图像。

    示例：
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """使用 trunk 和 neck 网络初始化用于特征提取和细化的 ImageEncoder。

        此编码器将 trunk 特征提取网络与 neck 特征细化及位置编码生成网络结合，并可选择丢弃最低分辨率特征。

        参数：
            trunk (nn.Module): 用于初始特征提取的 trunk 网络。
            neck (nn.Module): 用于特征细化和位置编码生成的 neck 网络。
            scalp (int): 要丢弃的最低分辨率特征层级数量。
        """
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."
        )

    def forward(self, sample: torch.Tensor):
        """通过 trunk 和 neck 网络编码输入，返回多尺度特征和位置编码。"""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 丢弃最低分辨率特征
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """用于目标检测模型多尺度特征融合的特征金字塔网络（FPN）neck 变体。

    此 FPN 变体移除输出卷积，并使用可配置插值（默认双线性）调整特征尺寸，类似于 ViT 位置嵌入插值。

    属性：
        position_encoding (PositionEmbeddingSine): 正弦位置编码模块。
        convs (nn.ModuleList): 每个 backbone 层级对应的卷积层列表。
        backbone_channel_list (列表[int]): backbone 的通道维度列表。
        fpn_interp_model (str): FPN 特征缩放的插值模式。
        fuse_type (str): 特征融合类型，可选 'sum' 或 'avg'。
        fpn_top_down_levels (列表[int]): 输出中启用自顶向下特征传播的层级。

    方法：
        forward: 通过 FPN neck 执行前向传播。

    示例：
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: list[int] | None = None,
    ):
        """初始化改进的特征金字塔网络（FPN）neck。

        此 FPN 变体移除输出卷积，并使用可配置插值（默认双线性）调整特征尺寸，类似于 ViT 位置嵌入插值。

        参数：
            d_model (int): 模型维度。
            backbone_channel_list (列表[int]): backbone 的通道维度列表。
            kernel_size (int): 卷积层的卷积核尺寸。
            stride (int): 卷积层的步幅。
            padding (int): 卷积层的填充。
            fpn_interp_model (str): FPN 特征缩放的插值模式。
            fuse_type (str): 特征融合类型，可选 'sum' 或 'avg'。
            fpn_top_down_levels (列表[int] | None): 输出中启用自顶向下特征传播的层级。
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type

        # 输出中启用自顶向下特征传播的层级
        # 例如 fpn_top_down_levels 为 [2, 3] 时，仅第 2、3 层启用自顶向下传播，
        # 第 0、1 层只保留来自对应 backbone 层的横向特征
        if fpn_top_down_levels is None:
            # 默认在所有层级启用自顶向下特征
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: list[torch.Tensor]):
        """通过特征金字塔网络（FPN）neck 执行前向传播。

        此方法将 backbone 输入张量列表送入 FPN，应用横向连接和自顶向下特征融合，生成输出特征图及对应的位置编码。

        参数：
            xs (列表[torch.Tensor]): backbone 输入张量列表，每个张量形状为 (B, C, H, W)。

        返回：
            out (列表[torch.Tensor]): FPN 处理后的输出特征图列表，每个形状为 (B, d_model, H, W)。
            pos (列表[torch.Tensor]): 与每个输出特征图对应的位置编码列表。

        示例：
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # FPN 前向传播
        # 参见 https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # 按自顶向下顺序传播（从低分辨率到高分辨率）
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=x.dtype),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class Hiera(nn.Module):
    """用于图像处理任务高效提取多尺度特征的层次化视觉 Transformer。

    此类实现 Hiera 模型，这是一种为高效提取多尺度特征而设计的层次化视觉 Transformer 架构。它将一系列 Transformer 块组织为多个阶段，并支持可选池化和全局注意力机制。

    属性：
        window_spec (tuple[int, ...]): 每个阶段的窗口尺寸。
        q_stride (tuple[int, int]): 阶段之间的下采样步幅。
        stage_ends (列表[int]): 每个阶段最后一个块的索引。
        q_pool_blocks (列表[int]): 应用池化的块索引。
        return_interm_layers (bool): 是否返回中间层输出。
        patch_embed (PatchEmbed): 图像块嵌入模块。
        global_att_blocks (tuple[int, ...]): 使用全局注意力的块索引。
        window_pos_embed_bkg_spatial_size (tuple[int, int]): 窗口位置嵌入背景的空间尺寸。
        pos_embed (nn.Parameter): 背景位置嵌入。
        pos_embed_window (nn.Parameter): 窗口位置嵌入。
        blocks (nn.ModuleList): MultiScaleBlock 模块列表。
        channel_list (列表[int]): 每个阶段的输出通道维度列表。

    方法：
        _get_pos_embed: 通过插值并组合窗口嵌入和背景嵌入生成位置嵌入。
        forward: 执行 Hiera 模型的前向传播。

    示例：
        >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output_features = model(input_tensor)
        >>> for feat in output_features:
        ...     print(feat.shape)
    """

    def __init__(
        self,
        embed_dim: int = 96,  # 初始嵌入维度
        num_heads: int = 1,  # 初始注意力头数
        drop_path_rate: float = 0.0,  # 随机深度概率
        q_pool: int = 3,  # 查询池化阶段数量
        q_stride: tuple[int, int] = (2, 2),  # 阶段之间的下采样步幅
        stages: tuple[int, ...] = (2, 3, 16, 3),  # 每个阶段的块数量
        dim_mul: float = 2.0,  # 阶段切换时的维度倍率
        head_mul: float = 2.0,  # 阶段切换时的头数倍率
        window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14),
        # 未使用全局注意力时每个阶段的窗口尺寸
        window_spec: tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # 这些块使用全局注意力
        global_att_blocks: tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # 返回每个阶段的特征
    ):
        """初始化用于高效提取多尺度特征的层次化视觉 Transformer Hiera 模型。

        Hiera 是一种为图像处理任务高效提取多尺度特征而设计的层次化视觉 Transformer 架构。它将 Transformer 块组织为多个阶段，并支持可选池化和全局注意力机制。

        参数：
            embed_dim (int): 模型的初始嵌入维度。
            num_heads (int): 初始注意力头数。
            drop_path_rate (float): 随机深度概率。
            q_pool (int): 查询池化阶段数量。
            q_stride (tuple[int, int]): 阶段之间的下采样步幅。
            stages (tuple[int, ...]): 每个阶段的块数量。
            dim_mul (float): 阶段切换时的维度倍率。
            head_mul (float): 阶段切换时的头数倍率。
            window_pos_embed_bkg_spatial_size (tuple[int, int]): 窗口位置嵌入背景的空间尺寸。
            window_spec (tuple[int, ...]): 未使用全局注意力时每个阶段的窗口尺寸。
            global_att_blocks (tuple[int, ...]): 使用全局注意力的块索引。
            return_interm_layers (bool): 是否返回中间层输出。
        """
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        # 指定使用全局注意力的块
        self.global_att_blocks = global_att_blocks

        # 窗口位置嵌入（https://arxiv.org/abs/2311.05613）
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # 延后一块，因此下一阶段的第一个块使用上一阶段的初始窗口尺寸和当前阶段的最终窗口尺寸
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        """通过插值并组合窗口嵌入与背景嵌入生成位置嵌入。"""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """执行 Hiera 模型的前向传播，从输入图像中提取多尺度特征。

        参数：
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)，表示一批图像。

        返回：
            (列表[torch.Tensor]): 不同尺度的特征图列表，每个形状为 (B, C_i, H_i, W_i)，其中 C_i 为通道维度，H_i、W_i 为第 i 个尺度的空间维度。return_interm_layers 为 True 时，列表按分辨率从高到低排列；否则仅包含最终输出。

        示例：
            >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output_features = model(input_tensor)
            >>> for feat in output_features:
            ...     print(feat.shape)
        """
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # 添加位置嵌入
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
