# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from ultralytics.utils import WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    """文本编码模型的抽象基类。

    此类定义了视觉语言任务中文本编码模型的接口。子类必须实现 ``tokenize`` 和 ``encode_text`` 方法，
    用于完成文本分词和文本编码。

    方法：
        tokenize：将输入文本转换为模型可处理的词元。
        encode_text：将分词后的文本编码为归一化的特征向量。
    """

    def __init__(self):
        """初始化 TextModel 基类。"""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """将输入文本转换为模型处理所需的词元。"""

    @abstractmethod
    def encode_text(self, texts, dtype):
        """将分词后的文本编码为归一化的特征向量。"""


class CLIP(TextModel):
    """OpenAI CLIP（对比语言-图像预训练）文本编码器。

    此类基于 OpenAI 的 CLIP 模型实现文本编码器，可将文本转换为特征向量，并使其与共享嵌入空间中的对应
    图像特征保持对齐。

    属性：
        模型 (clip.model.CLIP)：已加载的 CLIP 模型。
        image_preprocess (callable)：图像预处理变换。
        device (torch.device)：加载模型所使用的设备。

    方法：
        tokenize：将输入文本转换为 CLIP 词元。
        encode_text：将分词后的文本编码为归一化的特征向量。

    示例：
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 CLIP 文本编码器。

        此类使用 OpenAI 的 CLIP 模型实现 TextModel 接口，用于文本编码。方法会加载指定尺寸的预训练 CLIP
        模型，并为文本编码任务完成准备工作。

        参数：
            size (str)：模型尺寸标识符，例如 ``'ViT-B/32'``。
            device (torch.device)：加载模型所使用的设备。
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device, download_root=str(WEIGHTS_DIR / "clip"))
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 CLIP 词元。

        参数：
            texts (str | list[str])：要分词的单个文本或文本列表。
            truncate (bool，可选)：是否截断超过 CLIP 上下文长度的文本。默认为 True，可避免输入过长导致运行时
                错误，同时允许显式关闭截断。

        返回：
            (torch.Tensor)：已分词的文本张量，形状为 ``(batch_size, context_length)``，可直接输入模型。

        示例：
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
            >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # 强制进行长度检查
            >>> print(strict_tokens.shape)  # 提示词少于 77 个词元时，形状和内容与 tokens 相同
        """
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量。

        此方法通过 CLIP 模型处理分词后的文本输入，生成特征向量，然后将其归一化为单位长度。这些归一化向量
        可用于文本与图像的相似度比较。

        参数：
            texts (torch.Tensor)：分词后的文本输入，通常由 ``tokenize()`` 方法创建。
            dtype (torch.dtype，可选)：输出特征的数据类型。

        返回：
            (torch.Tensor)：归一化的文本特征向量，长度为 1（L2 范数为 1）。

        示例：
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将图像编码为归一化的特征向量。

        此方法通过 CLIP 模型处理图像输入，生成特征向量，然后将其归一化为单位长度。这些归一化向量可用于
        文本与图像的相似度比较。

        参数：
            image (PIL.Image | torch.Tensor)：输入图像，可以是 PIL 图像或预处理后的张量。如果提供 PIL 图像，
                会使用模型的图像预处理函数将其转换为张量。
            dtype (torch.dtype，可选)：输出特征的数据类型。

        返回：
            (torch.Tensor)：归一化的图像特征向量，长度为 1（L2 范数为 1）。

        示例：
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """Apple MobileCLIP 文本编码器，用于高效的文本编码。

    此类使用 Apple 的 MobileCLIP 模型实现 TextModel 接口，为视觉语言任务提供高效的文本编码能力；与标准
    CLIP 模型相比，其计算开销更低。

    属性：
        模型 (mobileclip.model.MobileCLIP)：已加载的 MobileCLIP 模型。
        tokenizer (callable)：处理文本输入的分词器函数。
        device (torch.device)：加载模型所使用的设备。
        config_size_map (dict)：模型尺寸标识符到模型配置名称的映射。

    方法：
        tokenize：将输入文本转换为 MobileCLIP 词元。
        encode_text：将分词后的文本编码为归一化的特征向量。

    示例：
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 MobileCLIP 文本编码器。

        此类使用 Apple 的 MobileCLIP 模型实现 TextModel 接口，以高效完成文本编码。

        参数：
            size (str)：模型尺寸标识符，例如 ``'s0'``、``'s1'``、``'s2'``、``'b'`` 或 ``'blt'``。
            device (torch.device)：加载模型所使用的设备。
        """
        try:
            import mobileclip
        except ImportError:
            # 优先使用 Ultralytics 的分支，因为 Apple MobileCLIP 仓库依赖的 torchvision 版本不正确
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP 词元。

        参数：
            texts (list[str])：要分词的文本字符串列表。

        返回：
            (torch.Tensor)：已分词的文本输入，形状为 ``(batch_size, sequence_length)``。

        示例：
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量。

        参数：
            texts (torch.Tensor)：分词后的文本输入。
            dtype (torch.dtype，可选)：输出特征的数据类型。

        返回：
            (torch.Tensor)：已应用 L2 归一化的文本特征向量。

        示例：
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型尺寸
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """加载 MobileCLIP 的 TorchScript 跟踪版本。

    此类使用 Apple 的 MobileCLIP 模型实现 TextModel 接口，并以 TorchScript 格式提供经过优化的推理能力，
    从而高效完成视觉语言任务中的文本编码。

    属性：
        encoder (torch.jit.ScriptModule)：已加载的 TorchScript MobileCLIP 文本编码器。
        tokenizer (callable)：处理文本输入的分词器函数。
        device (torch.device)：加载模型所使用的设备。

    方法：
        tokenize：将输入文本转换为 MobileCLIP 词元。
        encode_text：将分词后的文本编码为归一化的特征向量。

    示例：
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """初始化 MobileCLIP TorchScript 文本编码器。

        此类使用 Apple 的 MobileCLIP 模型和 TorchScript 格式，以优化后的推理性能完成文本编码。

        参数：
            device (torch.device)：加载模型所使用的设备。
            weight (str)：TorchScript 模型权重文件的路径。
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP 词元。

        参数：
            texts (list[str])：要分词的文本字符串列表。
            truncate (bool，可选)：是否截断超过分词器上下文长度的文本。默认为 True，与 CLIP 的行为保持一致，
                可避免长文本导致运行时错误。

        返回：
            (torch.Tensor)：已分词的文本输入，形状为 ``(batch_size, sequence_length)``。

        示例：
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # 如果超过 77 个词元，将引发 RuntimeError
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量。

        参数：
            texts (torch.Tensor)：分词后的文本输入。
            dtype (torch.dtype，可选)：输出特征的数据类型。

        返回：
            (torch.Tensor)：已应用 L2 归一化的文本特征向量。

        示例：
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型尺寸
        """
        # 注意：这里无需再次归一化，因为归一化操作已内置于 TorchScript 模型中
        return self.encoder(texts).to(dtype)


def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """根据指定的变体构建文本编码模型。

    参数：
        variant (str)：模型变体，格式为 ``"base:size"``，例如 ``"clip:ViT-B/32"`` 或 ``"mobileclip:s0"``。
        device (torch.device，可选)：加载模型所使用的设备。

    返回：
        (TextModel)：已实例化的文本编码模型。

    示例：
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    else:
        raise ValueError(f"Unrecognized base model '{base}'. Supported models are 'clip', 'mobileclip', 'mobileclip2'.")
