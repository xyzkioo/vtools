# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.utils import IS_JETSON, LOGGER, is_jetson
from ultralytics.utils.torch_utils import unwrap_model

from .base import BaseBackend


class PyTorchBackend(BaseBackend):
    """用于原生模型执行的 PyTorch 推理后端。

    加载并执行原生 PyTorch 模型（.pt 检查点文件）或预加载的 nn.Module 实例推理。
    支持模型层融合、FP16 精度和 NVIDIA Jetson 兼容性。
    """

    def __init__(
        self,
        weight: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化 PyTorch 后端。

        参数：
            weight (str | Path | nn.Module): .pt 模型文件路径或预加载的 nn.Module 实例。
            device (torch.device): 执行推理的设备（例如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
            fuse (bool): 是否融合 Conv2D + BatchNorm 层以进行优化。
            verbose (bool): 是否输出详细的模型加载消息。
        """
        self.fuse = fuse
        self.verbose = verbose
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | torch.nn.Module) -> None:
        """从检查点文件或 nn.Module 实例加载 PyTorch 模型。

        参数：
            weight (str | torch.nn.Module): .pt 检查点路径或预加载的模块。
        """
        from ultralytics.nn.tasks import BaseModel, load_checkpoint

        if isinstance(weight, torch.nn.Module):
            if self.fuse and hasattr(weight, "fuse"):
                if IS_JETSON and is_jetson(jetpack=5):
                    weight = weight.to(self.device)
                weight = weight.fuse(verbose=self.verbose) if isinstance(weight, BaseModel) else weight.fuse()
            model = weight.to(self.device)
        else:
            model, _ = load_checkpoint(weight, device=self.device, fuse=self.fuse)

        # 提取模型属性
        if hasattr(model, "kpt_shape"):
            self.kpt_shape = model.kpt_shape
        self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
        self.channels = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
        model.half() if self.fp16 else model.float()

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.end2end = getattr(model, "end2end", False)
        self.base_model = isinstance(unwrap_model(model), BaseModel)

    def forward(
        self, im: torch.Tensor, augment: bool = False, embed: list | None = None, **kwargs: Any
    ) -> torch.Tensor | list[torch.Tensor]:
        """执行原生 PyTorch 推理，并支持增强和嵌入提取。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].
            augment (bool): Whether to apply test-time augmentation.
            embed (列表 | None): 用于提取嵌入的层索引列表，或 None。
            **kwargs (Any): 传递给模型 forward 方法的其他关键字参数。

        返回：
            (torch.Tensor | 列表[torch.Tensor]): 张量或张量列表形式的模型预测结果。
        """
        if not self.base_model:  # a foreign nn.Module defines no `augment`/`embed` contract to honor
            return self.model(im, **kwargs)
        return self.model(im, augment=augment, embed=embed, **kwargs)


class TorchScriptBackend(BaseBackend):
    """用于执行序列化模型的 PyTorch TorchScript 推理后端。

    加载并执行通过 torch.jit.trace 创建的 TorchScript 模型（.torchscript 文件）推理，或
    torch.jit.script. Supports FP16 precision and embedded metadata extraction.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False):
        """初始化 TorchScript 后端。

        参数：
            weight (str | Path): .torchscript 模型文件路径。
            device (torch.device): 执行推理的设备（例如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
        """
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str) -> None:
        """从 .torchscript 文件加载 TorchScript 模型，并读取可选的嵌入元数据。

        参数：
            weight (str): .torchscript 模型文件的路径。
        """
        import torchvision  # noqa - TorchScript 模型反序列化所需

        LOGGER.info(f"Loading {weight} for TorchScript inference...")
        self.model = torch.jit.load(weight, map_location=self.device)
        self.model.half() if self.fp16 else self.model.float()
        self.apply_metadata(self.read_metadata(weight))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """执行 TorchScript 推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (torch.Tensor | 列表[torch.Tensor]): 张量或张量列表形式的模型预测结果。
        """
        return self.model(im)
