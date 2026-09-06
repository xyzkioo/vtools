# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class TritonBackend(BaseBackend):
    """用于远程模型服务的 NVIDIA Triton Inference Server 后端。

    通过 HTTP 或 gRPC 协议连接到 NVIDIA Triton Inference Server 实例上托管的模型并执行推理。
    模型使用 triton:// URL 方案指定。
    """

    def load_model(self, weight: str | Path) -> None:
        """连接到 NVIDIA Triton Inference Server 上的远程模型。

        参数：
            weight (str | Path): Triton 模型 URL (e.g., 'triton://host:8000/model_name').
        """
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(weight)

        # 从 Triton 模型复制元数据
        if hasattr(self.model, "metadata"):
            self.apply_metadata(self.model.metadata)

    def forward(self, im: torch.Tensor) -> list:
        """通过 NVIDIA Triton Inference Server 执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表): 来自 Triton 服务器的 NumPy 数组列表形式的模型预测结果。
        """
        return self.model(im.cpu().numpy())
