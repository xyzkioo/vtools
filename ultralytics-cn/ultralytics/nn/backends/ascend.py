# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class AscendBackend(BaseBackend):
    """用于 CANN 离线模型的华为 Ascend NPU 推理后端。

    加载编译后的 .om 离线模型，并通过封装 CANN pyACL 绑定的 ais_bench 运行时在 Ascend AI 处理器上执行推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .om 文件的目录加载 Ascend 模型。

        参数：
            weight (str | Path): 包含 .om 离线模型的 Ascend 模型目录路径。

        异常：
            ImportError: 未安装 ``ais_bench`` Python 软件包时抛出。
            FileNotFoundError: 给定目录中找不到 .om 文件时抛出。
        """
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as e:
            raise ImportError(
                "Ascend inference requires the CANN runtime and `ais_bench` Python package. "
                "See https://docs.ultralytics.com/integrations/ascend#runtime-installation for instructions."
            ) from e

        LOGGER.info(f"Loading {weight} for Huawei Ascend inference...")

        w = Path(weight)
        found = next(w.rglob("*.om"), None)
        if found is None:
            raise FileNotFoundError(f"No .om file found in: {w}")

        self.model = InferSession(getattr(self.device, "index", None) or 0, str(found))

        self.apply_metadata(self.read_metadata(found))

    def __del__(self):
        """释放推理会话持有的 Ascend 设备端资源。"""
        if model := getattr(self, "model", None):
            model.free_resource()

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """在 Ascend NPU 上执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (np.ndarray | 列表[np.ndarray]): 单个数组或数组列表形式的模型预测结果。
        """
        y = self.model.infer([im.cpu().numpy()])
        return y[0] if len(y) == 1 else y
