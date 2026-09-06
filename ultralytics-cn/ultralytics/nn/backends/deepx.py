# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class DeepXBackend(BaseBackend):
    """用于 DEEPX 硬件加速器的 DEEPX NPU 推理后端。

    加载已编译的 DEEPX 模型（.dxnn 文件），并使用 DEEPX DX-Runtime 执行推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .dxnn 文件的目录加载 DEEPX 模型。

        参数：
            weight (str | Path): 包含 .dxnn 二进制文件的 DEEPX 模型目录路径。

        异常：
            ImportError: 未安装 ``dx_engine`` Python 软件包时抛出。
            FileNotFoundError: 给定目录中找不到 .dxnn 文件时抛出。
        """
        try:
            from dx_engine import InferenceEngine
        except ImportError as e:
            raise ImportError(
                "DEEPX inference requires the DEEPX DX-Runtime and `dx_engine` Python package. "
                "See https://docs.ultralytics.com/integrations/deepx#runtime-installation for installation instructions."
            ) from e

        LOGGER.info(f"Loading {weight} for DEEPX inference...")

        w = Path(weight)
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        self.apply_metadata(self.read_metadata(found))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """在 DEEPX NPU 上执行推理。

        根据 DEEPX 运行时约定，将每张图像从 BCHW 浮点格式 [0, 1] 转换为 HWC uint8 格式 [0, 255]，
        对每张图像运行引擎，然后沿批次维度堆叠输出。

        参数：
            im (torch.Tensor): 输入图像张量，格式为 BCHW，归一化到 [0, 1]。

        返回：
            (np.ndarray | 列表[np.ndarray]): 单个数组或数组列表形式的模型预测结果。
        """
        outputs = []
        for sample in im.cpu().numpy():
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
