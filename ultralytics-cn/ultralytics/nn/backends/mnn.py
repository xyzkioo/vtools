# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class MNNBackend(BaseBackend):
    """MNN（Mobile Neural Network）推理后端。

    使用阿里巴巴 MNN 框架加载并运行 MNN 模型（.mnn 文件），针对移动端和边缘部署进行优化，并支持配置线程数和精度。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .mnn 文件加载阿里巴巴 MNN 模型。

        参数：
            weight (str | Path): .mnn 模型文件路径。
        """
        LOGGER.info(f"Loading {weight} for MNN inference...")
        check_requirements("MNN")
        import MNN

        config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
        rt = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(weight, [], [], runtime_manager=rt, rearrange=True)
        self.expr = MNN.expr

        # 从 bizCode 加载元数据
        info = self.net.get_info()
        if "bizCode" in info:
            try:
                self.apply_metadata(json.loads(info["bizCode"]))
            except json.JSONDecodeError:
                pass

    def forward(self, im: torch.Tensor) -> list:
        """使用 MNN 运行时执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表): NumPy 数组列表形式的模型预测结果。
        """
        input_var = self.expr.const(im.data_ptr(), im.shape)
        output_var = self.net.onForward([input_var])
        # 注意：必须执行 copy()，否则在 ARM 设备上可能得到错误结果
        if output_var:
            return [x.read().copy() for x in output_var]
        if self.metadata.get("args", {}).get("nms") and self.task in {"detect", "pose"}:
            return [np.empty((im.shape[0], 0, 6))]
        raise RuntimeError("Alibaba MNN inference returned no output tensors.")
