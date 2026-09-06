# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_executorch_requirements

from .base import BaseBackend


class ExecuTorchBackend(BaseBackend):
    """用于设备端部署的 Meta ExecuTorch 推理后端。

    使用 ExecuTorch 运行时加载并执行 Meta ExecuTorch 模型（.pte 文件）推理。
    同时支持独立的 .pte 文件和带有元数据的目录型模型包。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .pte 文件或目录加载 ExecuTorch 模型。

        参数：
            weight (str | Path): .pte 模型文件或包含模型的目录路径。
        """
        LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
        check_executorch_requirements()

        from executorch.runtime import Runtime

        w = Path(weight)
        program = Runtime.get().load_program(str(next(w.rglob("*.pte")) if w.is_dir() else w))
        self.model = program.load_method("forward")
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """使用 ExecuTorch 运行时执行推理。

        参数：
            im (torch.Tensor): 输入图像张量，格式为 BCHW，已归一化到 [0, 1]。

        返回：
            (列表): ExecuTorch 输出值列表形式的模型预测结果。
        """
        return self.model.execute([im])
