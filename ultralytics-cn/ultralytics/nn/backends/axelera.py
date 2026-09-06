# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class AxeleraBackend(BaseBackend):
    """用于 Axelera Metis AI 加速器的 Axelera AI 推理后端。

    加载已编译的 Axelera 模型（.axm 文件），并使用 Axelera AI 运行时 SDK 执行推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .axm 文件的目录加载 Axelera 模型。

        参数：
            weight (str | Path): 包含 .axm 二进制文件的 Axelera 模型目录路径。
        """
        try:
            from axelera.runtime import op
        except ImportError:
            check_requirements(
                "axelera-rt==1.7.0",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
            )

        from axelera.runtime import op

        w = Path(weight)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        self.model = op.load(str(found)).optimized()

        self.apply_metadata(self.read_metadata(found))

    def forward(self, im: torch.Tensor) -> list:
        """在 Axelera 硬件加速器上执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表): 输出数组列表形式的模型预测结果。
        """
        return self.model(im.cpu())
