# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class NCNNBackend(BaseBackend):
    """用于移动端和嵌入式部署的腾讯 NCNN 推理后端。

    加载并执行腾讯 NCNN 模型（*_ncnn_model/ 目录）推理。
    针对移动平台进行优化，并在可用时支持可选的 Vulkan GPU 加速。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .param/.bin 文件对或模型目录加载 NCNN 模型。

        参数：
            weight (str | Path): .param 文件或包含 NCNN 模型文件的目录路径。
        """
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()

        # 如果可用则设置 Vulkan
        if isinstance(self.device, str) and self.device.startswith("vulkan"):
            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(int(self.device.split(":")[1]))
            self.device = torch.device("cpu")
        else:
            self.net.opt.use_vulkan_compute = False

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.param"))

        self.net.load_param(str(w))
        self.net.load_model(str(w.with_suffix(".bin")))

        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """使用 NCNN 运行时执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表[np.ndarray]): NumPy 数组列表形式的模型预测结果，每个输出层对应一个数组。
        """
        outputs = []
        for sample in im.cpu().numpy():
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], self.pyncnn.Mat(sample))
                # 按输出名称排序，暂时解决 pnnx 问题
                outputs.append([np.array(ex.extract(x)[1]) for x in sorted(self.net.output_names())])
        return [np.stack(y) for y in zip(*outputs)]
