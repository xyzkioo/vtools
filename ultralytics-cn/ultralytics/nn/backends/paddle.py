# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class PaddleBackend(BaseBackend):
    """百度 PaddlePaddle 推理后端。

    加载并执行百度 PaddlePaddle 模型（*_paddle_model/ 目录）推理。
    支持 CPU 和 GPU 执行，并自动配置设备和初始化内存池。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .json 和 .pdiparams 文件的目录加载百度 PaddlePaddle 模型。

        参数：
            weight (str | Path): 模型目录或 .pdiparams 文件的路径。
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"
        LOGGER.info(f"Loading {weight} for PaddlePaddle inference...")
        if cuda:
            check_requirements("paddlepaddle-gpu>=3.0.0,<3.3.0")
        elif ARM64:
            check_requirements("paddlepaddle==3.0.0")
        else:
            check_requirements("paddlepaddle>=3.0.0,<3.3.0")

        import paddle.inference as pdi

        w = Path(weight)
        model_file, params_file = None, None

        if w.is_dir():
            model_file = next(w.rglob("*.json"), None)
            params_file = next(w.rglob("*.pdiparams"), None)
        elif w.suffix == ".pdiparams":
            model_file = w.with_name("model.json")
            params_file = w

        if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
            raise FileNotFoundError(f"Paddle model not found in {w}. Both .json and .pdiparams files are required.")

        config = pdi.Config(str(model_file), str(params_file))
        if cuda:
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=self.device.index or 0)

        self.predictor = pdi.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_names = self.predictor.get_output_names()

        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """执行百度 PaddlePaddle 推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表[np.ndarray]): NumPy 数组列表形式的模型预测结果，每个输出句柄对应一个数组。
        """
        self.input_handle.copy_from_cpu(im.cpu().numpy().astype(np.float32, copy=False))
        self.predictor.run()
        return [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
