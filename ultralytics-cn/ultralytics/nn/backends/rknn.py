# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """用于 Rockchip NPU 硬件的 Rockchip RKNN 推理后端。

    使用 RKNN-Toolkit-Lite2 运行时加载并执行 RKNN 模型（.rknn 文件）推理。
    仅支持带有 NPU 硬件的 Rockchip 设备（例如 RK3588、RK3566）。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .rknn 文件或模型目录加载 Rockchip RKNN 模型。

        参数：
            weight (str | Path): .rknn 文件或包含模型的目录路径。

        异常：
            OSError: 如果当前设备不是 Rockchip 设备。
            RuntimeError: 模型加载或运行时初始化失败时抛出。
        """
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """在 Rockchip NPU 上执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BHWC format, normalized to [0, 1].

        返回：
            (列表): 输出数组列表形式的模型预测结果。
        """
        h, w = im.shape[1:3]
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        y = self.model.inference(inputs=im)
        # INT8 导出使用相对于输入的坐标，因此单个逐张量缩放因子即可保持类别分数不变。
        if (
            self.metadata.get("args", {}).get("quantize") == 8
            and self.task in {"detect", "segment", "pose", "obb"}
            and not self.end2end
        ):
            kpt_start = 4 + len(self.names)  # 姿态关键点位于边界框 (4) 和类别分数 (nc) 通道之后
            for x in y:
                if x.ndim == 3:
                    x[:, [0, 2]] *= w
                    x[:, [1, 3]] *= h
                    if self.task == "pose":
                        x[:, kpt_start::3] *= w
                        x[:, kpt_start + 1 :: 3] *= h
        return y
