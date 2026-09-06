# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class CoreMLBackend(BaseBackend):
    """用于 Apple 硬件的 CoreML 推理后端。

    使用 coremltools 库加载并执行 CoreML 模型（.mlpackage 文件）推理。
    支持静态和动态输入形状，并处理包含 NMS 的模型输出。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .mlpackage 文件加载 CoreML 模型。

        参数：
            weight (str | Path): .mlpackage 模型文件的路径。
        """
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        LOGGER.info(f"Loading {weight} for CoreML inference...")
        # 在神经引擎（CPU_AND_NE）上运行约比 CPU 快 3 倍；默认的 ComputeUnit.ALL / CPU_AND_GPU
        # 会因 macOS 主机上的 MPSGraph 编译器错误（coremltools 9.x）导致进程中止。CPU_AND_NE 需要 macOS >= 13，
        # 更低版本在下方回退到 CPU_ONLY。CoreML 推理仅支持 macOS，因此该逻辑适用于所有运行此后端的环境。
        # 例外：RT-DETR 在单独使用神经引擎时会损失 FP16 精度且速度更慢，因此通过 ALL 运行。
        meta = self.read_metadata(weight)
        default_unit = ct.ComputeUnit.ALL if meta.get("head") == "RTDETRDecoder" else ct.ComputeUnit.CPU_AND_NE
        try:
            self.model = ct.models.MLModel(weight, compute_units=default_unit)
        except Exception:
            self.model = ct.models.MLModel(weight, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.dynamic = spec.description.input[0].type.HasField("multiArrayType")

        self.apply_metadata(meta)

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """执行 CoreML 推理，并自动处理输入格式。

        参数：
            im (torch.Tensor): 输入图像张量，格式为 BHWC（由 AutoBackend 从 BCHW 转换而来）。

        返回：
            (np.ndarray | 列表[np.ndarray]): NumPy 数组或数组列表形式的模型预测结果。
        """
        im = im.cpu().numpy()
        h, w = im.shape[1:3]

        im = im.transpose(0, 3, 1, 2) if self.dynamic else Image.fromarray((im[0] * 255).astype("uint8"))
        y = self.model.predict({self.input_name: im})
        if "confidence" in y:  # NMS 包括
            from ultralytics.utils.ops import xywh2xyxy

            box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
            cls = y["confidence"].argmax(1, keepdims=True)
            y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
        else:
            y = list(y.values())
        if len(y) == 2 and len(y[1].shape) != 4:  # segmentation 模型
            y = list(reversed(y))
        return y
