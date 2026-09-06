# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics YOLO 推理后端。

此包为各种深度学习框架和硬件加速器提供模块化推理后端。
每个后端都实现 `BaseBackend` 接口，可独立使用，也可通过统一的 `AutoBackend` 调度器自动检测格式并路由推理。
"""

from .ascend import AscendBackend
from .axelera import AxeleraBackend
from .base import BaseBackend
from .coreml import CoreMLBackend
from .deepx import DeepXBackend
from .executorch import ExecuTorchBackend
from .hailo import HailoBackend
from .litert import LiteRTBackend
from .mnn import MNNBackend
from .ncnn import NCNNBackend
from .onnx import ONNXBackend, ONNXIMXBackend
from .openvino import OpenVINOBackend
from .paddle import PaddleBackend
from .pytorch import PyTorchBackend, TorchScriptBackend
from .qnn import QNNBackend
from .rknn import RKNNBackend
from .tensorflow import TensorFlowBackend
from .tensorrt import TensorRTBackend
from .triton import TritonBackend

__all__ = [
    "AscendBackend",
    "AxeleraBackend",
    "BaseBackend",
    "CoreMLBackend",
    "DeepXBackend",
    "ExecuTorchBackend",
    "HailoBackend",
    "LiteRTBackend",
    "MNNBackend",
    "NCNNBackend",
    "ONNXBackend",
    "ONNXIMXBackend",
    "OpenVINOBackend",
    "PaddleBackend",
    "PyTorchBackend",
    "QNNBackend",
    "RKNNBackend",
    "TensorFlowBackend",
    "TensorRTBackend",
    "TorchScriptBackend",
    "TritonBackend",
]
