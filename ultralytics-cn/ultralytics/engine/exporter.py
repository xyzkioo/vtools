# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
DEEPX                   | `deepx`                   | yolo26n_deepx_model/
Qualcomm QNN            | `qnn`                     | yolo26n_qnn.onnx
LiteRT                  | `litert`                  | yolo26n.tflite
Hailo                   | `hailo`                   | yolo26n_hailo_model/
Huawei Ascend           | `ascend`                  | yolo26n_ascend_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo26n.pt')
    results = model.export(format='onnx')
    results = model.export(format='onnx', quantize=8, data='coco8.yaml')  # INT8 ONNX

CLI:
    $ yolo mode=export model=yolo26n.pt format=onnx
    $ yolo mode=export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml

Inference:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (macOS-only)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # IMX
                         yolo26n_rknn_model         # RKNN
                         yolo26n_executorch_model   # ExecuTorch
                         yolo26n_axelera_model      # Axelera AI
                         yolo26n_deepx_model        # DEEPX
                         yolo26n_qnn.onnx           # Qualcomm QNN
                         yolo26n.tflite             # LiteRT
                         yolo26n_ascend_model       # Huawei Ascend
"""

from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ultralytics import __version__
from ultralytics.cfg import QUANTIZE_DOCS_URL, TASK2CALIBRATIONDATA, TASK2DATA, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend, check_class_names, default_class_names
from ultralytics.nn.modules import (
    OBB,
    OBB26,
    Attention,
    C2f,
    Classify,
    Depth,
    Detect,
    Pose,
    Pose26,
    RTDETRDecoder,
    Segment,
    Segment26,
    SemanticSegment,
)
from ultralytics.nn.tasks import ClassificationModel, DepthModel, DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_DOCKER,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    QNN_HTP_TARGETS,
    RKNN_CHIPS,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
    is_jetson,
)
from ultralytics.utils.checks import (
    IS_PYTHON_MINIMUM_3_9,
    IS_PYTHON_MINIMUM_3_13,
    check_imgsz,
    check_requirements,
    check_version,
    is_intel,
)
from ultralytics.utils.files import file_size
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_3,
    TORCH_2_8,
    TORCH_2_9,
    select_device,
)


def export_formats():
    """返回 Ultralytics YOLO 支持的导出格式字典。"""
    #          格式、参数、后缀、CPU、GPU、参数列表、环境
    x = [
        ["PyTorch", "-", ".pt", True, True, [], "base"],
        [
            "TorchScript",
            "torchscript",
            ".torchscript",
            True,
            True,
            ["batch", "quantize", "nms", "dynamic"],
            "base",
        ],
        [
            "ONNX",
            "onnx",
            ".onnx",
            True,
            True,
            ["batch", "data", "dynamic", "quantize", "opset", "simplify", "nms", "fraction"],
            "base",
        ],
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
            ["batch", "data", "dynamic", "quantize", "nms", "fraction"],
            "base",
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "data", "dynamic", "quantize", "opset", "simplify", "workspace", "nms", "fraction"],
            "base",
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "quantize", "nms"], "coreml"],
        [
            "TensorFlow SavedModel",
            "saved_model",
            "_saved_model",
            True,
            True,
            ["batch", "data", "fraction", "quantize", "opset", "keras", "nms"],
            "tensorflow",
        ],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch", "opset"], "tensorflow"],
        [
            "TensorFlow Edge TPU",
            "edgetpu",
            "_edgetpu.tflite",
            True,
            False,
            ["data", "fraction", "quantize", "opset"],
            "tensorflow",
        ],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"], "base"],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "dynamic", "quantize", "opset", "simplify", "nms"], "mnn"],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "quantize"], "ncnn"],
        ["IMX", "imx", "_imx_model", True, True, ["data", "quantize", "fraction", "nms"], "isolated-imx"],
        [
            "RKNN",
            "rknn",
            "_rknn_model",
            False,
            False,
            ["batch", "name", "quantize", "opset", "simplify", "data", "fraction"],
            "isolated-rknn",
        ],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"], "executorch"],
        [
            "Axelera AI",
            "axelera",
            "_axelera_model",
            False,
            False,
            ["batch", "quantize", "fraction", "data"],
            "isolated-axelera",
        ],
        [
            "DEEPX",
            "deepx",
            "_deepx_model",
            False,
            False,
            ["data", "quantize", "opset", "simplify", "optimize"],
            "isolated-deepx",
        ],
        [
            "Qualcomm QNN",
            "qnn",
            "_qnn.onnx",
            False,
            False,
            ["batch", "name", "quantize", "opset", "simplify", "fraction", "data"],
            "base",
        ],
        ["LiteRT", "litert", ".tflite", True, False, ["batch", "quantize", "data", "fraction"], "litert"],
        [
            "Hailo",
            "hailo",
            "_hailo_model",
            False,
            False,
            ["name", "quantize", "data", "fraction", "simplify", "conf", "iou"],
            "base",
        ],
        [
            "Huawei Ascend",
            "ascend",
            "_ascend_model",
            False,
            False,
            ["batch", "name", "quantize", "opset", "simplify", "nms"],
            "base",
        ],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments", "Env"], zip(*x)))


EXPORT_ENVS = {
    "base": {
        "python": None,
        "extras": ["export-base"],
        "torch": None,
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": [],
    },
    "tensorflow": {
        "python": "3.12",
        "extras": ["export-base", "export-tensorflow"],
        "torch": None,
        "requirements": [
            "onnx2tf>=1.26.3,<1.29.0",
            "tf_keras<=2.19.0",
            "sng4onnx>=1.0.1",
            "onnx_graphsurgeon>=0.3.26",
            "ai-edge-litert>=1.2.0",
            "onnxruntime",
            "protobuf>=5",
        ],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=saved_model model=yolo26n.pt imgsz=32"],
    },
    "coreml": {
        "python": "3.13",
        "extras": ["export-base", "export-coreml"],
        "torch": ">=2.12",
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=coreml model=yolo26n.pt imgsz=32"],
    },
    "mnn": {
        "python": "3.13",
        "extras": ["export-base"],
        "torch": None,
        "requirements": ["MNN>=2.9.6", "aliyun-log-python-sdk", "protobuf<6.0.0,>=3.20.3"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=mnn model=yolo26n.pt imgsz=32"],
    },
    "ncnn": {
        "python": "3.13",
        "extras": ["export-base"],
        "torch": None,
        "requirements": ["ncnn", "pnnx==20260526"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=ncnn model=yolo26n.pt imgsz=32"],
    },
    "executorch": {
        "python": "3.13",
        "extras": ["export-base", "export-executorch"],
        "torch": ">=2.12",
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=executorch model=yolo26n.pt imgsz=32"],
    },
    "isolated-imx": {
        "python": "3.11",
        "extras": ["export-base"],
        "torch": ">=2.9,<2.12",
        "requirements": [
            "model-compression-toolkit>=2.4.1",
            "edge-mdt-cl<1.1.0",
            "edge-mdt-tpc>=1.2.0",
            "pydantic<2.12",
            "imx500-converter[pt]>=3.17.3",
        ],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=imx model=yolo11n.pt imgsz=32 data=coco8.yaml"],
    },
    "isolated-rknn": {
        "python": "3.11",
        "extras": ["export-base"],
        "torch": "==2.4",
        "requirements": ["rknn-toolkit2>=2.3.2", "onnx>=1.16.1,<1.19.0", "setuptools<82"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=rknn model=yolo26n.pt imgsz=32 quantize=16"],
    },
    "isolated-axelera": {
        # Axelera devkit 1.7.0 未提供 Python 3.13 的 wheel 包。
        "python": "3.12",
        "extras": ["export-base"],
        # Axelera 导出要求 2.8.0 <= torch < 2.12.0。
        "torch": ">=2.8,<2.12",
        "requirements": [
            "axelera-devkit==1.7.0",
            "omnimalloc==0.5.0",
            "numpy<=2.3.5",
            "onnx>=1.12.0,<2.0.0",
            "onnxslim>=0.1.71",
        ],
        "indexes": [
            ("--extra-index-url", "https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple"),
        ],
        # 使用 Python protobuf 运行时，以兼容 Axelera 编译器。
        "env": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
        "smoke": ["yolo export format=axelera model=yolo26n.pt imgsz=64 data=coco8.yaml"],
    },
    "isolated-deepx": {
        # dx-com 2.3.0 未提供 Python 3.13 的 wheel 包。
        "python": "3.12",
        "extras": ["export-base", "export-deepx"],
        "torch": ">=2.8,<2.12",
        "requirements": [],
        "indexes": [
            ("--find-links", "https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html"),
        ],
        # DeepX 导出仅支持非 aarch64 架构的 Linux 系统。
        "env": {},
        "smoke": ["yolo export format=deepx model=yolo26n.pt imgsz=32 data=coco8.yaml"],
    },
    "litert": {
        "python": "3.13",
        "extras": ["export-base", "export-litert"],
        "torch": None,
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=litert model=yolo26n.pt imgsz=32"],
    },
}


# 各格式的导出精度支持。未设置或设置为 32 时请求 FP32，但 FP32_UNSUPPORTED_FORMATS 中列出的格式除外。
FP16_FORMATS = frozenset({"torchscript", "onnx", "openvino", "engine", "coreml", "mnn", "ncnn", "rknn", "ascend"})
INT8_FORMATS = frozenset(
    {
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "edgetpu",
        "mnn",
        "imx",
        "rknn",
        "axelera",
        "deepx",
        "hailo",
        "litert",
    }
)
W8A16_FORMATS = frozenset(
    {"coreml", "imx", "qnn", "litert"}
)  # INT8 权重 + 16 位激活（FP16；LiteRT 上为 INT16）
W8A32_FORMATS = frozenset({"litert"})  # INT8 权重 + FP32 激活（动态/仅权重 INT8，无需校准）
FP32_UNSUPPORTED_FORMATS = frozenset({"edgetpu", "imx", "rknn", "axelera", "deepx", "qnn", "hailo", "ascend"})
# 每种量化精度对应（标签、支持格式），用于在错误信息中列出有效选项。32/None（FP32）通用于除 FP32_UNSUPPORTED_FORMATS 外的格式。
QUANTIZE_PRECISIONS = (
    ("16 (FP16)", FP16_FORMATS),
    ("8 (INT8)", INT8_FORMATS),
    ("'w8a16' (INT8 weights + INT16 activations)", W8A16_FORMATS),
    ("'w8a32' (dynamic INT8)", W8A32_FORMATS),
)


def validate_args(format, passed_args, valid_args):
    """根据导出格式验证参数。

    参数：
        format (str): 导出格式。
        passed_args (SimpleNamespace): 导出期间使用的参数。
        valid_args (list): 该格式支持的有效参数列表。

    异常：
        AssertionError: 使用了不支持的参数，或该格式没有列出支持的参数时抛出。
    """
    # 特定格式的参数来自导出表；跳过推理参数和 quantize（已在上面验证）
    export_args = sorted(set().union(*export_formats()["Arguments"]) - {"conf", "iou", "name", "quantize"})

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # exporter defaults
    default_args = get_cfg(DEFAULT_CFG, custom)
    if passed_args.quantize is not None:  # 除 FP32_UNSUPPORTED_FORMATS 外，32/None（FP32）通用
        options = [label for label, formats in QUANTIZE_PRECISIONS if format in formats]
        if format not in FP32_UNSUPPORTED_FORMATS:
            options.append("32 (FP32)")
        hint = f"format='{format}' supports quantize={', '.join(options) or 'none'} (or None for FP32). See {QUANTIZE_DOCS_URL}"
        if passed_args.quantize == 16:  # FP16
            assert format in FP16_FORMATS, f"ERROR ❌️ quantize=16 (FP16) is not supported; {hint}"
        elif passed_args.quantize == 8:  # INT8
            assert format in INT8_FORMATS, f"ERROR ❌️ quantize=8 (INT8) is not supported; {hint}"
        elif passed_args.quantize == "w8a16":  # INT8 权重 + 16 位激活（FP16；LiteRT 上为 INT16）
            assert format in W8A16_FORMATS, f"ERROR ❌️ quantize='w8a16' is not supported; {hint}"
        elif passed_args.quantize == "w8a32":  # INT8 权重 + FP32 激活（动态/仅权重 INT8）
            assert format in W8A32_FORMATS, f"ERROR ❌️ quantize='w8a32' is not supported; {hint}"
        elif passed_args.quantize == 32:  # FP32
            assert format not in FP32_UNSUPPORTED_FORMATS, f"ERROR ❌️ quantize=32 (FP32) is not supported; {hint}"
    for arg in export_args:
        not_default = getattr(passed_args, arg, getattr(default_args, arg, None)) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO 导出装饰器，即 @try_export。"""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """导出模型。"""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # 导出的文件/目录，或 (文件/目录, *) 元组
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.1, f"{mb:.3f} MB output model too small (likely corrupt or unsupported ops)"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise

    return outer_func


class Exporter:
    """将 YOLO 模型导出为各种格式的类。

    此类支持将 YOLO 模型导出为 ONNX、TensorRT、CoreML、TensorFlow 等多种格式，
    并负责格式验证、设备选择、模型准备以及各支持格式的实际导出过程。

    属性：
        args (SimpleNamespace): 导出器配置参数。
        callbacks (dict): 不同导出事件的回调函数字典。
        im (torch.Tensor): 导出期间用于模型推理的输入张量。
        model (torch.nn.Module): 要导出的 YOLO 模型。
        file (Path): 待导出模型文件的路径。
        output_shape (tuple): 模型输出张量的形状。
        pretty_name (str): 用于显示的格式化模型名称。
        metadata (dict): 模型元数据，包括描述、作者和版本等。
        device (torch.device): 加载模型所在的设备。
        imgsz (list): 模型输入图像尺寸。

    方法：
        __call__：处理导出流程的主要导出方法。
        get_int8_calibration_dataloader：构建 INT8 校准数据加载器。
        export_torchscript：将模型导出为 TorchScript 格式。
        export_onnx：将模型导出为 ONNX 格式。
        export_openvino：将模型导出为 OpenVINO 格式。
        export_paddle：将模型导出为 PaddlePaddle 格式。
        export_mnn：将模型导出为 MNN 格式。
        export_ncnn：将模型导出为 NCNN 格式。
        export_coreml：将模型导出为 CoreML 格式。
        export_engine：将模型导出为 TensorRT 格式。
        export_saved_model：将模型导出为 TensorFlow SavedModel 格式。
        export_pb：将模型导出为 TensorFlow GraphDef 格式。
        export_edgetpu：将模型导出为 Edge TPU 格式。
        export_rknn：将模型导出为 RKNN 格式。
        export_imx：将模型导出为 IMX 格式。
        export_executorch：将模型导出为 ExecuTorch 格式。
        export_axelera：将模型导出为 Axelera 格式。
        export_deepx：将模型导出为 DEEPX 格式。

    示例：
        将 YOLO26 模型导出为 TorchScript 格式
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolo26n.pt")  # 导出为 yolo26n.torchscript

        使用指定参数导出
        >>> args = {"format": "onnx", "dynamic": True, "quantize": 8, "data": "coco8.yaml"}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolo26n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 Exporter 类。

        参数：
            cfg (str | Path | dict | SimpleNamespace, optional): 配置文件路径或配置对象。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """导出模型，并以字符串形式返回最终导出路径。

        返回：
            (str): 导出文件或目录的路径（最后一个导出产物）。
        """
        t = time.time()
        fmt = self.args.format.lower()  # 转换为小写
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        if fmt in {"huawei", "cann", "om"}:  # 'ascend' aliases
            fmt = self.args.format = "ascend"
        if fmt in {"tflite", "tfjs"}:  # 已弃用格式，已由统一的 Google LiteRT 导出替代
            LOGGER.warning(
                f"format='{fmt}' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT "
                f"format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert"
            )
            fmt = self.args.format = "litert"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # 可用的导出格式
        if fmt not in fmts:
            import difflib

            # 如果格式无效，则获取最接近的匹配项
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 匹配需要达到 60% 相似度
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        is_tf_format = fmt in {"saved_model", "pb", "edgetpu"}

        # Device
        self.dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # 先将整数/列表转换为字符串
            device_str = str(self.args.device)
            self.dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # 将设备更新为 "0"
            assert self.dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if fmt == "imx" and self.args.device is None and torch.cuda.is_available():
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # 将设备更新为 "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # 参数兼容性检查
        fmt_keys = dict(zip(fmts_dict["Argument"], fmts_dict["Arguments"]))[fmt]
        validate_args(fmt, self.args, fmt_keys)
        if fmt in {"deepx", "axelera", "imx", "edgetpu", "qnn", "hailo"} and self.args.quantize not in {8, "w8a16"}:
            if self.args.quantize == 32:
                raise ValueError(
                    f"{fmt} export only supports INT8, but got an explicit quantize=32 (FP32) request. "
                    f"See {QUANTIZE_DOCS_URL}"
                )
            LOGGER.warning(f"{fmt} export requires INT8 quantization, enabling it.")
            self.args.quantize = "w8a16" if fmt == "qnn" else 8
        if fmt in {"axelera", "hailo"} and not self.args.data:
            self.args.data = TASK2CALIBRATIONDATA.get(model.task)
        if fmt == "hailo":
            assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
            blocks = {str(x[2]) for x in model.yaml.get("backbone", []) + model.yaml.get("head", [])}
            family = Path(getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")).stem.lower() or (
                "yolov8" if "C2f" in blocks else "yolo11" if {"C3k2", "C2PSA"} <= blocks else ""
            )
            task26 = {Segment26: "segmentation", Pose26: "pose", OBB26: "OBB"}.get(type(model.model[-1]))
            if task26:
                raise ValueError(f"Hailo export does not currently support YOLO26 {task26} models.")
            if (
                model.task not in {"detect", "segment", "pose", "obb", "classify", "semantic", "depth"}
                or type(model.model[-1]) not in {Detect, Segment, Pose, OBB, Classify, SemanticSegment, Depth}
                or not family.startswith(("yolov8", "yolo11", "yolo26"))
            ):
                raise ValueError(
                    "Hailo export currently supports YOLOv8/YOLO11/YOLO26 detection and classification models, "
                    "YOLOv8/YOLO11 segmentation, pose, and OBB models, and YOLO26 semantic segmentation and depth "
                    "models."
                )
            if model.task in {"semantic", "depth"} and not family.startswith("yolo26"):
                raise ValueError(f"Hailo export supports {model.task} models only for YOLO26.")
            if self.args.end2end is not None:
                raise ValueError(
                    "Hailo export selects the model output path automatically; remove the end2end argument."
                )
            self.args.name = str(self.args.name or "hailo8l").lower()
            hailo_archs = ("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l")
            if self.args.name not in hailo_archs:
                raise ValueError(f"Invalid Hailo architecture '{self.args.name}'. Valid names are {hailo_archs}.")
        if fmt == "axelera" and model.task == "segment" and any(isinstance(m, Segment26) for m in model.modules()):
            raise ValueError("Axelera export does not currently support YOLO26 segmentation models.")
        if fmt == "imx":
            if model.task == "depth":
                raise ValueError("IMX export is not supported for depth models.")
            if not self.args.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu", "qnn"}:
                # 某些导出格式不支持 topk，因此禁用 end2end 分支
                model.end2end = False
                LOGGER.warning(f"{fmt.upper()} export does not support end2end models, disabling end2end branch.")
            if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
                # 静态激活量化会压缩端到端类别索引输出；先导出原始输出，稍后再运行 NMS
                model.end2end = False
                LOGGER.warning("LiteRT INT8 export does not support end2end models, disabling end2end branch.")
            if fmt == "engine":
                try:
                    import tensorrt as trt

                    if check_version(trt.__version__, "<8.5.0"):
                        # https://github.com/ultralytics/ultralytics/issues/24607
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT versions earlier than 8.5.0 do not support the Mod operator in end-to-end models, disabling the end2end branch. "
                            "Please upgrade TensorRT to 8.5.0 or later to enable end2end export."
                        )

                    if (
                        self.args.quantize == 8
                        and check_version(trt.__version__, ">=10.3.0,<10.4.0")  # JetPack 6 builds report 10.3.0.x
                        and is_jetson(jetpack=6)
                    ):
                        # https://github.com/ultralytics/ultralytics/issues/23841
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT 10.3.0 on JetPack 6 with int8 has known end2end build issues, disabling end2end branch. "
                            "For a fix, see https://docs.ultralytics.com/guides/nvidia-jetson#why-does-my-tensorrt-int8-export-disable-end2end-on-jetpack-6"
                            ""
                        )
                except ImportError:
                    pass
        if self.args.quantize == 16 and fmt == "torchscript" and self.device.type == "cpu":
            raise ValueError("FP16 TorchScript export is only supported on GPU, i.e. use device=0.")
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # 检查图像尺寸
        if fmt == "axelera" and min(self.imgsz) < 64:
            raise ValueError(f"Axelera export requires imgsz>=64, but got imgsz={self.imgsz}.")
        if fmt == "rknn":
            if self.args.quantize == 8 and model.task != "detect":
                raise ValueError(
                    "Rockchip RKNN INT8 export is only supported for detection models. "
                    "Use FP16 (quantize=16) for other tasks."
                )
            if not self.args.name:
                LOGGER.warning(
                    "Rockchip RKNN export requires a missing 'name' arg for processor type. "
                    "Using default name='rk3588'."
                )
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."
            )
            if self.args.name in {"rv1103", "rv1106", "rv1103b", "rv1106b"} and self.args.quantize != 8:
                if self.args.quantize not in {None, 8}:
                    raise ValueError(
                        f"Rockchip target '{self.args.name}' only supports INT8, but got quantize={self.args.quantize}. "
                        f"See {QUANTIZE_DOCS_URL}"
                    )
                LOGGER.warning(f"Rockchip target '{self.args.name}' requires INT8 quantization, enabling it.")
                self.args.quantize = 8
            elif self.args.quantize is None:
                self.args.quantize = 16
        if fmt == "ascend":
            # 不设置 SoC 允许列表：有效的 --soc_version 值取决于已安装的 Ascend-cann-kernels-* 软件包，
            # 硬编码列表会拒绝有效目标。ATC 会自行报告未知 SoC。
            if not self.args.name:
                LOGGER.warning(
                    "Huawei Ascend export requires a missing 'name' arg for the target SoC. "
                    "Using default name='Ascend310B4'."
                )
                self.args.name = "Ascend310B4"
            if not str(self.args.name).startswith("Ascend"):
                raise ValueError(
                    f"Invalid Ascend SoC name='{self.args.name}'. Expected a CANN --soc_version such as "
                    f"'Ascend310P3' or 'Ascend310B4'. See https://docs.ultralytics.com/integrations/ascend"
                )
            if self.args.quantize is None:
                self.args.quantize = 16  # Ascend AI Core 卷积只接受 FP16/INT8 输入，不接受 FP32
        if fmt == "qnn":
            if not self.args.name:
                LOGGER.warning(
                    "Qualcomm QNN export requires a missing 'name' arg for the target Hexagon HTP architecture. "
                    "Using default name='73' (Snapdragon 8 Gen 2)."
                )
                self.args.name = "73"
            self.args.name = str(self.args.name).lower().lstrip("v")  # 接受 '73'、'v73' 或受支持的 SoC
            assert self.args.name in QNN_HTP_TARGETS, (
                f"Invalid Qualcomm QNN target '{self.args.name}'. Valid targets are {tuple(QNN_HTP_TARGETS)}."
            )
        if self.args.nms and model.task in {"semantic", "depth"}:
            LOGGER.warning(f"'nms=True' is not valid for {model.task} models. Forcing 'nms=False'.")
            self.args.nms = False
        if fmt == "coreml" and self.args.nms and model.task not in {"detect", "segment", "pose"}:
            LOGGER.warning(
                "CoreML 'nms=True' is only supported for detect, segment and pose models. Forcing 'nms=False'."
            )
            self.args.nms = False
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert fmt != "onnx" or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # 为 NMS 导出设置 conf 默认值
        if fmt == "mnn" and self.args.nms:
            if self.args.dynamic:
                raise ValueError("Alibaba MNN export does not support combining 'dynamic=True' with 'nms=True'.")
            if model.task not in {"detect", "pose"}:
                raise ValueError("Alibaba MNN export with 'nms=True' only supports detect and pose models.")
        if fmt == "coreml":
            if self.args.nms and model.task != "detect" and self.args.quantize == 32:
                # CoreML 仅在 FP16 路径上计算 NMSModel 的数据相关形状，FP32 ML Program 会静默丢弃检测结果。
                # detect 不受影响，因为它的 NMS 是 Apple 流水线阶段。
                LOGGER.warning(f"CoreML 'nms=True' requires FP16 for {model.task} models. Forcing 'quantize=16'.")
                self.args.quantize = 16
            if self.args.batch > 1:
                assert self.args.dynamic, (
                    "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
                )
            if self.args.dynamic:
                assert not self.args.nms, (
                    "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
                )
                assert model.task != "classify" and not isinstance(model.model[-1], RTDETRDecoder), (
                    "'dynamic=True' is not supported for CoreML classification or RT-DETR models."
                )
        if (fmt in {"engine", "coreml"} or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if fmt == "edgetpu":
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # OpenVINO INT8 导出错误：https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.quantize in {8, "w8a16"} and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # 分配默认数据集
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
        # 在导出到 Intel CPU 时推荐使用 OpenVINO
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino"
                )
            SETTINGS["openvino_msg"] = False

        # 输入
        im = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # 更新模型
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse(imgsz=self.imgsz)

        if fmt == "imx":
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if fmt == "edgetpu":
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        if fmt == "executorch":
            from ultralytics.utils.export.executorch import executorch_wrapper

            model = executorch_wrapper(model)
        for m in model.modules():
            if isinstance(m, Attention) and fmt == "coreml" and self.args.format.lower() != "mlmodel":
                m.format = fmt
            if isinstance(m, (Classify, SemanticSegment, Depth)):
                m.export = True
                m.format = self.args.format
                # 语义 argmax 固化需要整数图输出；TensorRT 仅在 TRT>=10 支持 uint8 输出
                #（Jetson TRT 8.x 会拒绝该输出）。从软件包名称读取版本，避免在此处导入 tensorrt。
                if isinstance(m, SemanticSegment) and fmt == "engine":
                    cuda_major = (torch.version.cuda or "12").split(".")[0]
                    m.bake_argmax = check_version(f"tensorrt-cu{cuda_major}", ">=10.0.0") or check_version(
                        "tensorrt", ">=10.0.0"
                    )
            if isinstance(m, (Detect, RTDETRDecoder)):  # 包含 Segment、Pose、OBB 等所有 Detect 子类
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                # 将 max_det 限制为可用查询/锚框数量（TensorRT 兼容性要求）
                available = (
                    m.num_queries
                    if isinstance(m, RTDETRDecoder)
                    else sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                )
                m.max_det = min(self.args.max_det, available)
                m.agnostic_nms = self.args.agnostic_nms
                # CoreML 检测保留 IOSDetectModel 自身的 xywh 处理；分割/姿态通过
                # 与其他格式一样，NMSModel 需要 xyxy 边界框。
                m.xyxy = self.args.nms and (fmt != "coreml" or model.task != "detect")
                m.shape = None  # 为新的导出输入尺寸重置缓存形状
                if hasattr(model, "pe") and hasattr(m, "fuse") and not hasattr(m, "lrpc"):  # 用于 YOLOE 模型
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU 不支持 FlexSplitV，而 split 可以生成更简洁的 ONNX 图。
                m.forward = m.forward_split

        if model.task == "semantic" and fmt in {"qnn", "coreml", "ascend"}:
            # 面向 NPU 的语义导出使用紧凑的 uint8 类别图，而不是浮点 logits：输出 logits 会迫使使用方每帧在 CPU 上
            # 对约 2000 万个浮点数执行反量化和 argmax（在 Hexagon 上测得耗时波动为 123-1065 ms）。LiteRT 不应用此策略，
            # 因为其 GPU 委托无法编译 ArgMax（int64 索引），而完整图的 CPU 回退比 GPU logits 加使用方 argmax 更慢。
            # predict/val 两种形式均可接受。
            model = ClassMapModel(model)

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms and fmt not in {"coreml", "imx"} else model(im)
        if self.args.quantize == 16 and fmt in {"onnx", "torchscript"} and self.device.type != "cpu":
            im, model = im.half(), model.half()  # 转为 FP16

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().astimezone().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "head": type(model.model[-1]).__name__,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in self.args if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
            "end2end": getattr(model, "end2end", False),
        }  # 模型元数据
        if self.dla is not None:
            self.metadata["dla"] = self.dla  # 确保 `AutoBackend` 存在 DLA 设备时使用正确的设备
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape
            if hasattr(model, "kpt_names"):
                self.metadata["kpt_names"] = model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")

        # Export
        if is_tf_format:
            f, keras_model = self.export_saved_model()
            if fmt == "pb":
                f = self.export_pb(keras_model=keras_model)
            if fmt == "edgetpu":
                f = self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")
        else:
            f = getattr(self, f"export_{fmt}")()

        # Finish
        if f:
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            q = "quantize=16" if self.args.quantize == 16 else ""  # 提示 val/predict 使用 FP16 推理的标志
            inference_commands = (
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                if fmt in AutoBackend._BACKEND_MAP
                else ""
            )
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', Path(f).resolve())}"
                f"{inference_commands}"
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # 最终导出产物的路径

    def get_int8_calibration_dataloader(self, prefix=""):
        """构建并返回用于 INT8 模型校准的数据加载器。"""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        cfg = deepcopy(self.args)
        cfg.imgsz = max(self.imgsz)
        if self.model.task == "classify":
            import torchvision.transforms as T  # scope for faster 'import ultralytics'

            data = check_cls_dataset(self.args.data, split=self.args.split)
            dataset = ClassificationDataset(data[self.args.split or "val"], args=cfg, augment=False)
            if self.args.fraction < 1.0:
                dataset.samples = dataset.samples[: round(len(dataset.samples) * self.args.fraction)]
            # INT8 后端会将图像除以 255，因此像分类推理一样输出中心裁剪的 uint8 [0, 255] 图像
            dataset.torch_transforms = T.Compose([T.Resize(cfg.imgsz), T.CenterCrop(cfg.imgsz), T.PILToTensor()])
        else:
            data = check_det_dataset(self.args.data, split=self.args.split)
            dataset = build_yolo_dataset(
                cfg,
                data[self.args.split or "val"],
                self.args.batch,
                data,
                mode="val",
                fraction=self.args.fraction,
            )
        if hasattr(dataset, "transforms") and hasattr(dataset.transforms.transforms[0], "new_shape"):
            dataset.transforms.transforms[0].new_shape = self.imgsz  # 非正方形 imgsz 的 LetterBox
        n = len(dataset)
        if n < 1:
            raise ValueError(f"The calibration dataset must have at least 1 image, but found {n} images.")
        batch = min(self.args.batch, n)
        if n < self.args.batch:
            LOGGER.warning(
                f"{prefix} calibration dataset has only {n} images, reducing calibration batch size to {batch}."
            )
        if self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)  # 批次加载所需

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):  # noqa: B008
        """将 YOLO 模型导出为 TorchScript 格式。"""
        from ultralytics.utils.export.torchscript import torch2torchscript

        return torch2torchscript(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            output_file=self.file.with_suffix(".torchscript"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):  # noqa: B008
        """将 YOLO 模型导出为 ONNX 格式。"""
        requirements = ["onnx>=1.16.1,<1.19.0" if self.args.format == "rknn" else "onnx>=1.12.0,<2.0.0"]
        if self.args.simplify or (self.args.format == "onnx" and self.args.quantize == 8):
            # 将 onnxruntime 变体作为可互换候选项，使 AutoUpdate 保留已安装版本
            # （例如 QNN 导出的 onnxruntime-qnn），避免重新安装稳定版并破坏 ABI。
            ort = "onnxruntime-gpu" if "cuda" in self.device.type else "onnxruntime"
            requirements += [(ort, "onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn")]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.82"]
        check_requirements(requirements)
        import onnx

        from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

        opset = self.args.opset or best_onnx_opset(onnx)
        assert not isinstance(self.model.model[-1], RTDETRDecoder) or opset >= 16, "RTDETR export requires opset>=16"
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DepthModel):
                dynamic["output0"] = {0: "batch", 2: "height", 3: "width"}  # shape(1,1,640,640) 密集图，而不是锚框
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:
                dynamic["output0"].pop(2)
        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset  # 用于 NMSModel
            self.args.simplify = True  # 修复与 topk 相关的 OBB 运行时错误

        model = NMSModel(self.model, self.args) if self.args.nms else self.model
        # 按输入尺寸归一化坐标，使 RKNN 的逐张量 INT8 缩放保持类别分数不变。
        if (
            self.args.format == "rknn"
            and self.args.quantize == 8
            and self.model.task in {"detect", "segment", "pose", "obb"}
            and not self.metadata["end2end"]
        ):
            from ultralytics.utils.export.engine import _NormalizeCoords

            model = _NormalizeCoords(
                model,
                int(self.im.shape[2]),
                int(self.im.shape[3]),
                self.model.task,
                len(self.metadata["names"]),
                self.metadata.get("kpt_shape"),
            )

        with arange_patch(dynamic=bool(dynamic), quantize=self.args.quantize, fmt=self.args.format):
            torch2onnx(
                model,
                self.im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )

        # 检查
        model_onnx = onnx.load(f)  # 加载 ONNX 模型

        # 简化
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # CANN 要求 ONNX NonMaxSuppression 节点提供可选的分数阈值输入。分数已经在 NMSModel 中由 args.conf
        # 完成过滤，因此使用 0 可以保持图的语义不变。
        if self.args.format == "ascend":
            for i, node in enumerate(model_onnx.graph.node):
                if node.op_type == "NonMaxSuppression" and len(node.input) == 4:
                    threshold_name = f"ascend_nms_score_threshold_{i}"
                    node.input.append(threshold_name)
                    model_onnx.graph.initializer.append(
                        onnx.helper.make_tensor(threshold_name, onnx.TensorProto.FLOAT, [1], [0.0])
                    )

        # 元数据
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR 版本
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # CPU 导出时转换为 FP16（GPU 导出在跟踪期间已通过 model.half() 转为 FP16）
        if self.args.quantize == 16 and self.args.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
        del model_onnx
        if self.args.quantize == 8 and self.args.format == "onnx":
            from ultralytics.utils.export.onnx import onnx_int8_quantize

            source = Path(f)
            f_int8 = str(source.with_name(f"{source.stem}_int8{source.suffix}"))
            f = onnx_int8_quantize(
                source,
                f_int8,
                self.get_int8_calibration_dataloader(prefix),
                self._transform_fn,
                batch=0 if self.args.dynamic else self.args.batch,
                prefix=prefix,
            )
            source.unlink(missing_ok=True)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):  # noqa: B008
        """将 YOLO 模型导出为 OpenVINO 格式。"""
        from ultralytics.utils.export.openvino import torch2openvino

        # OpenVINO <= 2025.1.0 在 macOS 15.4+ 上会报错：https://github.com/openvinotoolkit/openvino/issues/30023
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

        def serialize(ov_model, file):
            """设置 RT 信息、序列化模型并保存元数据 YAML。"""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.save_model(ov_model, file, compress_to_fp16=self.args.quantize == 16)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # 添加 metadata.yaml

        calibration_dataset = None
        if self.args.quantize == 8:
            check_requirements("packaging>=23.2")  # 构建 nncf wheel 前必须先安装
            check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
            import nncf

            calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)

        ov_model = torch2openvino(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            dynamic=self.args.dynamic,
            quantize=self.args.quantize,
            calibration_dataset=calibration_dataset,
            int8_detect=isinstance(self.model.model[-1], Detect),
            prefix=prefix,
        )

        suffix = f"_{'int8_' if self.args.quantize == 8 else ''}openvino_model{os.sep}"
        f = str(self.file).replace(self.file.suffix, suffix)
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):  # noqa: B008
        """将 YOLO 模型导出为 PaddlePaddle 格式。"""
        from ultralytics.utils.export.paddle import torch2paddle

        return torch2paddle(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_litert(self, prefix=colorstr("LiteRT:")):  # noqa: B008
        """使用 litert_torch 将 YOLO 模型导出为 LiteRT 格式，并可选择 INT8 量化。

        支持 ``quantize=8``（静态 INT8，INT8 权重和 INT8 激活，需要校准 ``data``）、
        ``quantize='w8a16'``（静态量化，INT8 权重和 INT16 激活，需要校准 ``data``）以及
        ``quantize='w8a32'``（动态或仅权重量化 INT8，INT8 权重和 FP32 激活，不需要校准）。
        """
        assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
        from ultralytics.utils.export.litert import torch2litert

        return torch2litert(
            self.model,
            self.im,
            self.file,
            quantize=self.args.quantize,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix)
            if self.args.quantize in {8, "w8a16"}
            else None,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):  # noqa: B008
        """使用 MNN 将 YOLO 模型导出为 MNN 格式：https://github.com/alibaba/MNN。"""
        from ultralytics.utils.export.mnn import onnx2mnn

        return onnx2mnn(
            onnx_file=self.export_onnx(),
            output_file=self.file.with_suffix(".mnn"),
            quantize=self.args.quantize,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):  # noqa: B008
        """使用 PNNX 将 YOLO 模型导出为 NCNN 格式：https://github.com/pnnx/pnnx。"""
        from ultralytics.utils.export.ncnn import torch2ncnn

        return torch2ncnn(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_ncnn_model/"),
            quantize=self.args.quantize,
            metadata=self.metadata,
            device=self.device,
            prefix=prefix,
        )

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):  # noqa: B008
        """将 YOLO 模型导出为 CoreML 格式。"""
        mlmodel = self.args.format.lower() == "mlmodel"  # 请求旧版 *.mlmodel 导出格式
        from ultralytics.utils.export.coreml import IOSDetectModel, pipeline_coreml, torch2coreml

        # numpy 2.4.x 会破坏 coremltools 的 CoreML 导出：https://github.com/apple/coremltools/issues/2633
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        if self.args.nms and self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel)
        elif self.args.nms:  # 分割、姿态：将 NMS 固化到追踪图中，而不是使用 Apple 的 NMS 流水线，
            model = NMSModel(self.model, self.args)  # 后者无法在抑制过程中携带掩码/关键点
        else:
            model = self.model

        if self.args.dynamic:
            h, w = self.imgsz
            lb_h = lb_w = 32
            if getattr(self.model, "end2end", False):
                # 端到端图会将 TopK k=max_det 固化，因此声明的最小输入仍必须提供 >= k 个锚点，
                # 否则 CoreML 加载模型时会拒绝；根据跟踪时的默认尺寸按比例缩小范围
                stride = int(self.model.stride.max())
                r = self.model.model[-1].max_det / sum(int(h / s) * int(w / s) for s in self.model.stride.tolist())
                lb_h = max(lb_h, int(np.ceil(h * r**0.5 / stride)) * stride)
                lb_w = max(lb_w, int(np.ceil(w * r**0.5 / stride)) * stride)
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=lb_h, upper_bound=h * 2, default=h),
                    ct.RangeDim(lower_bound=lb_w, upper_bound=w * 2, default=w),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

        quantize = 16 if self.args.nms and not mlmodel and self.args.quantize is None else self.args.quantize
        self.metadata["args"]["quantize"] = quantize
        ct_model = torch2coreml(
            model=model,
            inputs=inputs,
            im=self.im,
            classifier_names=list(self.model.names.values()) if self.model.task == "classify" else None,
            mlmodel=mlmodel,
            quantize=quantize,
            metadata=self.metadata,
            prefix=prefix,
        )

        if self.args.nms and self.model.task == "detect":
            ct_model = pipeline_coreml(
                ct_model,
                self.output_shape,
                weights_dir=None if mlmodel else ct_model.weights_dir,
                metadata=self.metadata,
                mlmodel=mlmodel,
                iou=self.args.iou,
                conf=self.args.conf,
                agnostic_nms=self.args.agnostic_nms,
                prefix=prefix,
            )

        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # 保存 *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):  # noqa: B008
        """将 YOLO 模型导出为 TensorRT 格式：https://developer.nvidia.com/tensorrt。"""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # 在导入 TRT 前运行 https://github.com/ultralytics/ultralytics/issues/7016
        from ultralytics.utils.export.engine import onnx2engine

        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT 引擎文件
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
            self.args.quantize,
            self.args.dynamic,
            self.im.shape,
            dla=self.dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.quantize == 8 else None,
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):  # noqa: B008
        """将 YOLO 模型导出为 TensorFlow SavedModel 格式。"""
        assert not (MACOS and IS_PYTHON_MINIMUM_3_13), (
            "TensorFlow exports not supported on macOS with Python>=3.13: the ai-edge-litert macOS wheel fails to load "
            "(missing libpywrap_litert_common.dylib). TensorFlow export works on Linux Python 3.13."
        )
        from ultralytics.utils.export.tensorflow import onnx2saved_model

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # 删除输出目录

        # 导出到 TF
        images = None
        if self.args.quantize == 8 and self.args.data:
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
            )

        # 导出到 ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.args.opset = self.args.opset or 19
            assert self.args.opset <= 19, "RTDETR TensorFlow export requires opset<=19"
        self.args.simplify = True
        f_onnx = self.export_onnx()  # 确保 ONNX 可用
        keras_model = onnx2saved_model(
            f_onnx,
            f,
            quantize=self.args.quantize,
            images=images,
            disable_group_convolution=self.args.format == "edgetpu",
            cuda=self.device.type == "cuda",
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        # 添加 TFLite 元数据
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # 或 keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):  # noqa: B008
        """将 YOLO 模型导出为 TensorFlow GraphDef *.pb 格式：https://github.com/leimao/Frozen-Graph-TensorFlow。"""
        from ultralytics.utils.export.tensorflow import keras2pb

        return keras2pb(keras_model, output_file=self.file.with_suffix(".pb"), prefix=prefix)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):  # noqa: B008
        """将 YOLO 模型导出为 Axelera 格式。"""
        assert LINUX and not (ARM64 and IS_DOCKER), (
            "export is only supported on Linux and is not supported on ARM64 Docker."
        )
        assert TORCH_2_8, "export requires torch>=2.8.0."

        from ultralytics.utils.export.axelera import torch2axelera

        output_dir = self.file.parent / f"{self.file.stem}_axelera_model"
        return torch2axelera(
            model=self.model,
            output_dir=output_dir,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            model_name=self.file.stem,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):  # noqa: B008
        """将 YOLO 模型导出为 ExecuTorch *.pte 格式。"""
        assert TORCH_2_9, f"ExecuTorch requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        from ultralytics.utils.export.executorch import torch2executorch

        return torch2executorch(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_executorch_model/"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):  # noqa: B008
        """将 YOLO 模型导出为 Edge TPU 格式：https://coral.ai/docs/edgetpu/models-intro/。"""
        from ultralytics.utils.export.tensorflow import tflite2edgetpu

        output_file = tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        self._add_tflite_metadata(output_file)
        return output_file

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):  # noqa: B008
        """将 YOLO 模型导出为 RKNN 格式，并可选择 INT8 量化。"""
        from ultralytics.utils.export.rknn import onnx2rknn

        if self.args.opset and self.args.opset > 19:
            LOGGER.warning(f"{prefix} rknn-toolkit2 requires opset<=19, setting opset=19.")
        self.args.opset = min(self.args.opset or 19, 19)  # rknn-toolkit expects opset<=19
        self.im = self.im[:1]  # RKNN Toolkit 要求先使用批次为 1 的 ONNX 模型进行校准，再扩展批次
        f_onnx = self.export_onnx()
        output_dir = Path(str(self.file).replace(self.file.suffix, f"_rknn_model{os.sep}"))
        rknn_dataset = None
        if self.args.quantize == 8:
            dataloader = self.get_int8_calibration_dataloader(prefix)
            image_paths = getattr(dataloader.dataset, "im_files", None)
            if image_paths is None and hasattr(dataloader.dataset, "samples"):
                image_paths = [x[0] for x in dataloader.dataset.samples]
            if not image_paths:
                raise ValueError("RKNN INT8 export requires a calibration dataset with image file paths.")
            output_dir.mkdir(parents=True, exist_ok=True)
            rknn_dataset = output_dir / "dataset.txt"
            rknn_dataset.write_text("\n".join(str(Path(x).resolve()) for x in image_paths) + "\n")
        try:
            return onnx2rknn(
                onnx_file=f_onnx,
                output_dir=output_dir,
                name=self.args.name,
                quantize=self.args.quantize,
                batch=self.args.batch,
                dataset=rknn_dataset,
                metadata=self.metadata,
                prefix=prefix,
            )
        finally:
            if self.args.quantize == 8:  # INT8 图包含归一化坐标，因此不可复用
                Path(f_onnx).unlink(missing_ok=True)

    @try_export
    def export_ascend(self, prefix=colorstr("Ascend:")):  # noqa: B008
        """将 YOLO 模型导出为华为昇腾离线模型（.om）格式。"""
        from ultralytics.utils.export.ascend import _check_atc, onnx2ascend

        _check_atc()  # 在 ONNX 追踪前检查，避免缺少工具链时先进行完整导出
        if self.args.opset and self.args.opset > 17:
            LOGGER.warning(f"{prefix} the CANN ONNX parser requires opset<=17, setting opset=17.")
        self.args.opset = min(self.args.opset or 17, 17)
        return onnx2ascend(
            onnx_file=self.export_onnx(),
            output_dir=self.file.parent / f"{self.file.stem}_ascend_model",
            name=self.args.name,
            imgsz=self.imgsz,
            batch=self.args.batch,
            channels=self.im.shape[1],
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):  # noqa: B008
        """将 YOLO 模型导出为 IMX 格式。"""
        assert LINUX, (
            "Export only supported on Linux."
            "See https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter?version=3.17.3&progLang="
        )
        assert IS_PYTHON_MINIMUM_3_9, "IMX export is only supported on Python 3.9 or above."

        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
        from ultralytics.utils.export.imx import torch2imx

        return torch2imx(
            model=self.model,
            output_dir=str(self.file).replace(self.file.suffix, "_imx_model/"),
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            metadata=self.metadata,
            dataset=partial(self.get_int8_calibration_dataloader, prefix),
            prefix=prefix,
        )

    @try_export
    def export_deepx(self, prefix=colorstr("DEEPX:")):  # noqa: B008
        """将 YOLO 模型导出为 DEEPX 格式。"""
        assert LINUX and not ARM64, "DEEPX export only supported on non-aarch64 Linux"
        from ultralytics.utils.export.deepx import onnx2deepx

        f = self.export_onnx()
        return onnx2deepx(
            onnx_file=f,
            imgsz=self.imgsz,
            dataset=self.get_int8_calibration_dataloader(prefix),
            metadata=self.metadata,
            optimize=self.args.optimize,
            prefix=prefix,
        )

    @try_export
    def export_qnn(self, prefix=colorstr("Qualcomm QNN:")):  # noqa: B008
        """使用 ONNX Runtime QNN 将 YOLO 模型导出为 Qualcomm QNN 上下文二进制文件。"""
        from ultralytics.utils.export.qnn import onnx2qnn

        # 包装为适配 Hexagon 的输入输出：输入采用通道最后格式（语义类别图的包装与格式无关）。
        model, im = self.model, self.im
        try:
            self.model, self.im = QNNModel(model), im.permute(0, 2, 3, 1)
            f_onnx = self.export_onnx()
        finally:
            self.model, self.im = model, im
        return onnx2qnn(
            onnx_file=f_onnx,
            output_file=str(self.file.with_name(f"{self.file.stem}_qnn.onnx")),
            dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            name=self.args.name,
            metadata=self.metadata,
            batch=0 if self.args.dynamic else self.args.batch,
            prefix=prefix,
        )

    @try_export
    def export_hailo(self, prefix=colorstr("Hailo:")):  # noqa: B008
        """将 YOLO 模型导出为 Hailo 可执行格式（HEF）。"""
        try:
            import tensorflow as tf
            from hailo_sdk_client import ClientRunner
        except ImportError as e:
            raise ImportError("Hailo export requires the Hailo Dataflow Compiler.") from e

        calibration_dataloader = self.get_int8_calibration_dataloader(prefix)
        calibration_size = len(calibration_dataloader.dataset)
        LOGGER.warning(
            f"\nHailo level-2 optimization will use {calibration_size} calibration images. "
            "Hailo recommends at least 1,024 representative images for best accuracy. "
            'Pass data="path/to/dataset.yaml". '
            "See https://docs.ultralytics.com/integrations/hailo#export-a-hailo-hef-model"
        )
        head_index = len(self.model.model) - 1
        head = self.model.model[head_index]
        one2one = getattr(self.model, "end2end", False)
        task = self.model.task
        if task == "classify":
            # Classify 头以 Gemm -> Softmax 结束；在 Softmax 处截断，使 HEF 返回与 PyTorch 模型相同的 (1, nc) 概率。
            # DFC 会将 softmax 转换为原生层。
            end_nodes = [f"/model.{head_index}/Softmax"]
        elif task == "semantic":
            # 多类别 Hailo-15/10（DFC 5.x）头可在芯片上编译双线性上采样和 ArgMax。Hailo-8/8L
            #（DFC 3.x）无法编译 Resize，单类别头也使用阈值而不是 ArgMax，因此两者都在分类器 logits 处截断并在主机端执行归约。
            head.bake_argmax = head.nc > 1 and self.args.name in {"hailo10h", "hailo15h", "hailo15l"}
            end_nodes = [
                f"/model.{head_index}/ArgMax"
                if head.bake_argmax
                else f"/model.{head_index}/classifier/classifier.1/Conv"
            ]
        elif task == "depth":
            # Depth 头以 clamp/exp、对数仿射校准和 4 倍上采样结束。在最终 logit 卷积处截断
            #（head.3，密集解码器的最后一层），使 a16 HEF 携带原始 logit，主机端
            # 与 Depth.forward 一致。字符串约定与 detect 的 `.2/Conv` 相同，不新增检测头属性。
            end_nodes = [f"/model.{head_index}/head/head.3/Conv"]
        else:
            scales = range(len(head.one2one_cv2 if one2one else head.cv2))
            if one2one:
                end_nodes = [
                    f"/model.{head_index}/one2one_cv{branch}.{i}/one2one_cv{branch}.{i}.2/Conv"
                    for branch in (2, 3)
                    for i in scales
                ]
            elif task in {"segment", "pose", "obb"}:
                # 每个尺度包含 reg/cls/extra 三元组（extra 为掩码系数、关键点或角度）；分割任务还会添加原型。
                end_nodes = [
                    f"/model.{head_index}/cv{branch}.{i}/cv{branch}.{i}.2/Conv" for i in scales for branch in (2, 3, 4)
                ]
                if task == "segment":
                    end_nodes.append(f"/model.{head_index}/proto/cv3/act/Mul")
            else:
                end_nodes = [
                    f"/model.{head_index}/cv{branch}.{i}/cv{branch}.{i}.2/Conv" for i in scales for branch in (2, 3)
                ]
        self.args.opset = 11
        f_onnx = Path(self.export_onnx())
        output_dir = self.file.parent / f"{self.file.stem}_hailo_model"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            runner = ClientRunner(hw_arch=self.args.name)
            runner.translate_onnx_model(str(f_onnx), self.file.stem, end_node_names=end_nodes)
            model_script = [
                "normalization1 = normalization([0, 0, 0], [255, 255, 255])",
                "model_optimization_flavor(optimization_level=2)",
                f"post_quantization_optimization(finetune, policy=enabled, dataset_size={calibration_size})",
            ]
            if one2one or task == "depth":
                # 输出使用 a16：无 NMS 检测 logits 和单个稠密深度 logits 都需要更宽的激活范围
                # （a8 会压缩深度图；已在 Hailo-8L 上验证）。
                outputs = ", ".join(f"output_layer{i + 1}" for i in range(len(end_nodes)))
                model_script.append(f"quantization_param([{outputs}], precision_mode=a16_w16)")
            elif task in {"classify", "semantic"}:
                pass  # softmax/class-map 已经是图输出，无需修改 NMS 或激活函数
            else:
                outputs = [layer.inputs[0].rsplit("/", 1)[-1] for layer in runner.get_hn_model().get_output_layers()]
                if task in {"segment", "pose", "obb"}:
                    # 仅将 sigmoid 固化到类别卷积中（每个尺度 reg/cls/extra 三元组的第 1 个位置）。
                    # 掩码系数、原型、关键点和角度保持原始形式，在主机端解码。
                    model_script.extend(
                        f"change_output_activation({outputs[i]}, sigmoid)" for i in range(1, 3 * len(scales), 3)
                    )
                else:
                    nms_config = output_dir / "nms_config.json"
                    nms_config.write_text(
                        json.dumps(
                            {
                                "nms_scores_th": self.args.conf if self.args.conf is not None else 0.25,
                                "nms_iou_th": self.args.iou,
                                "image_dims": self.imgsz,
                                "max_proposals_per_class": 100,
                                "classes": len(self.model.names),
                                "regression_length": 16,
                                "background_removal": False,
                                "background_removal_index": 0,
                                "bbox_decoders": [
                                    {
                                        "name": f"bbox_decoder_{stride}",
                                        "stride": stride,
                                        "reg_layer": outputs[i * 2],
                                        "cls_layer": outputs[i * 2 + 1],
                                    }
                                    for i, stride in enumerate(int(x) for x in head.stride)
                                ],
                            },
                            indent=2,
                        )
                    )
                    model_script.extend(
                        f"change_output_activation({outputs[i]}, sigmoid)" for i in range(1, len(outputs), 2)
                    )
                    model_script.append(f'nms_postprocess("{nms_config}", meta_arch=yolov8, engine=cpu)')
                    model_script.append("allocator_param(width_splitter_defuse=disabled)")
            runner.load_model_script("\n".join(model_script))

            def calibration_dataset():
                for batch in calibration_dataloader:
                    for image in batch["img"].permute(0, 2, 3, 1).numpy().astype(np.float32):
                        yield image, {}

            runner.optimize(
                lambda: tf.data.Dataset.from_generator(
                    calibration_dataset,
                    output_signature=(tf.TensorSpec(shape=(*self.imgsz, 3), dtype=tf.float32), {}),
                )
            )
            (output_dir / f"{self.file.stem}.hef").write_bytes(runner.compile())
            YAML.save(
                output_dir / "metadata.yaml",
                {
                    **self.metadata,
                    "hailo_arch": self.args.name,
                    "nms": task == "detect" and not one2one,
                    "semantic_baked": task == "semantic" and head.bake_argmax,
                    # Depth 学习得到的对数仿射校准在主机端执行，因此必须随元数据保存。
                    **({"cal_a": float(head.cal_a), "cal_b": float(head.cal_b)} if task == "depth" else {}),
                },
            )
            return str(output_dir)
        finally:
            f_onnx.unlink(missing_ok=True)

    def _add_tflite_metadata(self, file):
        """按照 https://ai.google.dev/edge/litert/models/metadata 为 *.tflite 模型添加元数据。"""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    @staticmethod
    def _transform_fn(data_item) -> np.ndarray:
        """用于 INT8 校准的量化预处理变换（Axelera、OpenVINO、ONNX、QNN）。"""
        data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
        assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
        im = data_item.numpy().astype(np.float32) / 255.0  # 将 uint8 转为 fp16/32，并将 0 - 255 转为 0.0 - 1.0
        return im[None] if im.ndim == 3 else im

    def add_callback(self, event: str, callback):
        """将给定回调添加到指定事件。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """执行指定事件的所有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)


class ExportWrapper(torch.nn.Module):
    """导出阶段模型包装器的基类：保存被包装模型并转发属性查找。

    子类会根据特定部署约定（布局、输出归约）调整融合 YOLO 模型的推理输入输出，
    同时让导出器能够像操作原模型一样操作包装器。
    """

    def __init__(self, model):
        """包装准备导出的融合 YOLO `model`。"""
        super().__init__()
        # 使用私有名称保存，使属性转发将 `wrapper.model` 解析为被包装模型自身的
        # `model`（即 nn.Sequential），从而保持 `self.model.model[-1]` 等导出器代码不变。
        self._model = model
        self.task = model.task

    def __getattr__(self, name):
        """将属性查找（model、names、stride、yaml、args 等）转发到被包装模型。"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


class QNNModel(ExportWrapper):
    """为 Qualcomm QNN 导出包装使用通道末尾推理输入的 YOLO 模型。

    标准 ONNX 导出会对其进行跟踪（`export_qnn` 使用通道末尾的虚拟输入替换原模型）。图像图接收 `[N, H, W, C]`
    格式，这是 Hexagon HTP 的原生布局，也是摄像头流程的输出布局；因此 ONNX Runtime 的布局转换器会在生成上下文时
    将包装器的 Transpose 折叠到 NPU 分区中，NPU 和调用应用都无需在每次推理时额外承担布局转换开销。

    属性：
        task (str): The wrapped model's task, forwarded for the ONNX export plumbing.
    """

    def forward(self, x):
        """对归一化到 [0, 1] 的通道末尾 `[N, H, W, C]` 输入执行推理。"""
        return self._model(x.permute(0, 3, 1, 2))  # 被包装模型采用 NCHW，转置会折叠到 NPU 图中


class ClassMapModel(ExportWrapper):
    """将语义分割 logits 归约为紧凑的整数类别图，以便导出。

    此包装器用于 QNN、Core ML 和 Ascend 语义分割导出，此时 argmax 在 NPU 上执行。部署端需要逐像素类别索引，
    若传输浮点 logits，则每帧都需要在调用方 CPU 上对大型张量（标准 640px 移动端输入约 800 万个值）执行反量化和
    argmax，这在移动端 NPU 上实测速度较慢且波动很大。argmax 不可放在模型自身的 forward 中，因为它不可微
    （训练需要 logits），因此像 `NMSModel` 只在导出时添加抑制一样，在此处附加该操作。

    属性：
        task (str): 被包装模型的任务（"semantic"）。
        dtype (torch.dtype): 类别索引数据类型；类别数超过 256 时使用 int32，否则使用 uint8。
    """

    def __init__(self, model):
        """包装融合语义分割 `model`，使导出结果输出类别索引而不是 logits。"""
        super().__init__(model)
        # 与 int32 相比，uint8 可将 NPU 到 CPU 的输出传输量降至四分之一，Core ML 会按规范将其提升为 int32；
        # 只有类别超过 256 个、uint8 索引可能产生歧义时才使用 int32。
        self.dtype = torch.uint8 if len(model.names) <= 256 else torch.int32

    def forward(self, x):
        """运行被包装模型，返回 `[N, H, W]` 整数类别图，而不是浮点 logits。"""
        y = self._model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        # 单通道（二分类）模型对 logit 应用阈值，与 nc == 1 时的 predict/val 语义保持一致
        return (y.argmax(1) if y.shape[1] > 1 else y[:, 0].gt(0)).to(self.dtype)


class NMSModel(torch.nn.Module):
    """为 Detect、Segment、Pose 和 OBB 内置 NMS 的模型包装器。"""

    def __init__(self, model, args):
        """初始化 NMSModel。

        参数：
            model (torch.nn.Module): 要使用 NMS 后处理进行包装的模型。
            args (SimpleNamespace): 导出参数。
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format == "saved_model"

    def forward(self, x):
        """执行带 NMS 后处理的推理，支持 Detect、Segment、OBB 和 Pose。

        参数：
            x (torch.Tensor): 预处理张量，形状为 (B, C, H, W)。

        返回：
            (torch.Tensor | tuple): 形状为 (B, max_det, 4 + 2 + extra_shape) 的张量，其中 B 为批次大小；
                对于分割模型，则返回 (detections, proto) 元组。
        """
        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = {"device": pred.device, "dtype": pred.dtype}
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # 从 shape(1,84,6300) 转换为 shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # Segment、OBB、Pose 的额外输出
        if self.args.dynamic and self.args.batch > 1:  # 由于循环展开，批次大小必须始终保持一致
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        if self.args.dynamic and self.args.format == "onnx" and self.obb:
            pred = torch.cat((pred, pred.new_zeros(pred.shape[0], self.args.max_det * 5, pred.shape[2])), dim=1)
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        # 输出形状为 (N, max_det, 4 个坐标 + 1 个类别分数 + 1 个类别标签 + extra_shape)。
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # 掩码为空时会出现 TFLite GatherND 错误
                score *= mask
                # 显式指定长度，否则会重塑失败，固定为 `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # 经过试验，`8` 是获得正确 OBB NMS 结果的最小值
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # 为 NMS 归一化边界框，因为类别偏移量过大会导致 int8 量化问题
            nmsbox = multiplier * (nmsbox / torch._shape_as_tensor(x)[2:].max().to(**kwargs))
            if not self.args.agnostic_nms:  # 按类别执行 NMS
                end = 2 if self.obb else 4
                # 必须完全显式展开，否则会在 reshape 时出错
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.quantize == 8)  # OpenVINO INT8 使用 triu 时出错
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # 用零填充到 max_det 大小以避免重塑错误。对展平的一维视图进行填充，而不是填充二维张量的第 0 维，
            # 因为 CoreML 的 MIL 转换仅支持对一维张量进行动态填充。
            c = dets.shape[-1]
            pad = (0, (self.args.max_det - dets.shape[0]) * c)
            out[i] = torch.nn.functional.pad(dets.reshape(-1), pad).reshape(self.args.max_det, c)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
