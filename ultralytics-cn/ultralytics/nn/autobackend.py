# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import is_url

from .backends import (
    AscendBackend,
    AxeleraBackend,
    CoreMLBackend,
    DeepXBackend,
    ExecuTorchBackend,
    HailoBackend,
    LiteRTBackend,
    MNNBackend,
    NCNNBackend,
    ONNXBackend,
    ONNXIMXBackend,
    OpenVINOBackend,
    PaddleBackend,
    PyTorchBackend,
    QNNBackend,
    RKNNBackend,
    TensorFlowBackend,
    TensorRTBackend,
    TorchScriptBackend,
    TritonBackend,
)


def check_class_names(names: list | dict) -> dict[int, str]:
    """检查类别名称，并在需要时将其转换为字典格式。

    参数：
        names (列表 | dict): 类别名称，可以是列表或字典格式。

    返回：
        (dict): 字典格式的类别名称，键为整数，值为字符串。

    异常：
        KeyError: 如果类别索引超出数据集范围。
    """
    if isinstance(names, list):  # 名称是列表
        names = dict(enumerate(names))  # 转换为字典
    if isinstance(names, dict):
        # 依次将字符串键转换为整数（例如 '0' 转为 0），并将非字符串值转换为字符串（例如 True 转为 'True'）
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if not n:
            raise KeyError("0-class dataset, at least one class name is required in your dataset YAML.")
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # ImageNet 类别代码，例如 'n01440764'
            from ultralytics.utils import ROOT, YAML

            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # 人类可读的名称
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """从 YAML 文件加载类别名称；如果加载失败，则返回数字类别名称。

    参数：
        data (str | Path, 可选): 包含类别名称的 YAML 文件路径。

    返回：
        (dict): 将类别索引映射到类别名称的字典。
    """
    if data:
        try:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # 上述操作失败时返回默认名称


class AutoBackend(nn.Module):
    """动态选择后端，以运行 Ultralytics YOLO 模型推理。

    AutoBackend 类为各种推理引擎提供统一抽象层，支持多种格式，每种格式的命名约定如下：

        支持的格式和命名约定：
            | 格式                  | 文件后缀          |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |
            | Triton Inference      | triton://模型    |
            | ExecuTorch            | *.pte             |
            | Axelera AI            | *_axelera_model/  |
            | DEEPX                 | *_deepx_model/    |
            | Qualcomm QNN          | *_qnn.onnx        |
            | LiteRT                | *.tflite          |
            | Hailo                 | *_hailo_model/    |
            | Huawei Ascend         | *_ascend_model/   |

    属性：
        backend (BaseBackend): 加载的推理后端实例。
        format (str): 模型格式（例如 'pt'、'onnx'、'engine'）。
        model: 底层模型（PyTorch 后端为 nn.Module，其他后端为后端实例）。
        device (torch.device): 加载模型所在的设备（CPU 或 GPU）。
        task (str): 模型执行的任务类型（detect、segment、semantic、classify、pose、obb）。
        names (dict): 模型可检测类别的名称字典。
        stride (int): 模型步幅，YOLO 模型通常为 32。
        fp16 (bool): 模型是否使用半精度（FP16）推理。
        nhwc (bool): 模型是否要求使用 NHWC 而不是 NCHW 输入格式。

    方法：
        forward：对输入图像运行推理。
        from_numpy：将 NumPy 数组转换为模型设备上的张量。
        warmup：使用虚拟输入预热模型。
        _model_type：根据文件路径确定模型类型。

    示例：
        >>> model = AutoBackend(model="yolo26n.pt", device="cuda")
        >>> results = model(img)
    """

    _BACKEND_MAP = {
        "pt": PyTorchBackend,
        "torchscript": TorchScriptBackend,
        "onnx": ONNXBackend,
        "dnn": ONNXBackend,  # 特殊情况：使用 DNN 的 ONNX
        "openvino": OpenVINOBackend,
        "engine": TensorRTBackend,
        "coreml": CoreMLBackend,
        "saved_model": TensorFlowBackend,
        "pb": TensorFlowBackend,
        "edgetpu": TensorFlowBackend,
        "paddle": PaddleBackend,
        "mnn": MNNBackend,
        "ncnn": NCNNBackend,
        "imx": ONNXIMXBackend,
        "rknn": RKNNBackend,
        "triton": TritonBackend,
        "executorch": ExecuTorchBackend,
        "axelera": AxeleraBackend,
        "deepx": DeepXBackend,
        "qnn": QNNBackend,
        "litert": LiteRTBackend,
        "hailo": HailoBackend,
        "ascend": AscendBackend,
    }

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device | None = None,
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化用于推理的 AutoBackend。

        参数：
            model (str | torch.nn.Module): 模型权重文件路径或模块实例。
            device (torch.device): 运行模型的设备。
            dnn (bool): 是否使用 OpenCV DNN 模块执行 ONNX 推理。
            data (str | Path, 可选): 包含类别名称的附加 data.yaml 文件路径。
            fp16 (bool): 是否启用半精度推理，仅部分后端支持。
            fuse (bool): 是否融合 Conv2D 和 BatchNorm 层以进行优化。
            verbose (bool): 是否启用详细日志。
        """
        super().__init__()
        device = device or torch.device("cpu")
        # 根据路径或 URL 确定模型格式
        format = "pt" if isinstance(model, nn.Module) else self._model_type(model, dnn)

        # 检查该格式是否支持 FP16
        fp16 &= format in {"pt", "torchscript", "onnx", "openvino", "engine", "triton"}

        # 设置设备
        if (
            isinstance(device, torch.device)
            and torch.cuda.is_available()
            and device.type != "cpu"
            and format not in {"pt", "torchscript", "engine", "onnx", "paddle"}
        ):
            device = torch.device("cpu")

        # 选择并初始化适当的后端
        backend_kwargs = {"device": device, "fp16": fp16}

        if format not in self._BACKEND_MAP:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{model}' is not a supported model format. "
                f"Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )
        if format == "pt":
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif format in {"saved_model", "pb", "edgetpu", "dnn"}:
            backend_kwargs["format"] = format
        self.backend = self._BACKEND_MAP[format](model, **backend_kwargs)

        self.nhwc = format in {"coreml", "saved_model", "pb", "edgetpu", "rknn"}
        self.format = format

        # 确保后端具有名称（如果元数据未设置，则使用默认名称）
        if not self.backend.names:
            self.backend.names = default_class_names(data)
        self.backend.names = check_class_names(self.backend.names)

    def __getattr__(self, name: str) -> Any:
        """将属性访问委托给后端。

        这样 AutoBackend 可以透明地公开后端属性，而无需显式复制。

        参数：
            name: 要查找的属性名称。

        返回：
            后端中的属性值。

        异常：
            AttributeError: 如果后端中找不到该属性。
        """
        if "backend" in self.__dict__ and hasattr(self.backend, name):
            return getattr(self.backend, name)
        return super().__getattr__(name)

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> Any:
        """在 AutoBackend 模型上运行推理。

        参数：
            im (torch.Tensor): 要执行推理的图像张量。
            augment (bool): 是否在推理期间执行数据增强。
            embed (列表, 可选): 要返回嵌入的层索引列表。
            **kwargs (Any): 模型配置的其他关键字参数。

        返回：
            (Any): 原始模型输出，其中 NumPy 数组会转换为 `self.device` 上的张量。
        """
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # 将 torch BCHW 转为 numpy BHWC，形状为 (1, 320, 192, 3)
        if self.backend.fp16 and im.dtype != torch.float16:
            im = im.half()

        # 根据后端类型构建 forward 关键字参数
        forward_kwargs = {}
        if self.format == "pt":
            forward_kwargs = {"augment": augment, "embed": embed, **kwargs}

        y = self.backend.forward(im, **forward_kwargs)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # 分割任务且名称未定义
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 116, 8400), (1, 32, 160, 160)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: Any) -> Any:
        """在可能的情况下，将后端输出归一化到模型所在设备。

        参数：
            x (Any): 要归一化的后端输出。

        返回：
            (Any): `self.device` 上的张量；非张量输出保持不变。
        """
        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640), im: torch.Tensor | None = None) -> None:
        """通过执行一次或多次前向传播预热模型。

        参数：
            imgsz (tuple[int, int, int, int]): 虚拟输入形状，格式为（批次、通道、高度、宽度）。
            im (torch.Tensor, 可选): 要复用的输入张量，而不是重新分配虚拟张量。
        """
        from ultralytics.utils.nms import non_max_suppression

        if not self.end2end:
            import torchvision  # noqa（此处导入会触发 nms.py 使用 torchvision NMS）
        if self.format in {"pt", "torchscript", "onnx", "engine", "saved_model", "pb", "triton"} and (
            self.device.type != "cpu" or self.format == "triton"
        ):
            im = (
                im
                if im is not None
                else torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
            )
            for _ in range(2 if self.format == "torchscript" else 1):
                self.forward(im)  # warmup 模型
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 根据经验，16 个边界框效果最好
                warmup_boxes[:, :4] *= im.shape[-1]
                non_max_suppression(warmup_boxes)  # warmup NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt", dnn: bool = False) -> str:
        """接收模型文件路径，并返回模型格式字符串。

        参数：
            p (str): 模型文件路径。
            dnn (bool): 是否使用 OpenCV DNN 模块执行 ONNX 推理。

        返回：
            (str): 模型格式字符串（例如 'pt'、'onnx'、'engine'、'triton'）。

        示例：
            >>> fmt = AutoBackend._model_type("path/to/model.onnx")
            >>> assert fmt == "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        format = next((f for i, f in enumerate(export_formats()["Argument"]) if types[i]), None)
        if name.endswith("_qnn.onnx"):  # 否则 QNN 上下文二进制文件会匹配普通的 '.onnx' 后缀
            format = "qnn"
        elif name.endswith(".tflite") and not name.endswith("_edgetpu.tflite"):
            format = "litert"  # 裸 .tflite 文件（包括旧版 TFLite 导出文件）通过 LiteRT 加载
        elif format == "-":
            format = "pt"
        elif format == "onnx" and dnn:
            format = "dnn"
        elif not any(types):
            from urllib.parse import urlsplit

            url = urlsplit(p)
            if bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}:
                format = "triton"
        return format

    def eval(self) -> AutoBackend:
        """如果后端支持，则将后端模型设置为评估模式。"""
        if hasattr(self.backend, "model") and hasattr(self.backend.model, "eval"):
            self.backend.model.eval()
        return super().eval()

    def _apply(self, fn) -> AutoBackend:
        """对后端模型的参数、缓冲区和张量应用函数。

        此方法扩展父类的 _apply 方法，同时对后端模型应用函数并更新后端设备。
        它通常用于将模型移动到其他设备或修改模型精度等操作。

        参数：
            fn (Callable): 要应用于模型张量的函数，通常是 to()、cpu()、cuda()、half() 或 float() 等方法。

        返回：
            (AutoBackend): 已应用函数且属性已更新的模型实例。
        """
        super()._apply(fn)
        if hasattr(self.backend, "model") and isinstance(self.backend.model, nn.Module):
            self.backend.model._apply(fn)
            self.backend.device = next(self.backend.model.parameters()).device  # 移动后更新设备
        return self
