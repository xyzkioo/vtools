# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import contextlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import YAML


def _read_proto_map(file: Path, path: tuple[int, ...]) -> dict:
    """读取嵌套字段路径中的 protobuf ``map<string, string>``，无需导入对应格式的框架。

    参数：
        file (Path): protobuf 文件路径，例如 ONNX 或 CoreML 模型。
        path (tuple[int, ...]): 要逐层进入的字段编号，最后一个字段包含重复的 ``key``/``value`` 条目。

    返回：
        (dict): 以字符串键值对形式返回的映射条目。
    """
    import mmap

    def fields(buf):
        """遍历消息中的每个长度分隔字段，并跳过 varint 字段，返回 ``(编号, payload)``。"""
        i = 0

        def varint():
            """解码当前偏移处的 base-128 varint。"""
            nonlocal i
            v = shift = 0
            while buf[i] & 0x80:
                v, i, shift = v | (buf[i] & 0x7F) << shift, i + 1, shift + 7
            v, i = v | buf[i] << shift, i + 1
            return v

        while i < len(buf):
            tag = varint()
            if tag & 7 == 0:  # varint 字段，例如 ONNX ir_version
                varint()
            elif tag & 7 == 2:  # 长度分隔字段，例如嵌套消息、字符串或权重数据块
                n = varint()
                yield tag >> 3, buf[i : i + n]  # 返回 memoryview 切片，避免复制大型 payload
                i += n
            else:
                return  # 这些 protobuf 消息不包含固定宽度字段

    with open(file, "rb") as f:
        messages = [memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))]
    for number in path:
        messages = [payload for m in messages for n, payload in fields(m) if n == number]
    return {bytes(e[1]).decode(): bytes(e.get(2, b"")).decode() for e in map(dict, map(fields, messages))}


class BaseBackend(ABC):
    """所有推理后端的基类。

    此抽象类定义所有推理后端必须实现的接口，并提供模型加载、元数据处理和设备管理等通用功能。

    属性：
        model: 底层推理模型或运行时会话。
        device (torch.device): 执行推理的设备。
        fp16 (bool): 是否使用 FP16（半精度）推理。
        nhwc (bool): 模型是否需要 NHWC 输入格式，而不是 NCHW。
        stride (int): 模型步长，YOLO 模型通常为 32。
        names (dict): 将类别索引映射到类别名称的字典。
        task (str | None): 任务类型（detect、segment、semantic、classify、pose、obb）。
        batch (int): 推理批次大小。
        imgsz (tuple): 输入图像尺寸，格式为 (height, width)。
        channels (int): 输入通道数，RGB 输入通常为 3。
        end2end (bool): 模型是否包含端到端 NMS 后处理。
        dynamic (bool): 模型是否支持动态输入形状。
        base_model (bool): 已加载模型是否为 Ultralytics `BaseModel`，从而支持 `augment` 和 `embed` 前向参数。
        metadata (dict): 包含导出配置的模型元数据字典。
    """

    def __init__(self, weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False):
        """使用通用属性初始化基础后端并加载模型。

        参数：
            weight (str | torch.nn.Module): 模型权重文件路径或 PyTorch 模块实例。
            device (torch.device | str): 执行推理的设备（例如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
        """
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.channels = 3
        self.end2end = False
        self.dynamic = False
        self.base_model = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)

    @abstractmethod
    def load_model(self, weight: str | torch.nn.Module) -> None:
        """从权重文件或模块实例加载模型。

        参数：
            weight (str | torch.nn.Module): 模型权重文件路径或 PyTorch 模块。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, im: torch.Tensor) -> Any:
        """对输入图像张量执行推理。

        参数：
            im (torch.Tensor): BCHW 格式的输入图像张量，已归一化到 [0, 1]。

        返回：
            (Any): 模型前向传播的原始输出，可能还需要后处理。
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """允许直接调用后端实例执行推理，并将参数转发给 `forward` 方法。
        """
        return self.forward(*args, **kwargs)

    @staticmethod
    def engine_header(file: str | Path) -> tuple[int, dict]:
        """读取 Ultralytics ``.engine`` 导出文件写在序列化引擎之前的元数据头。

        参数：
            file (str | Path): TensorRT 引擎文件路径。

        返回：
            (tuple[int, dict]): 引擎数据的字节偏移量和头部元数据；没有头部时返回 ``(0, {})``。
        """
        with open(file, "rb") as f:
            n = int.from_bytes(f.read(4), byteorder="little")  # 4 字节小端 JSON 长度（如果存在头部）
            if 0 < n <= f.seek(0, 2) - 4:  # 长度超出文件范围时，不视为头部
                f.seek(4)
                with contextlib.suppress(ValueError):  # 引擎数据不是 JSON，因此只有真正的头部才能解析成功
                    return 4 + n, json.loads(f.read(n))
        return 0, {}

    @staticmethod
    def read_metadata(file: str | Path) -> dict:
        """从导出文件中读取 Ultralytics 元数据，无需加载导出文件或导入对应框架。

        单文件格式会将元数据嵌入带长度前缀的 JSON 头（``.engine``）、ZIP 条目（``.torchscript``、``.tflite``）
        或 protobuf 字符串映射条目（``.onnx``、``.mlpackage``）中；其他格式会在导出文件旁边或内部写入 ``metadata.yaml``。
        MNN 将元数据保存在 flatbuffer 的 ``bizCode`` 字段中，Triton 则通过 HTTP 提供元数据，因此这里不读取这两种格式。

        参数：
            file (str | Path): 已导出模型文件或目录的路径。

        返回：
            (dict): 解析后的元数据；对于第三方导出文件或不支持嵌入元数据的旧导出文件，返回空字典。
        """
        import zipfile

        p = Path(file)
        try:
            if p.suffix == ".engine":  # 4 字节小端长度，后面紧跟相应长度的 JSON 数据
                return BaseBackend.engine_header(p)[1]
            if p.suffix in {".tflite", ".torchscript"}:  # 元数据追加在模型 ZIP 中，或保存在 ZIP 内部
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if "metadata.json" in names:  # litert-torch 和单文件 tflite 导出格式
                        return json.loads(z.read("metadata.json"))
                    name = next((n for n in names if n.endswith("extra/config.txt")), None)  # torch.jit extra 文件
                    return json.loads(z.read(name)) if name else ast.literal_eval(z.read(names[0]).decode())
            if p.suffix == ".onnx" or p.name.endswith("_imx_model"):  # IMX 会将 ONNX 打包在目录中
                return _read_proto_map(next(p.glob("*.onnx")) if p.is_dir() else p, (14,))  # 元数据属性
            if p.suffix in {".mlpackage", ".mlmodel"}:  # description.metadata.userDefined
                return _read_proto_map(next(p.rglob("*.mlmodel")) if p.is_dir() else p, (2, 100, 100))
            sidecar = (p if p.is_dir() else p.parent) / "metadata.yaml"  # openvino、ncnn、paddle、saved_model 等格式
            if p.suffix == ".pb":  # 冻结图会将元数据保存在同级 saved_model 目录中
                sidecar = next(p.resolve().parent.rglob(f"{p.stem}_saved_model*/metadata.yaml"), sidecar)
            return YAML.load(sidecar) if sidecar.exists() else {}
        except Exception:  # 第三方、截断或不包含元数据的文件
            return {}

    def apply_metadata(self, metadata: dict | None) -> None:
        """处理模型元数据，并将其应用到后端属性。

        此方法会转换常见元数据字段（例如 stride、batch 和 names）的类型，并将它们设置为实例属性；
        同时根据导出参数解析端到端 NMS 和动态形状设置。

        参数：
            metadata (dict | None): 包含模型导出元数据键值对的字典。
        """
        if not metadata:
            return

        # 保存原始元数据
        self.metadata = metadata

        # 转换已知字段的类型
        for k, v in metadata.items():
            if k in {"stride", "batch", "channels"}:
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                metadata[k] = ast.literal_eval(v)

        # 处理包含端到端 NMS 的导出模型
        metadata["end2end"] = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
        metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)

        # 将所有元数据字段应用为后端属性
        for k, v in metadata.items():
            setattr(self, k, v)
