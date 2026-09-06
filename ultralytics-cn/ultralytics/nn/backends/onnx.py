# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend

# ONNX Runtime 输出类型字符串 -> (torch dtype, numpy dtype)，用于 IO 绑定。
_ORT_DTYPES = {
    "tensor(float16)": (torch.float16, np.float16),
    "tensor(float)": (torch.float32, np.float32),
    "tensor(double)": (torch.float64, np.float64),
    "tensor(uint8)": (torch.uint8, np.uint8),
    "tensor(int8)": (torch.int8, np.int8),
    "tensor(int32)": (torch.int32, np.int32),
    "tensor(int64)": (torch.int64, np.int64),
}


class ONNXBackend(BaseBackend):
    """支持可选 OpenCV DNN 的 Microsoft ONNX Runtime 推理后端。

    使用带 CUDA/CoreML 执行提供程序的 Microsoft ONNX Runtime，或使用 OpenCV DNN 执行轻量级 CPU 推理，
    加载并执行 ONNX 模型（.onnx 文件）。支持静态输入形状下用于优化 GPU 推理的 IO 绑定。
    """

    def __init__(
        self,
        weight: str | Path,
        device: torch.device,
        fp16: bool = False,
        format: str = "onnx",
        session_options: object | None = None,
    ):
        """初始化 ONNX 后端。

        参数：
            weight (str | Path): .onnx 模型文件路径。
            device (torch.device): 执行推理的设备。
            fp16 (bool): 是否使用 FP16 半精度推理。
            format (str): Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN.
            session_options (对象 | None): 可选 ONNX Runtime session options.
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        self.session_options = session_options
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """使用 ONNX Runtime 或 OpenCV DNN 加载 ONNX 模型。

        参数：
            weight (str | Path): .onnx 模型文件路径。
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"

        self.apply_metadata(self.read_metadata(weight))

        if self.format == "dnn":
            # OpenCV DNN 后端
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # ONNX Runtime 后端
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            # 选择执行提供程序
            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                    self.device = torch.device("cpu")
                    cuda = False

            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with "
                f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )

            try:
                self.session = onnxruntime.InferenceSession(weight, self.session_options, providers=providers)
            except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf as e:
                # ONNX Runtime 会将无法解析的图报告为原始 protobuf 错误，既不说明问题，也不提供解决办法。
                # 这里只捕获这一种错误；其他加载失败通常与执行提供程序或模型支持有关，此时应保留运行时自己的错误信息。
                raise TypeError(
                    f"ERROR ❌️ {weight} is not a loadable ONNX model — the file is empty, truncated or corrupted "
                    f"({type(e).__name__}: {e}).\nRecommend fixes are to re-export it with "
                    f"'yolo export model=yolo26n.pt format=onnx', or to re-download the file."
                ) from e
            self.output_names = [x.name for x in self.session.get_outputs()]

                # 检查是否为动态形状
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type

            # 为 CUDA 设置 IO 绑定
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    torch_dtype, np_dtype = _ORT_DTYPES.get(output.type, (torch.float32, np.float32))
                    y_tensor = torch.empty(output.shape, dtype=torch_dtype).to(self.device)
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=self.device.index if cuda else 0,
                        element_type=np_dtype,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def forward(
        self, im: torch.Tensor | dict[str, torch.Tensor | np.ndarray]
    ) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        """使用 IO 绑定（CUDA）或标准会话执行来运行 ONNX 推理。

        参数：
            im (torch.Tensor | dict): 输入图像张量，格式为 BCHW 且归一化到 [0, 1]；也可以是字典映射
                将输入名称映射到张量/数组，用于多输入 ONNX Runtime 模型。

        返回：
            (torch.Tensor | 列表[torch.Tensor] | np.ndarray): 张量或 NumPy 数组形式的模型预测结果。
        """
        if self.format == "dnn":
        # OpenCV DNN 后端
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()

        # ONNX Runtime 后端
        if isinstance(im, dict):  # multi-输入 模型
            im = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in im.items()}
            return self.session.run(self.output_names, im)

        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
            self.io.bind_input(
                name="images",
                device_type=im.device.type,
                device_id=im.device.index if im.device.type == "cuda" else 0,
                element_type=np.float16 if self.fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            return self.bindings
        else:
            return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})


class ONNXIMXBackend(ONNXBackend):
    """用于 NXP i.MX 处理器的 ONNX IMX 推理后端。

    扩展 `ONNXBackend`，支持面向 NXP i.MX 边缘设备的量化模型。
    使用 MCT（Model Compression Toolkit）量化器和自定义 NMS 操作来优化推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 IMX 模型目录加载量化的 ONNX 模型。

        参数：
            weight (str | Path): 包含 .onnx 文件的 IMX 模型目录路径。
        """
        check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort  # noqa - register custom NMS ops

        w = Path(weight)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False

        self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
        """执行 IMX 推理，并针对检测、姿态和分割任务拼接对应输出。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (np.ndarray | 列表[np.ndarray] | tuple[np.ndarray, ...]): Task-formatted 模型 预测结果.
        """
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})

        if self.task == "detect":
            # 边界框, conf, cls
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif self.task == "pose":
            # 边界框, conf, kpts
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif self.task == "segment":
            return (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )
        return y
