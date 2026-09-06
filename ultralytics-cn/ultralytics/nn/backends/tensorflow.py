# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class TensorFlowBackend(BaseBackend):
    """支持多种序列化格式的 Google TensorFlow 推理后端。

    加载并运行 SavedModel、GraphDef（.pb）和 Edge TPU 格式的 Google TensorFlow 模型推理，
    同时处理量化模型的反量化和任务特定的输出格式化。
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "saved_model"):
        """初始化 Google TensorFlow 后端。

        参数：
            weight (str | Path): SavedModel 目录、.pb 文件或 Edge TPU .tflite 文件的路径。
            device (torch.device): 执行推理的设备。
            fp16 (bool): 是否使用 FP16 半精度推理。
            format (str): 模型格式，可选 "saved_model"、"pb" 或 "edgetpu"。
        """
        assert format in {"saved_model", "pb", "edgetpu"}, f"Unsupported TensorFlow format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """以 SavedModel、GraphDef 或 Edge TPU 格式加载 Google TensorFlow 模型。

        参数：
            weight (str | Path): 模型文件或目录的路径。
        """
        if self.format in {"saved_model", "pb"}:
            import tensorflow as tf

        if self.format == "saved_model":
            LOGGER.info(f"Loading {weight} for TensorFlow SavedModel inference...")
            self.model = tf.saved_model.load(weight)
            self.apply_metadata(self.read_metadata(weight))
        elif self.format == "pb":
            LOGGER.info(f"Loading {weight} for TensorFlow GraphDef inference...")
            from ultralytics.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """裁剪到指定的输入和输出节点，将 TensorFlow 冻结图包装为推理模型。"""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()
            with open(weight, "rb") as f:
                gd.ParseFromString(f.read())
            self.frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            self.apply_metadata(self.read_metadata(weight))
        else:  # edgetpu
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate

                self.tf = None
            except ImportError:
                import tensorflow as tf

                self.tf = tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

            device = self.device[3:] if str(self.device).startswith("tpu") else ":0"
            LOGGER.info(f"Loading {weight} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
            delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                platform.system()
            ]
            self.interpreter = Interpreter(
                model_path=str(weight),
                experimental_delegates=[load_delegate(delegate, options={"device": device})],
            )
            self.device = torch.device("cpu")  # 从 PyTorch 角度看，Edge TPU 在 CPU 上运行

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.apply_metadata(self.read_metadata(weight))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """执行 Google TensorFlow 推理，并根据格式完成对应的执行和输出后处理。

        参数：
            im (torch.Tensor): 输入图像张量，格式为 BHWC（由 AutoBackend 从 BCHW 转换而来）。

        返回：
            (列表[np.ndarray]): NumPy 数组列表形式的模型预测结果。
        """
        im = im.cpu().numpy()
        if self.format == "saved_model":
            y = self.model.serving_default(im)
            if not isinstance(y, list):
                y = [y]
        elif self.format == "pb":
            import tensorflow as tf

            y = self.frozen_func(x=tf.constant(im))
        else:
            h, w = im.shape[1:3]

            details = self.input_details[0]
            is_int = details["dtype"] in {np.int8, np.int16}

            if is_int:
                scale, zero_point = details["quantization"]
                im = (im / scale + zero_point).astype(details["dtype"])

            self.interpreter.set_tensor(details["index"], im)
            self.interpreter.invoke()

            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output["index"])
                if self.task == "semantic" and x.ndim == 3:
                    # 固化的 argmax 类别图 [B, H, W] 包含整数类别 ID，而不是边界框或量化 logits：
                    # 跳过反量化和 xywh 反归一化，否则会破坏索引并造成溢出。
                    y.append(x)
                    continue
                if is_int:
                    scale, zero_point = output["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale
                if x.ndim == 3:
                    # 根据图像尺寸对 xywh 反归一化
                    if x.shape[-1] == 6 or self.end2end:
                        x[:, :, [0, 2]] *= w
                        x[:, :, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, :, 6::3] *= w
                            x[:, :, 7::3] *= h
                    else:
                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, 5::3] *= w
                            x[:, 6::3] *= h
                y.append(x)

        if self.task == "segment":  # 分割任务的 (det, proto) 输出顺序相反
            if len(y[1].shape) != 4:
                y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
            if y[1].shape[-1] == 6:  # end-to-end 模型
                y = [y[1]]
            else:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
        elif self.task in {"semantic", "depth"} and len(y) == 1 and y[0].ndim == 4:
            y[0] = np.transpose(y[0], (0, 3, 1, 2))  # NHWC → NCHW for semantic segmentation and depth anything logits
        return [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
