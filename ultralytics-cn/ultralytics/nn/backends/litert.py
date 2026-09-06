# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, NUM_THREADS
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class LiteRTBackend(BaseBackend):
    """Google LiteRT（原 TensorFlow Lite）推理后端。

    加载并执行通过 ai-edge-litert/litert-torch 导出的 LiteRT 模型（.tflite 文件）推理。
    Ultralytics 导出模型保留浮点图输入输出（内部权重/激活值可能为 int8/int16）；同时通过在边界处执行反量化或量化，
    处理图输入输出为 int8/int16 的全整型 .tflite 模型（旧版 onnx2tf 或第三方导出）。边界框和关键点坐标根据图像尺寸反归一化。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .tflite 文件加载 LiteRT 模型。

        参数：
            weight (str | Path): .tflite 模型文件路径（元数据嵌入为 metadata.json 条目）。
        """
        check_requirements("ai-edge-litert>=2.1.4")
        from ai_edge_litert.interpreter import Interpreter

        tflite_file = Path(weight)

        LOGGER.info(f"Loading {tflite_file} for LiteRT inference...")
        self.interpreter = Interpreter(str(tflite_file), num_threads=NUM_THREADS)  # 启用多核推理
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # 旧版 onnx2tf TFLite 导出使用 NHWC；litert-torch 导出使用 NCHW。根据输入张量判断布局，
        # 使两种导出路径都能通过此后端加载（检测输出和 proto 输出在两种路径下布局一致）。
        self.nhwc = self.input_details[0]["shape"][-1] == 3

        self.apply_metadata(self.read_metadata(tflite_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """使用 LiteRT 解释器执行推理。

        边界框和姿态关键点坐标导出时归一化到 [0, 1]（使 INT8 量化保留类别分数精度），
        然后在此处根据输入图像尺寸反归一化，与 TensorFlow Lite 后端保持一致。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表[np.ndarray]): NumPy 数组列表形式的模型预测结果。
        """
        im = im.cpu().numpy()  # BCHW 格式
        if self.nhwc:
            im = im.transpose(0, 2, 3, 1)  # 旧版 onnx2tf TFLite 需要将 BCHW 转为 BHWC
        h, w = im.shape[1:3] if self.nhwc else im.shape[2:4]
        details = self.input_details[0]
        # Ultralytics 导出保持浮点输入输出，但对全整型 .tflite（旧版 onnx2tf 或第三方导出）也在此处量化，
        # 因为这类模型的计算图输入是 int8/int16。
        if details["dtype"] in {np.int8, np.int16}:
            scale, zero_point = details["quantization"]
            im = (im / scale + zero_point).astype(details["dtype"])
        self.interpreter.set_tensor(details["index"], im)
        self.interpreter.invoke()

        kpt_start = 4 + len(self.names)  # 姿态关键点位于边界框（4）和类别分数（nc）通道之后
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            if self.task == "semantic" and x.ndim == 3:
                # 旧版 onnx2tf 将 argmax 固化为整数 ID 类别图 [B, H, W]：跳过 xywh 反归一化，避免索引损坏和溢出。
                y.append(x)
                continue
            if output["dtype"] in {np.int8, np.int16}:  # 对全整型 .tflite 输出执行反量化（见输入说明）
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale
            # 根据图像尺寸对 xywh（以及姿态关键点）反归一化。litert-torch 端到端输出已经是 NMS 后的
            # 像素坐标（batch, max_det, 6+），因此保持不变；旧版 onnx2tf TFLite 连端到端输出也进行了归一化，
            # 所以需要在最后一个维度上反归一化。
            if x.ndim == 3 and not self.end2end:
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == "pose":
                    x[:, kpt_start::3] *= w
                    x[:, kpt_start + 1 :: 3] *= h
            elif x.ndim == 3 and self.end2end and self.nhwc:
                x[:, :, [0, 2]] *= w
                x[:, :, [1, 3]] *= h
                if self.task == "pose":  # NMS 后格式为 [B, N, 边界框（4）+置信度+类别+关键点]，关键点从 6 开始
                    x[:, :, 6::3] *= w
                    x[:, :, 7::3] *= h
            y.append(x)

        if self.task == "segment" and y[0].ndim == 4:  # 调整为（检测结果，原型掩码）的顺序
            y = [y[1], y[0]]
        # litert-torch 导出使用 NCHW；旧版 onnx2tf 的掩码/logits 使用 NHWC，需要将最后通道转到第一维。
        if self.nhwc:
            if self.task == "segment" and len(y) > 1 and y[1].ndim == 4:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # protos NHWC → NCHW
            elif self.task in {"semantic", "depth"} and len(y) == 1 and y[0].ndim == 4:
                y[0] = np.transpose(y[0], (0, 3, 1, 2))  # logits NHWC → NCHW

        return y
