# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class QNNBackend(BaseBackend):
    """用于 Snapdragon 硬件的 Qualcomm QNN 推理后端。

    使用 ONNX Runtime 和 QNN 执行提供程序插件（`onnxruntime-qnn`）加载并运行 Ultralytics QNN 导出生成的
    QNN 上下文二进制文件（`*_qnn.onnx`）。推理通过 HTP（NPU）后端运行在 Qualcomm Snapdragon 设备上，
    包括 Android、Snapdragon Windows 和 Qualcomm Linux 开发板。
    """

    def load_model(self, weight: str | Path) -> None:
        """使用 ONNX Runtime 的 QNN 执行提供程序插件加载 QNN context-binary 模型。

        参数：
            weight (str | Path): `*_qnn.onnx` 文件路径。

        异常：
            OSError: 无法注册 QNN 执行提供程序时抛出（例如当前设备不是 Snapdragon 硬件）。
        """
        check_requirements("onnxruntime-qnn")
        import onnxruntime

        from ultralytics.utils.export.qnn import qnn_library_paths

        onnx_file = Path(weight)
        LOGGER.info(f"Loading {onnx_file} for Qualcomm QNN inference...")

        # 注册 QNN EP（库由插件辅助程序或 onnxruntime/capi 目录解析）并选择它；
        # 当 QNN 已内置于 ONNX Runtime 时，ep_library 为 None，无需注册插件
        ep_name = "QNNExecutionProvider"
        ep_library, htp_backend = qnn_library_paths()
        ep_options = {"backend_path": htp_backend}
        options = onnxruntime.SessionOptions()
        if ep_library:
            onnxruntime.register_execution_provider_library(ep_name, ep_library)
            devices = [d for d in onnxruntime.get_ep_devices() if d.ep_name == ep_name]
            if not devices:
                raise OSError(
                    "QNN Execution Provider registered but no QNN devices were found. Run on a Qualcomm Snapdragon "
                    "device with 'onnxruntime-qnn' installed."
                )
            options.add_provider_for_devices(devices, ep_options)
            self.session = onnxruntime.InferenceSession(str(onnx_file), sess_options=options)
        else:
            self.session = onnxruntime.InferenceSession(
                str(onnx_file), sess_options=options, providers=[ep_name], provider_options=[ep_options]
            )
        self.output_names = [x.name for x in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape  # 通道-last exports take [N, H, W, C] 输入
        self.nhwc = len(shape) == 4 and shape[3] in {1, 3} and shape[1] not in {1, 3}
        self.apply_metadata(self.read_metadata(onnx_file))

    def forward(self, im: torch.Tensor) -> list:
        """在 Qualcomm QNN 运行时上执行推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表): 输出数组列表形式的模型预测结果。
        """
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)
        return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
