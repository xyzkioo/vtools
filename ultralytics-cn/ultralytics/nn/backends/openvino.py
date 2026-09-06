# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class OpenVINOBackend(BaseBackend):
    """用于 Intel 硬件加速的 Intel OpenVINO 推理后端。

    使用 Intel OpenVINO IR 模型（*_openvino_model/ 目录）加载并运行推理。
    支持自动设备选择和 Intel 专用设备指定。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .xml/.bin 文件对或模型目录加载 Intel OpenVINO IR 模型。

        参数：
            weight (str | Path): .xml 文件路径，或包含 OpenVINO 模型文件的目录。
        """
        LOGGER.info(f"Loading {weight} for OpenVINO inference...")
        check_requirements("openvino>=2024.0.0")
        import openvino as ov

        core = ov.Core()
        fallback_device = "CPU" if core.available_devices == ["CPU"] else "AUTO"
        device_name = fallback_device

        if isinstance(self.device, str) and self.device.startswith("intel"):
            device_name = self.device.split(":")[1].upper()
            self.device = torch.device("cpu")
            if not any(d == device_name or d.startswith(f"{device_name}.") for d in core.available_devices):
                LOGGER.warning(f"OpenVINO device '{device_name}' not available. Using '{fallback_device}' instead.")
                device_name = fallback_device

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.xml"))

        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

        self.apply_metadata(self.read_metadata(w))

        # OpenVINO CPU 插件在 Intel AMX CPU（Sapphire Rapids 及更新型号）上运行动态形状 INT8 模型时会段错误，
        # 参见 https://github.com/openvinotoolkit/openvino/issues/37577，因此改为在 forward() 中按输入形状重塑并重新编译，
        # 将其作为静态模型运行。
        cpuinfo = Path("/proc/cpuinfo")
        self.read_model = (
            partial(core.read_model, model=str(w), weights=w.with_suffix(".bin"))
            if LINUX
            and device_name in {"CPU", "AUTO"}
            and ov_model.input().get_partial_shape().is_dynamic
            and any(op.get_type_name() == "FakeQuantize" for op in ov_model.get_ops())
            and cpuinfo.exists()
            and "amx_int8" in cpuinfo.read_text()
            else None
        )
        if self.read_model is not None:
            self.dynamic = False  # fixed letterbox shapes so recompiles stay rare

        # 强制同步推理，因为 AsyncInferQueue 可能在 Intel 和 AMD CPU 上无限挂起，参见
        # https://github.com/ultralytics/ultralytics/issues/25923.
        self.inference_mode = "LATENCY"
        config = {"PERFORMANCE_HINT": self.inference_mode}
        if LINUX and ARM64 and device_name == "CPU":
            config["EXECUTION_MODE_HINT"] = ov.properties.hint.ExecutionMode.ACCURACY
            config["INFERENCE_PRECISION_HINT"] = ov.Type.f32
        if (
            self.task == "classify"
            and device_name.startswith("NPU")
            and "NPU_TURBO" in core.get_property(device_name, "SUPPORTED_PROPERTIES")
        ):
            config["NPU_TURBO"] = "YES"

        self.compile_model = partial(core.compile_model, device_name=device_name, config=config)
        self.ov_compiled_model = self.compile_model(ov_model)
        LOGGER.info(
            f"Using OpenVINO {self.inference_mode} mode for batch={self.batch} inference on "
            f"{', '.join(self.ov_compiled_model.get_property('EXECUTION_DEVICES'))}..."
        )
        self.input_name = self.ov_compiled_model.input().get_any_name()
        self.ov = ov

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """执行 Intel OpenVINO 推理。

        参数：
            im (torch.Tensor): 输入图像 张量 in BCHW format, normalized to [0, 1].

        返回：
            (列表[np.ndarray]): NumPy 数组列表形式的模型预测结果，每个输出层对应一个数组。
        """
        im = im.cpu().numpy().astype(np.float32, copy=False)
        if self.read_model is not None and self.ov_compiled_model.input().get_partial_shape() != self.ov.PartialShape(
            im.shape
        ):
            ov_model = self.read_model()
            ov_model.reshape(list(im.shape))
            self.ov_compiled_model = self.compile_model(ov_model)

        if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
            # 对较大批次使用异步推理
            n = im.shape[0]
            results = [None] * n

            def callback(request, userdata):
                """将异步推理结果存储到预分配结果列表的指定索引处。"""
                results[userdata] = request.results

            async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
            async_queue.set_callback(callback)

            for i in range(n):
                async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)
            async_queue.wait_all()

            y = [list(r.values()) for r in results]
            y = [np.concatenate(x) for x in zip(*y)]
        else:
            # LATENCY 模式使用同步推理
            y = list(self.ov_compiled_model(im).values())
        return y
