# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_dir: Path | str | None = None,
    dynamic: bool = False,
    quantize: int | str | None = None,
    calibration_dataset: Any | None = None,
    int8_detect: bool = False,
    prefix: str = "",
) -> Any:
    """将 PyTorch 模型导出为 OpenVINO 格式，并支持可选 INT8 量化。

    参数：
        model (torch.nn.Module): 要导出的模型（可能已使用 NMS 包装）。
        im (torch.Tensor | 列表[torch.Tensor] | tuple[torch.Tensor, ...]): 用于跟踪的示例输入张量。
        output_dir (Path | str | None): 保存导出 OpenVINO 模型的目录。
        dynamic (bool): 是否使用动态输入形状。
        quantize (int | str | None): 精度方案，例如 16 表示 FP16，8 表示 INT8。
        calibration_dataset (nncf.Dataset | None): INT8 校准数据集（``quantize=8`` 时必需）。
        int8_detect (bool): INT8 量化期间是否让检测头保持浮点精度。
        prefix (str): 日志消息前缀。

    返回：
        (ov.Model): 转换后的 OpenVINO 模型。
    """
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")

    input_shape = [i.shape for i in im] if isinstance(im, (list, tuple)) else im.shape
    # 向 OpenVINO 传入已经完成跟踪的 ScriptModule（torchscript/coreml 导出采用相同的跟踪方式），而不是原始
    # nn.Module，这样它就不会在内部使用 check_trace=True 再次跟踪；该跟踪并比较的健全性检查在 NMS 模型上
    # 具有非确定性，并会失败并提示“Graphs differed across invocations!”。check_trace=False 会跳过我们自己的相同检查。
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    ov_model = ov.convert_model(ts, input=None if dynamic else input_shape, example_input=im)
    if quantize == 8:
        import nncf

        ignored_scope = None
        if int8_detect:
            operations = ov_model.get_ordered_ops()
            sigmoid_names = [op.get_friendly_name() for op in operations if op.get_type_name() == "Sigmoid"]
            head_scope = sigmoid_names[-1].split("/", 1)[0]
            ignored_scope = nncf.IgnoredScope(
                names=[
                    op.get_friendly_name()
                    for op in operations
                    if op.get_type_name() == "Sigmoid"
                    or op.get_friendly_name().startswith((f"{head_scope}/", f"{head_scope}.dfl"))
                ]
            )
        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            # 与其他 INT8 后端一样使用完整数据集进行校准，而不是采用 nncf 默认的 300 个批次。
            subset_size=calibration_dataset.get_length() or 300,
            ignored_scope=ignored_scope,
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "model.xml"
        ov.save_model(ov_model, output_file, compress_to_fp16=quantize == 16)
    return ov_model
