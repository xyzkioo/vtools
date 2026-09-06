# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER


def onnx2mnn(
    onnx_file: str,
    output_file: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """将 ONNX 模型转换为 MNN 格式。

    参数：
        onnx_file (str): 源 ONNX 文件路径。
        output_file (Path | str): 保存导出 MNN 模型的路径。
        quantize (int | str | None): 精度方案，例如 16 表示 FP16，8 表示 INT8 权重。
        metadata (dict | None): 通过 ``--bizCode`` 嵌入的可选元数据。
        prefix (str): 日志消息前缀。

    返回：
        (str): 导出的 ``.mnn`` 文件路径。
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_1_10

    assert TORCH_1_10, "MNN export requires torch>=1.10.0 to avoid segmentation faults"
    assert Path(onnx_file).exists(), f"failed to export ONNX file: {onnx_file}"

    check_requirements("MNN>=2.9.6")
    import MNN
    from MNN.tools import mnnconvert

    LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")
    mnn_args = [
        "",
        "-f",
        "ONNX",
        "--modelFile",
        onnx_file,
        "--MNNModel",
        str(output_file),
        "--bizCode",
        json.dumps(metadata or {}),
    ]
    if quantize == 8:
        mnn_args.extend(("--weightQuantBits", "8"))
    if quantize == 16:
        mnn_args.append("--fp16")
    mnnconvert.convert(mnn_args)
    # 删除模型转换优化过程中生成的临时文件
    convert_scratch = Path(output_file).parent / ".__convert_external_data.bin"
    if convert_scratch.exists():
        convert_scratch.unlink()
    return str(output_file)
