# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str:
    """使用 PNNX 将 PyTorch 模型导出为 NCNN 格式。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型。
        im (torch.Tensor): 用于跟踪的示例输入张量。
        output_dir (Path | str): 保存导出 NCNN 模型的目录。
        quantize (int | str | None): 量化方案，例如 16 表示 FP16。
        metadata (dict | None): 保存为 ``metadata.yaml`` 的可选元数据。
        device (torch.device | None): 模型所在的设备。
        prefix (str): 日志消息前缀。

    返回：
        (str): 导出的 ``_ncnn_model`` 目录路径。
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("ncnn", cmds="--no-deps")  # 不安装依赖，避免安装 opencv-python
    # 在 PNNX 20260704 修复 NCNN 推理段错误前固定该版本：https://github.com/pnnx/pnnx/issues/293
    check_requirements("pnnx==20260526")
    import ncnn
    import pnnx

    LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
    output_dir = Path(output_dir)

    ncnn_args = {
        "ncnnparam": (output_dir / "model.ncnn.param").as_posix(),
        "ncnnbin": (output_dir / "model.ncnn.bin").as_posix(),
        "ncnnpy": (output_dir / "model_ncnn.py").as_posix(),
    }
    pnnx_args = {
        "ptpath": (output_dir / "model.pt").as_posix(),
        "pnnxparam": (output_dir / "model.pnnx.param").as_posix(),
        "pnnxbin": (output_dir / "model.pnnx.bin").as_posix(),
        "pnnxpy": (output_dir / "model_pnnx.py").as_posix(),
        "pnnxonnx": (output_dir / "model.pnnx.onnx").as_posix(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)  # 创建 ncnn_model 目录
    device_type = device.type if device is not None else "cpu"
    pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=quantize == 16, device=device_type)

    for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
        Path(f_debug).unlink(missing_ok=True)

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)  # add metadata.yaml
    return str(output_dir)
