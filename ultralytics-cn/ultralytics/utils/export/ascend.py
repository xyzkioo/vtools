# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


def _check_atc() -> None:
    """如果 CANN ATC 编译器不在 PATH 中，则抛出异常。"""
    if not shutil.which("atc"):
        raise FileNotFoundError(
            "Ascend export requires the CANN toolkit 'atc' compiler, which was not found on PATH. Install CANN and "
            "source its environment, e.g. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. "
            "See https://docs.ultralytics.com/integrations/ascend"
        )


def onnx2ascend(
    onnx_file: str | Path,
    output_dir: str | Path,
    name: str,
    imgsz: tuple[int, int],
    batch: int = 1,
    channels: int = 3,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """使用 CANN ATC 编译器将 ONNX 模型转换为华为 Ascend 离线模型（.om）。

    参数：
        onnx_file (str | Path): 输入 ONNX 模型路径。
        output_dir (str | Path): 写入编译后 .om 模型的目录。
        name (str): 通过 ``--soc_version`` 传给 ATC 的目标 Ascend SoC，例如 ``"Ascend310B4"``。
        imgsz (tuple[int, int]): 导出图像尺寸，格式为 ``(高度, 宽度)``。
        batch (int, 可选): 写入离线模型的静态批次大小。默认为 1。
        channels (int, 可选): 与跟踪 ONNX 图匹配的输入通道数。默认为 3。
        metadata (dict | None, 可选): 保存为 YAML 的可选元数据。默认为 None。
        prefix (str, 可选): 日志前缀。默认为 ""。

    返回：
        (str): 导出的 Ascend 模型目录路径。
    """
    _check_atc()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "atc",
        f"--model={Path(onnx_file).resolve()}",  # 使用绝对路径：ATC 的工作目录为 output_dir
        "--framework=5",  # 5 = ONNX
        f"--output={output_dir / f'{Path(onnx_file).stem}_{name}'}",  # ATC 会追加 .om 后缀
        "--input_format=NCHW",
        f"--input_shape=images:{batch},{channels},{imgsz[0]},{imgsz[1]}",
        f"--soc_version={name}",
        "--precision_mode=force_fp16",  # Ascend AI Core convolutions reject FP32 inputs
    ]  # argv 列表 avoids shell metacharacter issues in onnx_file/output_dir 路径
    LOGGER.info(f"\n{prefix} starting export with ATC for {name}...")
    LOGGER.info(f"{prefix} running '{shlex.join(cmd)}'")
    with tempfile.TemporaryDirectory() as scratch:  # ATC 会在其工作目录写入 kernel_meta/ 和 fusion_result.json
        subprocess.run(cmd, check=True, cwd=scratch)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
