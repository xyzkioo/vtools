# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, TORCH_VERSION


def torch2torchscript(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_file: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """将 PyTorch 模型导出为 TorchScript 格式。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型（可能已使用 NMS 包装）。
        im (torch.Tensor): 用于跟踪的示例输入张量。
        output_file (Path | str): 保存导出 TorchScript 模型的路径。
        metadata (dict | None): 要嵌入 TorchScript 存档的可选元数据。
        prefix (str): 日志消息前缀。

    返回：
        (str): 导出的 ``.torchscript`` 文件路径。
    """
    LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")

    output_file = str(output_file)
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    extra_files = {"config.txt": json.dumps(metadata or {})}  # torch._C.ExtraFilesMap()
    ts.save(output_file, _extra_files=extra_files)
    return output_file
