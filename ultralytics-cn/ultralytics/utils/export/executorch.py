# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ultralytics.nn.modules import Pose, Pose26
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_executorch_requirements


def executorch_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """应用 ExecuTorch 专用模型补丁，以保证导出和运行时兼容性。"""
    import types

    for m in model.modules():
        if not isinstance(m, Pose):
            continue
        m.kpts_decode = types.MethodType(partial(_executorch_kpts_decode, is_pose26=type(m) is Pose26), m)
    return model


def _executorch_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    """为 ExecuTorch 导出解码姿态关键点，并使用 XNNPACK 安全的广播方式。"""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    y = kpts.view(bs, *self.kpt_shape, -1)

    # XNNPACK 要求广播时维度显式匹配，因此将二维张量扩展为四维。
    anchors = self.anchors[None, None]
    strides = self.strides[None, None]
    a = ((y[:, :, :2] + anchors) if is_pose26 else (y[:, :, :2] * 2.0 + (anchors - 0.5))) * strides
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def torch2executorch(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """将 PyTorch 模型导出为 ExecuTorch 格式。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型。
        im (torch.Tensor): 用于跟踪和导出的示例输入张量。
        output_dir (Path | str): 保存导出 ExecuTorch 模型的目录。
        metadata (dict | None, 可选): 要保存为 YAML 的可选元数据。
        prefix (str, 可选): 日志消息前缀。

    返回：
        (str): 导出的 ExecuTorch 模型目录路径。
    """
    check_executorch_requirements()
    from executorch import version as executorch_version
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    LOGGER.info(f"\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pte_file = output_dir / "model.pte"
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (im,)),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    pte_file.write_bytes(et_program.buffer)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
