# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import ARM64, IS_JETSON, LOGGER, YAML


def torch2paddle(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """使用 X2Paddle 将 PyTorch 模型导出为 PaddlePaddle 格式。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型。
        im (torch.Tensor): 用于跟踪的示例输入张量。
        output_dir (Path | str): 保存导出 PaddlePaddle 模型的目录。
        metadata (dict | None): 保存为 ``metadata.yaml`` 的可选元数据。
        prefix (str): 日志消息前缀。

    返回：
        (str): 导出的 ``_paddle_model`` 目录路径。
    """
    assert not IS_JETSON, "Jetson Paddle exports not supported yet"
    from ultralytics.utils.checks import check_requirements

    check_requirements(
        (
            # 这些候选项可互换，避免已安装的变体被重复安装（它们都提供 'paddle'）
            (
                "paddlepaddle-gpu>=3.0.0,<3.3.0"  # pin <3.3.0 https://github.com/PaddlePaddle/Paddle/issues/77340
                if im.device.type == "cuda"
                else "paddlepaddle==3.0.0"  # ARM64 固定使用 3.0.0
                if ARM64
                else "paddlepaddle>=3.0.0,<3.3.0",  # pin <3.3.0 https://github.com/PaddlePaddle/Paddle/issues/77340
                "paddlepaddle==3.0.0" if ARM64 else "paddlepaddle>=3.0.0,<3.3.0",
                "paddlepaddle-gpu>=3.0.0,<3.3.0",
            ),
            "x2paddle",
        )
    )

    import x2paddle
    from x2paddle.convert import pytorch2paddle
    from x2paddle.op_mapper.pytorch2paddle import prim2code

    # x2paddle 1.6.0 为池化生成的代码通过 locals() 读取 exec() 结果，而 PEP 667 在 Python 3.13 中破坏了该行为。
    # 该断言只会重新检查 torch 在跟踪期间已经验证过的池化参数，因此跳过生成它。
    prim_assert = prim2code.prim_assert
    prim2code.prim_assert = lambda layer, **kwargs: None

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")

    try:
        pytorch2paddle(module=model, save_dir=output_dir, jit_type="trace", input_examples=[im])  # 导出
    finally:
        prim2code.prim_assert = prim_assert
    if metadata:
        YAML.save(Path(output_dir) / "metadata.yaml", metadata)  # 添加 metadata.yaml
    return str(output_dir)
