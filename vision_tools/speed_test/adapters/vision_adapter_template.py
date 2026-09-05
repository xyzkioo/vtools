#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自定义 PyTorch 视觉模型 adapter 模板。

将本文件复制为 ``my_adapter.py`` 后修改 ``build_model`` 和
``make_inputs``，再运行：

    python benchmark_pytorch.py \
        --adapter my_adapter.py \
        --model flower=/path/to/best.pth \
        --input-size 832 --precision fp32 --precision fp16

函数参数名不必完全一致，测速脚本会按函数声明传入可用参数。
复杂的检测/分割输出不需要为了测速强行转换成 YOLO Results；只有你想
统计目标数量时，才实现 ``count_outputs``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from core.vision_benchmark_common import BenchmarkConfig, InputBundle, dtype_for_precision


def build_model(weights: Path, device: torch.device) -> nn.Module:
    """构建网络并加载权重；请按你的模型修改这里。"""
    # 情况 A：checkpoint 本身保存的是完整 nn.Module，可以直接使用。
    try:
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(weights), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device).eval()
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        return checkpoint["model"].to(device).eval()

    # 情况 B：state_dict。把下面两行替换成你的模型构造函数和权重键名。
    # model = MyVisionModel(num_classes=1)
    # state_dict = checkpoint.get("state_dict", checkpoint)
    # model.load_state_dict(state_dict, strict=True)
    raise RuntimeError(
        "当前模板没有定义你的网络结构。请在 build_model() 中实例化模型后 load_state_dict。"
    )


def make_inputs(
    batch_size: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    source: Any = None,
    for_export: bool = False,
) -> InputBundle:
    """构造与模型 forward 签名一致的输入。

    默认示例是 [0, 1] 范围的 NCHW 图像。如果模型需要 ImageNet 均值方差、
    list[Tensor]、额外 mask 或多个输入，请在此处替换。
    """
    del source, for_export
    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.rand(
        (batch_size, 3, image_size[0], image_size[1]),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    return InputBundle((image,), batch_size=batch_size, description="NCHW image")


def prepare_model(model: nn.Module, device: torch.device, precision: str, fuse: bool = False) -> nn.Module:
    """可选：自定义部署准备，例如 fuse、eval 和 dtype 设置。"""
    del fuse
    model.to(device).eval()
    if precision == "fp16":
        model.half()
    elif precision == "bf16":
        model.bfloat16()
    return model


def predict(model: nn.Module, source: Any, config: BenchmarkConfig) -> Any:
    """可选：要测试真实预处理时，在这里实现完整 pipeline。"""
    # 只有启用真实图片 pipeline 时才需要 OpenCV；纯模型测速不依赖它。
    try:
        import cv2
    except ImportError as error:
        raise ImportError("该模板的图片 pipeline 需要 opencv-python；只测模型调用时可删除 predict()") from error

    # 这里给出一个常见的单张 OpenCV 图片示例；视频/目录请自行扩展。
    if isinstance(source, str):
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(source)
    elif source is None:
        bundle = make_inputs(
            config.batch_size,
            config.image_size,
            config.device,
            dtype_for_precision(config.precision, config.device),
        )
        return model(*bundle.args, **bundle.kwargs)
    else:
        image = source

    # BGR -> RGB、缩放、归一化；请按训练时的预处理替换。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.width, config.height))
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    tensor = tensor.unsqueeze(0).repeat(config.batch_size, 1, 1, 1).to(config.device)
    if config.precision == "fp16":
        tensor = tensor.half()
    elif config.precision == "bf16":
        tensor = tensor.bfloat16()
    return model(tensor)


def count_outputs(outputs: Any) -> Optional[int]:
    """可选：返回每张图的预测数量；未知时返回 None。"""
    del outputs
    return None


# 一致性检查的可选钩子（需要时再取消注释并实现，不能仅返回 ...）：
#
# def make_validation_inputs(source, config):
#     """真实图像预处理放这里；source 是 Path 或 None，返回 InputBundle。
#
#     检查器只调用一次，再克隆同一份输入给 PyTorch/TensorRT。
#     多输入、非 RGB、均值/方差归一化等必须与你的训练/部署约定一致。
#     """
#     return your_preprocess(source, config)
#
# def prepare_validation_model(model, config):
#     # 若自定义导出切换了 export 模式，在此使参考模型采用相同 forward。
#     return model
#
# def validation_forward(model, inputs):
#     # 自定义导出 wrapper 的等价 PyTorch 路径；不要在这里单独重读图片。
#     return model(*inputs.args, **inputs.kwargs)
#
# def validation_outputs(outputs, output_names):
#     # 输出必须是 {ONNX 输出名: Tensor}，包含全部 output_names。
#     # 默认导出无需此钩子；如果改变了输出顺序/结构，请显式按语义映射。
#     return {"logits": outputs["logits"], "mask": outputs["mask"]}
