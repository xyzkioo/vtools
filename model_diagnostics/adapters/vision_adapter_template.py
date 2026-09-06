#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vtools-compatible detection adapter template.

Copy this file to ``adapters/my_detector.py`` and modify ``build_model`` plus
``postprocess_detections``.  Then select ``mode: C`` and change
``mode_c.weights`` and ``mode_c.adapter`` in ``config/pycharm_run.yaml``.

The hook names and signatures intentionally follow
``vtools/speed_test/adapters/vision_adapter_template.py``:

    build_model(weights, device)
    make_inputs(batch_size, image_size, device, dtype, source=None, for_export=False)
    prepare_model(model, device, precision, fuse=False)
    predict(model, source, config)                 # optional
    postprocess_detections(outputs, source, image_info, config)  # required for raw outputs
    count_outputs(outputs)                         # optional

The diagnostics runner calls ``predict`` for each validation image with
``source`` as a string path and expects the final detections in original-image
pixel coordinates.  It does not apply a model-specific confidence filter;
keep low-score detections so the diagnostic threshold sweep can measure
low-confidence misses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from core.vision_benchmark_common import BenchmarkConfig, InputBundle, dtype_for_precision


def build_model(weights: Path, device: torch.device) -> nn.Module:
    """Build the model and load its checkpoint; replace this for state_dicts."""
    try:
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(weights), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device).eval()
    if isinstance(checkpoint, dict):
        for key in ("model", "ema"):
            if isinstance(checkpoint.get(key), nn.Module):
                return checkpoint[key].to(device).eval()

    # For a state_dict, instantiate your network here:
    # model = MyDetector(num_classes=1)
    # state_dict = checkpoint.get("state_dict", checkpoint)
    # model.load_state_dict(state_dict, strict=True)
    raise RuntimeError("请在 build_model() 中实例化网络并加载 state_dict")


def make_inputs(
    batch_size: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    source: Any = None,
    for_export: bool = False,
) -> InputBundle:
    """Return the exact inputs expected by your model forward method."""
    del source, for_export
    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.rand((batch_size, 3, image_size[0], image_size[1]), generator=generator, dtype=torch.float32)
    image = image.to(device=device, dtype=dtype)
    return InputBundle((image,), batch_size=batch_size, description="NCHW image")


def prepare_model(model: nn.Module, device: torch.device, precision: str, fuse: bool = False) -> nn.Module:
    """Optional deployment preparation, such as eval/fuse/half."""
    model.to(device).eval()
    if fuse and callable(getattr(model, "fuse", None)):
        try:
            model.fuse(verbose=False)
        except TypeError:
            model.fuse()
    if precision == "fp16":
        model.half()
    elif precision == "bf16":
        model.bfloat16()
    return model


def predict(model: nn.Module, source: Any, config: BenchmarkConfig) -> Any:
    """Run preprocessing and forward; return raw model output if desired.

    Replace this preprocessing with the exact training/deployment pipeline. The
    output can remain raw if ``postprocess_detections`` below converts it.
    """
    if source is None:
        inputs = make_inputs(config.batch_size, config.image_size, config.device, dtype_for_precision(config.precision, config.device), source=None)
        return model(*inputs.args, **inputs.kwargs)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("该 adapter 的图片输入需要 opencv-python") from exc
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(str(source))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.width, config.height))
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    tensor = tensor.unsqueeze(0).to(config.device)
    if config.precision == "fp16":
        tensor = tensor.half()
    elif config.precision == "bf16":
        tensor = tensor.bfloat16()
    return model(tensor)


def postprocess_detections(
    outputs: Any,
    source: Any,
    image_info: dict[str, Any],
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    """Convert raw output to canonical detections in original-image pixels.

    Return dictionaries with ``bbox=[x1,y1,x2,y2]``, ``score`` and
    ``class_id``. Undo every resize/letterbox operation here. Do not filter by
    ``config.conf`` if low-confidence diagnostics are required; the diagnostic
    engine applies its own thresholds later.
    """
    del outputs, source, image_info, config
    # Example:
    # return [{"bbox": [x1, y1, x2, y2], "score": float(score), "class_id": int(cls)}]
    raise NotImplementedError("请把模型输出转换为 [{bbox, score, class_id}, ...]")


def count_outputs(outputs: Any) -> Optional[int]:
    """Optional vtools hook; diagnostics does not require it."""
    del outputs
    return None
