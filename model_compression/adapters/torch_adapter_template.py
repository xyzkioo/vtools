"""自定义 PyTorch adapter 模板。

复制此文件后实现 ``build_model``、``load_weights`` 和可选的
``build_dataloader``，在配置中把 ``model.adapter`` 写成该文件路径。
"""

from __future__ import annotations

from typing import Any, Mapping


def build_model(weights: str, device: Any = "cpu"):
    """返回已加载或待加载的 ``torch.nn.Module``。"""

    del weights, device
    raise NotImplementedError("请在 adapter 中实现 build_model(weights, device)")


def load_weights(model: Any, weights: str, device: Any = "cpu") -> None:
    """将权重加载到 build_model 返回的模型。"""

    import torch

    payload = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("模板默认只处理 state_dict；完整模型请实现 load_model")
    model.load_state_dict(payload)


def build_dataloader(split: str, config: Mapping[str, Any], batch_size: int = 1):
    """可选：返回一个产生 ``(images, labels)`` 的 DataLoader。"""

    raise NotImplementedError(f"请为 {split} 划分实现 build_dataloader")
