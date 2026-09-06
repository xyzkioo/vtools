# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO 模型的深度估计预测器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class DepthPredictor(BasePredictor):
    """YOLO 深度估计模型的预测器。

    根据 RGB 图像生成逐像素深度图。

    示例：
        >>> from ultralytics.models.yolo.depth import DepthPredictor
        >>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
        >>> results = predictor("image.jpg")
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None
    ) -> None:
        """初始化 DepthPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"

    def postprocess(
        self, preds: torch.Tensor | tuple | list, img: torch.Tensor, orig_imgs: list[np.ndarray] | torch.Tensor
    ) -> list[Results]:
        """将深度预测结果后处理为 Results 对象。"""
        depth_maps = preds[0] if isinstance(preds, (tuple, list)) else preds  # (B, 1, H, W)
        if depth_maps.ndim == 3:
            depth_maps = depth_maps.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        # 恢复模型输入分辨率，使所有后端都能在缩放回原始图像前裁剪 letterbox 填充区域。
        depth_maps = ops.scale_masks(depth_maps, img.shape[2:], padding=False)

        if not isinstance(orig_imgs, list):  # torch.Tensor source (B, 3, H, W)
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for i, orig_img in enumerate(orig_imgs):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            depth = ops.scale_masks(depth_maps[i : i + 1].float(), orig_img.shape[:2])
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, depth=depth.squeeze()))

        return results
