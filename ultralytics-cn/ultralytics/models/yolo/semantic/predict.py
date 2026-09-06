# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class SemanticSegmentationPredictor(BasePredictor):
    """语义分割模型的预测器。

    此预测器处理模型输出，生成逐像素类别标签图。

    示例：
        >>> from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
        >>> args = dict(model="yolo26n-sem.pt", source="path/to/image.jpg")
        >>> predictor = SemanticSegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 SemanticSegmentationPredictor。

        参数：
            cfg (dict): 预测器配置。
            overrides (dict, 可选): 配置覆盖项。
            _callbacks (dict, 可选): 回调函数。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semantic"

    @staticmethod
    def _class_map_dtype(num_classes: int) -> torch.dtype:
        """返回适用于语义类别 ID 的最小整数数据类型。"""
        return torch.uint8 if num_classes <= 256 else torch.int16 if num_classes <= 32768 else torch.int32

    def postprocess(self, preds, img, orig_imgs):
        """将模型输出转换为语义分割结果。

        参数：
            preds (torch.Tensor | tuple): 模型输出 logits [B, nc, H, W] 或内置类别图 [B, H, W]。
            img (torch.Tensor): 预处理后的输入图像张量。
            orig_imgs (列表 | torch.Tensor): 原始图像。

        返回：
            (列表[Results]): 包含语义掩码的 Results 对象列表。
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        classes = (
            torch.as_tensor(self.args.classes, device=preds.device).flatten()
            if self.args.classes is not None and len(self.model.names) > 1
            else None
        )

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            class_map_input = pred.ndim == 2  # 图内 ArgMax 导出的结果直接生成 [H, W] 类别图
            pred = (pred[None, None] if class_map_input else pred[None]).float()
            if class_map_input:
                # OpenCV DNN 可能以浮点数返回类别图，但这些值仍是离散 ID，因此只能使用最近邻缩放。
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="nearest")
                class_map = ops.scale_masks(pred, orig_img.shape[:2], mode="nearest")[0, 0]
                class_map = class_map.to(self._class_map_dtype(int(class_map.max().item()) + 1))
            else:
                # pred： [1, nc, H, W] logits。先上采样到输入分辨率，使 LetterBox 填充为整数。
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="bilinear")
                # 移除 letterbox 填充，然后缩放到原始图像尺寸。
                pred = ops.scale_masks(pred, orig_img.shape[:2])[0]
                dtype = self._class_map_dtype(max(pred.shape[0], 2))
                class_map = pred.argmax(0).to(dtype) if pred.shape[0] > 1 else pred.gt(0).squeeze(0).to(dtype)
            if classes is not None:  # 仅保留选定类别，其余类别标记为忽略
                class_map[~(class_map.unsqueeze(-1) == classes).any(-1)] = 255
            results.append(Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map))
        return results
