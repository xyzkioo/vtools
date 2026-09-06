# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):
    """用于对象检测的 Ultralytics YOLO NAS 验证器。

    此类继承 Ultralytics 模型包中的 DetectionValidator，负责后处理 YOLO NAS 模型生成的原始预测结果。
    它执行非极大值抑制以移除重叠和低置信度边界框，最终生成检测结果。

    属性：
        args (Namespace): 包含后处理配置的命名空间，例如置信度和 IoU 阈值。
        lb (torch.Tensor): 用于多标签 NMS 的可选张量。

    示例：
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> validator = model.validator
        >>> # 假设 raw_preds 已经可用
        >>> final_preds = validator.postprocess(raw_preds)

    注意：
        此类通常不会直接实例化，而是在 NAS 类内部使用。
    """

    def postprocess(self, preds_in):
        """对预测输出应用非极大值抑制。"""
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框从 xyxy 格式转换为 xywh 格式
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 拼接边界框和分数并调整维度顺序
        return super().postprocess(preds)
