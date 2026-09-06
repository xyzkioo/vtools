# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class NASPredictor(DetectionPredictor):
    """用于对象检测的 Ultralytics YOLO NAS 预测器。

    此类继承 Ultralytics 引擎中的 DetectionPredictor，负责后处理 YOLO NAS 模型生成的原始预测结果。
    它会执行非极大值抑制，并缩放边界框以匹配原始图像尺寸。

    属性：
        args (Namespace): 包含后处理配置的命名空间，包括置信度阈值、IoU 阈值、类别无关 NMS 标志、
            最大检测数和类别过滤选项。
        model (torch.nn.Module): 用于推理的 YOLO NAS 模型。
        batch (列表): 待处理的输入批次。

    示例：
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> predictor = model.predictor

        Assume that raw_preds, img, orig_imgs are available
        >>> results = predictor.postprocess(raw_preds, img, orig_imgs)

    注意：
        通常不会直接实例化此类；它由 NAS 类在内部使用。
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """后处理 NAS 模型预测结果，生成最终检测结果。

        此方法接收 YOLO NAS 模型的原始预测结果，转换边界框格式并执行后处理，
        生成可供 Ultralytics 结果可视化和分析工具使用的最终检测结果。

        参数：
            preds_in (列表): NAS 模型的原始预测结果，通常包含边界框和类别分数。
            img (torch.Tensor): 输入模型的图像张量，形状为 (B, C, H, W)。
            orig_imgs (列表 | torch.Tensor | np.ndarray): 预处理前的原始图像，用于将坐标缩放回原始尺寸。

        返回：
            (列表): 包含批次中每张图像处理后预测结果的 Results 对象列表。

        示例：
            >>> predictor = NAS("yolo_nas_s").predictor
            >>> results = predictor.postprocess(raw_preds, img, orig_imgs)
        """
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框从 xyxy 格式转换为 xywh 格式
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 拼接边界框和类别分数
        return super().postprocess(preds, img, orig_imgs)
