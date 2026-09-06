# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """继承 BasePredictor、用于生成预测结果的 RT-DETR（实时检测 Transformer）预测器。

    此类利用 Vision Transformer 在保持高精度的同时提供实时对象检测，
    支持高效混合编码和 IoU 感知查询选择等关键特性。

    属性：
        imgsz (int): 用于推理的图像尺寸（必须为正方形并进行缩放填充）。
        args (dict): 预测器的参数覆盖项。
        model (torch.nn.Module): 已加载的 RT-DETR 模型。
        batch (列表): 当前处理的输入批次。

    方法：
        postprocess: 对模型原始预测结果进行后处理，生成边界框和置信度分数。
        pre_transform: 在将输入图像送入模型推理前执行预处理变换。

    示例：
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """后处理模型的原始预测结果，生成边界框和置信度分数。

        此方法根据 `self.args` 中指定的置信度和类别过滤检测结果。
        它将模型预测结果（已由解码头选出 top-k）转换为 Results 对象，并生成正确缩放的边界框。

        参数：
            preds (列表 | tuple): 模型输出的 [预测结果, extra]，预测结果形状为 (bs, num_queries, 6)，
                格式为 [cx, cy, w, h, 分数, 类别]。
            img (torch.Tensor): 处理后的输入图像，形状为 (N, 3, H, W)。
            orig_imgs (列表 | torch.Tensor): 未处理的原始图像。

        返回：
            (列表[Results]): 包含后处理边界框、置信度分数和类别标签的 Results 对象列表。
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, orig_img, img_path in zip(bboxes, scores, labels, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (label == torch.tensor(self.args.classes, device=label.device)).any(1) & idx
            pred = torch.cat([bbox, score, label], dim=-1)[idx][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # scale x coordinates to original 宽度
            pred[..., [1, 3]] *= oh  # scale y coordinates to original 高度
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """在将输入图像送入模型推理前执行预变换。

        输入图像会通过 letterbox 变换，以确保图像为正方形比例并进行缩放填充。

        参数：
            im (列表[np.ndarray]): 输入图像，形状为 [(H, W, 3) x N]。

        返回：
            (列表): 可供模型推理使用的预变换图像列表。
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
