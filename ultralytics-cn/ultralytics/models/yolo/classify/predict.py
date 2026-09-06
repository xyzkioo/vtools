# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import torch
from PIL import Image

from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """继承 BasePredictor、用于根据分类模型生成预测结果的类。

    此预测器处理分类模型的专用需求，包括图像预处理和预测结果后处理，以生成分类结果。

    属性：
        args (dict): 预测器配置参数。

    方法：
        preprocess: 将输入图像转换为模型兼容格式。
        postprocess: 将模型预测结果处理为 Results 对象。

    示例：
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.classify import ClassificationPredictor
        >>> args = dict(model="yolo26n-cls.pt", source=ASSETS)
        >>> predictor = ClassificationPredictor(overrides=args)
        >>> predictor.predict_cli()

    注意：
        - 也可以将 Torchvision 分类模型传给 'model' 参数，例如 model='resnet18'。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用指定配置初始化 ClassificationPredictor，并将任务设置为 'classify'。

        此构造函数初始化用于分类任务的 ClassificationPredictor 实例。无论输入配置如何，
        它都会确保任务设置为 'classify'。

        参数：
            cfg (dict): 包含预测设置的默认配置字典。
            overrides (dict, 可选): 优先于 cfg 的配置覆盖项。
            _callbacks (dict, 可选): 预测期间执行的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def setup_source(self, source):
        """设置输入源、推理模式和分类变换。"""
        super().setup_source(source)
        transforms = getattr(self.model.model, "transforms", None)  # YAML 构建模型和旧版检查点中可能不存在
        size = getattr(transforms.transforms[0], "size", max(self.imgsz)) if transforms is not None else None
        self.transforms = (
            transforms if size == max(self.imgsz) and self.model.format == "pt" else classify_transforms(self.imgsz)
        )

    def preprocess(self, img):
        """将输入图像转换为模型兼容的张量格式，并执行适当的归一化。"""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # 将 uint8 转换为 fp16/32
        return img

    def postprocess(self, preds, img, orig_imgs):
        """将预测结果处理为包含分类概率的 Results 对象。

        参数：
            preds (torch.Tensor): 模型输出的原始预测结果。
            img (torch.Tensor): 预处理后的输入图像。
            orig_imgs (列表[np.ndarray] | torch.Tensor): 预处理前的原始图像。

        返回：
            (列表[Results]): 包含每张图像分类结果的 Results 对象列表。
        """
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
