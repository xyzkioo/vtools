# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PosePredictor(DetectionPredictor):
    """继承 DetectionPredictor、用于根据姿态模型生成预测结果的类。

    此类专用于姿态估计，在继承 DetectionPredictor 标准对象检测能力的同时处理关键点检测。

    属性：
        args (namespace): 预测器配置参数。
        model (torch.nn.Module): 已加载的具备关键点检测能力的 YOLO 姿态模型。

    方法：
        construct_result: 根据预测结果构建包含关键点的结果对象。

    示例：
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo26n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化用于姿态估计任务的 PosePredictor。

        设置 PosePredictor 实例，将其配置为姿态检测任务，并处理 Apple MPS 的设备特定警告。

        参数：
            cfg (Any): 预测器配置。
            overrides (dict, 可选): 优先于 cfg 的配置覆盖项。
            _callbacks (dict, 可选): 预测期间调用的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

    def construct_result(self, pred, img, orig_img, img_path):
        """根据预测结果构建包含关键点的结果对象。

        此方法扩展父类实现，从预测结果中提取关键点数据并添加到结果对象。

        参数：
            pred (torch.Tensor): 预测边界框、分数和关键点，形状为 (N, 6+K*D)，其中 N 为检测数量、
                K 为关键点数量、D 为关键点维度。
            img (torch.Tensor): 处理后的输入图像张量，形状为 (B, C, H, W)。
            orig_img (np.ndarray): 未处理的原始图像 NumPy 数组。
            img_path (str): 原始图像文件路径。

        返回：
            (Results): 包含原始图像、图像路径、类别名称、边界框和关键点的结果对象。
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # 从预测结果中提取关键点，并根据模型的关键点形状调整维度
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # 缩放关键点坐标以匹配原始图像尺寸
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
