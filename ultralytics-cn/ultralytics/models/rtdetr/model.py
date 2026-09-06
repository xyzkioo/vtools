# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

参考：
    https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """百度 RT-DETR 模型接口，一种基于 Vision Transformer 的实时对象检测器。

    此模型在保持高精度的同时提供实时性能，支持高效混合编码、IoU 感知查询选择和可调节推理速度。

    属性：
        model (str): 预训练模型路径。

    方法：
        task_map: 返回 RT-DETR 任务到对应 Ultralytics 类的任务映射。

    示例：
        使用预训练模型初始化 RT-DETR
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """使用给定的预训练模型文件初始化 RT-DETR 模型。

        参数：
            model (str): 预训练模型路径，支持 .pt、.yaml 和 .yml 格式。
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """返回 RT-DETR 任务到对应 Ultralytics 类的任务映射。

        返回：
            (dict): 将任务名称映射到 RT-DETR 模型对应 Ultralytics 任务类的字典。
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
