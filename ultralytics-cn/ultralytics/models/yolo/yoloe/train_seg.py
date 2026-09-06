# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.segment import SegmentationTrainer

from .train import YOLOEPETrainer, YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """用于 YOLOE 分割模型的训练器。"""


class YOLOEPESegTrainer(SegmentationTrainer):
    """以线性探测方式微调 YOLOE 分割模型。"""

    get_model = YOLOEPETrainer.get_model  # 共享线性探测构建器；SegmentationTrainer 仍是唯一基类


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """用于从头训练 YOLOE 分割模型的训练器，不使用预训练权重。"""


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """具备视觉提示（VP）能力的 YOLOE 分割模型训练器。"""
