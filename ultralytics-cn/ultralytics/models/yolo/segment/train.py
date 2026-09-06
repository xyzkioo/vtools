# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """继承 DetectionTrainer、用于训练分割模型的训练器类。

    此训练器专门处理分割任务，在检测训练器基础上扩展模型初始化、验证和可视化等分割专用功能。

    属性：
        loss_names (tuple[str]): 损失组件名称，来自 criterion 返回的损失字典。

    示例：
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """初始化 a SegmentationTrainer 对象.

        参数：
            cfg (dict): Configuration 字典 with default 训练 settings.
            overrides (dict, 可选): Dictionary of 参数 overrides for the default 配置.
            _callbacks (dict, 可选): Dictionary of callback functions to be executed 训练期间.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """使用指定配置和权重初始化并返回 SegmentationModel。

        参数：
            cfg (dict | str, 可选): Model 配置. Can be a 字典, a 路径 to a YAML 文件, or None.
            权重 (str | Path, 可选): 路径： pretrained 权重 文件.
            verbose (bool): Whether to display 模型 information during initialization.

        返回：
            (SegmentationModel): Initialized segmentation 模型 with loaded 权重 if specified.

        示例：
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
        """
        model = self.set_model_names_for_load(
            SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于验证 YOLO 模型的 SegmentationValidator 实例。"""
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
