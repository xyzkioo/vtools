# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """继承 DetectionTrainer 的训练器，用于训练旋转边界框（OBB）模型。

    此训练器专门处理能够检测任意角度目标的 YOLO 模型，而不仅是检测轴对齐矩形。

    属性：
        loss_names (tuple): 损失组件名称，来自损失函数返回的损失字典。

    方法：
        get_model：使用指定配置和权重返回初始化后的 OBBModel。
        get_validator：返回用于验证 YOLO 模型的 OBBValidator 实例。

    示例：
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """初始化用于训练旋转边界框（OBB）模型的 OBBTrainer 对象。

        参数：
            cfg (dict, 可选): 训练器配置字典，包含训练参数和模型配置。
            overrides (dict, 可选): 覆盖配置的参数字典，其中的值优先于 cfg 中的值。
            _callbacks (dict, 可选): 训练期间调用的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """使用指定配置和权重返回初始化后的 OBBModel。

        参数：
            cfg (str | dict, 可选): 模型配置，可以是 YAML 配置文件路径、包含配置参数的字典，或使用默认配置的 None。
            weights (str | Path, 可选): 预训练权重文件路径；为 None 时使用随机初始化。
            verbose (bool): 初始化期间是否显示模型信息。

        返回：
            (OBBModel): 使用指定配置和权重初始化后的 OBBModel。

        示例：
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
        """
        model = self.set_model_names_for_load(
            OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于验证 YOLO 模型的 OBBValidator 实例。"""
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
