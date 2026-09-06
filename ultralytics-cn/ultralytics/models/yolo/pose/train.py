# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, RANK


class PoseTrainer(yolo.detect.DetectionTrainer):
    """继承 DetectionTrainer 的训练器，用于训练 YOLO 姿态估计模型。

    此训练器专门处理姿态估计任务，负责模型训练、验证，以及姿态关键点和边界框的可视化。

    属性：
        args (dict): 用于训练的配置参数。
        model (PoseModel): 正在训练的姿态估计模型。
        data (dict): 数据集配置，包括关键点形状信息。
        loss_names (tuple): 损失组件名称，来自损失函数返回的损失字典。

    方法：
        get_model：使用指定配置获取姿态估计模型。
        set_model_attributes：在模型上设置关键点形状属性。
        get_validator：创建用于模型评估的验证器实例。
        plot_training_samples：可视化带有关键点的训练样本。
        get_dataset：获取数据集并确保包含 kpt_shape 键。

    示例：
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化用于训练 YOLO 姿态估计模型的 PoseTrainer 对象。

        参数：
            cfg (dict, 可选): 包含训练参数的默认配置字典。
            overrides (dict, 可选): 用于覆盖默认配置的参数字典。
            _callbacks (dict, 可选): 训练期间执行的回调函数字典。

        注意：
            无论 overrides 中提供了什么值，此训练器都会自动将任务设置为 'pose'。
            由于姿态模型存在已知问题，使用 Apple MPS 设备时会发出警告。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """使用指定配置和权重获取姿态估计模型。

        参数：
            cfg (str | Path | dict, 可选): 模型配置文件路径或配置字典。
            weights (str | Path, 可选): 模型权重文件路径。
            verbose (bool): 是否显示模型信息。

        返回：
            (PoseModel): 初始化后的姿态估计模型。
        """
        model = self.set_model_names_for_load(
            PoseModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """设置 PoseModel 的关键点形状属性。"""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.kpt_oks_sigmas = self.data.get("kpt_oks_sigmas")
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """返回用于验证的 PoseValidator 实例。"""
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """获取数据集并确保其中包含必需的 `kpt_shape` 键。

        返回：
            (dict): 包含训练、验证和测试数据集及类别名称的字典。

        异常：
            KeyError: 数据集中不存在 `kpt_shape` 键时抛出。
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose")
        return data
