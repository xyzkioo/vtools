# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """用于对象检测的 YOLO-NAS 模型。

    此类为 YOLO-NAS 模型提供接口，并继承 Ultralytics 引擎中的 `Model` 类，
    用于使用预训练或自定义训练的 YOLO-NAS 模型执行对象检测。

    属性：
        model (torch.nn.Module): 已加载的 YOLO-NAS 模型。
        task (str): 模型任务类型，默认为 'detect'。
        predictor (NASPredictor): 用于生成预测结果的预测器实例。
        validator (NASValidator): 用于模型验证的验证器实例。

    方法：
        info: 记录模型信息并返回模型详情。

    示例：
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

    注意：
        YOLO-NAS 模型仅支持预训练模型，不要提供 YAML 配置文件。
    """

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        """使用提供的模型或默认模型初始化 NAS 模型。"""
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """加载已有 NAS 模型权重，或使用预训练权重创建新的 NAS 模型。

        参数：
            weights (str): 模型权重文件路径或模型名称。
            task (str, 可选): 模型任务类型。
        """
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch_load(attempt_download_asset(weights))
        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # 统一模型属性以保证兼容性
        self.model.fuse = lambda verbose=True, imgsz=640: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # 用于 info()
        self.model.yaml = {}  # 用于 info()
        self.model.pt_path = str(weights)  # 用于 export()
        self.model.task = "detect"  # 用于 export()
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 用于 export()
        self.model.eval()

    def info(self, detailed: bool = False, verbose: bool = True) -> tuple:
        """记录模型信息。

        参数：
            detailed (bool): 是否显示模型详细信息。
            verbose (bool): 是否输出详细日志。

        返回：
            (tuple): 模型信息元组，内容为（层数、参数量、梯度数、GFLOPs）。
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """返回将任务映射到对应预测器和验证器类的字典。"""
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
