# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """用于 Segment Anything 任务的 FastSAM 模型接口。

    此类继承基础 Model 类，为 FastSAM（Fast Segment Anything Model）实现提供专用功能，
    支持高效、准确的图像分割以及可选提示。

    属性：
        模型 (str): 预训练 FastSAM 模型文件路径。
        task (str): 任务类型，FastSAM 模型设置为 "分割段"。

    方法：
        predict: 对图像或视频源执行分割预测，支持可选提示。
        task_map: 返回将分割任务映射到预测器和验证器类的映射。

    示例：
        初始化 FastSAM 模型并执行预测
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("FastSAM-x.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

        使用边界框提示执行预测
        >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
    """

    def __init__(self, model: str | Path = "FastSAM-x.pt"):
        """使用指定的预训练权重初始化 FastSAM 模型。"""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM only supports pre-trained weights."
        super().__init__(model=model, task="segment")

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes: list | None = None,
        points: list | None = None,
        labels: list | None = None,
        texts: list | None = None,
        **kwargs: Any,
    ):
        """对图像或视频源执行分割预测。

        支持使用边界框、点、标签和文本进行提示分割。此方法会打包这些提示，
        并将其传递给父类的 predict 方法处理。

        参数：
            source (str | PIL.Image | np.ndarray): 预测输入源，可以是文件路径、URL、PIL 图像或 numpy 数组。
            stream (bool): 是否为视频输入启用实时流式模式。
            bboxes (列表, 可选): 提示分割使用的边界框坐标，格式为 [[x1, y1, x2, y2]]。
            points (列表, 可选): 提示分割使用的点坐标，格式为 [[x, y]]。
            labels (列表, 可选): 提示分割使用的类别标签。
            texts (列表, 可选): 用于分割引导的文本提示。
            **kwargs (Any): 传递给预测器的其他关键字参数。

        返回：
            (列表): 包含预测结果的 Results 对象列表。
        """
        prompts = {"bboxes": bboxes, "points": points, "labels": labels, "texts": texts}
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """返回将 segment 任务映射到对应预测器和验证器类的字典。"""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
