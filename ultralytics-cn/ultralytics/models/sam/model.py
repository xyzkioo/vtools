# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM 模型接口。

此模块提供 Ultralytics Segment Anything Model（SAM）接口，用于实时图像分割。
SAM 模型支持灵活的提示分割，已在 SA-1B 数据集上训练，并具备零样本能力，
能够在没有先验知识的情况下适应新的图像分布和任务。

主要特性：
    - 支持提示的分割
    - 实时性能
    - 零样本迁移能力
    - 在 SA-1B 数据集上训练
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .predict import Predictor, SAM2Predictor, SAM3Predictor


class SAM(Model):
    """用于实时图像分割任务的 SAM（Segment Anything Model）接口类。

    此类提供 Ultralytics Segment Anything Model（SAM）接口，支持基于提示的灵活图像分析。
    它支持边界框、点和标签等多种提示，并具备零样本能力。

    属性：
        model (torch.nn.Module): 已加载的 SAM 模型。
        is_sam2 (bool): 指示模型是否为 SAM2 变体。
        task (str): 任务类型；SAM 模型设置为 "segment"。

    方法：
        predict: 对给定图像或视频源执行分割预测。
        info: 记录 SAM 模型信息。

    示例：
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """初始化 SAM（Segment Anything Model）实例。

        参数：
            model (str): 预训练 SAM 模型文件路径，扩展名应为 .pt 或 .pth。

        异常：
            NotImplementedError: 模型文件扩展名不是 .pt 或 .pth 时抛出。
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        self.is_sam2 = "sam2" in Path(model).stem
        self.is_sam3 = "sam3" in Path(model).stem
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """将指定权重加载到 SAM 模型中。

        参数：
            weights (str): 权重文件路径，应为包含模型参数的 .pt 或 .pth 文件。
            task (str | None): 任务名称。若提供，则指定要加载模型的具体任务。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")
        """
        if self.is_sam3:
            from .build_sam3 import build_interactive_sam3

            self.model = build_interactive_sam3(weights)
        else:
            from .build import build_sam  # slow import

            self.model = build_sam(weights)

    def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """对给定图像或视频源执行分割预测。

        参数：
            source (str | PIL.Image | np.ndarray): 图像或视频文件路径、PIL.Image 对象或 np.ndarray 对象。
            stream (bool): 为 True 时启用实时流式处理。
            bboxes (列表[列表[float]] | None): 用于提示分割的边界框坐标列表。
            points (列表[列表[float]] | None): 用于提示分割的点列表。
            labels (列表[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 传递给预测过程的其他关键字参数。

        返回：
            (列表): 模型预测结果。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"Detected {len(r.masks)} masks")
        """
        overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024}
        kwargs = {**overrides, **kwargs, "retina_masks": True}
        prompts = {"bboxes": bboxes, "points": points, "labels": labels}
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """对给定图像或视频源执行分割预测。

        此方法是 'predict' 方法的别名，为调用 SAM 模型执行分割任务提供便捷方式。

        参数：
            source (str | PIL.Image | np.ndarray | None): 图像或视频文件路径、PIL.Image 对象或 np.ndarray 对象。
            stream (bool): 为 True 时启用实时流式处理。
            bboxes (列表[列表[float]] | None): 用于提示分割的边界框坐标列表。
            points (列表[列表[float]] | None): 用于提示分割的点列表。
            labels (列表[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 要传递给 predict 方法的其他关键字参数。

        返回：
            (列表): 模型预测结果，通常包含分割掩码和其他相关信息。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"Detected {len(results[0].masks)} masks")
        """
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed: bool = False, verbose: bool = True):
        """记录 SAM 模型信息。

        参数：
            detailed (bool): 为 True 时显示模型层和操作的详细信息。
            verbose (bool): 为 True 时将信息输出到控制台。

        返回：
            (tuple): 包含模型信息的元组（模型的字符串表示）。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # 打印摘要信息
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
        """提供从 'segment' 任务到对应 'Predictor' 的映射。

        返回：
            (dict[str, dict[str, type[Predictor]]]): 将 'segment' 任务映射到对应 Predictor 类的字典。
                对于 SAM2 模型，该映射指向 SAM2Predictor，否则指向标准 Predictor。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'分割段': {'predictor': <类别 'ultralytics.models.sam.predict.Predictor'>}}
        """
        return {
            "segment": {"predictor": SAM2Predictor if self.is_sam2 else SAM3Predictor if self.is_sam3 else Predictor}
        }
