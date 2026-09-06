# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.backends.base import BaseBackend
from ultralytics.nn.tasks import (
    ClassificationModel,
    DepthModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    SemanticSegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """YOLO（You Only Look Once）对象检测模型。

    此类为 YOLO 模型提供统一接口，并根据模型文件名自动切换到专用模型类型（YOLOWorld 或 YOLOE）。
    它支持多种计算机视觉任务，包括对象检测、实例分割、语义分割、图像分类、姿态估计和有向边界框检测。

    属性：
        model: 已加载的 YOLO 模型实例。
        task: 任务类型（detect、segment、semantic、classify、pose、obb）。
        overrides: 模型配置覆盖项。

    方法：
        __init__: 初始化 YOLO 模型并自动识别模型类型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。

    示例：
        加载预训练的 YOLO26n 检测模型
        >>> model = YOLO("yolo26n.pt")

        加载预训练的 YOLO26n 分割模型
        >>> model = YOLO("yolo26n-seg.pt")

        根据 YAML 配置初始化
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """初始化 YOLO 模型。

        此构造函数初始化 YOLO 模型，并根据模型文件名自动切换到专用模型类型（YOLOWorld 或 YOLOE）。

        参数：
            model (str | Path): 模型名称或模型文件路径，例如 'yolo26n.pt'、'yolo26n.yaml'。
            task (str, 可选): YOLO 任务类型，例如 'detect'、'segment'、'classify'、'pose'、'obb'；默认为根据模型自动识别。
            verbose (bool): 加载模型时是否显示详细信息。
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是 YOLOWorld PyTorch 模型
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是 YOLOE PyTorch 模型
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # 继续执行默认的 YOLO 初始化
            super().__init__(model=model, task=task, verbose=verbose)
            head = self.model.model[-1]._get_name() if hasattr(self.model, "model") else ""
            if "RTDETR" in (head or BaseBackend.read_metadata(self.model).get("head", "")):  # 如果是 RTDETR 检测头
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将任务映射到对应的模型、训练器、验证器和预测器类。"""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
            "depth": {
                "model": DepthModel,
                "trainer": yolo.depth.DepthTrainer,
                "validator": yolo.depth.DepthValidator,
                "predictor": yolo.depth.DepthPredictor,
            },
            "semantic": {
                "model": SemanticSegmentationModel,
                "trainer": yolo.semantic.SemanticSegmentationTrainer,
                "validator": yolo.semantic.SemanticSegmentationValidator,
                "predictor": yolo.semantic.SemanticSegmentationPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World 对象检测模型。

    YOLO-World 是一种开放词汇对象检测模型，可以根据文本描述检测对象，无需针对特定类别进行训练。
    它扩展了 YOLO 架构，以支持实时开放词汇检测。

    属性：
        model: 已加载的 YOLO-World 模型实例。
        task: 对象检测任务始终设置为 'detect'。
        overrides: 模型配置覆盖项。

    方法：
        __init__: 使用预训练模型文件初始化 YOLOv8-World 模型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。
        set_classes: 设置模型用于检测的类别名称。

    示例：
        加载 YOLOv8-World 模型
        >>> model = YOLOWorld("yolov8s-world.pt")

        设置用于检测的自定义类别
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """使用预训练模型文件初始化 YOLOv8-World 模型。

        加载用于对象检测的 YOLOv8-World 模型。如果未提供自定义类别名称，则分配默认 COCO 类别名称。

        参数：
            model (str | Path): 预训练模型文件路径，支持 *.pt 和 *.yaml 格式。
            verbose (bool): 为 True 时在初始化期间打印额外信息。
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # 没有自定义名称时分配默认 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将任务映射到对应的模型、训练器、验证器和预测器类。"""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.world.WorldValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """设置模型用于检测的类别名称。

        参数：
            classes (list[str]): 类别名称列表，例如 ["person"]。
        """
        self.model.set_classes(classes)
        # 如果提供了背景类别，则移除背景类别
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # 重置预测器中的类别名称
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE 对象检测与实例分割模型。

    YOLOE 是增强版 YOLO 模型，支持目标检测和实例分割任务，并提供视觉提示、文本提示以及视觉和文本位置嵌入等能力。

    属性：
        model: 已加载的 YOLOE 模型实例。
        task: 任务类型（`detect` 或 `segment`）。
        overrides: 模型的配置覆盖项。

    方法：
        __init__: 使用预训练模型文件初始化 YOLOE 模型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。
        get_text_pe: 获取给定文本的位置嵌入。
        get_visual_pe: 获取给定图像和视觉特征的位置嵌入。
        set_vocab: 为 YOLOE 模型设置词汇表和类别名称。
        get_vocab: 获取给定类别名称对应的词汇表，并在融合检测头后作为模型类别。
        set_classes: 设置模型用于检测的类别名称和嵌入向量。
        save_prompt_embeddings: 将当前提示嵌入和类别名称保存到 NPZ 文件。
        load_prompt_embeddings: 从 NPZ 文件加载提示嵌入和类别名称。
        val: 使用文本提示或视觉提示验证模型。
        predict: 在图像、视频、目录、流等输入上执行预测。

    示例：
        加载 YOLOE 分割模型：
        >>> model = YOLOE("yoloe-11s-seg.pt")

        使用视觉提示进行预测，其中 `cls` 为每个边界框对应的类别索引：
        >>> from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        >>> prompts = {"bboxes": np.array([[10, 20, 100, 200]]), "cls": np.array([0])}
        >>> results = model.predict("image.jpg", visual_prompts=prompts, predictor=YOLOEVPSegPredictor)

        将模型重新参数化为不依赖提示的模型，此后模型不再接受提示输入：
        >>> names = ["person", "car", "dog"]
        >>> model.set_vocab(model.get_vocab(names), names)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """使用预训练模型文件初始化 YOLOE 模型。

        参数：
            model (str | Path): 预训练模型文件的路径，支持 `*.pt` 和 `*.yaml` 格式。
            task (str, 可选): 模型任务类型；设置为 `None` 时自动检测。
            verbose (bool): 是否在初始化过程中输出额外信息。
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将任务映射到对应的模型、训练器、验证器和预测器类。"""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """获取给定文本的位置嵌入。"""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """获取给定图像和视觉特征的视觉位置嵌入。

        此方法根据输入图像从视觉特征中提取位置嵌入，要求模型必须是 YOLOEModel 的实例。

        参数：
            img (torch.Tensor): 输入图像张量。
            visual (torch.Tensor): 从图像中提取的视觉特征。

        返回：
            (torch.Tensor): 视觉位置嵌入。

        示例：
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: torch.nn.ModuleList, names: list[str]) -> None:
        """根据给定类别名称，将模型重新参数化为不依赖提示的模型。

        词汇表是针对相同名称由 `get_vocab` 返回的融合分类层，而不是类别名称本身。
        模型必须是 YOLOEModel 的实例。

        参数：
            vocab (torch.nn.ModuleList): 对 `names` 调用 `get_vocab` 返回的融合分类层。
            names (列表[str]): 模型可以检测或分类的类别名称列表。

        异常：
            AssertionError: 当模型不是 YOLOEModel 实例时抛出。

        示例：
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> names = ["person", "car", "dog"]
            >>> model.set_vocab(model.get_vocab(names), names)
        """
        assert isinstance(self.model, YOLOEModel)
        names = check_class_names(names)
        self.predictor = None  # 委托模型会直接将检测头重新参数化
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """获取给定类别名称对应的词汇表，并在融合检测头后将其作为模型类别。"""
        assert isinstance(self.model, YOLOEModel)
        self.predictor = None  # 委托模型会直接融合可提示检测头
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """设置模型用于检测的类别名称和嵌入向量。

        参数：
            classes (列表[str]): 类别名称列表，例如 `["person"]`。
            embeddings (torch.Tensor, 可选): 与类别名称对应的嵌入向量。
        """
        # 确保类别列表中不包含背景类别
        assert " " not in classes
        assert isinstance(self.model, YOLOEModel)
        names = self.model.names.values() if isinstance(self.model.names, dict) else self.model.names
        if embeddings is not None or sorted(names) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # 未提供嵌入向量时生成文本嵌入
            self.model.set_classes(classes, embeddings)

        # 同步当前预测器中的类别名称
        if self.predictor:
            self.predictor.model.names = self.model.names

    def _prompt_embedding_model(self) -> str:
        """返回用于将提示嵌入绑定到当前模型的检查点标识符。"""
        source = self.overrides.get("pretrained") or getattr(self.model, "pt_path", None) or self.ckpt_path
        source = source if isinstance(source, (str, Path)) else self.model.yaml["yaml_file"]
        model = Path(source).stem
        return model[:-4] if model.endswith("-seg") else model

    def save_prompt_embeddings(self, file: str | Path) -> Path:
        """将当前提示嵌入和类别名称保存到 NPZ 文件。

        参数：
            file (str | Path): 目标 NPZ 文件路径。

        返回：
            (Path): 已保存 NPZ 文件的路径。

        异常：
            ValueError: 尚未设置提示嵌入，或提示嵌入无效时抛出。
        """
        assert isinstance(self.model, YOLOEModel)
        embeddings = getattr(self.model, "pe", None)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prompt embeddings must be set before they can be saved.")
        names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        if embeddings.shape[1] != len(names) or not torch.isfinite(embeddings).all():
            raise ValueError("Prompt embeddings must be finite and match the number of class names.")

        file = Path(file)
        if file.suffix.lower() != ".npz":
            raise ValueError(f"Prompt embedding file must have an '.npz' suffix, not '{file.suffix}'.")
        np.savez_compressed(
            file,
            embeddings=embeddings.detach().cpu().float().numpy(),
            names=np.asarray(names, dtype=np.str_),
            model=np.asarray(self._prompt_embedding_model(), dtype=np.str_),
        )
        return file

    def load_prompt_embeddings(self, file: str | Path) -> None:
        """从与模型绑定的 NPZ 文件加载提示嵌入和类别名称。

        参数：
            file (str | Path): 由 `save_prompt_embeddings` 创建的源 NPZ 文件路径。

        异常：
            ValueError: 文件无效，或文件属于其他 YOLOE 架构时抛出。
        """
        assert isinstance(self.model, YOLOEModel)
        with np.load(file, allow_pickle=False) as data:
            if set(data.files) != {"embeddings", "names", "model"}:
                raise ValueError("Prompt embedding file must contain 'embeddings', 'names', and 'model'.")
            embeddings, names, model = data["embeddings"], data["names"], data["model"]

        if model.ndim != 0 or model.dtype.kind != "U":
            raise ValueError("Prompt embedding model identifier must be a scalar string.")
        model_name = str(model.item())
        if model_name != self._prompt_embedding_model():
            raise ValueError(
                f"Prompt embeddings for model '{model_name}' cannot be loaded into '{self._prompt_embedding_model()}'."
            )
        if names.ndim != 1 or names.dtype.kind != "U":
            raise ValueError("Prompt embedding class names must be a one-dimensional string array.")
        if embeddings.dtype != np.float32 or embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prompt embeddings must be a float32 array with shape (1, classes, dimensions).")
        if embeddings.shape[1] != len(names) or embeddings.shape[2] != self.model.model[-1].embed:
            raise ValueError("Prompt embedding shape does not match the class names or model embedding dimension.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Prompt embeddings must contain only finite values.")
        self.set_classes(names.tolist(), torch.from_numpy(embeddings.copy()).to(next(self.model.parameters()).device))

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """使用文本提示或视觉提示验证模型。

        参数：
            validator (callable, 可选): 可调用的验证器函数；为 `None` 时加载默认验证器。
            load_vp (bool): 是否加载视觉提示；为 `False` 时使用文本提示。
            refer_data (str, 可选): 视觉提示所需参考数据的路径。
            **kwargs (Any): 用于覆盖默认设置的其他关键字参数。

        返回：
            (dict): 验证过程中计算得到的指标统计信息。
        """
        custom = {"rect": not load_vp}  # 方法默认设置
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 右侧参数具有更高优先级

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, np.ndarray | list[np.ndarray]] | None = None,
        refer_image=None,
        predictor=yolo.yoloe.YOLOEVPDetectPredictor,
        **kwargs,
    ):
        """在图像、视频、目录、数据流等输入上执行预测。

        参数：
            source (str | int | PIL.Image | np.ndarray, 可选): 预测输入源。可接受图像路径、目录路径、URL、
                YouTube 数据流、PIL 图像、NumPy 数组或摄像头索引。
            stream (bool): 是否以流式方式返回预测结果。为 `True` 时，结果会在计算完成后通过生成器逐个产出。
            visual_prompts (dict[str, np.ndarray | 列表[np.ndarray]]): 包含视觉提示的字典。非空时必须包含 `bboxes`
                和 `cls` 键；对于显式列表、元组或在未设置 `refer_image` 时的四维张量输入，二者可以是扁平数组，
                也可以按图像分别提供一个数组。
            refer_image (str | PIL.Image | np.ndarray, 可选): 视觉提示使用的参考图像。
            predictor (callable): 处理视觉提示预测结果的自定义预测器类，默认为 `YOLOEVPDetectPredictor`。
            **kwargs (Any): 传递给预测器的其他关键字参数。

        返回：
            (列表 | generator): `stream=True` 时返回 Results 对象列表或 Results 对象生成器。

        示例：
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # 使用视觉提示，其中 `cls` 为每个边界框对应的类别索引
            >>> from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            >>> prompts = {"bboxes": np.array([[10, 20, 100, 200]]), "cls": np.array([0])}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts, predictor=YOLOEVPSegPredictor)
        """
        visual_prompts = visual_prompts if visual_prompts is not None else {}
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            bboxes, classes = visual_prompts["bboxes"], visual_prompts["cls"]
            assert all(hasattr(x, "__len__") and getattr(x, "ndim", 1) > 0 for x in (bboxes, classes)), (
                "Expected non-scalar 'bboxes' and 'cls' visual prompts"
            )
            assert len(bboxes) == len(classes) > 0, "Expected an equal, non-zero number of boxes and classes"
            nested = yolo.yoloe.YOLOEVPDetectPredictor.is_per_image(visual_prompts)  # 每张图像对应一个提示数组
            assert not isinstance(source, np.ndarray) or source.ndim != 4, "4-D NumPy sources are not supported"
            per_image_source = isinstance(source, (list, tuple)) or (
                isinstance(source, torch.Tensor) and source.ndim == 4
            )
            assert not nested or (refer_image is None and per_image_source), (
                "Expected flat 'bboxes' and 'cls' arrays for a non-sequence source or when refer_image is set"
            )
            multi = nested
            pairs = list(zip(bboxes, classes)) if multi else [(bboxes, classes)]
            assert not multi or len(pairs) == len(source), (
                f"Expected one prompt per source image, but got {len(pairs)} prompts for {len(source)} images"
            )
            assert all(
                getattr(b, "ndim", 2) == 2
                and (not multi or b.shape[1:] == (4,))
                and getattr(c, "ndim", 1) == 1
                and len(b) == len(c)
                and all(np.isscalar(x) and not isinstance(x, (str, bytes)) for x in c)
                for b, c in pairs
            ), "Expected non-string scalar class indices for each bounding box"
            per_image = [len(set(c.tolist() if isinstance(c, np.ndarray) else c)) for _, c in pairs]
            assert all(per_image), "Expected at least one class per image"
            num_cls = max(per_image)
            if type(self.predictor) is not predictor:
                args = get_cfg(overrides={**self.overrides, **kwargs})
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": kwargs.get("verbose", self.overrides.get("verbose", refer_image is None)),
                        "batch": 1,
                        "device": args.device,
                        "quantize": args.quantize,
                        "imgsz": args.imgsz,
                    },
                    _callbacks=self.callbacks,
                )

            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model, verbose=self.predictor.args.verbose)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # 注意：将视频或数据流的第一帧设置为推理参考图像
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # 重置预测器
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # 没有视觉提示时重置预测器
        self.overrides["agnostic_nms"] = True  # YOLOE 默认使用类别无关的 NMS

        return super().predict(source, stream, **kwargs)
