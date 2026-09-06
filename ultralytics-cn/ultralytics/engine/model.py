# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, _handle_deprecation, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import BaseModel, guess_model_task, load_checkpoint, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    PLATFORM_URL,
    RANK,
    SETTINGS,
    YAML,
    callbacks,
    checks,
)
from ultralytics.utils.torch_utils import unwrap_model


class Model(torch.nn.Module):
    """实现 YOLO 模型的基类，为不同模型类型统一 API。

    此类为 YOLO 模型的训练、验证、预测、导出和基准测试等操作提供通用接口，同时支持从本地文件或 Triton Server
    加载的不同类型模型。

    属性：
        callbacks (dict)：模型运行期间各类事件的回调函数字典。
        predictor (BasePredictor)：用于执行预测的预测器对象。
        model (torch.nn.Module)：底层 PyTorch 模型。
        trainer (BaseTrainer)：用于训练模型的训练器对象。
        ckpt (dict)：模型从 *.pt 文件加载时的检查点数据。
        cfg (str)：模型从 *.yaml 文件加载时的配置。
        ckpt_path (str)：检查点文件路径。
        overrides (dict)：模型配置的覆盖参数字典。
        metrics (ultralytics.utils.metrics.DetMetrics)：最新的训练或验证指标。
        task (str)：模型执行的任务类型。
        model_name (str)：模型名称。

    方法：
        __call__：predict 方法的别名，使模型实例可以直接调用。
        _new：根据配置文件初始化新模型。
        _load：从检查点文件加载模型。
        _check_is_pytorch_model：确保当前模型是 PyTorch 模型。
        reset_weights：将模型权重重置为初始状态。
        load：从指定文件加载模型权重。
        save：将当前模型状态保存到文件。
        info：记录或返回模型信息。
        fuse：融合 Conv2d 和 BatchNorm2d 层，以优化推理。
        predict：对给定图像源执行预测。
        track：执行目标跟踪。
        val：在数据集上验证模型。
        benchmark：在不同导出格式上测试模型性能。
        export：将模型导出为不同格式。
        train：在数据集上训练模型。
        tune：执行超参数调优。
        _apply：将函数应用于模型张量。
        add_callback：为事件添加回调函数。
        clear_callback：清除事件的所有回调。
        reset_callbacks：将回调重置为默认函数。

    示例：
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: str | Path | Model = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
    ) -> None:
        """初始化 YOLO 模型类的新实例。

        此构造函数根据提供的模型路径或名称设置模型，支持本地文件和 Triton Server 模型等不同模型来源，
        并初始化模型的关键属性，为训练、预测和导出等操作做好准备。

        参数：
            model (str | Path | Model)：要加载或创建的模型路径或名称。可以是本地文件路径、Triton Server 模型，
                或已经初始化的 Model 实例。
            task (str，可选)：模型执行的具体任务。为 None 时从配置中推断。
            verbose (bool)：为 True 时，在模型初始化和后续操作期间启用详细输出。

        异常：
            FileNotFoundError：指定的模型文件不存在或无法访问。
            ValueError：模型文件或配置无效，或不受支持。
        """
        if isinstance(model, Model):
            self.__dict__ = model.__dict__  # 接受已经初始化的模型
            return
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # 复用预测器
        self.model = None  # 模型对象
        self.trainer = None  # 训练器对象
        self.ckpt = {}  # 从 *.pt 加载时使用
        self.cfg = None  # 从 *.yaml 加载时使用
        self.ckpt_path = None
        self.overrides = {}  # 训练器对象的覆盖参数
        self.metrics = None  # 验证/训练指标
        self.task = task  # 任务类型
        self.model_name = None  # 模型名称
        model = str(model).strip()

        # 检查是否为 Triton Server 模型
        if self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # 未明确指定时设置 task=detect
            return

        # 加载或创建新的 YOLO 模型
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 避免确定性警告
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # 删除 super().training，以便访问 self.model.training
        del self.training

    def __call__(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
        """predict 方法的别名，使模型实例可以直接调用并执行预测。

        此方法允许直接调用模型实例并传入必要参数，从而简化预测过程。

        参数：
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple)：用于预测的图像源。
                可以是文件路径、URL、PIL 图像、NumPy 数组、PyTorch 张量，或这些对象组成的列表/元组。
            stream (bool)：为 True 时，将输入源作为连续流进行预测。
            **kwargs (Any)：用于配置预测过程的其他关键字参数。

        返回：
            (Iterator[ultralytics.engine.results.Results | torch.Tensor] | list[ultralytics.engine.results.Results] |
            list[torch.Tensor])：预测结果或嵌入；`stream=True` 时以流式方式返回。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"图像中检测到 {len(r)} 个目标")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """检查给定模型字符串是否为 Triton Server URL。

        此静态方法使用 urllib.parse.urlsplit() 解析模型字符串，判断其是否表示有效的 Triton Server URL。

        参数：
            model (str)：待检查的模型字符串。

        返回：
            (bool)：模型字符串为有效 Triton Server URL 时返回 True，否则返回 False。

        示例：
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            True
            >>> Model.is_triton_model("yolo26n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """初始化新模型，并根据模型定义推断任务类型。

        此方法根据给定配置文件创建新的模型实例，加载模型配置，在未指定任务类型时进行推断，
        并使用任务映射中的对应类初始化模型。

        参数：
            cfg (str)：YAML 格式模型配置文件路径。
            task (str，可选)：模型执行的具体任务。为 None 时从配置中推断。
            model (type[torch.nn.Module]，可选)：自定义模型类。提供时将替代任务映射中的默认模型类。
            verbose (bool)：为 True 时，在加载期间显示模型信息。

        异常：
            ValueError：配置文件无效或无法推断任务类型。
            ImportError：指定任务所需的依赖未安装。

        示例：
            >>> model = Model()
            >>> model._new("yolo26n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # 构建模型
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # 以下内容用于支持从 YAML 文件导出
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 合并默认参数和模型参数（优先使用模型参数）
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """从检查点文件加载模型，或根据权重文件初始化模型。

        此方法支持从 .pt 检查点文件或其他格式的权重文件加载模型，并根据加载的权重设置模型、任务和相关属性。

        参数：
            weights (str)：待加载模型权重文件的路径。
            task (str，可选)：模型关联的任务。为 None 时从模型中推断。

        异常：
            FileNotFoundError：指定的权重文件不存在或无法访问。
            ValueError：权重文件格式不受支持或无效。

        示例：
            >>> model = Model()
            >>> model._load("yolo26n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(checks.REMOTE_FILE_PREFIXES):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # 下载并返回本地文件
        weights = checks.check_model_file_from_stem(weights)  # 添加后缀，例如 yolo26n -> yolo26n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # 所有情况下都会执行，与上面的调用不重复
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """检查模型是否为 PyTorch 模型；如果不是则抛出 TypeError。

        此方法验证模型是否为 PyTorch 模块或 .pt 文件，确保需要 PyTorch 模型的操作只作用于兼容的模型类型。

        异常：
            TypeError：模型不是 PyTorch 模块或 .pt 文件。错误消息会提供支持的模型格式和操作的详细信息。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model._check_is_pytorch_model()  # 未引发错误
            >>> model = Model("yolo26n.onnx")
            >>> model._check_is_pytorch_model()  # 引发 TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo26n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> Model:
        """将模型权重重置为初始状态。

        此方法遍历模型中的所有模块，如果模块具有 `reset_parameters` 方法，则重置其参数；同时确保所有参数的
        `requires_grad` 为 True，使其可以在训练期间更新。

        返回：
            (Model)：权重已重置的模型实例。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: str | Path = "yolo26n.pt") -> Model:
        """将指定权重文件中的参数加载到模型中。

        此方法支持从文件或权重对象直接加载权重，并按名称和形状匹配参数后传递到模型。

        参数：
            weights (str | Path)：权重文件路径或权重对象。

        返回：
            (Model)：已加载权重的模型实例。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = Model()
            >>> model.load("yolo26n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # 记住用于 DDP 训练的权重
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """将当前模型状态保存到文件。

        此方法将模型检查点（ckpt）导出到指定文件名，并包含日期、Ultralytics 版本、许可证信息和文档链接等元数据。

        参数：
            filename (str | Path)：保存模型的文件名。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().astimezone().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int | list[int, int] = 640):
        """显示模型信息。

        此方法根据传入参数提供模型概览或详细信息，并可控制输出的详细程度。

        参数：
            detailed (bool)：为 True 时显示模型层和参数的详细信息。
            verbose (bool)：为 True 时打印信息并返回模型摘要；为 False 时返回 None。
            imgsz (int | list[int, int])：用于计算 FLOPs 的输入图像尺寸。

        返回：
            (tuple)：包含层数（int）、参数量（int）、梯度数量（int）和 GFLOPs（float）的元组。
                verbose 为 False 时返回 None。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model.info()  # 打印模型摘要并返回元组
            >>> model.info(detailed=True)  # 打印详细信息并返回元组
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose, imgsz=imgsz)

    def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640) -> Model:
        """融合模型中的 Conv2d 和 BatchNorm2d 层，以优化推理。

        此方法遍历模型模块，将连续的 Conv2d 和 BatchNorm2d 层融合为单层。通过减少前向传播所需的操作数量和
        内存访问次数，这种融合可以显著提升推理速度。

        融合过程通常会将 BatchNorm2d 的参数（均值、方差、权重和偏置）折叠到前置 Conv2d 层的权重和偏置中，
        最终得到一个同时执行卷积和归一化的 Conv2d 层。

        参数：
            verbose (bool)：是否在融合后打印模型信息。
            imgsz (int | list[int, int])：用于计算 FLOPs 的输入图像尺寸。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model.fuse()
            >>> # 模型现已融合，可以进行优化推理
        """
        self._check_is_pytorch_model()
        # DistillationModel 会融合到学生模型中，因此采用返回的模型
        self.model = self.model.fuse(verbose=verbose, imgsz=imgsz)
        return self

    def embed(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[torch.Tensor] | list[torch.Tensor]:
        """根据提供的图像源生成图像嵌入。

        此方法封装 `predict()` 方法，返回图像源的特征嵌入。默认从模型倒数第二层提取嵌入，
        可在 `kwargs` 中传入 `embed=[layer_index]` 选择指定层。

        参数：
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor)：用于生成嵌入的图像源。
                可以是文件路径、URL、NumPy 数组等。
            stream (bool)：为 True 时以流式方式返回预测结果。
            **kwargs (Any)：用于配置嵌入过程的其他关键字参数。

        返回：
            (Iterator[torch.Tensor] | list[torch.Tensor])：图像嵌入；`stream=True` 时以流式方式返回。

        异常：
            TypeError：模型不是 Ultralytics PyTorch 模型。导出格式和第三方模块不会暴露可用于提取嵌入的中间层。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> results = model.predict(image)
            >>> print(embeddings[0].shape)
            >>> print(results[0].boxes.shape)
        """
        self._check_is_pytorch_model()
        model = unwrap_model(self.model)
        if not isinstance(model, BaseModel):  # 例如 super-gradients YOLO-NAS 模块，该模块没有层列表
            raise TypeError(f"model='{type(model).__name__}' is not an Ultralytics model and cannot be embedded.")
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(model.model) - 2]  # 未传入索引时嵌入倒数第二层
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
        """使用 YOLO 模型对给定图像源执行预测。

        此方法支持通过关键字参数进行各种配置，也支持使用自定义预测器或默认预测器处理不同类型的图像源，
        并可以流式模式运行。

        参数：
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple)：用于预测的图像源。
                支持文件路径、URL、PIL 图像、NumPy 数组和 PyTorch 张量等多种类型。
            stream (bool)：为 True 时，将输入源作为连续流进行预测。
            predictor (BasePredictor，可选)：用于执行预测的自定义预测器实例；为 None 时使用默认预测器。
            **kwargs (Any)：用于配置预测过程的其他关键字参数，包括用于返回指定层特征嵌入的 `embed`。

        返回：
            (Iterator[ultralytics.engine.results.Results | torch.Tensor] | list[ultralytics.engine.results.Results] |
            list[torch.Tensor])：预测结果或嵌入；`stream=True` 时以流式方式返回。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # 打印检测边界框

        注意：
            - 如果未提供 `source`，则默认使用 ASSETS 常量，并发出警告。
            - 如果预测器尚未创建，则此方法会创建新的预测器，并在每次调用时更新其参数。
            - 对于 SAM 类型模型，可以通过关键字参数传入 `prompts`。
        """
        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True, "embed": None}
        prompts = kwargs.pop("prompts", None)  # 用于 SAM 类型模型
        args = {**self.overrides, **custom, **kwargs}  # 右侧参数具有最高优先级

        if not self.predictor or self.predictor.args.device != args.get("device", self.predictor.args.device):
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # 预测器已经设置时才更新参数
            save_keys = ("project", "name", "save_dir", "exist_ok")
            prev_save_args = tuple(getattr(self.predictor.args, k, None) for k in save_keys)
            setup_keys = ("device", "dnn", "data", "end2end", "compile", "channels_last", "quantize")
            base_args = {
                **DEFAULT_CFG_DICT,
                **self.overrides,
                **{k: getattr(self.predictor.args, k) for k in setup_keys},
            }
            if hasattr(self.predictor.model, "imgsz") and not self.predictor.model.dynamic:
                base_args["imgsz"] = self.predictor.args.imgsz
            self.predictor.args = get_cfg(base_args, {**custom, **kwargs})
            if self.predictor.args.show:
                self.predictor.args.show = checks.check_imshow(warn=True)
            if prev_save_args != tuple(getattr(self.predictor.args, k, None) for k in save_keys):
                self.predictor.save_dir = get_save_dir(self.predictor.args)
            if getattr(self.model, "end2end", False):
                self.model.set_head_attr(
                    max_det=max(self.predictor.args.max_det, 300), agnostic_nms=self.predictor.args.agnostic_nms
                )
        if prompts and hasattr(self.predictor, "set_prompts"):  # 用于 SAM 类型模型
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[Results]:
        """使用已注册的跟踪器对指定输入源执行目标跟踪。

        此方法使用模型预测器和可选的已注册跟踪器执行目标跟踪，支持文件路径、视频流等不同输入源，并可通过关键字
        参数进行自定义。每次调用都会注册或刷新跟踪回调，因此后续设置的 `persist` 参数也会生效。

        参数：
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor，可选)：目标跟踪的输入源，可以是文件路径、
                URL 或视频流。
            stream (bool)：为 True 时，将输入源作为连续视频流处理。
            persist (bool)：为 True 时，在多次调用之间保留跟踪器状态。
            **kwargs (Any)：用于配置跟踪过程的其他关键字参数。

        返回：
            (list[ultralytics.engine.results.Results])：跟踪结果列表，每个元素都是 Results 对象。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # 打印跟踪 ID

        注意：
            - 此方法为基于 ByteTrack 的跟踪设置默认置信度阈值 0.1。
            - 关键字参数中会显式设置跟踪模式。
            - 视频跟踪的批次大小设置为 1。
        """
        from ultralytics.trackers import register_tracker

        register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # 基于 ByteTrack 的方法需要以低置信度预测作为输入
        kwargs["batch"] = kwargs.get("batch") or 1  # 视频跟踪的批次大小为 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """使用指定数据集和验证配置验证模型。

        此方法支持通过各种设置自定义验证过程，也支持使用自定义验证器或默认验证方式。方法会合并默认配置、
        方法默认参数和用户提供的参数，以配置验证过程。

        参数：
            validator (ultralytics.engine.validator.BaseValidator，可选)：用于验证模型的自定义验证器实例。
            **kwargs (Any)：用于自定义验证过程的任意关键字参数。

        返回：
            (ultralytics.utils.metrics.DetMetrics)：验证过程得到的验证指标。具体指标类型取决于任务（例如
                DetMetrics、SegmentMetrics、PoseMetrics 或 ClassifyMetrics）。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # 打印 mAP50-95
        """
        custom = {"rect": True}  # 方法默认参数
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 右侧参数具有最高优先级

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def calibrate(self, data=None, **kwargs: Any):
        """在小型带标注数据集上拟合仅缩放的深度校准（仅适用于 depth 任务）。

        方法先执行一次验证，再通过 :func:`fit_calibration_selective` 根据真实深度拟合全局对数仿射变换
        ``d' = exp(a·log d + b)``，采用与训练器自动校准相同的“仅在有帮助时校准”策略（根据留出集 δ1 选择恒等
        变换或仅缩放变换），并将 ``(a, b)`` 写入检测头的 ``cal_a``/``cal_b`` 缓冲区。此过程不进行梯度训练，
        也不会修改解码器权重，因此不会破坏相对深度结构。之后调用 ``model.save(...)`` 保存校准结果。

        参数：
            data (str，可选)：提供带标注校准划分的数据集 YAML 文件。
            **kwargs (Any)：其他验证参数（例如 ``imgsz``、``batch``、``device``、``split``）。

        返回：
            (tuple | None)：拟合得到的 ``(a, b)``；如果有效深度像素所在图像少于 2 张，则返回 ``None``。

        示例：
            >>> model = YOLO("yolo26s-depth.pt")
            >>> model.calibrate(data="my_depth_dataset.yaml")
            >>> model.save("yolo26s-depth-calibrated.pt")
        """
        self._check_is_pytorch_model()
        if self.task != "depth":
            raise ValueError(f"calibrate() is only supported for depth models (task='depth'), got task={self.task!r}.")
        from ultralytics.models.yolo.depth.calibrate import _depth_head, fit_calibration_selective

        if _depth_head(self.model) is None:
            raise ValueError("Model has no Depth head with calibration buffers (cal_a/cal_b).")
        args = {**self.overrides, **kwargs, "mode": "val", "task": "depth"}
        if data is not None:
            args["data"] = data
        validator = self._smart_load("validator")(args=args, _callbacks=self.callbacks)
        validator(model=self.model)  # 构建数据加载器，并使用当前校准结果报告指标
        res = fit_calibration_selective(
            self.model, validator.dataloader, validator.device, max_depth=validator.data.get("max_depth") or 100.0
        )
        if res is None:
            return None
        LOGGER.info("Call model.save(...) to persist the calibration.")
        return res["a"], res["b"]

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        """在不同导出格式上测试模型性能。

        此方法评估模型在 ONNX、TorchScript 等不同导出格式下的性能，并使用 ultralytics.utils.benchmarks 模块
        中的 benchmark 函数。基准测试配置由默认配置值、模型参数、方法默认参数和用户提供的其他关键字参数共同决定。

        参数：
            data (str | None)：用于基准测试的数据集路径。为 None 时使用任务的默认数据集。
            format (str)：指定基准测试的导出格式名称。
            verbose (bool)：是否打印详细的基准测试信息。
            **kwargs (Any)：用于自定义基准测试过程的任意关键字参数。常用选项包括：
                - imgsz (int | list[int])：基准测试的图像尺寸。
                - quantize (int | str)：精度，例如 16（FP16）或 8（INT8）；32/None 表示 FP32。
                - device (str)：运行基准测试的设备，例如 'cpu' 或 'cuda'。

        返回：
            (polars.DataFrame)：包含每种格式基准测试结果的 Polars DataFrame，包括文件大小、指标和推理时间。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, quantize=16)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        from .exporter import export_formats

        custom = {"verbose": False}  # 方法默认参数
        kwargs = _handle_deprecation(kwargs)  # 合并前先转发旧版标志（例如 half/int8 -> quantize）
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format.lower(), [])) - {
            "batch",
            "data",
            "quantize",
        }
        export_kwargs = {k: v for k, v in args.items() if k in export_args}  # quantize 参数会在下面显式传入
        return benchmark(
            model=self,
            data=data,  # 未传入 data 参数时设置为 None，以使用默认数据集
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            quantize=args.get("quantize"),
            **export_kwargs,
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """将模型导出为适合部署的其他格式。

        此方法将模型导出为 ONNX、TorchScript 等多种格式，用于部署，并使用 Exporter 类执行导出过程，
        同时合并模型覆盖参数、方法默认参数和用户提供的其他参数。

        参数：
            **kwargs (Any)：导出配置的任意关键字参数。常用选项包括：
                - format (str)：导出格式，例如 'onnx'、'engine' 或 'coreml'。
                - quantize (int | str)：精度，例如 16（FP16）或 8（INT8）；32/None 表示 FP32。
                - device (str)：执行导出的设备。
                - workspace (int)：TensorRT 引擎的最大内存工作区大小。
                - nms (bool)：向模型添加非极大值抑制（NMS）模块。
                - simplify (bool)：简化 ONNX 模型。

        返回：
            (str)：导出模型文件的路径。

        异常：
            TypeError：模型不是 PyTorch 模型。
            ValueError：指定了不受支持的导出格式。
            RuntimeError：导出过程因错误而失败。
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter, export_formats

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # 重置以避免多 GPU 错误
            "verbose": False,
        }  # 方法默认参数
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # 右侧参数具有最高优先级
        try:
            return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)
        except Exception:
            formats = export_formats()
            export_format = args.get("format", DEFAULT_CFG_DICT["format"])
            format_name = dict(zip(formats["Argument"], formats["Format"])).get(
                str(export_format).lower(), export_format
            )
            LOGGER.info(f"Export to {format_name} in the cloud with Ultralytics Platform: {PLATFORM_URL}")
            raise

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """使用指定的数据集和训练配置训练模型。

        此方法支持通过多种设置自定义训练过程，也支持使用自定义训练器或默认训练方式；同时处理从检查点恢复训练、
        训练后更新模型和配置、检查 pip 更新，以及合并默认配置、方法默认参数和用户提供参数等场景。

        参数：
            trainer (BaseTrainer，可选)：用于模型训练的自定义训练器实例；为 None 时使用默认训练器。
            **kwargs (Any)：训练配置的任意关键字参数。常用选项包括：
                - data (str)：数据集配置文件路径。
                - epochs (int)：训练轮数。
                - batch (int)：训练批次大小。
                - imgsz (int)：输入图像尺寸。
                - device (str)：训练设备，例如 'cuda' 或 'cpu'。
                - workers (int)：数据加载工作线程数量。
                - optimizer (str)：训练使用的优化器。
                - lr0 (float)：初始学习率。
                - patience (int)：训练提前停止前等待无明显改进的 epoch 数量。
                - augmentations (list[Callable])：训练期间应用的数据增强函数列表。

        返回：
            (ultralytics.utils.metrics.DetMetrics | dict | None)：训练成功且有指标时返回训练指标，否则返回 None。
                具体指标类型取决于任务。当 `data` 是数据集列表或元组时，会依次微调基础模型，并返回
                {dataset: metrics} 字典。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
            >>> multi = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=3)  # 在多个数据集上微调
        """
        self._check_is_pytorch_model()
        checks.check_pip_update_available()

        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # 注意：处理 cfg 中包含 data 的情况。
            "data": (overrides.get("data") if kwargs.get("cfg") else None)
            or DEFAULT_CFG_DICT["data"]
            or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # 方法默认参数
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # 优先使用最右侧的参数
        if isinstance(args.get("data"), (list, tuple)):  # 在多个数据集上微调同一个基础模型
            from ultralytics.engine.trainer import MultiTrainer

            use_python_trainer = trainer is not None or self.callbacks != callbacks.get_default_callbacks()
            self.trainer = MultiTrainer(
                (trainer or self._smart_load("trainer")) if use_python_trainer else None,
                args,
                self.model,
                _callbacks=self.callbacks,
            )
            self.metrics = self.trainer.train()
            return self.metrics
        pretrained = kwargs.get("pretrained", overrides.get("pretrained", True) if kwargs.get("cfg") else True)
        if args.get("resume") is True:  # resume=True（布尔值）使用当前模型作为检查点
            if self.ckpt and self.ckpt.get("epoch", -1) >= 0 and self.ckpt.get("optimizer") is not None:
                args["resume"] = self.ckpt_path
            else:
                LOGGER.warning(
                    f"model '{self.ckpt_path}' is not a resumable training checkpoint "
                    f"(missing epoch/optimizer state). Use 'resume' only to continue incomplete training. "
                    f"Starting new training instead."
                )
                args["resume"] = False

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume") and self.ckpt:
            # 复用已加载的检查点模型，避免训练器设置期间再次解析远程权重来源。
            weights = None if pretrained is False else self.model
            if isinstance(pretrained, (str, Path)):
                weights, _ = load_checkpoint(pretrained)
            self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)
            self.model = self.trainer.model
            self.predictor = None  # 当前模块替换了缓存预测器所封装的模块

        self.trainer.train()
        # 训练后更新模型和 cfg
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Training completed but no checkpoint was saved. Expected {self.trainer.best} or {self.trainer.last}."
                )
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.predictor = None  # 检查点再次替换了模块，也覆盖 resume 和 YAML 运行
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.overrides["model"] = str(ckpt)  # 重置过程会删除该值，train() 和 tune() 会重新读取
            self.metrics = getattr(self.trainer.validator, "metrics", None)
            if self.metrics is None and self.ckpt:  # 在 DDP 下从检查点恢复（验证器运行在子进程中）
                self.metrics = self.ckpt.get("train_metrics")
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """对模型执行超参数调优，并可选择使用 Ray Tune。

        此方法支持两种超参数调优方式：Ray Tune 或自定义调优方式。启用 Ray Tune 时，调用
        ultralytics.utils.tuner 模块中的 run_ray_tune 函数；否则使用内部 Tuner 类。方法会合并默认参数、覆盖参数
        和自定义参数，以配置调优过程。

        参数：
            use_ray (bool)：是否使用 Ray Tune 进行超参数调优；为 False 时使用内部调优方式。
            iterations (int)：执行调优的迭代次数。
            *args (Any)：传递给调优器的其他位置参数。
            **kwargs (Any)：调优配置的其他关键字参数，会与模型覆盖参数和默认参数合并。

        返回：
            (ray.tune.ResultGrid | None)：use_ray=True 时返回包含超参数搜索结果的 ResultGrid；
                use_ray=False 时返回 None，并将最佳超参数保存到 YAML。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # 使用 Ray Tune 进行更高级的超参数搜索
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, *args, iterations=iterations, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # 方法默认参数
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # 右侧参数具有最高优先级
            return Tuner(args=args, _callbacks=self.callbacks)(iterations=iterations)

    def _apply(self, fn) -> Model:
        """将函数应用于模型参数、缓冲区和张量。

        此方法扩展父类的 _apply 方法，额外重置预测器并更新模型覆盖参数中的设备。通常用于将模型移动到其他设备
        或更改模型精度等操作。

        参数：
            fn (Callable)：应用于模型张量的函数，通常是 to()、cpu()、cuda()、half() 或 float() 等方法。

        返回：
            (Model)：已应用函数并更新属性的模型实例。

        异常：
            TypeError：模型不是 PyTorch 模型。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # 将模型移动到 GPU
        """
        self._check_is_pytorch_model()
        super()._apply(fn)
        self.predictor = None  # 设备可能已改变，因此重置预测器
        self.overrides["device"] = self.device  # 原来是 str(self.device)，例如 device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> dict[int, str]:
        """获取已加载模型关联的类别名称。

        如果模型中定义了类别名称，此属性会返回这些名称，并使用 ultralytics.nn.autobackend 模块中的
        check_class_names 函数检查其有效性。如果预测器尚未初始化，则会先设置预测器再获取类别名称。

        返回：
            (dict[int, str])：与模型关联的类别名称字典，键为类别索引，值为对应的类别名称。

        异常：
            AttributeError：模型或预测器没有 'names' 属性。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # 调用 predict() 前，导出格式不会定义预测器
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)  # 不要修改 self.predictor.model 的参数
            return predictor.model.names
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """获取模型参数所在的设备。

        此属性确定模型参数当前存储的设备（CPU 或 GPU），仅适用于 torch.nn.Module 实例。

        返回：
            (torch.device | None)：模型所在的设备（CPU/GPU）；如果模型不是 torch.nn.Module 实例，则返回 None。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # CUDA 可用时
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """获取应用于已加载模型输入数据的变换。

        如果模型中定义了变换，此属性会返回这些变换。变换通常包括调整大小、归一化和数据增强等预处理步骤，
        这些步骤会在输入数据送入模型前执行。

        返回：
            (object | None)：模型的变换对象（如果可用），否则返回 None。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"模型变换：{transforms}")
            ... else:
            ...     print("此模型未定义变换。")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """为指定事件添加回调函数。

        此方法用于注册自定义回调函数，在训练或推理等模型操作的特定事件发生时触发。回调可以扩展并自定义模型
        生命周期各阶段的行为。

        参数：
            event (str)：要附加回调的事件名称，必须是 Ultralytics 框架识别的有效事件名称。
            func (Callable)：要注册的回调函数，在指定事件发生时调用。

        示例：
            >>> def on_train_start(trainer):
            ...     print("训练即将开始！")
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """清除指定事件注册的所有回调函数。

        此方法移除与给定事件关联的所有自定义回调和默认回调，并将该事件的回调列表重置为空列表。

        参数：
            event (str)：要清除回调的事件名称，必须是 Ultralytics 回调系统识别的有效事件名称。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # 'on_train_start' 的所有回调现已移除

        注意：
            - 此方法会同时影响用户添加的自定义回调和 Ultralytics 框架提供的默认回调。
            - 调用此方法后，在重新添加回调之前不会执行该事件的任何回调。
            - 请谨慎使用，因为它会移除所有回调，包括某些操作正常运行所必需的回调。
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """将所有回调重置为默认函数。

        此方法恢复所有事件的默认回调函数，并移除之前添加的自定义回调。它遍历 callbacks.default_callbacks 中的
        默认事件，将当前回调替换为默认回调。模型生命周期中的 on_train_start、on_epoch_end 等事件的预定义回调
        都保存在该字典中。

        当需要在自定义修改后恢复原始回调集合时，此方法可以确保不同运行或实验之间的行为一致。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # 所有回调现在都已重置为默认函数
        """
        for event in callbacks.default_callbacks:
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        """加载 PyTorch 模型检查点时重置指定参数。

        此方法筛选输入参数字典，仅保留模型加载所需的一组关键参数，丢弃不必要或可能冲突的设置。

        参数：
            args (dict[str, Any])：包含各种模型参数和设置的字典。

        返回：
            (dict[str, Any])：仅包含输入参数中指定键的新字典。

        示例：
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # 加载 PyTorch 模型时只保留这些参数
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """根据模型任务智能加载对应模块。

        此方法根据模型当前任务和给定键动态选择并返回正确的模块（模型、训练器、验证器或预测器），并通过
        task_map 字典确定指定任务需要加载的模块。

        参数：
            key (str)：要加载的模块类型，必须是 'model'、'trainer'、'validator' 或 'predictor' 之一。

        返回：
            (object)：与指定键和当前任务对应的模块类。

        异常：
            NotImplementedError：当前任务不支持指定键。

        示例：
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # 获取函数名称。
            raise NotImplementedError(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.") from e

    @property
    def task_map(self) -> dict:
        """提供模型任务到不同模式对应类的映射。

        此属性返回一个字典，将每个受支持的任务（例如 detect、segment、semantic、classify）映射到嵌套字典。
        嵌套字典包含不同运行模式（model、trainer、validator、predictor）及其对应的类实现。

        该映射支持根据模型任务和目标运行模式动态加载合适的类，使 Ultralytics 框架能够灵活、可扩展地处理各种任务。

        返回：
            (dict[str, dict[str, Any]])：将任务名称映射到嵌套字典的字典。每个嵌套字典将 'model'、'trainer'、
                'validator' 和 'predictor' 键映射到该任务对应的类实现。

        示例：
            >>> model = Model("yolo26n.pt")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """将模型设置为评估模式。

        此方法将模型切换到评估模式，影响 Dropout 和批归一化等在训练与评估期间行为不同的层。评估模式下，
        这些层使用运行统计量而不是计算批次统计量，并禁用 Dropout 层。

        返回：
            (Model)：已设置为评估模式的模型实例。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> model.eval()
            >>> # 模型现已进入评估模式，可用于推理
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """允许通过 Model 类直接访问底层模型的属性。

        此方法允许通过 Model 实例直接访问底层模型属性。它首先检查请求的属性是否为 'model'；如果是，则从模块
        字典中返回模型，否则将属性查找委托给底层模型。

        参数：
            name (str)：要获取的属性名称。

        返回：
            (Any)：请求的属性值。

        异常：
            AttributeError：请求的属性不存在于模型中。

        示例：
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.stride)  # 访问 model.stride 属性
            >>> print(model.names)  # 访问 model.names 属性
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
