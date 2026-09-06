# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics Results, Boxes, Masks, SemanticMask, Keypoints, Probs, and OBB classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict
"""

from __future__ import annotations

from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    """基础张量类，提供便于操作和设备管理的附加方法。

    此类为类似张量的对象提供基础功能，支持 PyTorch 张量和 NumPy 数组，并提供在设备之间移动数据及转换数据类型的方法。

    属性：
        data (torch.Tensor | np.ndarray): 预测数据，例如边界框、掩码或关键点。
        orig_shape (tuple[int, int]): 图像原始形状，通常为（高度、宽度）。

    方法：
        cpu：返回存储在 CPU 内存中的张量副本。
        numpy：将张量转换为 NumPy 数组并返回副本。
        cuda：将张量移动到 GPU 内存，必要时返回新实例。
        to：返回移动到指定设备并转换为指定数据类型的张量副本。

    示例：
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用预测数据和图像原始形状初始化 BaseTensor。

        参数：
            data (torch.Tensor | np.ndarray): 预测数据，例如边界框、掩码或关键点。
            orig_shape (tuple[int, int]): 图像原始形状，格式为（高度、宽度）。
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """返回底层数据张量的形状。

        返回：
            (tuple[int, ...]): 数据张量的形状。

        示例：
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            torch.Size([100, 4])
        """
        return self.data.shape

    def cpu(self):
        """返回存储在 CPU 内存中的张量副本。

        返回：
            (BaseTensor): 将数据张量移动到 CPU 内存后的新 BaseTensor 对象。

        示例：
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """将数据转换为 NumPy 数组并返回对象副本。

        返回：
            (BaseTensor): `data` 为 NumPy 数组的新实例。

        示例：
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """将张量移动到 GPU 内存。

        返回：
            (BaseTensor): 数据已移动到 GPU 内存的新 BaseTensor 实例。

        示例：
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """返回移动到指定设备并转换为指定数据类型的张量副本。

        参数：
            *args (Any): 要传递给 torch.Tensor.to() 的可变长度参数列表。
            **kwargs (Any): 要传递给 torch.Tensor.to() 的任意关键字参数。

        返回：
            (BaseTensor): 数据已移动到指定设备和/或转换为指定数据类型的新 BaseTensor 实例。

        示例：
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """返回底层数据张量的长度。

        返回：
            (int): 数据张量第一维的元素数量。

        示例：
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """返回包含数据张量中指定索引元素的新 BaseTensor 实例。

        参数：
            idx (int | list[int] | torch.Tensor): 要从数据张量中选择的索引。

        返回：
            (BaseTensor): 包含索引数据的新 BaseTensor 实例。

        示例：
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # 选择第一行
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class _DenseResultTensor(BaseTensor):
    """表示每张图像零个或一个稠密结果的 BaseTensor，可避免按行索引破坏结果。"""

    def __len__(self) -> int:
        """如果映射包含数据则返回 1；如果之前的索引操作将其清空则返回 0。"""
        return int(self.data.shape[0] > 0)

    def __getitem__(self, idx):
        """选择该结果的任意索引都返回完整映射；空选择则返回空映射。"""
        idx = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx  # NumPy 将原始布尔张量读取为整数
        empty = np.size(np.arange(len(self))[idx]) == 0  # 根据逻辑长度检查任意类型的 idx 是否越界
        return self.__class__(self.data[:0] if empty else self.data, self.orig_shape)


class SemanticMask(_DenseResultTensor):
    """单张图像的语义分割类别图。"""


class DepthMap(_DenseResultTensor):
    """单张图像的逐像素深度图（单位：米），形状为 (H, W)。"""


class Results(SimpleClass, DataExportMixin):
    """用于存储和操作推理结果的类。

    此类提供处理各种 Ultralytics 模型推理结果的完整功能，包括目标检测、实例分割、语义分割、分类、姿态估计和有向
    边界框检测，并支持可视化、数据导出及多种坐标变换。

    属性：
        orig_img (np.ndarray): 以 NumPy 数组表示的原始图像。
        orig_shape (tuple[int, int]): 图像原始形状，格式为（高度、宽度）。
        boxes (Boxes | None): 检测到的边界框。
        masks (Masks | None): 分割掩码。
        probs (Probs | None): 分类概率。
        keypoints (Keypoints | None): 检测到的关键点。
        obb (OBB | None): 有向边界框。
        semantic_mask (SemanticMask | None): 语义分割类别图。
        depth (DepthMap | None): 逐像素深度图。
        speed (dict): 包含推理速度信息的字典。
        names (dict): 将类别索引映射到类别名称的字典。
        path (str): 输入图像文件路径。
        save_dir (str | None): 保存结果的目录。

    方法：
        update: 使用新的检测数据更新 Results 对象。
        cpu: 返回所有张量移至 CPU 内存后的 Results 对象副本。
        numpy: 将 Results 对象中的所有张量转换为 NumPy 数组。
        cuda: 将 Results 对象中的所有张量移至 GPU 内存。
        to: 将所有张量移至指定设备并转换为指定数据类型。
        new: 使用相同的图像、路径、类别名称和速度属性创建新的 Results 对象。
        plot: 在输入的 BGR 图像上绘制检测结果。
        show: 显示带有推理结果标注的图像。
        save: 将带有推理结果标注的图像保存到文件。
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Polars DataFrame.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.

    示例：
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # 获取第一个结果
        >>> boxes = result.boxes  # 获取第一个结果的边界框
        >>> masks = result.masks  # 获取第一个结果的掩码
        >>> for result in results:
        ...     result.plot()  # 绘制检测结果
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
        semantic_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> None:
        """初始化用于存储和操作推理结果的 Results 类。

        参数：
            orig_img (np.ndarray): 以 NumPy 数组表示的原始图像。
            path (str): 图像文件路径。
            names (dict): 类别名称字典。
            boxes (torch.Tensor | None): 每个检测结果的边界框坐标二维张量。
            masks (torch.Tensor | None): 检测掩码三维张量，每个掩码都是二值图像。
            probs (torch.Tensor | None): 分类任务中每个类别概率的一维张量。
            keypoints (torch.Tensor | None): 每个检测结果关键点坐标的二维张量。
            obb (torch.Tensor | None): 每个检测结果有向边界框坐标的二维张量。
            semantic_mask (torch.Tensor | None): 语义分割结果类别 ID 的二维张量。
            depth (torch.Tensor | None): 逐像素深度值的二维浮点张量，形状为 (H, W)。
            speed (dict | None): 包含预处理、推理和后处理速度的字典（毫秒/图像）。

        注意：
            默认姿态模型的人体姿态关键点索引如下：
            0：鼻子，1：左眼，2：右眼，3：左耳，4：右耳
            5：左肩，6：右肩，7：左肘，8：右肘
            9：左腕，10：右腕，11：左髋，12：右髋
            13：左膝，14：右膝，15：左踝，16：右踝
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # 原始尺寸的边界框
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # 原始尺寸或 imgsz 尺寸的掩码
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.semantic_mask = SemanticMask(semantic_mask, self.orig_shape) if semantic_mask is not None else None
        self.depth = DepthMap(depth, self.orig_shape) if depth is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb", "semantic_mask", "depth"

    def __getitem__(self, idx):
        """返回推理结果中指定索引对应的 Results 对象。

        参数：
            idx (int | slice): 要从 Results 对象中获取的索引或切片。

        返回：
            (Results): 包含指定推理结果子集的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> single_result = results[0]  # 获取第一个结果
            >>> subset_results = results[1:4]  # 获取结果切片
        """
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        """返回 Results 对象中的结果数量。

        返回：
            (int): 结果数量，由 (boxes、masks、probs、keypoints、obb、semantic_mask 或 depth) 中第一个非空属性的
                长度决定。空 Results 对象返回 0。

        示例：
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 6))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)
        return 0

    def update(
        self,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ):
        """使用新的检测数据更新 Results 对象。

        此方法可更新 Results 对象的边界框、掩码、关键点、概率和有向边界框（OBB），并确保边界框被裁剪到图像原始形状内。

        参数：
            boxes (torch.Tensor | None): 形状为 (N, 6) 的边界框坐标和置信度张量，格式为 (x1, y1, x2, y2, conf, class)。
            masks (torch.Tensor | None): 形状为 (N, H, W) 的分割掩码张量。
            probs (torch.Tensor | None): 形状为 (num_classes,) 的类别概率张量。
            obb (torch.Tensor | None): 形状为 (N, 7) 或 (N, 8) 的有向边界框坐标张量。
            keypoints (torch.Tensor | None): 形状为 (N, K, 3) 的关键点张量，其中人体关键点 K=17。
            semantic_mask (torch.Tensor | None): 形状为 (H, W) 的语义分割类别 ID 张量。
            depth (torch.Tensor | None): 形状为 (H, W) 的逐像素深度值张量。

        示例：
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = Probs(probs)
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)
        if semantic_mask is not None:
            self.semantic_mask = SemanticMask(semantic_mask, self.orig_shape)
        if depth is not None:
            self.depth = DepthMap(depth, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """对所有非空属性应用函数，并返回属性已修改的新 Results 对象。

        .to()、.cuda()、.cpu() 等方法会在内部调用此方法。

        参数：
            fn (str): 要应用的函数名称。
            *args (Any): 要传递给函数的可变长度参数列表。
            **kwargs (Any): 要传递给函数的任意关键字参数。

        返回：
            (Results): 属性已由所应用函数修改的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is None:
                continue
            setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """返回一个 Results 对象副本，并将其中的所有张量移动到 CPU 内存。

        此方法创建新的 Results 对象，并将所有张量属性（boxes、masks、probs、keypoints、obb）转移到 CPU 内存，
        适用于将数据从 GPU 移到 CPU 以便进一步处理或保存。

        返回：
            (Results): 所有张量属性均位于 CPU 内存中的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> cpu_result = results[0].cpu()  # 将第一个结果移动到 CPU
            >>> print(cpu_result.boxes.device)  # 输出：cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """将 Results 对象中的所有张量转换为 NumPy 数组。

        返回：
            (Results): 所有张量均已转换为 NumPy 数组的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        注意：
            此方法会创建新的 Results 对象，不会修改原对象，适用于与基于 NumPy 的库交互或需要执行 CPU 操作的场景。
        """
        return self._apply("numpy")

    def cuda(self):
        """将 Results 对象中的所有张量移动到 GPU 内存。

        返回：
            (Results): 所有张量均已移动到 CUDA 设备的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # 将第一个结果移动到 GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移动到 GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """将 Results 对象中的所有张量移动到指定设备并转换为指定数据类型。

        参数：
            *args (Any): 要传递给 torch.Tensor.to() 的可变长度参数列表。
            **kwargs (Any): 要传递给 torch.Tensor.to() 的任意关键字参数。

        返回：
            (Results): 所有张量均已移动到指定设备并转换为指定数据类型的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # 将第一个结果移动到 GPU
            >>> result_cpu = results[0].to("cpu")  # 将第一个结果移动到 CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # 将第一个结果转换为半精度
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """使用相同的图像、路径、名称和速度属性创建新的 Results 对象。

        返回：
            (Results): 从原实例复制属性得到的新 Results 对象。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """在输入 BGR 图像上绘制检测结果。

        参数：
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | torch.Tensor | None): Image to plot on. Tensor images must be contiguous HWC BGR uint8. If
                None, uses the original image.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

        返回：
            (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

        示例：
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot(pil=True)
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        if pred_boxes is not None and (show_boxes or (pred_masks and show_masks)):
            pred_boxes = pred_boxes.cpu()  # 一次主机传输，避免在颜色和标签循环中逐框同步 GPU
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # 分类任务默认 pil=True
            example=names,
        )

        # 绘制分割结果
        if pred_masks and show_masks:
            pred_mask_data = torch.as_tensor(pred_masks.data)  # 对 torch 为无操作，可将 numpy() 结果转换为张量
            idx = (
                pred_boxes.id
                if pred_boxes and pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_mask_data, colors=[colors(x, True) for x in idx])

        # 绘制检测结果
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c = int(d.cls.item())  # .item() 同时适用于 torch 和 numpy；numpy 2.4 要求 int()/float() 输入 0 维值
                d_conf, id = float(d.conf.item()) if conf else None, int(d.id.item()) if d.is_track else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else (f"{d_conf:.2f}" if conf else None)
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )

        # 绘制分类结果
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA 框

        # 绘制语义分割结果
        if self.semantic_mask and show_masks:
            sem_mask = self.semantic_mask.data
            if isinstance(sem_mask, torch.Tensor):
                sem_mask = sem_mask.cpu().numpy()
            annotator.semantic_mask(sem_mask, alpha=0.5)

        # 绘制深度结果——将着色后的深度热力图叠加到图像上
        if self.depth and show_masks:
            d = self.depth.data
            d = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
            annotator.depth_map(d)

        # 绘制姿态结果
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.cpu().numpy().data)):  # one host transfer, no per-kpt syncs
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # 显示结果
        if show:
            annotator.show(self.path)

        # 保存结果
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.result(pil)

    def show(self, *args, **kwargs):
        """显示带有推理结果标注的图像。

        此方法在原始图像上绘制检测结果并显示图像，可直接查看模型预测结果。

        参数：
            *args (Any): 传递给 `plot()` 方法的可变长度参数列表。
            **kwargs (Any): 传递给 `plot()` 方法的任意关键字参数。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # 显示第一个结果
            >>> for result in results:
            ...     result.show()  # 显示所有结果
        """
        self.plot(*args, show=True, **kwargs)

    def save(self, filename: str | None = None, *args, **kwargs) -> str:
        """将带有推理结果标注的图像保存到文件。

        此方法在原始图像上绘制检测结果，将标注后的图像保存到文件。它使用 `plot` 方法生成标注图像，
        然后将其保存到指定文件名。

        参数：
            filename (str | None): 保存标注图像的文件名。如果为 None，则根据原始图像路径生成默认文件名。
            *args (Any): 传递给 `plot` 方法的可变长度参数列表。
            **kwargs (Any): 传递给 `plot` 方法的任意关键字参数。

        返回：
            (str): 图像保存到的文件名。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # 或使用自定义绘图参数
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
            >>> # 如果目录不存在，将自动创建
            >>> result.save("path/to/annotated_image.jpg")
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
        self.plot(*args, save=True, filename=filename, **kwargs)
        return filename

    def verbose(self) -> str:
        """返回每个任务的日志字符串，详细描述检测和分类结果。

        此方法生成易于阅读的字符串，用于概括检测和分类结果。其中包含每个类别的检测数量，
        以及分类任务中概率最高的类别。

        返回：
            (str): 包含结果摘要的格式化字符串。检测任务中包含每个类别的检测数量，分类任务中包含概率最高的 5 个类别。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        注意：
            - 检测任务没有检测结果时，方法返回 "(no detections), "。
            - 分类任务返回概率最高的 5 个类别及其对应类别名称。
            - 返回字符串使用逗号分隔，并以逗号和空格结尾。
        """
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return "" if self.probs is not None else "(no detections), "
        if self.probs is not None:
            return f"{', '.join(f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5)}, "
        if boxes:
            counts = torch.as_tensor(boxes.cls, dtype=torch.int64).bincount()  # 对 torch 不执行操作，对 numpy() 进行转换
            return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)
        if self.depth is not None:
            d = self.depth.data
            d = d[d > 0]
            return f"depth {float(d.min()):.2f}-{float(d.max()):.2f}m, " if len(d) else "depth (no valid pixels), "
        if self.semantic_mask is not None:
            return ""

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """将检测结果保存到文本文件。

        参数：
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        返回：
            (str): Path to the saved text file.

        示例：
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        注意：
            - 文件中每个检测或分类结果占一行，结构如下：
              - 检测结果：`class x_center y_center width height [confidence] [track_id]`
              - 分类结果：`confidence class_name`
              - 掩码和关键点的具体格式会相应变化。
            - 掩码轮廓点数少于 3 的检测结果会被省略，因为较短的行不能构成多边形；如果没有剩余行，则不会写入文件。
            - 如果输出目录不存在，函数会自动创建。
            - save_conf 为 False 时，输出中不包含置信度分数。
            - 不会覆盖文件已有内容，新结果会追加到文件末尾。
            - 此方法不支持语义分割任务。
        """
        if self.semantic_mask is not None:
            LOGGER.warning("Semantic Segmentation task does not support `save_txt`.")
            return str(txt_file)
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # 分类
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # 检测/分割/姿态
            boxes = boxes.cpu()  # 一次主机传输，避免下面循环中逐框同步 GPU
            kpts = kpts.cpu() if kpts is not None else None
            segments = masks.xyn if masks else None
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls.item()), float(d.conf.item()), int(d.id.item()) if d.is_track else None
                line = (c, *(d.xyxyxyxyn.reshape(-1) if is_obb else d.xywhn.reshape(-1)))
                if segments is not None:
                    seg = segments[j]
                    if len(seg) < 3:  # 少于 3 个点不是多边形，写入的行也无法被数据加载器接受
                        continue
                    line = (c, *seg.copy().reshape(-1))  # 反转 mask.xyn，从 (n,2) 变为 (n*2)
                if kpts is not None:
                    kpt = kpts[j].xyn
                    if kpts[j].has_visible:
                        kpt = torch.cat((torch.as_tensor(kpt), torch.as_tensor(kpts[j].conf)[..., None]), 2)
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建目录
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)

    def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
        """将裁剪后的检测图像保存到指定目录。

        此方法将检测目标的裁剪图像保存到指定目录。每个裁剪图像都保存在以目标类别命名的子目录中，
        文件名根据输入的 file_name 生成。

        参数：
            save_dir (str | Path): 保存裁剪图像的目录路径。
            file_name (str | Path): 保存裁剪图像所使用的基础文件名。

        示例：
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

        注意：
            - 此方法不支持分类、有向边界框（OBB）或语义分割任务。
            - 裁剪图像保存为 `save_dir/class_name/file_name.jpg`。
            - 如果必要的子目录不存在，方法会自动创建。
            - 裁剪前会复制原始图像，以避免修改原图。
        """
        if self.probs is not None:
            LOGGER.warning("Classify task does not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task does not support `save_crop`.")
            return
        if self.semantic_mask is not None:
            LOGGER.warning("Semantic Segmentation task does not support `save_crop`.")
            return
        if self.depth is not None:
            LOGGER.warning("Depth task does not support `save_crop`.")
            return
        for d in self.boxes.cpu():  # 一次主机传输，避免下面循环中逐框同步 GPU
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls.item())] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]:
        """将推理结果转换为摘要字典，并可选择对边界框坐标进行归一化。

        此方法创建检测字典列表，每个字典包含一个检测或分类结果的信息。
        对于分类任务，返回概率最高的 5 个类别及其置信度；对于检测任务，包含类别信息和边界框坐标，
        并可选择包含掩码轮廓和关键点。

        参数：
            normalize (bool): 是否根据图像尺寸对边界框坐标进行归一化。
            decimals (int): 输出值保留的小数位数。

        返回：
            (list[dict[str, Any]]): 字典列表，每个字典包含一个检测或分类结果的摘要信息。
                字典结构取决于任务类型（分类或检测）和可用信息（边界框、掩码、关键点）。

        示例：
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     summary = result.summary()
            ...     print(summary)
        """
        # 创建检测字典列表
        results = []
        if self.probs is not None:
            # 返回前 5 个分类结果
            for class_id, conf in zip(self.probs.top5, self.probs.top5conf.tolist()):
                class_id = int(class_id)
                results.append(
                    {
                        "name": self.names[class_id],
                        "class": class_id,
                        "confidence": round(conf, decimals),
                    }
                )
            return results

        if self.semantic_mask is not None:
            # 返回语义分割中每个类别的像素覆盖率
            mask = self.semantic_mask.data
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            unique, counts = np.unique(mask, return_counts=True)
            total = mask.size
            for class_id, count in zip(unique.tolist(), counts.tolist()):
                if len(self.names) == 1:
                    if class_id != 1:  # 跳过二值背景和忽略标签
                        continue
                    class_id = 0
                elif class_id not in self.names:  # 跳过忽略标签（例如 255）
                    continue
                results.append(
                    {
                        "name": self.names[class_id],
                        "class": class_id,
                        "pixel_ratio": round(count / total, decimals),
                    }
                )
            return results

        if self.depth is not None:
            # 深度是密集的逐像素图，而不是逐实例结果，因此不计入摘要。
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        if data:
            data = data.cpu()  # 一次主机传输，避免下面循环中逐行同步 GPU
        kpts = self.keypoints
        if kpts is not None:
            kpts = kpts.cpu()  # 同样避免下面逐行关键点同步 GPU
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy；跟踪时还包括 track_id、conf、class_id
            class_id, conf = int(row.cls.item()), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # 跟踪 ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).astype(float).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).astype(float).round(decimals).tolist(),
                }
            if kpts is not None:
                kpt = kpts[i]
                k = kpt.data[0]
                k = k.cpu().numpy() if isinstance(k, torch.Tensor) else k
                result["keypoints"] = {
                    "x": (k[:, 0] / w).astype(float).round(decimals).tolist(),
                    "y": (k[:, 1] / h).astype(float).round(decimals).tolist(),
                }
                if kpt.has_visible:
                    result["keypoints"]["visible"] = k[:, 2].astype(float).round(decimals).tolist()
            results.append(result)

        return results


class Boxes(BaseTensor):
    """用于管理和操作检测边界框的类。

    此类提供完整的检测边界框处理功能，包括坐标、置信度分数、类别标签和可选的跟踪 ID。
    支持多种边界框格式，并提供在不同坐标系之间便捷操作和转换的方法。

    属性：
        data (torch.Tensor | np.ndarray): 包含检测边界框及相关数据的原始张量。
        orig_shape (tuple[int, int]): 原始图像尺寸（高度、宽度）。
        is_track (bool): 指示边界框数据中是否包含跟踪 ID。
        xyxy (torch.Tensor | np.ndarray): [x1, y1, x2, y2] 格式的边界框。
        conf (torch.Tensor | np.ndarray): 每个边界框的置信度分数。
        cls (torch.Tensor | np.ndarray): 每个边界框的类别标签。
        id (torch.Tensor | None): 每个边界框的跟踪 ID（如果可用）。
        xywh (torch.Tensor | np.ndarray): [x, y, width, height] 格式的边界框。
        xyxyn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 [x1, y1, x2, y2] 边界框。
        xywhn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 [x, y, width, height] 边界框。

    方法：
        cpu: 返回所有张量位于 CPU 内存中的对象副本。
        numpy: 返回所有张量转换为 NumPy 数组后的对象副本。
        cuda: 返回所有张量位于 GPU 内存中的对象副本。
        to: 返回张量位于指定设备并采用指定数据类型的对象副本。

    示例：
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # 高度、宽度
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测边界框数据和图像原始形状初始化 Boxes 类。

        此类管理检测边界框，便于访问和操作边界框坐标、置信度分数、类别标识及可选的跟踪 ID。
        支持绝对坐标和归一化坐标等多种边界框格式。

        参数：
            boxes (torch.Tensor | np.ndarray): 形状为 (num_boxes, 6) 或 (num_boxes, 7) 的检测边界框张量或 NumPy 数组。
                各列应包含 [x1, y1, x2, y2,（可选）track_id, confidence, class]。
            orig_shape (tuple[int, int]): 原始图像形状（高度、宽度），用于归一化。
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy、track_id、conf、cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """返回 [x1, y1, x2, y2] 格式的边界框。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (n, 4) 的张量或数组，包含 [x1, y1, x2, y2] 格式的边界框坐标，
                其中 n 为边界框数量。

        示例：
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """返回每个检测边界框的置信度分数。

        返回：
            (torch.Tensor | np.ndarray): 包含每个检测结果置信度分数的一维张量或数组，形状为 (N,)，
                其中 N 为检测结果数量。

        示例：
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """返回表示每个边界框类别预测结果的类别 ID 张量。

        返回：
            (torch.Tensor | np.ndarray): 包含每个检测边界框类别 ID 的张量或数组。
                形状为 (N,)，其中 N 为边界框数量。

        示例：
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """如果存在，则返回每个检测边界框的跟踪 ID。

        返回：
            (torch.Tensor | np.ndarray | None): 如果启用跟踪，则返回包含每个边界框跟踪 ID 的张量或数组，否则返回 None。
                形状为 (N,)，其中 N 为边界框数量。

        示例：
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        注意：
            - 只有启用跟踪（即 `is_track` 为 True）时，此属性才可用。
            - 跟踪 ID 通常用于在视频分析的多帧之间关联检测结果。
        """
        return self.data[:, -3] if self.is_track else None

    @cached_property
    def xywh(self) -> torch.Tensor | np.ndarray:
        """将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, width, height] 格式。

        返回：
            (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center
                are the coordinates of the center point of the bounding box, width, height are the dimensions of the
                bounding box and the shape of the returned tensor is (N, 4), where N is the number of boxes.

        示例：
            >>> boxes = Boxes(
            ...     torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]]), orig_shape=(480, 640)
            ... )
            >>> xywh = boxes.xywh
        """
        return ops.xyxy2xywh(self.xyxy)

    @cached_property
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """返回相对于图像原始尺寸归一化后的边界框坐标。

        此属性计算并返回 [x1, y1, x2, y2] 格式的边界框坐标，并根据原始图像尺寸归一化到 [0, 1] 范围。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框坐标，其中 N 为边界框数量。
                每行包含归一化到 [0, 1] 的 [x1, y1, x2, y2] 值。

        示例：
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @cached_property
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """返回 [x, y, width, height] 格式的归一化边界框。

        此属性计算并返回 [x_center, y_center, width, height] 格式的归一化边界框坐标，
        所有值均相对于原始图像尺寸。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框，其中 N 为边界框数量。
                每行包含根据原始图像尺寸归一化到 [0, 1] 的 [x_center, y_center, width, height] 值。

        示例：
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """用于存储和操作检测掩码的类。

    此类继承 BaseTensor，用于处理分割掩码，并提供像素坐标与归一化坐标之间的转换方法。

    属性：
        data (torch.Tensor | np.ndarray): 包含掩码数据的原始张量或数组。
        orig_shape (tuple[int, int]): 原始图像形状（高度、宽度）。
        xy (list[np.ndarray]): 像素坐标格式的轮廓列表。
        xyn (list[np.ndarray]): 归一化坐标格式的轮廓列表。

    方法：
        cpu: 返回掩码张量位于 CPU 内存中的 Masks 对象副本。
        numpy: 返回掩码张量转换为 NumPy 数组后的 Masks 对象副本。
        cuda: 返回掩码张量位于 GPU 内存中的 Masks 对象副本。
        to: 返回掩码张量位于指定设备并采用指定数据类型的 Masks 对象副本。

    示例：
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测掩码数据和图像原始形状初始化 Masks 类。

        参数：
            masks (torch.Tensor | np.ndarray): 形状为 (num_masks, height, width) 的检测掩码。
            orig_shape (tuple[int, int]): 原始图像形状（高度、宽度），用于归一化。
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @cached_property
    def xyn(self) -> list[np.ndarray]:
        """返回分割掩码的归一化 xy 坐标。

        此属性计算并缓存分割掩码的归一化 xy 坐标。坐标相对于原始图像形状进行归一化。

        返回：
            (list[np.ndarray]): NumPy 数组列表，每个数组包含一个分割掩码的归一化 xy 坐标。
                每个数组形状为 (N, 2)，其中 N 为掩码轮廓的点数。

        示例：
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # 第一个掩码的归一化坐标
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @cached_property
    def xy(self) -> list[np.ndarray]:
        """返回掩码张量中每个分割区域的 [x, y] 像素坐标。

        此属性计算并返回 Masks 对象中每个分割掩码的像素坐标列表。
        坐标会缩放到与原始图像尺寸一致。

        返回：
            (list[np.ndarray]): NumPy 数组列表，每个数组包含一个分割掩码的 [x, y] 像素坐标。
                每个数组形状为 (N, 2)，其中 N 为轮廓点数。

        示例：
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # 掩码数量
            >>> print(xy_coords[0].shape)  # 第一个掩码坐标的形状
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """用于存储和操作检测关键点的类。

    此类用于处理关键点数据，包括坐标操作、归一化和置信度值，并支持带可选可见性信息的关键点检测结果。

    属性：
        data (torch.Tensor): 包含关键点数据的原始张量。
        orig_shape (tuple[int, int]): 原始图像尺寸（高度、宽度）。
        has_visible (bool): 指示关键点是否包含可见性信息。
        xy (torch.Tensor): [x, y] 格式的关键点坐标。
        xyn (torch.Tensor): 相对于 orig_shape 归一化的 [x, y] 格式关键点坐标。
        conf (torch.Tensor | None): 每个关键点的置信度值（如果可用）。

    方法：
        cpu: 返回关键点张量位于 CPU 内存中的副本。
        numpy: 返回关键点张量转换为 NumPy 数组后的副本。
        cuda: 返回关键点张量位于 GPU 内存中的副本。
        to: 返回关键点张量位于指定设备并采用指定数据类型的副本。

    示例：
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape = (480, 640)  # 原始图像形状（高度、宽度）
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # 访问 xy 坐标
        >>> print(keypoints.conf)  # Access confidence values
        >>> keypoints_cpu = keypoints.cpu()  # 将关键点移动到 CPU
    """

    def __init__(self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测关键点和图像原始尺寸初始化 Keypoints 对象。

        此方法处理输入关键点张量，同时支持二维和三维格式。

        参数：
            keypoints (torch.Tensor | np.ndarray): 包含关键点数据的张量或数组，形状可以是：
                - (num_objects, num_keypoints, 2)，仅包含 x、y 坐标；
                - (num_objects, num_keypoints, 3)，包含 x、y 坐标和置信度分数。
            orig_shape (tuple[int, int]): 原始图像尺寸（高度、宽度）。
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @cached_property
    def xy(self) -> torch.Tensor | np.ndarray:
        """返回关键点的 x、y 坐标。

        返回：
            (torch.Tensor | np.ndarray): 包含关键点 x、y 坐标的张量或数组，形状为 (N, K, 2)，
                其中 N 为检测结果数量，K 为每个检测结果的关键点数量。

        示例：
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # 第一个检测结果的关键点 x、y 坐标

        注意：
            - 返回坐标以像素为单位，并相对于原始图像尺寸表示。
            - 此属性使用 LRU 缓存来提升重复访问时的性能。
        """
        return self.data[..., :2]

    @cached_property
    def xyn(self) -> torch.Tensor | np.ndarray:
        """返回相对于图像原始尺寸归一化后的关键点坐标（x、y）。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, K, 2) 的张量或数组，包含归一化关键点坐标；
                N 为实例数量，K 为关键点数量，最后一维包含范围为 [0, 1] 的 [x, y] 值。

        示例：
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @cached_property
    def conf(self) -> torch.Tensor | np.ndarray | None:
        """返回每个关键点的置信度值。

        返回：
            (torch.Tensor | np.ndarray | None): 如果可用，则返回包含每个关键点置信度分数的张量或数组，否则返回 None。
                批量数据的形状为 (num_detections, num_keypoints)，单个检测结果的形状为 (num_keypoints,)。

        示例：
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """用于存储和操作分类概率的类。

    此类继承 BaseTensor，提供访问和操作分类概率的方法，包括最高概率和前五名预测结果。

    属性：
        data (torch.Tensor | np.ndarray): 包含分类概率的原始张量或数组。
        orig_shape (tuple[int, int] | None): 原始图像形状（高度、宽度），此类中不使用。
        top1 (int): 概率最高类别的索引。
        top5 (list[int]): 按概率排序的前五名类别索引。
        top1conf (torch.Tensor | np.ndarray): 概率最高类别的置信度分数。
        top5conf (torch.Tensor | np.ndarray): 前五名类别的置信度分数。

    方法：
        cpu: 返回概率张量位于 CPU 内存中的副本。
        numpy: 返回概率张量转换为 NumPy 数组后的副本。
        cuda: 返回概率张量位于 GPU 内存中的副本。
        to: 返回概率张量位于指定设备并采用指定数据类型的副本。

    示例：
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs: torch.Tensor | np.ndarray, orig_shape: tuple[int, int] | None = None) -> None:
        """使用分类概率初始化 Probs 类。

        此类存储并管理分类概率，便于访问最高概率预测结果及其置信度。

        参数：
            probs (torch.Tensor | np.ndarray): 一维分类概率张量或数组。
            orig_shape (tuple[int, int] | None): 原始图像形状（高度、宽度）。此类中不使用，但为与其他结果类保持一致而保留。
        """
        super().__init__(probs, orig_shape)

    @cached_property
    def top1(self) -> int:
        """返回概率最高的类别索引。

        返回：
            (int): 概率最高类别的索引。

        示例：
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        return int(self.data.argmax())

    @cached_property
    def top5(self) -> list[int]:
        """返回概率最高的前 5 个类别索引。

        返回：
            (list[int]): 包含前五名类别概率索引的列表，按降序排列。

        示例：
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()  # 这种方式同时适用于 torch 和 numpy。

    @cached_property
    def top1conf(self) -> torch.Tensor | np.ndarray:
        """返回概率最高类别的置信度分数。

        此属性从分类结果中获取预测概率最高类别的置信度分数（概率）。

        返回：
            (torch.Tensor | np.ndarray): 包含最高概率类别置信度分数的张量。

        示例：
            >>> results = model("image.jpg")  # 对图像进行分类
            >>> probs = results[0].probs  # 获取分类概率
            >>> top1_confidence = probs.top1conf  # 获取排名第一类别的置信度
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
        """
        return self.data[self.top1]

    @cached_property
    def top5conf(self) -> torch.Tensor | np.ndarray:
        """返回前 5 个分类预测结果的置信度分数。

        此属性获取模型预测的前五名类别概率对应的置信度分数，便于快速访问最可能的类别及其置信度。

        返回：
            (torch.Tensor | np.ndarray): 包含前五名预测类别置信度分数的张量或数组，按概率降序排列。

        示例：
            >>> results = model("image.jpg")
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # 打印前 5 个类别的置信度分数
        """
        return self.data[self.top5]


class OBB(BaseTensor):
    """用于存储和操作有向边界框（OBB）的类。

    此类用于处理有向边界框，包括不同格式之间的转换、归一化以及访问边界框的各种属性。
    同时支持跟踪和非跟踪场景。

    属性：
        data (torch.Tensor): 包含边界框坐标及相关数据的原始 OBB 张量。
        orig_shape (tuple[int, int]): 原始图像尺寸（高度、宽度）。
        is_track (bool): 指示边界框数据中是否包含跟踪 ID。
        xywhr (torch.Tensor | np.ndarray): [x_center, y_center, width, height, rotation] 格式的边界框。
        conf (torch.Tensor | np.ndarray): 每个边界框的置信度分数。
        cls (torch.Tensor | np.ndarray): 每个边界框的类别标签。
        id (torch.Tensor | np.ndarray): 每个边界框的跟踪 ID（如果可用）。
        xyxyxyxy (torch.Tensor | np.ndarray): 8 点 [x1, y1, x2, y2, x3, y3, x4, y4] 格式的边界框。
        xyxyxyxyn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 8 点坐标。
        xyxy (torch.Tensor | np.ndarray): [x1, y1, x2, y2] 格式的轴对齐边界框。

    方法：
        cpu: 返回所有张量位于 CPU 内存中的 OBB 对象副本。
        numpy: 返回所有张量转换为 NumPy 数组后的 OBB 对象副本。
        cuda: 返回所有张量位于 GPU 内存中的 OBB 对象副本。
        to: 返回张量位于指定设备并采用指定数据类型的 OBB 对象副本。

    示例：
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr、conf、cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用有向边界框数据和图像原始形状初始化 OBB（Oriented Bounding Box）实例。

        此类存储并操作用于目标检测任务的有向边界框（OBB），并提供访问和转换 OBB 数据的各种属性与方法。

        参数：
            boxes (torch.Tensor | np.ndarray): 包含检测边界框的张量或 NumPy 数组，形状为 (num_boxes, 7) 或 (num_boxes, 8)。
                最后两列包含置信度和类别值；如果存在，倒数第三列包含跟踪 ID，第五列包含旋转角度。
            orig_shape (tuple[int, int]): 原始图像尺寸，格式为（高度、宽度）。

        异常：
            AssertionError: 每个边界框的值数量不是 7 或 8 时抛出。
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh、rotation、track_id、conf、cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> torch.Tensor | np.ndarray:
        """返回 [x_center, y_center, width, height, rotation] 格式的边界框。

        返回：
            (torch.Tensor | np.ndarray): 包含 [x_center, y_center, width, height, rotation] 格式有向边界框的张量或数组。
                形状为 (N, 5)，其中 N 为边界框数量。

        示例：
            >>> results = model("image.jpg")
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])

        注意：
            预测结果不会按照训练标签使用的长边约定进行规范化，因此宽度可能小于高度，旋转角度也可能从短边开始测量。
            如果只关心几何形状，请使用 `xyxyxyxy`；如需规范形式，请使用 `ops.xyxyxyxy2xywhr(obb.xyxyxyxy)`。
        """
        return self.data[:, :5]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """返回有向边界框（OBB）的置信度分数。

        此属性获取每个 OBB 检测结果对应的置信度值，置信度分数表示模型对该检测结果的确信程度。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N,) 的张量或数组，包含 N 个检测结果的置信度分数，
                每个分数的范围为 [0, 1]。

        示例：
            >>> results = model("image.jpg")
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """返回有向边界框的类别值。

        返回：
            (torch.Tensor | np.ndarray): 包含每个有向边界框类别值的张量或数组。
                形状为 (N,)，其中 N 为边界框数量。

        示例：
            >>> results = model("image.jpg")
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """返回有向边界框的跟踪 ID（如果存在）。

        返回：
            (torch.Tensor | np.ndarray | None): 包含每个有向边界框跟踪 ID 的张量或数组。
                如果跟踪 ID 不可用，则返回 None。

        示例：
            >>> results = model.track("path/to/video.mp4")  # 执行带跟踪的推理
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"Tracking IDs: {track_ids}")
        """
        return self.data[:, -3] if self.is_track else None

    @cached_property
    def xyxyxyxy(self) -> torch.Tensor | np.ndarray:
        """将 OBB 格式转换为旋转边界框使用的 8 点（xyxyxyxy）坐标格式。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, 4, 2) 的 xyxyxyxy 格式旋转边界框，其中 N 为边界框数量。
                4 个点 (x, y) 按渲染后的逆时针顺序排列，从边界框坐标系中 (+w/2, +h/2) 的角点开始，
                因此图像中的起始角点取决于旋转角度。

        示例：
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @cached_property
    def xyxyxyxyn(self) -> torch.Tensor | np.ndarray:
        """将旋转边界框转换为归一化的 xyxyxyxy 格式。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, 4, 2) 的归一化 xyxyxyxy 格式旋转边界框，其中 N 为边界框数量。
                每个边界框由 4 个点 (x, y) 表示，并相对于原始图像尺寸进行归一化。

        示例：
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 个随机 OBB
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @cached_property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """将有向边界框（OBB）转换为 xyxy 格式的轴对齐边界框。

        此属性计算每个有向边界框的最小外接矩形，并以 xyxy 格式（x1, y1, x2, y2）返回。
        这对于需要轴对齐边界框的操作很有用，例如与非旋转边界框计算 IoU。

        返回：
            (torch.Tensor | np.ndarray): 形状为 (N, 4) 的 xyxy 格式轴对齐边界框，其中 N 为边界框数量。
                每行包含 [x1, y1, x2, y2] 坐标。

        示例：
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo26n-obb.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        注意：
            - 此方法使用最小外接矩形近似 OBB。
            - 返回格式与标准目标检测指标和可视化工具兼容。
            - 此属性使用缓存来提升重复访问时的性能。
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
