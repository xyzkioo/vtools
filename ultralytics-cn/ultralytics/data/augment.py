# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr, deprecation_warn
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:
    """Ultralytics 图像变换的基类。

    此类为图像、对象实例和语义分割掩码提供统一的变换接口。简单变换应重写 `apply_image`、`apply_instances`
    和/或 `apply_semantic`；需要在图像与标注修改之间共享状态的复杂变换，则应直接重写 `__call__`。

    方法：
        get_params: 计算图像、对象实例和语义掩码之间共享的变换参数。
        apply_image: 对 labels['img'] 中的图像应用变换。
        apply_instances: 对 labels['instances'] 中的对象实例应用变换。
        apply_semantic: 对 labels['semantic_mask'] 中的语义掩码应用变换。
        __call__: 组织完整的变换流程。
    """

    def __call__(self, labels):
        """对标签字典应用变换。

        参数：
            labels (dict): 包含 'img'，以及可选 'instances' 和 'semantic_mask' 的字典。

        返回：
            (dict): 变换后的标签字典。
        """
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        return labels

    def get_params(self, labels):
        """计算并返回变换参数。

        此方法允许在图像、对象实例和语义掩码变换之间共享随机状态或已计算的矩阵（例如仿射矩阵和翻转决策）。

        参数：
            labels (dict): 输入标签字典。

        返回：
            (dict): 传递给 apply_image、apply_instances 和 apply_semantic 的参数。
        """
        return {}

    def apply_image(self, labels, params=None):
        """对图像应用变换。

        参数：
            labels (dict): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 更新后的标签字典。
        """
        return labels

    def apply_instances(self, labels, params=None):
        """对对象实例应用变换。

        参数：
            labels (dict): 包含 'instances' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 更新后的标签字典。
        """
        return labels

    def apply_semantic(self, labels, params=None):
        """对语义分割掩码应用变换。

        参数：
            labels (dict): 包含 'semantic_mask' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 更新后的标签字典。
        """
        return labels

    def apply_depth(self, labels, params=None):
        """对深度图应用变换。

        参数：
            labels (dict): 包含 'depth' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 更新后的标签字典。
        """
        return labels


class Compose:
    """用于组合多个图像变换的类。

    属性：
        transforms (list[Callable]): 按顺序应用的变换函数列表。

    方法：
        __call__: 对输入数据依次应用一系列变换。
        append: 向现有变换列表追加新变换。
        insert: 在指定索引处插入新变换。
        __getitem__: 使用索引获取单个变换或一组变换。
        __setitem__: 使用索引设置单个变换或一组变换。
        tolist: 将变换列表转换为标准 Python 列表。

    示例：
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """使用变换列表初始化 Compose 对象。

        参数：
            transforms (list[Callable]): 按顺序调用的变换对象列表。
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """对输入数据依次应用一系列变换。

        此方法会按照顺序将 Compose 对象中的每个变换应用到输入数据。

        参数：
            data (Any): 待变换的输入数据。其类型取决于变换列表中的具体变换，可以是任意类型。

        返回：
            (Any): 按顺序应用所有变换后的数据。

        示例：
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """向现有变换列表追加新变换。

        参数：
            transform (BaseTransform): 要添加到组合中的变换。

        示例：
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """在现有变换列表的指定索引处插入新变换。

        参数：
            index (int): 插入新变换的位置索引。
            transform (BaseTransform): 要插入的变换对象。

        示例：
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        """使用索引获取单个变换或一组变换。

        参数：
            index (int | list[int]): 要获取的变换索引，或索引列表。

        返回：
            (Compose | Any): index 为列表时返回新的 Compose 对象，为整数时返回单个变换。

        异常：
            AssertionError: 当 index 不是 int 或 list 类型时抛出。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # 直接返回 RandomPerspective 变换
            >>> multiple_transforms = compose[[0, 1]]  # 返回包含 RandomFlip 和 RandomPerspective 的 Compose 对象
        """
        assert isinstance(index, (int, list)), f"索引必须是 list 或 int 类型，但当前为 {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: list | int, value: list | int) -> None:
        """使用索引设置组合中的一个或多个变换。

        参数：
            index (int | list[int]): 要设置变换的索引或索引列表。
            value (Any | list[Any]): 要设置到指定索引处的变换或变换列表。

        异常：
            AssertionError: 当索引类型无效、value 类型与索引类型不匹配或索引越界时抛出。

        示例：
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # 替换第二个变换
            >>> compose[[0, 1]] = [NewTransform1(), NewTransform2()]  # 替换前两个变换
        """
        assert isinstance(index, (int, list)), f"索引必须是 list 或 int 类型，但当前为 {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"索引和 value 必须是相同类型，但当前分别为 {type(index)} 和 {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"列表索引 {i} 超出范围，变换列表长度为 {len(self.transforms)}。"
            self.transforms[i] = v

    def tolist(self):
        """将变换列表转换为标准 Python 列表。

        返回：
            (list): 包含 Compose 实例中所有变换对象的列表。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """返回 Compose 对象的字符串表示。

        返回：
            (str): 包含变换列表的 Compose 对象字符串表示。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> "RandomFlip" in repr(compose) and "RandomPerspective" in repr(compose)
            True
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform(BaseTransform):
    """CutMix、MixUp 和 Mosaic 等混合变换的基类。

    此类为数据集混合变换提供基础实现，负责根据概率应用变换，并管理多张图像及其标签的混合。

    属性：
        dataset (Any): 包含图像和标签的数据集对象。
        pre_transform (Callable | None): 混合前可选的预处理变换。
        p (float): 应用混合变换的概率。

    方法：
        __call__: 对输入标签应用混合变换。
        get_params: 准备混合标签并更新文本标签。
        get_indexes: 获取待混合图像索引的方法。
        _update_label_text: 更新混合图像的文本标签。

    示例：
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def apply_image(self, labels, params=None):
        ...         # 在此处实现自定义图像混合
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """初始化 CutMix、MixUp 和 Mosaic 等混合变换的 BaseMixTransform 对象。

        此类用于在图像处理流程中实现混合变换。

        参数：
            dataset (Any): 包含待混合图像和标签的数据集对象。
            pre_transform (Callable | None): 混合前可选的预处理变换。
            p (float): 应用混合变换的概率，取值范围应为 [0.0, 1.0]。
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p
        self.preserve_obb = getattr(dataset, "use_obb", False)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对标签数据应用预处理以及 CutMix、MixUp 或 Mosaic 变换。

        此方法根据概率决定是否应用混合变换。应用时会选择其他图像，按需执行预处理，然后完成混合变换。

        参数：
            labels (dict[str, Any]): 包含图像标签数据的字典。

        返回：
            (dict[str, Any]): 变换后的标签字典，其中可能包含其他图像的混合数据。

        示例：
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """准备混合标签并更新文本标签。

        参数：
            labels (dict[str, Any]): 包含图像标签数据的字典。

        返回：
            (dict[str, Any]): 传递给 apply_image、apply_instances 和 apply_semantic 的参数。
        """
        # 获取一张或三张其他图像的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取 Mosaic、CutMix 或 MixUp 所需的图像信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # 更新类别和文本标签
        self._update_label_text(labels)
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        """获取 Mosaic 增强所需的随机索引。

        返回：
            (int): 数据集中的随机索引。

        示例：
            >>> transform = BaseMixTransform(dataset)
            >>> index = transform.get_indexes()
            >>> print(index)  # 7
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        """更新图像增强中混合标签的文本和类别 ID。

        此方法处理输入标签字典及混合标签中的 'texts' 和 'cls' 字段，创建统一的文本标签集合，并相应更新类别 ID。

        参数：
            labels (dict[str, Any]): 包含标签信息的字典，必须包含 'texts' 和 'cls' 字段，也可以包含由其他标签字典组成的
                'mix_labels' 字段。

        返回：
            (dict[str, Any]): 文本标签已统一且类别 ID 已更新的标签字典。

        示例：
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = BaseMixTransform._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = [*labels["texts"], *(item for x in labels["mix_labels"] for item in x["texts"])]
        mix_texts = [list(x) for x in dict.fromkeys(tuple(x) for x in mix_texts)]
        text2id = {tuple(text): i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """对图像数据集应用 Mosaic 增强。

    此类通过将多张（4 张或 9 张）图像组合成一张 Mosaic 图像来执行数据增强，并按照指定概率应用该增强。

    属性：
        dataset: 应用 Mosaic 增强的数据集。
        imgsz (int): 单张图像经过 Mosaic 流程后的图像尺寸（高度和宽度）。
        p (float): 应用 Mosaic 增强的概率，取值范围为 0-1。
        n (int): 网格大小，可为 4（2x2）或 9（3x3）。
        border (tuple[int, int]): 高度和宽度方向的边界大小。

    方法：
        get_indexes: 返回数据集中的随机索引列表。
        get_params: 计算 Mosaic 布局参数。
        apply_image: 创建画布并将图像粘贴到 Mosaic 图像中。
        apply_instances: 拼接并裁剪 Mosaic 中的对象实例。
        _update_labels: 使用填充量更新标签。
        _cat_labels: 拼接标签，并裁剪超出 Mosaic 边界的实例。

    示例：
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """初始化 Mosaic 增强对象。

        此类通过将多张（4 张或 9 张）图像组合成一张 Mosaic 图像来执行数据增强，并按照指定概率应用该增强。

        参数：
            dataset (Any): 应用 Mosaic 增强的数据集。
            imgsz (int): 单张图像经过 Mosaic 流程后的图像尺寸（高度和宽度）。
            p (float): 应用 Mosaic 增强的概率，取值范围为 0-1。
            n (int): 网格大小，可为 4（2x2）或 9（3x3）。
        """
        assert 0 <= p <= 1.0, f"概率必须在 [0, 1] 范围内，但当前为 {p}。"
        assert n in {4, 9}, "网格大小必须为 4 或 9。"
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """返回 Mosaic 增强所需的数据集随机索引列表。

        此方法根据 `buffer_enabled` 属性，从缓冲区或整个数据集中选择随机图像索引，用于创建 Mosaic 增强图像。

        返回：
            (list[int]): 随机图像索引列表。列表长度为 n-1；当 n 为 4 或 9 时，分别对应额外使用 3 张或 8 张图像。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # 输出：3
        """
        if self.buffer_enabled:  # 从缓冲区选择图像
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # 从整个数据集中选择图像
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 Mosaic 布局参数。

        参数：
            labels (dict[str, Any]): 输入标签字典。

        返回：
            (dict[str, Any]): 包含 `layout` 的参数字典，其中记录每个图像块的几何信息。
        """
        params = super().get_params(labels)
        assert labels.get("rect_shape") is None, "rect 和 mosaic 不能同时使用。"
        assert len(labels.get("mix_labels", [])), "没有可用于 Mosaic 增强的其他图像。"

        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # 左上
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # 右上
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # 左下
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # 右下
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = x1a - x1b
                padh = y1a - y1b
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1a": x1a,
                        "y1a": y1a,
                        "x2a": x2a,
                        "y2a": y2a,
                        "x1b": x1b,
                        "y1b": y1b,
                        "x2b": x2b,
                        "y2b": y2b,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
        elif self.n == 9:
            hp, wp = -1, -1
            h0, w0 = None, None
            for i in range(9):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # 中心
                    c = s, s, s + w, s + h
                    h0, w0 = h, w
                elif i == 1:  # 上方
                    c = s, s - h, s + w, s
                elif i == 2:  # 右上
                    c = s + wp, s - h, s + wp + w, s
                elif i == 3:  # 右侧
                    c = s + w0, s, s + w0 + w, s + h
                elif i == 4:  # 右下
                    c = s + w0, s + hp, s + w0 + w, s + hp + h
                elif i == 5:  # 下方
                    c = s + w0 - w, s + h0, s + w0, s + h0 + h
                elif i == 6:  # 左下
                    c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
                elif i == 7:  # 左侧
                    c = s - w, s + h0 - h, s, s + h0
                elif i == 8:  # 左上
                    c = s - w, s + h0 - hp - h, s, s + h0 - hp
                padw, padh = c[:2]
                x1, y1, x2, y2 = (max(x, 0) for x in c)
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
                hp, wp = h, w
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 Mosaic 增强。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回且包含 `layout` 的参数。

        返回：
            (dict): 包含 Mosaic 图像的更新后标签字典。
        """
        layout = params["layout"]
        if self.n == 4:
            img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img4
        elif self.n == 9:
            img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对对象实例应用 Mosaic 增强。

        参数：
            labels (dict[str, Any]): 包含 'instances' 和 'cls' 的字典。
            params (dict | None): 由 get_params 返回且包含 `layout` 的参数。

        返回：
            (dict): 包含拼接后对象实例的更新后标签字典。
        """
        layout = params["layout"]
        mosaic_labels = []
        for item in layout:
            if self.n == 4:
                padw = item["padw"]
                padh = item["padh"]
            else:  # n == 9 时
                padw = item["padw"] + self.border[0]
                padh = item["padh"] + self.border[1]
            labels_patch = self._update_labels(item["labels_patch"], padw, padh, item.get("img_shape"))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        labels.update(final_labels)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对语义掩码应用 Mosaic 增强。

        参数：
            labels (dict[str, Any]): 包含 'semantic_mask' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含拼接后语义掩码的更新后标签字典。
        """
        if labels.get("semantic_mask") is None and all(
            m.get("semantic_mask") is None for m in labels.get("mix_labels", [])
        ):
            return labels

        layout = params["layout"]
        if self.n == 4:
            mask4 = np.full((self.imgsz * 2, self.imgsz * 2), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask4
        elif self.n == 9:
            mask9 = np.full((self.imgsz * 3, self.imgsz * 3), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                mask9[y1:y2, x1:x2] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int, img_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """使用填充值更新标签坐标。

        此方法通过添加填充值调整标签中对象实例的边界框坐标；如果坐标之前已归一化，还会将其反归一化。

        参数：
            labels (dict[str, Any]): 包含图像和实例信息的字典。
            padw (int): 添加到 x 坐标上的宽度填充值。
            padh (int): 添加到 y 坐标上的高度填充值。
            img_shape (tuple[int, int] | None): 原始图像块的可选尺寸 (h, w)。这是因为 apply_image 可能会在
                apply_instances 执行前用 Mosaic 画布覆盖 labels["img"]。

        返回：
            (dict): 对象实例坐标已调整的标签字典。

        示例：
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = img_shape if img_shape is not None else labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """拼接并处理 Mosaic 增强的标签。

        此方法合并 Mosaic 增强所用多张图像的标签，将实例裁剪到 Mosaic 边界内，并移除面积为零的边界框。

        参数：
            mosaic_labels (list[dict[str, Any]]): Mosaic 中每张图像对应的标签字典列表。

        返回：
            (dict[str, Any]): 包含 Mosaic 图像拼接及处理后标签的字典，包括：
                - im_file (str)：Mosaic 中第一张图像的文件路径。
                - ori_shape (tuple[int, int])：第一张图像的原始尺寸。
                - resized_shape (tuple[int, int])：Mosaic 图像的尺寸 (imgsz * 2, imgsz * 2)。
                - cls (np.ndarray)：拼接后的类别标签。
                - instances (Instances)：拼接后的实例标注。
                - texts (list[str], optional)：原始标签中存在时保留的文本标签。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # 生成最终标签
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
        }
        final_labels["instances"].clip(imgsz, imgsz, preserve_obb=self.preserve_obb)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """对图像数据集应用 MixUp 增强。

    此类实现论文 [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) 中描述的 MixUp
    增强技术。MixUp 使用随机权重混合两张图像及其标签。

    属性：
        dataset (Any): 应用 MixUp 增强的数据集。
        pre_transform (Callable | None): MixUp 前可选的预处理变换。
        p (float): 应用 MixUp 增强的概率。

    方法：
        get_params: 计算包括混合比例在内的 MixUp 参数。
        apply_image: 使用 MixUp 混合图像。
        apply_instances: 拼接 MixUp 的对象实例。

    示例：
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        """初始化 MixUp 增强对象。

        MixUp 是一种图像增强技术，通过对两张图像的像素值和标签进行加权求和来合并它们。本实现用于 Ultralytics YOLO 框架。

        参数：
            dataset (Any): 应用 MixUp 增强的数据集。
            pre_transform (Callable | None): MixUp 前应用于图像的可选预处理变换。
            p (float): 对图像应用 MixUp 增强的概率，取值范围必须为 [0, 1]。
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 MixUp 参数。

        参数：
            labels (dict[str, Any]): 输入标签字典。

        返回：
            (dict[str, Any]): 包含混合比例 `r` 的参数字典。
        """
        params = super().get_params(labels)
        params["r"] = np.random.beta(32.0, 32.0)
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """使用 MixUp 混合图像。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回且包含 `r` 的参数。

        返回：
            (dict): 包含混合图像的更新后标签字典。
        """
        r = params["r"]
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """拼接 MixUp 的对象实例。

        参数：
            labels (dict[str, Any]): 包含 'instances' 和 'cls' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含拼接后对象实例的更新后标签字典。
        """
        labels2 = labels["mix_labels"][0]
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对语义分割掩码应用 MixUp 增强。

        参数：
            labels (dict[str, Any]): 主图像标签，包含 'semantic_mask' 和 'mix_labels'。
            params (dict[str, Any] | None): 包含 `r`（混合比例）的参数字典，默认为 None。

        返回：
            (dict[str, Any]): 更新后的标签字典；当 r < 0.5 时，语义掩码会替换为混合图像的掩码。
        """
        if labels.get("semantic_mask") is None:
            return labels
        labels2 = labels["mix_labels"][0]
        if labels2.get("semantic_mask") is None:
            return labels
        r = params["r"]
        # 使用权重较大的图像掩码，避免类别索引出现小数
        if r < 0.5:
            labels["semantic_mask"] = labels2["semantic_mask"].copy()
        return labels


class CutMix(BaseMixTransform):
    """按照论文 https://arxiv.org/abs/1905.04899 的描述，对图像数据集应用 CutMix 增强。

    CutMix 从一张图像中随机选取矩形区域，并用另一张图像的对应区域替换它，同时按照混合区域面积的比例调整标签。

    属性：
        dataset (Any): 应用 CutMix 增强的数据集。
        pre_transform (Callable | None): CutMix 前可选的预处理变换。
        p (float): 应用 CutMix 增强的概率。
        beta (float): 用于采样混合比例的 Beta 分布参数。
        num_areas (int): 尝试裁剪和混合的区域数量。

    方法：
        get_params: 计算包括裁剪区域和筛选后索引在内的 CutMix 参数。
        apply_image: 将辅助图像的图像块复制到主图像中。
        apply_instances: 裁剪并拼接 CutMix 的对象实例。
        _rand_bbox: 生成裁剪区域的随机边界框坐标。

    示例：
        >>> from ultralytics.data.augment import CutMix
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0, beta: float = 1.0, num_areas: int = 3) -> None:
        """初始化 CutMix 增强对象。

        参数：
            dataset (Any): 应用 CutMix 增强的数据集。
            pre_transform (Callable | None): CutMix 前可选的预处理变换。
            p (float): 应用 CutMix 增强的概率。
            beta (float): 用于采样混合比例的 Beta 分布参数。
            num_areas (int): 尝试裁剪和混合的区域数量。
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """生成裁剪区域的随机边界框坐标。

        参数：
            width (int): 图像宽度。
            height (int): 图像高度。

        返回：
            (tuple[int]): 边界框坐标 (x1, y1, x2, y2)。
        """
        # 从 Beta 分布采样混合比例
        lam = np.random.beta(self.beta, self.beta)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # 随机中心点
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # 边界框坐标
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 CutMix 参数。

        参数：
            labels (dict[str, Any]): 输入标签字典。

        返回：
            (dict[str, Any]): 包含 'skip'、'area' 和 'indexes2' 的参数字典。
        """
        params = super().get_params(labels)
        h, w = labels["img"].shape[:2]

        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)  # (self.num_areas, num_boxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            params["skip"] = True
            return params

        labels2 = labels["mix_labels"][0]
        area = cut_areas[np.random.choice(idx)]  # 随机选择一个区域
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1))[0]
        if len(indexes2) == 0:
            params["skip"] = True
            return params

        params["area"] = area
        params["indexes2"] = indexes2
        params["w"] = w
        params["h"] = h
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 CutMix。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含混合图像的更新后标签字典。
        """
        if params.get("skip"):
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对对象实例应用 CutMix。

        参数：
            labels (dict[str, Any]): 包含 'instances' 和 'cls' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含混合对象实例的更新后标签字典。
        """
        if params.get("skip"):
            return labels
        labels2 = labels["mix_labels"][0]
        w, h = params["w"], params["h"]
        area = params["area"]
        indexes2 = params["indexes2"]

        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)

        x1, y1, x2, y2 = area.astype(np.int32)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1, preserve_obb=self.preserve_obb)
        if self.preserve_obb:
            indexes2 = indexes2[instances2.remove_zero_area_boxes()]
        instances2.add_padding(x1, y1)

        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对语义分割掩码应用 CutMix 增强。

        参数：
            labels (dict[str, Any]): 主图像标签，包含 'semantic_mask' 和 'mix_labels'。
            params (dict[str, Any] | None): 参数字典，包含 'area'（边界框坐标）和 'skip'（布尔标志），默认为 None。

        返回：
            (dict[str, Any]): 更新后的标签字典，其中语义掩码区域已替换为混合图像的掩码。
        """
        if params.get("skip"):
            return labels
        if labels.get("semantic_mask") is None:
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        if labels2.get("semantic_mask") is not None:
            mask = labels["semantic_mask"].copy()
            mask[y1:y2, x1:x2] = labels2["semantic_mask"][y1:y2, x1:x2]
            labels["semantic_mask"] = mask
        return labels


class RandomPerspective(BaseTransform):
    """对图像及其对应标注执行随机透视和仿射变换。

    此类会对图像及其边界框、分割段和关键点应用随机旋转、平移、缩放、剪切和透视变换，可用于对象检测和实例分割任务的数据增强流程。

    属性：
        degrees (float): 随机旋转的最大绝对角度范围。
        translate (float): 最大平移量相对于图像尺寸的比例。
        scale (float): 缩放因子范围，例如 scale=0.1 表示 0.9-1.1。
        shear (float): 最大剪切角度，单位为度。
        perspective (float): 透视畸变因子。
        size (tuple[int, int] | None): 输出尺寸 (width, height)；为 None 时使用输入图像尺寸。

    方法：
        get_params: 计算仿射变换矩阵及相关参数。
        apply_image: 使用仿射矩阵扭曲图像。
        apply_instances: 变换边界框、分割段和关键点。
        apply_semantic: 对语义分割掩码执行变换。
        apply_bboxes: 使用仿射矩阵变换边界框。
        apply_segments: 变换分割段并生成新的边界框。
        apply_keypoints: 使用仿射矩阵变换关键点。
        box_candidates: 根据尺寸和宽高比筛选变换后的边界框。

    示例：
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float | tuple[float, float] = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        size: tuple[int, int] | None = None,
        preserve_obb: bool = False,
    ):
        """使用变换参数初始化 RandomPerspective 对象。

        此类对图像及其对应的边界框、分割段和关键点执行随机透视与仿射变换，包括旋转、平移、缩放和剪切。

        参数：
            degrees (float): 随机旋转的角度范围。
            translate (float): 随机平移占总宽度和总高度的比例。
            scale (float | tuple[float, float]): 缩放因子区间。为浮点数时，例如 0.5 表示在 50%-150% 之间缩放；为元组时，表示绝对的 (min, max) 缩放因子。
            shear (float): 剪切强度（角度，单位为度）。
            perspective (float): 透视畸变因子。
            size (tuple[int, int] | None): 输出尺寸 (width, height)；为 None 时使用输入图像尺寸。
            preserve_obb (bool): 当变换后的分割段越过图像边界时，是否保留有向框方向。
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.size = size
        self.preserve_obb = preserve_obb

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        """计算仿射变换矩阵，但不应用该矩阵。

        参数：
            img (np.ndarray): 用于确定中心和尺寸的输入图像。
            size (tuple[int, int]): 输出图像尺寸 (width, height)，用于限制平移变换。

        返回：
            (M, scale)：3x3 变换矩阵和缩放因子。
        """
        # 中心平移
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x 方向平移（像素）
        C[1, 2] = -img.shape[0] / 2  # y 方向平移（像素）

        # 透视变换
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x 方向透视（绕 y 轴）
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y 方向透视（绕 x 轴）

        # 旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        if isinstance(self.scale, (tuple, list)):
            s = random.uniform(self.scale[0], self.scale[1])
        else:
            s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 剪切
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x 方向剪切（度）
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y 方向剪切（度）

        # 平移
        T = np.eye(3, dtype=np.float32)

        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]  # x 方向平移（像素）
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]  # y 方向平移（像素）

        # 组合变换矩阵
        M = T @ S @ R @ P @ C  # 运算顺序（从右到左）非常重要
        return M, s

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算图像和对象实例之间共享的仿射变换参数。

        参数：
            labels (dict[str, Any]): 包含 'img' 的输入标签字典。

        返回：
            (dict): 包含 'M'（仿射矩阵）、'scale'、'orig_shape' 和 'size' 的参数字典。
        """
        img = labels["img"]
        if (rect_shape := labels.get("rect_shape")) is not None:  # rect 优先级更高
            size = (int(rect_shape[1]), int(rect_shape[0]))  # 将 rect 模式批次尺寸 (h, w) 转为 (w, h)
        else:
            size = (img.shape[1], img.shape[0]) if self.size is None else self.size  # w, h
        orig_shape = img.shape[:2]
        M, scale = self._compute_affine_matrix(img, size)
        return {"M": M, "scale": scale, "orig_shape": orig_shape, "size": size}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用仿射扭曲。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回且包含 'M' 和 'size' 的参数。

        返回：
            (dict): 包含扭曲图像和 'resized_shape' 的更新后标签字典。
        """
        img = labels["img"]
        M = params["M"]
        size = params["size"]
        # 4 个值：cv2 会按每 4 个通道为一组平铺 borderValue，因此 3 元组会将多光谱图像的每第 4 个通道置零
        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114, 114))
        else:  # 仿射变换
            img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114, 114))
        if img.ndim == 2:
            img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对对象实例应用仿射变换。"""
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])

        M = params["M"]
        scale = params["scale"]

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # 如果存在分割段，则使用分割段更新边界框。
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M, params["size"])

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, params["size"])
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # 裁剪到图像边界
        new_instances.clip(*params["size"], preserve_obb=self.preserve_obb)

        # 筛选对象实例
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # 使原始边界框与新边界框具有相同的缩放比例
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """对边界框应用仿射变换。

        此函数使用指定的变换矩阵对一组边界框应用仿射变换。

        参数：
            bboxes (np.ndarray): 形状为 (N, 4) 的 xyxy 格式边界框，N 表示边界框数量。
            M (np.ndarray): 形状为 (3, 3) 的仿射变换矩阵。

        返回：
            (np.ndarray): 形状为 (N, 4) 的变换后 xyxy 格式边界框。

        示例：
            >>> rp = RandomPerspective()
            >>> bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
            >>> M = np.eye(3, dtype=np.float32)
            >>> transformed_bboxes = rp.apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1、x2y2、x1y2、x2y1
        xy = xy @ M.T  # 应用变换
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # 透视重缩放或仿射变换

        # 创建新边界框
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(
        self, segments: np.ndarray, M: np.ndarray, size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """变换分割段并计算其边界框。"""
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # 应用变换
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, size[0], size[1]) for xy in segments], 0)
        if not self.preserve_obb:
            segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
            segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """对关键点应用仿射变换。

        此方法使用指定的仿射变换矩阵变换输入关键点；必要时处理透视重缩放，并将变换后落在图像边界外的关键点设为不可见。

        参数：
            keypoints (np.ndarray): 形状为 (N, K, 3) 的关键点数组；N 为实例数，K 为每个实例的关键点数，3 表示 (x, y, visibility)。
            M (np.ndarray): 3x3 仿射变换矩阵。
            size (tuple[int, int]): 输出图像尺寸 (width, height)，用于判断关键点是否可见。

        返回：
            (np.ndarray): 形状与输入相同 (N, K, 3) 的变换后关键点数组。

        示例：
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # 应用变换
        xy = xy[:, :2] / xy[:, 2:3]  # 透视重缩放或仿射变换
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > size[0]) | (xy[:, 1] > size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对语义分割掩码应用仿射变换。

        参数：
            labels (dict[str, Any]): 包含 'semantic_mask' 的字典。
            params (dict | None): 由 get_params 返回且包含 'M' 和 'size' 的参数。

        返回：
            (dict): 包含变换后语义掩码的更新后标签字典。
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        M = params["M"]
        size = params["size"]
        if (size[0] != mask.shape[1] or size[1] != mask.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                mask = cv2.warpPerspective(mask, M, dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
            else:
                mask = cv2.warpAffine(mask, M[:2], dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
        labels["semantic_mask"] = mask
        return labels

    def apply_depth(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对度量深度图应用相同的投影扭曲。

        深度值仍以米为单位，仅对其空间位置进行扭曲。最近邻插值可避免在稀疏区域或无效区域邻接真实深度时产生虚假的近零“有效”像素。
        """
        depth = labels.get("depth")
        if depth is None:
            return labels

        M = params["M"]
        size = params["size"]
        if (size[0] != depth.shape[1] or size[1] != depth.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                depth = cv2.warpPerspective(depth, M, dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
            else:
                depth = cv2.warpAffine(depth, M[:2], dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
        labels["depth"] = depth
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """根据尺寸和宽高比条件计算可供后续处理的候选边界框。

        此方法比较增强前后的边界框，判断它们是否满足宽度、高度、宽高比和面积阈值，用于筛除在增强过程中严重变形或缩小的边界框。

        参数：
            box1 (np.ndarray): Original boxes before augmentation, shape (4, N) where N is the number of boxes. Format
                is [x1, y1, x2, y2] in absolute coordinates.
            box2 (np.ndarray): 变换后的边界框，形状为 (4, N)，格式为绝对坐标 [x1, y1, x2, y2]。
            wh_thr (int): 宽度和高度像素阈值，任一维度小于该值的边界框会被拒绝。
            ar_thr (int): 宽高比阈值，宽高比大于该值的边界框会被拒绝。
            area_thr (float): 面积比例阈值，面积比例（新面积/旧面积）小于该值的边界框会被拒绝。
            eps (float): 用于避免除零的小量。

        返回：
            (np.ndarray): 形状为 (N,) 的布尔数组，用于指示哪些边界框是候选框。True 表示满足所有条件。

        示例：
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [ True  True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # 宽高比
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # 候选框


class RandomHSV(BaseTransform):
    """随机调整图像的色调、饱和度和值（HSV）通道。

    此类在 hgain、sgain 和 vgain 设定的范围内对图像应用随机 HSV 增强。

    属性：
        hgain (float): 色调的最大变化量，通常取值范围为 [0, 1]。
        sgain (float): 饱和度的最大变化量，通常取值范围为 [0, 1]。
        vgain (float): 明度的最大变化量，通常取值范围为 [0, 1]。

    方法：
        apply_image: 对图像应用随机 HSV 增强。

    示例：
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> labels = augmenter(labels)
        >>> augmented_image = labels["img"]
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """初始化用于随机 HSV（色调、饱和度、明度）增强的 RandomHSV 对象。

        此类在指定范围内随机调整图像的 HSV 通道。

        参数：
            hgain (float): 色调的最大变化量，取值范围应为 [0, 1]。
            sgain (float): 饱和度的最大变化量，取值范围应为 [0, 1]。
            vgain (float): 明度的最大变化量，取值范围应为 [0, 1]。
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply_image(self, labels, params: dict[str, Any] | None = None):
        """在预定义范围内对图像应用随机 HSV 增强。

        此方法通过随机调整输入图像的色调、饱和度和明度（HSV）通道来修改图像，调整范围由初始化时的 hgain、sgain 和 vgain 确定。

        参数：
            labels (dict[str, Any]): 包含图像数据和元数据的字典，必须包含值为 NumPy 数组的 'img' 键。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 包含 HSV 增强后图像的标签字典。

        示例：
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> labels = hsv_augmenter.apply_image(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if img.shape[-1] != 3:  # 仅对 3 通道（BGR）图像应用
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype  # uint8 类型

            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # 随机增益
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # ultralytics<=8.3.78 中的原始色调实现
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # 防止纯白色发生变色，8.3.79 版本引入

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # 无需返回值
        return labels


class RandomFlip(BaseTransform):
    """按照指定概率对图像执行随机水平或垂直翻转。

    此类执行随机图像翻转，并同步更新边界框和关键点等对象实例标注。

    属性：
        p (float): 执行翻转的概率，必须介于 0 和 1 之间。
        direction (str): 翻转方向，可为 'horizontal' 或 'vertical'。
        flip_idx (array-like): 翻转关键点时使用的索引映射（如适用）。

    方法：
        __call__: 对图像及其标注应用随机翻转变换。

    示例：
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: list[int] | None = None) -> None:
        """使用概率和方向初始化 RandomFlip 类。

        此类按照指定概率对图像执行随机水平或垂直翻转，并相应更新对象实例（边界框、关键点等）。

        参数：
            p (float): 执行翻转的概率，必须介于 0 和 1 之间。
            direction (str): 翻转方向，必须为 'horizontal' 或 'vertical'。
            flip_idx (list[int] | None): 翻转关键点时使用的索引映射（如有）。

        异常：
            AssertionError: 当 direction 不是 'horizontal' 或 'vertical'，或 p 不在 0 到 1 之间时抛出。
        """
        assert direction in {"horizontal", "vertical"}, f"方向必须为 `horizontal` 或 `vertical`，但当前为 {direction}"
        assert 0 <= p <= 1.0, f"概率必须在 [0, 1] 范围内，但当前为 {p}。"

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算随机翻转参数。

        参数：
            labels (dict[str, Any]): 包含 'img' 和 'instances' 的输入标签字典。

        返回：
            (dict): 包含 'flip'（bool）、'h'、'w'、'direction' 和 'flip_idx' 的参数字典。
        """
        img = labels["img"]
        instances = labels["instances"]
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return {
            "flip": random.random() < self.p,
            "h": h,
            "w": w,
            "direction": self.direction,
            "flip_idx": self.flip_idx,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对图像应用翻转。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含翻转后（或未改变）图像的更新后标签字典。
        """
        img = labels["img"]
        if params["flip"]:
            if params["direction"] == "vertical":
                img = np.flipud(img)
            elif params["direction"] == "horizontal":
                img = np.fliplr(img)
        labels["img"] = img
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对对象实例应用翻转。

        参数：
            labels (dict[str, Any]): 包含 'instances' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含翻转后（或未改变）对象实例的更新后标签字典。
        """
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        if params["flip"]:
            if params["direction"] == "vertical":
                instances.flipud(params["h"])
            elif params["direction"] == "horizontal":
                instances.fliplr(params["w"])
            if params["flip_idx"] is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, params["flip_idx"], :])
        labels["instances"] = instances
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对语义分割掩码应用翻转。

        参数：
            labels (dict[str, Any]): 包含 'semantic_mask' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含翻转后（或未改变）语义掩码的更新后标签字典。
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        if params["flip"]:
            if params["direction"] == "vertical":
                labels["semantic_mask"] = np.flipud(labels["semantic_mask"])
            elif params["direction"] == "horizontal":
                labels["semantic_mask"] = np.fliplr(labels["semantic_mask"])
        return labels

    def apply_depth(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对配对的度量深度图应用翻转。

        参数：
            labels (dict[str, Any]): 包含 'depth' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含翻转后（或未改变）深度图的更新后标签字典。
        """
        if labels.get("depth") is None:
            return labels
        if params["flip"]:
            if params["direction"] == "vertical":
                labels["depth"] = np.flipud(labels["depth"])
            elif params["direction"] == "horizontal":
                labels["depth"] = np.fliplr(labels["depth"])
        return labels


class LetterBox(BaseTransform):
    """用于检测、实例分割和姿态估计的图像缩放与填充变换。

    此类在保持宽高比的同时将图像缩放并填充到指定尺寸，并同步更新对应的标签和边界框。

    属性：
        new_shape (tuple): 缩放目标尺寸 (height, width)。
        auto (bool): 是否使用最小矩形填充。
        scale_fill (bool): 是否将图像拉伸到 new_shape。
        scaleup (bool): 是否允许放大图像；为 False 时只缩小图像。
        stride (int): 用于将填充量取整的步长。
        center (bool): 是否居中放置图像；为 False 时对齐左上角。

    方法：
        __call__: 缩放并填充图像，同时更新标签和边界框。

    示例：
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """初始化用于缩放和填充图像的 LetterBox 对象。

        此类用于对象检测、实例分割和姿态估计任务的图像缩放与填充，支持自动调整尺寸、拉伸填充和保持比例填充等模式。

        参数：
            new_shape (tuple[int, int]): 缩放目标尺寸 (height, width)。
            auto (bool): 为 True 时使用最小矩形调整尺寸；为 False 时直接使用 new_shape。
            scale_fill (bool): 为 True 时将图像拉伸到 new_shape，不进行填充。
            scaleup (bool): 为 True 时允许放大；为 False 时只缩小图像。
            center (bool): 为 True 时居中放置图像；为 False 时将图像放在左上角。
            stride (int): 模型步长（例如 YOLOv5 使用 32）。
            padding_value (int): 图像填充值，默认为 114。
            interpolation (int): 缩放时使用的插值方法，默认为 cv2.INTER_LINEAR。
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # 将图像居中或放置在左上角
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: dict[str, Any] | None = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        """缩放并填充图像，用于对象检测、实例分割或姿态估计任务。

        此方法对输入图像执行保持宽高比的缩放，并添加填充以适配目标尺寸，同时相应更新关联标签。

        参数：
            labels (dict[str, Any] | None): 包含图像数据和关联标签的字典；为 None 时使用空字典。
            image (np.ndarray | None): NumPy 格式的输入图像；为 None 时从 'labels' 中获取图像。

        返回：
            (dict[str, Any] | np.ndarray)：提供 'labels' 时返回包含缩放填充后图像、更新标签和额外元数据的字典；'labels' 为空时仅返回缩放填充后的图像。

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        if return_image_only:
            return labels["img"]
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算保持比例填充的参数。

        参数：
            labels (dict[str, Any]): 包含 'img' 的输入标签字典。

        返回：
            (dict): 包含 'orig_shape'、'new_shape'、'ratio'、填充量和缩放信息的参数字典。
        """
        img = labels["img"]
        shape = img.shape[:2]  # 当前尺寸 [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 缩放比例（新尺寸 / 旧尺寸）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # 只缩小，不放大（可获得更好的验证集 mAP）
            r = min(r, 1.0)

        # 计算填充量
        ratio = r, r  # 宽度和高度缩放比例
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽度和高度填充量
        if self.auto:  # 最小矩形
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # 宽度和高度填充量
        elif self.scale_fill:  # 拉伸
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度和高度缩放比例

        if self.center:
            dw /= 2  # 将宽度填充分配到两侧
            dh /= 2

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)

        return {
            "orig_shape": shape,
            "new_shape": new_shape,
            "ratio": ratio,
            "new_unpad": new_unpad,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """缩放并填充图像。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含缩放填充后图像的更新后标签字典。
        """
        img = labels["img"]
        shape = img.shape[:2]
        new_unpad = params["new_unpad"]

        if shape[::-1] != new_unpad:  # 缩放
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        h, w, c = img.shape
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # 多光谱图像
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        labels["img"] = img
        labels["resized_shape"] = params["new_shape"]
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对语义分割掩码应用保持比例填充。

        参数：
            labels (dict[str, Any]): 包含 'semantic_mask' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含缩放填充后语义掩码的更新后标签字典。
        """
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        shape = params["orig_shape"]
        new_unpad = params["new_unpad"]
        if shape[::-1] != new_unpad:
            mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        labels["semantic_mask"] = mask
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """在保持比例填充后更新对象实例坐标。

        参数：
            labels (dict[str, Any]): 包含 'instances' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含变换后对象实例的更新后标签字典。
        """
        if "instances" in labels:
            labels = self._update_labels(labels, params["ratio"], params["left"], params["top"], params["orig_shape"])
        if labels.get("ratio_pad"):
            gain_h, gain_w = labels["ratio_pad"]
            ratio_w, ratio_h = params["ratio"]
            labels["ratio_pad"] = (
                (gain_h * ratio_h, gain_w * ratio_w),
                (params["left"], params["top"]),
            )  # 用于评估
        return labels

    @staticmethod
    def _update_labels(
        labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float, orig_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """对图像应用保持比例填充后更新标签。

        此方法根据保持比例填充过程中执行的缩放和填充，修改标签中对象实例的边界框坐标。

        参数：
            labels (dict[str, Any]): 包含图像标签和对象实例的字典。
            ratio (tuple[float, float]): 应用于图像的宽度和高度缩放比例。
            padw (float): 添加到图像上的宽度填充值。
            padh (float): 添加到图像上的高度填充值。
            orig_shape (tuple[int, int]): 缩放前的原始图像尺寸 (height, width)。

        返回：
            (dict[str, Any]): 对象实例坐标已修改的更新后标签字典。

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh, (480, 640))
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*orig_shape[::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):
    """对图像数据集应用 Copy-Paste 增强的类。

    此类实现论文《Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation》
    （https://arxiv.org/abs/2012.07177）中描述的 Copy-Paste 增强技术。`flip` 模式会粘贴当前图像对象的镜像副本，
    `mixup` 模式会粘贴从随机数据集样本中选取的对象。

    属性：
        dataset (Any): 应用 Copy-Paste 增强的数据集。
        pre_transform (Callable | None): Copy-Paste 前可选的预处理变换。
        p (float): 符合条件并被粘贴的对象比例；在 `mixup` 模式下也表示应用增强的概率。

    方法：
        get_params: 计算包括选中实例和掩码在内的 Copy-Paste 参数。
        apply_image: 绘制轮廓并粘贴 Copy-Paste 像素。
        apply_instances: 拼接选中的 Copy-Paste 对象实例。

    示例：
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        """使用数据集、预处理变换、粘贴比例和模式初始化 CopyPaste 对象。"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        if mode not in ("flip", "mixup"):
            raise ValueError(f"Expected `mode` to be `flip` or `mixup`, but got {mode}.")
        self.mode = mode

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对图像及其标签应用 Copy-Paste 增强。"""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            params = self.get_params(labels)
            labels = self.apply_image(labels, params)
            labels = self.apply_instances(labels, params)
            labels = self.apply_semantic(labels, params)
            return labels
        return super().__call__(labels)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 Copy-Paste 参数。

        参数：
            labels (dict[str, Any]): 输入标签字典。

        返回：
            (dict[str, Any]): 包含 'instances2'、'selected' 和 'im_new' 的参数字典。
        """
        params = {}
        if self.mode == "mixup":
            params = super().get_params(labels)
            labels2 = labels.get("mix_labels", [{}])[0]
        else:
            labels2 = {}

        h, w = labels["img"].shape[:2]
        instances = deepcopy(labels["instances"])
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        instances2 = deepcopy(labels2.get("instances")) if labels2 else None
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)

        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]
        indexes = indexes[np.argsort(ioa.max(1)[indexes])]
        selected = indexes[: round(self.p * len(indexes))]

        im_new = np.zeros((h, w), np.uint8)

        params["instances"] = instances
        params["instances2"] = instances2
        params["selected"] = selected
        params["im_new"] = im_new
        params["labels2_cls"] = labels2.get("cls")
        params["labels2_img"] = labels2.get("img")
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 Copy-Paste。

        参数：
            labels (dict[str, Any]): 包含 'img' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含粘贴对象的更新后标签字典。
        """
        im = labels["img"].copy()

        instances2 = params["instances2"]
        selected = params["selected"]
        im_new = params["im_new"]

        for j in selected:
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, 1, cv2.FILLED)

        result = params.get("labels2_img")
        if result is None:
            result = cv2.flip(im, 1)
        if result.ndim == 2:
            result = result[..., None]

        i = im_new.astype(bool)
        im[i] = result[i]
        labels["img"] = im
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对对象实例应用 Copy-Paste。

        参数：
            labels (dict[str, Any]): 包含 'instances' 和 'cls' 的字典。
            params (dict | None): 由 get_params 返回的参数。

        返回：
            (dict): 包含拼接后对象实例的更新后标签字典。
        """
        instances = params["instances"]
        instances2 = params["instances2"]
        selected = params["selected"]
        cls = labels["cls"]
        labels2_cls = params.get("labels2_cls")

        if len(selected):
            cls = np.concatenate((cls, (labels2_cls if labels2_cls is not None else cls)[selected]), axis=0)
            instances = Instances.concatenate([instances, instances2[selected]], axis=0)

        labels["cls"] = cls
        labels["instances"] = instances
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对语义分割掩码应用 Copy-Paste。"""
        mask = labels.get("semantic_mask")
        if mask is None:
            return labels

        source = labels.get("mix_labels", [{}])[0].get("semantic_mask") if self.mode == "mixup" else cv2.flip(mask, 1)
        if source is None:
            return labels
        pasted = params["im_new"].astype(bool)
        mask = mask.copy()
        mask[pasted] = source[pasted]
        labels["semantic_mask"] = mask
        return labels


class Albumentations(BaseTransform):
    """使用 Albumentations 执行图像增强的变换类。

    此类使用 Albumentations 库执行多种图像变换，包括模糊、中值模糊、灰度转换、对比度受限自适应直方图均衡化（CLAHE）、
    随机亮度和对比度调整、RandomGamma，以及通过压缩降低图像质量。

    属性：
        p (float): 应用变换的概率。
        transform (albumentations.Compose): 组合后的 Albumentations 变换。
        contains_spatial (bool): 指示变换是否包含空间操作。

    方法：
        __call__: 对输入标签应用 Albumentations 变换。

    示例：
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    注意：
        - 需要 Albumentations 1.0.3 或更高版本。
        - 空间变换会采用特殊处理，以确保边界框兼容性。
        - 默认情况下，部分变换以很低的概率（0.01）应用。
    """

    def __init__(self, p: float = 1.0, transforms: list | None = None, flip_idx: list[int] | None = None) -> None:
        """初始化用于 YOLO 边界框格式参数的 Albumentations 变换对象。

        此类使用 Albumentations 库执行多种图像增强，包括模糊、中值模糊、灰度转换、对比度受限自适应直方图均衡化、
        随机亮度和对比度调整、RandomGamma，以及通过压缩降低图像质量。

        参数：
            p (float): 应用增强的概率，必须介于 0 和 1 之间。
            transforms (list | None): 自定义 Albumentations 变换，可以是变换对象或检查点中保存的 `A.to_dict()` 字典；为 None 时使用默认变换。
            flip_idx (list[int] | None): 反射变换使用的关键点索引映射。
        """
        self.p = p
        self.flip_idx = flip_idx
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import os

            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # 禁止显示 Albumentations 升级提示
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # 版本要求
            if transforms and isinstance(transforms[0], dict):
                transforms = [A.from_dict(t) for t in transforms]  # 恢复训练器序列化的变换
            topology_changing = getattr(A, "RandomGridShuffle", ())

            def transform_types(t) -> tuple[bool, list]:
                """递归遍历组合变换，并返回空间变换标志及会改变拓扑的变换。"""
                nested = [transform_types(x) for x in t.transforms] if isinstance(t, A.BaseCompose) else []
                return (
                    isinstance(t, A.DualTransform) or any(x[0] for x in nested),
                    ([t] if isinstance(t, topology_changing) else []) + [y for x in nested for y in x[1]],
                )

            # 如果提供了自定义变换则使用自定义变换，否则使用默认变换
            T = (
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_range=(75, 100), p=0.0),
                ]
                if transforms is None
                else transforms
            )

            # 组合变换
            transform_types = [transform_types(transform) for transform in T]
            self.contains_spatial = any(x[0] for x in transform_types)
            self.topology_transforms = [transform for x in transform_types for transform in x[1]]
            for transform in self.topology_transforms:
                transform.set_deterministic(True, save_key="topology")
            self.transform = (
                A.Compose(
                    T,
                    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels", "idx"]),
                    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["pidx"]),
                )
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # albumentations>=1.4.21 中的确定性变换需要此设置
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # 未安装相关包，跳过
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入标签应用 Albumentations 变换。

        此方法使用 Albumentations 库对输入图像及其对应标签执行一系列增强，可同时支持空间变换和非空间变换。

        参数：
            labels (dict[str, Any]): 包含图像数据和标注的字典，预期包含以下键：
                - 'img'：表示图像的 np.ndarray。
                - 'cls'：类别标签 np.ndarray。
                - 'instances'：包含边界框和其他实例信息的对象。
                - 'semantic_mask'：可选的语义类别 ID np.ndarray。
                - 'depth'：可选的度量深度值 np.ndarray。

        返回：
            (dict[str, Any]): 包含增强后图像和更新后标注的输入字典。

        示例：
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(
            ...         bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]]), segments=np.zeros((0, 1000, 2))
            ...     ),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        注意：
            - 此方法以 self.p 的概率应用变换。
            - 空间变换会更新边界框，非空间变换只修改图像。
            - 需要安装 Albumentations 库。
        """
        if self.transform is None or random.random() >= self.p:
            return labels

        im = labels["img"]
        if im.shape[2] != 3:  # 仅对 3 通道图像应用 Albumentations
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            key = "semantic_mask" if labels.get("semantic_mask") is not None else "depth"
            mask = labels.get(key)
            instances = labels["instances"]
            instances.convert_bbox("xywh")
            instances.normalize(*im.shape[:2][::-1])
            segments, keypoints = instances.segments, instances.keypoints
            h, w = im.shape[:2]
            points = segments.reshape(-1, 2)
            if keypoints is not None:
                points = np.concatenate((points, keypoints[..., :2].reshape(-1, 2)))
            points = (points * (w, h)).astype(np.float32)
            annotation_points = len(points)
            if keypoints is not None:
                points = np.concatenate((points, np.array(((0, 0), (w, 0), (0, h)), dtype=np.float32)))
            new = self.transform(
                image=im,
                bboxes=instances.bboxes,
                class_labels=cls,
                idx=np.arange(len(cls)),
                keypoints=points,
                pidx=np.arange(len(points)),
                **({"topology": {}} if self.topology_transforms else {}),
                **({"mask": mask} if mask is not None else {}),
            )
            if (segments.size or keypoints is not None) and new.get("topology"):
                raise NotImplementedError("RandomGridShuffle 无法保留多边形或关键点拓扑结构")
            if mask is not None or len(new["class_labels"]) or not len(cls):
                h, w = new["image"].shape[:2]
                i = np.array(new["idx"], dtype=int)
                n = segments.size // 2
                lost = np.ones(len(points), bool)
                lost[np.array(new["pidx"], dtype=int)] = False
                moved = points.copy()
                moved[~lost] = np.array(new["keypoints"], dtype=np.float32)
                if n:
                    segment_lost = lost[:n].reshape(segments.shape[:2])
                    segment_points = moved[:n].reshape(segments.shape)
                    i = i[~segment_lost.all(1)[i]]
                    for segment, missing in zip(segment_points, segment_lost):
                        v = np.flatnonzero(~missing)
                        if len(v) and missing.any():
                            segment[missing] = segment[
                                v[np.searchsorted(v, np.flatnonzero(missing)).clip(0, len(v) - 1)]
                            ]
                    moved[:n] = segment_points.reshape(-1, 2)
                if keypoints is not None:
                    xy = moved[n:annotation_points].reshape(*keypoints.shape[:2], 2)[i]
                    out = ((xy < 0) | (xy > (w, h))).any(-1, keepdims=True)
                    gone = lost[n:annotation_points].reshape(*keypoints.shape[:2], 1)[i] | out
                    keypoints = np.concatenate((xy.clip(0, (w, h)), np.where(gone, 0, keypoints[i][..., 2:])), -1)
                    anchors = moved[annotation_points:]
                    a, b = anchors[1] - anchors[0], anchors[2] - anchors[0]
                    reflected = not lost[annotation_points:].any() and a[0] * b[1] - a[1] * b[0] < 0
                    if self.flip_idx and reflected:
                        keypoints = np.ascontiguousarray(keypoints[:, self.flip_idx])
                if n:
                    segments = moved[:n].reshape(segments.shape)[i]
                    bboxes = np.array([segment2box(s, w, h) for s in segments], np.float32).reshape(-1, 4)
                    segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
                    segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
                    instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
                    instances.normalize(w, h)
                else:
                    if keypoints is not None:
                        keypoints[..., 0] /= w
                        keypoints[..., 1] /= h
                    instances.update(np.array(new["bboxes"], dtype=np.float32).reshape(-1, 4), keypoints=keypoints)
                labels["img"] = new["image"]
                labels["cls"] = cls[i].reshape(-1, 1)
                labels["instances"] = instances
                if mask is not None:
                    labels[key] = new["mask"]
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # 应用变换

        return labels


class Format(BaseTransform):
    """用于格式化对象检测、实例分割和姿态估计任务图像标注的类。

    此类将图像和实例标注标准化，以便 PyTorch DataLoader 的 `collate_fn` 使用。

    属性：
        bbox_format (str): 边界框格式，可选 'xywh' 或 'xyxy'。
        normalize (bool): 是否归一化边界框。
        return_mask (bool): 是否返回用于分割的实例掩码。
        return_keypoint (bool): 是否返回用于姿态估计的关键点。
        return_obb (bool): 是否返回有向边界框。
        mask_ratio (int): 掩码下采样比例。
        mask_overlap (bool): 是否允许掩码重叠。
        batch_idx (bool): 是否保留批次索引。
        bgr (float): 返回 BGR 图像的概率。

    方法：
        __call__: 格式化包含图像、类别、边界框以及可选掩码和关键点的标签字典。
        _format_img: 将图像从 NumPy 数组转换为 PyTorch 张量。
        _format_segments: 将多边形点转换为位图掩码。

    示例：
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: bool = False,
        return_keypoint: bool = False,
        return_obb: bool = False,
        mask_ratio: int = 4,
        mask_overlap: bool = True,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        """使用图像和实例标注格式化参数初始化 Format 类。

        此类为对象检测、实例分割和姿态估计任务标准化图像与实例标注，使其能够被 PyTorch DataLoader 的 `collate_fn` 使用。

        参数：
            bbox_format (str): 边界框格式，可选 'xywh'、'xyxy' 等。
            normalize (bool): 是否将边界框归一化到 [0,1]。
            return_mask (bool): 为 True 时返回用于分割任务的实例掩码。
            return_keypoint (bool): 为 True 时返回用于姿态估计任务的关键点。
            return_obb (bool): 为 True 时返回有向边界框。
            mask_ratio (int): 掩码下采样比例。
            mask_overlap (bool): 为 True 时允许掩码重叠。
            batch_idx (bool): 为 True 时保留批次索引。
            bgr (float): 返回 BGR 图像而非 RGB 图像的概率。
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # 仅训练检测任务时设为 False
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # 保留批次索引
        self.bgr = bgr

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算图像和实例格式化之间共享的参数。

        提取图像尺寸，并从标签中取出实例标注；同时转换边界框格式并将坐标反归一化，以便后续创建张量。

        参数：
            labels (dict[str, Any]): 包含 'img'、'cls' 和 'instances' 的输入标签字典。

        返回：
            (dict[str, Any]): 包含 'h'、'w'、'cls'、'instances' 和 'nl' 的参数字典。
        """
        img = labels.get("img")
        h, w = img.shape[:2] if img is not None else (0, 0)
        cls = labels.pop("cls", np.array([]))
        instances = labels.pop("instances", None)
        if instances is not None:
            instances.convert_bbox(format=self.bbox_format)
            instances.denormalize(w, h)
        return {"h": h, "w": w, "cls": cls, "instances": instances, "nl": len(instances) if instances else 0}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """将图像从 NumPy 数组格式转换为 PyTorch 张量格式。

        参数：
            labels (dict[str, Any]): 包含 NumPy 数组格式 'img' 的字典。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 包含 PyTorch 张量格式 'img' 的更新后标签字典。
        """
        img = labels.pop("img", None)
        if img is not None:
            labels["img"] = self._format_img(img)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """将实例标注格式化为 PyTorch 张量。

        将类别标签、边界框、掩码和关键点转换为适合 PyTorch DataLoader 整理的张量。

        参数：
            labels (dict[str, Any]): 用于写入格式化张量的字典。
            params (dict[str, Any]): 由 get_params 返回且包含 'h'、'w'、'cls'、'instances' 和 'nl' 的参数。

        返回：
            (dict[str, Any]): 包含格式化实例张量的更新后标签字典。
        """
        cls = params.get("cls", np.array([]))
        instances = params.get("instances")
        assert instances is not None, "Format.apply_instances 需要 instances。"
        h = params.get("h", 0)
        w = params.get("w", 0)
        nl = params.get("nl", 0)

        if self.return_mask:
            if self.mask_ratio > min(h, w):
                raise ValueError(
                    f"mask_ratio={self.mask_ratio} downsamples imgsz={(h, w)} masks to zero size; use mask_ratio <= {min(h, w)}"
                )
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
                cls_tensor = torch.from_numpy(cls.squeeze(1))
                if not masks.shape[0] or not cls_tensor.numel():
                    sem_masks = torch.zeros(h // self.mask_ratio, w // self.mask_ratio)
                elif self.mask_overlap:
                    sem_masks = cls_tensor[masks[0].long() - 1]  # 从 (1, H, W) 实例索引生成 (H, W)
                else:
                    # 创建与 mask_overlap=True 一致的语义掩码
                    sem_masks = (masks * cls_tensor[:, None, None]).max(0).values  # 从 (N, H, W) 二值掩码生成 (H, W)
                    overlap = masks.sum(dim=0) > 1  # (H, W)
                    if overlap.any():
                        weights = masks.sum(axis=(1, 2))
                        weighted_masks = masks * weights[:, None, None]  # (N, H, W)
                        weighted_masks[masks == 0] = weights.max() + 1  # 处理背景
                        smallest_idx = weighted_masks.argmin(dim=0)  # (H, W)
                        sem_masks[overlap] = cls_tensor[smallest_idx[overlap]]
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, h // self.mask_ratio, w // self.mask_ratio)
                sem_masks = torch.zeros(h // self.mask_ratio, w // self.mask_ratio)
            labels["masks"] = masks
            labels["sem_masks"] = sem_masks.float()
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl, 1)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = (
                torch.empty(0, 3) if instances.keypoints is None else torch.from_numpy(instances.keypoints)
            )
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = xyxyxyxy2xywhr(torch.from_numpy(instances.segments))
        # 注意：需要将 xywhr 格式的有向框归一化，以保持宽高一致性
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # 之后即可使用 collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        """将图像从 NumPy 数组格式转换为 YOLO 使用的 PyTorch 张量格式。

        此函数执行以下操作：
        1. 确保图像为 3 维，必要时添加通道维度。
        2. 将图像从 HWC 转置为 CHW 格式。
        3. 根据 bgr 概率可选地反转颜色通道（例如从 BGR 转为 RGB）。
        4. 将图像转换为连续数组。
        5. 将 NumPy 数组转换为 PyTorch 张量。

        参数：
            img (np.ndarray): NumPy 格式输入图像，形状为 (H, W, C) 或 (H, W)。

        返回：
            (torch.Tensor): 格式化后的 PyTorch 张量，形状为 (C, H, W)。

        示例：
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = img[..., None]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        return img

    def _format_segments(
        self, instances: Instances, cls: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        """将多边形分割段转换为位图掩码。

        参数：
            instances (Instances): 包含分割段信息的对象。
            cls (np.ndarray): 每个实例的类别标签。
            w (int): 图像宽度。
            h (int): 图像高度。

        返回：
            masks (np.ndarray)：形状为 (N, H, W) 的位图掩码；当 mask_overlap 为 True 时形状为 (1, H, W)。
            instances (Instances)：更新后的实例对象；当 mask_overlap 为 True 时分割段已排序。
            cls (np.ndarray)：更新后的类别标签；当 mask_overlap 为 True 时已排序。

        注意：
            - 当 self.mask_overlap 为 True 时，掩码会重叠并按面积排序。
            - 当 self.mask_overlap 为 False 时，每个掩码单独表示。
            - 掩码会按照 self.mask_ratio 进行下采样。
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


class SemanticFormat(Format):
    """用于语义分割的格式化变换，将图像和掩码转换为张量。

    此变换会将经过保持比例填充的语义掩码调整到与图像相同的尺寸，并将两者转换为适当的张量格式。
    """

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """格式化语义分割所需的图像和语义掩码。

        参数：
            labels (dict[str, Any]): 包含 'img' 和 'semantic_mask' 的字典。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 包含张量格式 'img' 和 'semantic_mask' 的更新后标签字典。
        """
        img = labels.pop("img", None)
        if img is not None:
            labels["img"] = self._format_img(img)
        mask = labels.get("semantic_mask")
        if mask is not None:
            labels["semantic_mask"] = torch.from_numpy(mask.copy()).to(torch.int32)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """移除语义分割不需要的实例级键。

        参数：
            labels (dict[str, Any]): 待清理的字典。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 移除无用键后的更新后标签字典。
        """
        for k in ("cls", "instances", "resized_shape", "ori_shape", "ratio_pad"):
            labels.pop(k, None)
        return labels


class LoadVisualPrompt(BaseTransform):
    """根据边界框或掩码创建供模型输入使用的视觉提示。"""

    def __init__(self, scale_factor: float = 1 / 8) -> None:
        """使用缩放因子初始化 LoadVisualPrompt。

        参数：
            scale_factor (float): 输入图像尺寸的缩放因子。
        """
        self.scale_factor = scale_factor

    @staticmethod
    def make_mask(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """根据边界框创建二值掩码。

        参数：
            boxes (torch.Tensor): xyxy 格式的边界框，形状为 (N, 4)。
            h (int): 掩码高度。
            w (int): 掩码宽度。

        返回：
            (torch.Tensor): 形状为 (N, h, w) 的二值掩码。
        """
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 形状为 (n,1,1)
        r = torch.arange(w)[None, None, :]  # 行坐标，形状为 (1,1,w)
        c = torch.arange(h)[None, :, None]  # 列坐标，形状为 (1,h,1)

        return (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算视觉提示参数。

        参数：
            labels (dict[str, Any]): 输入标签字典。

        返回：
            (dict): 包含 'imgsz'、'bboxes'、'masks' 和 'cls' 的参数字典。
        """
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None
        if "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * torch.tensor(imgsz)[[1, 0, 1, 0]]  # 将边界框反归一化
        elif "masks" in labels:
            masks = labels["masks"]

        cls = labels["cls"].squeeze(-1).to(torch.int)
        return {"imgsz": imgsz, "bboxes": bboxes, "masks": masks, "cls": cls}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """创建视觉提示并将其添加到标签中。

        参数：
            labels (dict[str, Any]): 包含图像数据和标注的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 添加视觉提示后的更新后标签字典。
        """
        visuals = self.get_visuals(params["cls"], params["imgsz"], bboxes=params["bboxes"], masks=params["masks"])
        labels["visuals"] = visuals
        return labels

    def get_visuals(
        self,
        category: int | np.ndarray | torch.Tensor,
        shape: tuple[int, int],
        bboxes: np.ndarray | torch.Tensor = None,
        masks: np.ndarray | torch.Tensor = None,
    ) -> torch.Tensor:
        """根据边界框或掩码生成视觉掩码。

        参数：
            category (int | np.ndarray | torch.Tensor): 对象的类别标签。
            shape (tuple[int, int]): 图像尺寸 (height, width)。
            bboxes (np.ndarray | torch.Tensor, optional): 对象的 xyxy 格式边界框。
            masks (np.ndarray | torch.Tensor, optional): 对象掩码。

        返回：
            (torch.Tensor): 包含每个类别视觉掩码的张量。

        异常：
            ValueError: 未提供 bboxes 和 masks 时抛出。
        """
        masksz = (int(shape[0] * self.scale_factor), int(shape[1] * self.scale_factor))
        if bboxes is not None:
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes)
            bboxes *= self.scale_factor
            masks = self.make_mask(bboxes, *masksz).float()
        elif masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)  # (N, H, W)
            masks = F.interpolate(masks.unsqueeze(1), masksz, mode="nearest").squeeze(1).float()
        else:
            raise ValueError("LoadVisualPrompt 的标签中必须包含 bboxes 或 masks")
        if not isinstance(category, torch.Tensor):
            category = torch.tensor(category, dtype=torch.int)
        cls_unique, inverse_indices = torch.unique(category, sorted=True, return_inverse=True)
        # 注意：RandomLoadText 生成的 `cls` 索引应当是连续的。
        # 如果 len(cls_unique)：
        #     断言 len(cls_unique) == cls_unique[-1] + 1，否则：
        #         f"类别索引应为连续范围，但当前为 {cls_unique}"
        #     )
        visuals = torch.zeros(cls_unique.shape[0], *masksz)
        for idx, mask in zip(inverse_indices, masks):
            visuals[idx] = torch.logical_or(visuals[idx], mask)
        return visuals


class RandomLoadText(BaseTransform):
    """随机采样正样本文本和负样本文本，并相应更新类别索引。

    此类从给定的类别文本集合中采样文本，包括图像中存在的正样本和不存在的负样本。它会根据采样文本更新类别索引，
    还可以选择将文本列表填充到固定长度。

    属性：
        prompt_format (str): 文本提示的格式字符串。
        neg_samples (tuple[int, int]): 随机采样负样本的数量范围。
        max_samples (int): 单张图像中不同文本样本的最大数量。
        padding (bool): 是否将文本填充到 max_samples。
        padding_value (list[str]): padding 为 True 时使用的填充文本。

    方法：
        __call__: 处理输入标签，并返回更新后的类别和文本。

    示例：
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: list[str] | None = None,
    ) -> None:
        """初始化 RandomLoadText 类，以随机采样正样本文本和负样本文本。

        此类用于随机采样正样本和负样本文本，并根据采样结果相应更新类别索引，可用于基于文本的对象检测任务。

        参数：
            prompt_format (str): 提示格式字符串，应包含一对用于插入文本的花括号 {}。
            neg_samples (tuple[int, int]): 随机采样负样本的数量范围，第一个整数表示最小数量，第二个整数表示最大数量。
            max_samples (int): 单张图像中不同文本样本的最大数量。
            padding (bool): 是否将文本填充到 max_samples；为 True 时文本数量始终等于 max_samples。
            padding_value (list[str]): padding 为 True 时使用的填充文本。
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value if padding_value is not None else [""]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算文本采样参数。

        参数：
            labels (dict[str, Any]): 包含 'texts'、'cls' 和 'instances' 的输入标签字典。

        返回：
            (dict): 包含 'valid_idx'、'new_cls' 和 'texts' 的参数字典。
        """
        assert "texts" in labels, "标签中未找到 texts。"
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        # 随机处理
        # random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])

        # 当一个类别存在多个提示时，随机选择其中一个
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += random.choices(self.padding_value, k=num_padding)

        assert len(texts) == self.max_samples

        return {"valid_idx": valid_idx, "new_cls": np.array(new_cls), "texts": texts}

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """根据采样文本筛选对象实例并更新类别标签。

        参数：
            labels (dict[str, Any]): 包含 'instances' 和 'cls' 的字典。
            params (dict): 由 get_params 返回的参数。

        返回：
            (dict): 包含筛选后对象实例以及新类别和文本条目的更新后标签字典。
        """
        labels["instances"] = labels["instances"][params["valid_idx"]]
        labels["cls"] = params["new_cls"]
        labels["texts"] = params["texts"]
        return labels


def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace):
    """为训练过程应用一系列图像变换。

    此函数组合多种图像增强技术，为 YOLO 训练准备图像，包括 Mosaic、Copy-Paste、随机透视、MixUp 和多种颜色调整。

    参数：
        dataset (Dataset): 包含图像数据和标注的数据集对象。
        imgsz (int): 缩放目标图像尺寸。
        hyp (IterableSimpleNamespace): 控制各项变换参数的超参数命名空间。

    返回：
        (Compose): 应用于数据集的组合图像变换。

    示例：
        >>> from ultralytics.cfg import DEFAULT_CFG
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, imgsz=640)
        >>> hyp = IterableSimpleNamespace(
        ...     **{
        ...         **vars(DEFAULT_CFG),
        ...         "mosaic": 1.0,
        ...         "copy_paste": 0.5,
        ...         "degrees": 10.0,
        ...         "translate": 0.2,
        ...         "scale": 0.9,
        ...     }
        ... )
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])

        >>> # 使用自定义 Albumentations
        >>> import albumentations as A
        >>> augmentations = [A.Blur(p=0.01), A.CLAHE(p=0.01)]
        >>> hyp.augmentations = augmentations
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        size=(imgsz, imgsz),
        preserve_obb=getattr(dataset, "use_obb", False),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(dataset, p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # 用于关键点增强
    if getattr(dataset, "use_keypoints", False):
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # fliplr 和 flipud 都需要 flip_idx
            LOGGER.warning("data.yaml 中未定义 'flip_idx' 数组，将禁用 'fliplr' 和 'flipud' 增强。")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None), flip_idx=flip_idx),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # 图像变换


# 分类数据增强 ---------------------------------------------------------------------------------------------------------
def classify_transforms(
    size: tuple[int, int] | int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: float | None = None,
):
    """创建用于分类任务的组合图像变换。

    此函数生成一组 torchvision 变换，用于在分类模型评估或推理前预处理图像，包括缩放、中心裁剪、张量转换和归一化。

    参数：
        size (tuple[int, int] | int): 变换后图像的目标尺寸。为整数时表示短边尺寸；为元组时表示 (height, width)。
        mean (tuple[float, float, float]): 归一化时使用的各 RGB 通道均值。
        std (tuple[float, float, float]): 归一化时使用的各 RGB 通道标准差。
        interpolation (str): 插值方法，可为 'NEAREST'、'BILINEAR' 或 'BICUBIC'。
        crop_fraction (float | None): 已弃用，未来版本将移除。

    返回：
        (torchvision.transforms.Compose): 组合后的 torchvision 变换。

    示例：
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # 局部导入以加快 'import ultralytics'

    scale_size = size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)

    if crop_fraction:
        deprecation_warn("crop_fraction")

    # 正方形目标使用标量短边模式（保持宽高比）；非正方形目标缩放到精确的 (h, w)。
    resize = scale_size[0] if scale_size[0] == scale_size[1] else scale_size
    tfl = [
        T.Resize(resize, interpolation=getattr(T.InterpolationMode, interpolation)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    return T.Compose(tfl)


# 分类训练数据增强 -----------------------------------------------------------------------------------------------------
def classify_augmentations(
    size: int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    scale: tuple[float, float] | None = None,
    ratio: tuple[float, float] | None = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str | None = None,
    hsv_h: float = 0.015,  # 图像 HSV 色调增强（比例）
    hsv_s: float = 0.4,  # 图像 HSV 饱和度增强（比例）
    hsv_v: float = 0.4,  # 图像 HSV 明度增强（比例）
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """创建用于分类任务的组合图像增强变换。

    此函数生成适合训练分类模型的图像变换集合，包括缩放、翻转、颜色抖动、自动增强和随机擦除等选项。

    参数：
        size (int): 变换后图像的目标尺寸。
        mean (tuple[float, float, float]): 归一化时使用的各 RGB 通道均值。
        std (tuple[float, float, float]): 归一化时使用的各 RGB 通道标准差。
        scale (tuple[float, float] | None): 裁剪区域占原图面积比例的范围。
        ratio (tuple[float, float] | None): 裁剪区域宽高比的范围。
        hflip (float): 水平翻转概率。
        vflip (float): 垂直翻转概率。
        auto_augment (str | None): 自动增强策略，可为 'randaugment'、'augmix'、'autoaugment' 或 None。
        hsv_h (float): 图像 HSV 色调增强因子。
        hsv_s (float): 图像 HSV 饱和度增强因子。
        hsv_v (float): 图像 HSV 明度增强因子。
        force_color_jitter (bool): 启用自动增强时是否仍应用颜色抖动。
        erasing (float): 随机擦除概率。
        interpolation (str): 插值方法，可为 'NEAREST'、'BILINEAR' 或 'BICUBIC'。

    返回：
        (torchvision.transforms.Compose): 组合后的图像增强变换。

    示例：
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # 未安装 Albumentations 时使用的变换
    import torchvision.transforms as T  # 局部导入以加快 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_augmentations() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # ImageNet 默认面积比例范围
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # ImageNet 默认宽高比范围
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"提供的参数必须是字符串，但当前类型为 {type(auto_augment)}"
        # 启用 AA/RA 时通常会禁用颜色抖动；保留此选项可覆盖该行为而不破坏旧版超参数配置
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" 需要 torchvision >= 0.11.0，已禁用该选项。')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" 需要 torchvision >= 0.13.0，已禁用该选项。')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" 需要 torchvision >= 0.10.0，已禁用该选项。')

        else:
            raise ValueError(
                f'无效的 auto_augment 策略：{auto_augment}。应为 "randaugment"、'
                f'"augmix"、"autoaugment" 或 None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


class DepthFormat(Format):
    """用于单目深度估计的格式化变换：通过 Format 处理图像，并缩放和张量化深度图。

    此变换与 SemanticFormat 类似：基类 Format.apply_image 将图像从 HWC BGR 转换为 CHW RGB 张量，
    apply_depth 钩子（由 BaseTransform.__call__ 在 apply_image 后调用）将配对深度图调整到填充后图像的尺寸，
    并输出为形状 (1, H, W) 的浮点张量。
    """

    def apply_depth(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """使用最近邻插值将深度图调整到格式化图像尺寸，并输出形状为 (1, H, W) 的浮点张量。

        参数：
            labels (dict[str, Any]): 包含 'img'（已为 CHW 张量）以及可选 'depth' 的字典。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 包含形状为 (1, H, W) 浮点张量格式 'depth' 的更新后标签字典。
        """
        depth = labels.get("depth")
        if depth is None or "img" not in labels:
            return labels
        _, h, w = labels["img"].shape
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        labels["depth"] = torch.from_numpy(np.ascontiguousarray(depth[None])).float()
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """移除深度估计不需要的实例级键。

        参数：
            labels (dict[str, Any]): 待清理的字典。
            params (dict[str, Any] | None): 为兼容接口保留但未使用的参数。

        返回：
            (dict[str, Any]): 移除无用键后的更新后标签字典。
        """
        for k in ("cls", "instances", "resized_shape", "ori_shape", "ratio_pad"):
            labels.pop(k, None)
        return labels


# 注意：保留此类以维持向后兼容
class ClassifyLetterBox:
    """用于分类任务图像缩放和填充的类。

    此类用于组成变换流程，例如 T.Compose([LetterBox(size), ToTensor()])，会在保持原始宽高比的同时将图像缩放并填充到指定尺寸。

    属性：
        h (int): 图像目标高度。
        w (int): 图像目标宽度。
        auto (bool): 为 True 时根据 stride 自动计算短边。
        stride (int): 步长值，在 auto 为 True 时使用。

    方法：
        __call__: 对输入图像应用保持比例填充变换。

    示例：
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size: int | tuple[int, int] = (640, 640), auto: bool = False, stride: int = 32):
        """初始化用于图像预处理的 ClassifyLetterBox 对象。

        此类用于分类任务的图像变换流程，在保持原始宽高比的同时将图像缩放并填充到指定尺寸。

        参数：
            size (int | tuple[int, int]): 保持比例填充后的目标尺寸。为整数时创建 (size, size) 的正方形图像；为元组时应为 (height, width)。
            auto (bool): 为 True 时根据 stride 自动计算短边。
            stride (int): 步长值，在 auto 为 True 时使用。
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # 传入最大尺寸整数，并根据 stride 自动计算短边
        self.stride = stride  # 与 auto 配合使用

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """使用保持比例填充方法缩放并填充图像。

        此方法在保持宽高比的同时将输入图像缩放到指定尺寸范围内，然后填充缩放后的图像以匹配目标尺寸。

        参数：
            im (np.ndarray): NumPy 格式输入图像，形状为 (H, W, C)。

        返回：
            (np.ndarray): 缩放填充后的 NumPy 图像，形状为 (hs, ws, 3)，其中 hs 和 ws 分别为目标高度和宽度。

        示例：
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # 新旧尺寸比例
        h, w = round(imh * r), round(imw * r)  # 缩放后图像尺寸

        # 计算填充尺寸
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # 创建填充后的图像
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# 注意：保留此类以维持向后兼容
class CenterCrop:
    """对分类任务图像执行中心裁剪。

    此类从输入图像中心裁剪区域，在保持宽高比的同时将其调整到指定尺寸，用于组成诸如 T.Compose([CenterCrop(size), ToTensor()]) 的变换流程。

    属性：
        h (int): 裁剪后图像的目标高度。
        w (int): 裁剪后图像的目标宽度。

    方法：
        __call__: 对输入图像应用中心裁剪变换。

    示例：
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size: int | tuple[int, int] = (640, 640)):
        """初始化用于图像预处理的 CenterCrop 对象。

        此类用于组成变换流程，例如 T.Compose([CenterCrop(size), ToTensor()])，会从输入图像中心裁剪指定尺寸的区域。

        参数：
            size (int | tuple[int, int]): 裁剪目标尺寸。为整数时执行 (size, size) 的正方形裁剪；为类似 (h, w) 的序列时，将其作为输出尺寸。
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im: Image.Image | np.ndarray) -> np.ndarray:
        """对输入图像执行中心裁剪。

        此方法从图像中心裁剪最大的正方形区域，并将其缩放到指定尺寸。

        参数：
            im (np.ndarray | PIL.Image.Image): 输入图像，可以是形状为 (H, W, C) 的 NumPy 数组或 PIL 图像对象。

        返回：
            (np.ndarray): 中心裁剪并缩放后的 NumPy 图像，形状为 (self.h, self.w, C)。

        示例：
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):  # 必要时将 PIL 图像转换为 NumPy 数组
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # 最小尺寸
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


# 注意：保留此类以维持向后兼容
class ToTensor:
    """将图像从 NumPy 数组转换为 PyTorch 张量。

    此类用于组成变换流程，例如 T.Compose([LetterBox(size), ToTensor()])。

    属性：
        half (bool): 为 True 时将图像转换为半精度（float16）。

    方法：
        __call__: 将输入图像转换为张量。

    示例：
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

        注意：
        输入图像应为 BGR 格式，形状为 (H, W, C)。
        输出张量为 BGR 格式，形状为 (C, H, W)，并归一化到 [0, 1]。
    """

    def __init__(self, half: bool = False):
        """初始化用于将图像转换为 PyTorch 张量的 ToTensor 对象。

        此类用于 Ultralytics YOLO 框架的图像预处理流程，可将 NumPy 数组或 PIL 图像转换为 PyTorch 张量，并支持转换为半精度（float16）。

        参数：
            half (bool): 为 True 时将张量转换为半精度（float16）。
        """
        super().__init__()
        self.half = half

    def __call__(self, im: np.ndarray) -> torch.Tensor:
        """将图像从 NumPy 数组转换为 PyTorch 张量。

        此方法将输入图像从 NumPy 数组转换为 PyTorch 张量，可选地转换为半精度并执行归一化，同时将图像从 HWC 转置为 CHW 格式。

        参数：
            im (np.ndarray): BGR 顺序的 NumPy 输入图像，形状为 (H, W, C)。

        返回：
            (torch.Tensor): 转换后的 PyTorch 张量，类型为 float32 或 float16，归一化到 [0, 1]，形状为 (C, H, W)，并保持 BGR 顺序。

        示例：
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1)))  # HWC 转 CHW，并确保连续
        im = torch.from_numpy(im)  # 转换为 torch 张量
        im = im.half() if self.half else im.float()  # uint8 转为 fp16/32
        im /= 255.0  # 0-255 转为 0.0-1.0
        return im
