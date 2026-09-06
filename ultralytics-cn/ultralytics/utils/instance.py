# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import abc
from itertools import repeat
from numbers import Number

import cv2
import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """创建一个函数，将输入转换为 n 元组；单个值会被重复填充。"""

    def parse(x):
        """解析输入并返回 n 元组；单个值会重复 n 次。"""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` 表示左上角和右下角坐标。
# `xywh` 表示中心点 x、中心点 y、宽度和高度（YOLO 格式）。
# `ltwh` 表示左上角坐标、宽度和高度（COCO 格式）。
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes", "Instances")  # 元组或列表


class Bboxes:
    """用于处理多种格式边界框的类。

    此类支持 'xyxy'、'xywh' 和 'ltwh' 等边界框格式，并提供格式转换、缩放和面积计算方法。边界框数据应以 NumPy 数组形式提供。

    属性：
        bboxes (np.ndarray): 以形状 (N, 4) 的二维 NumPy 数组存储的边界框。
        format (str): 边界框格式（'xyxy'、'xywh' 或 'ltwh'）。

    方法：
        convert: 将边界框从一种格式转换为另一种格式。
        areas: 计算边界框面积。
        mul: 将边界框坐标乘以缩放因子。
        add: 为边界框坐标添加偏移量。
        concatenate: 拼接多个 Bboxes 对象。

    示例：
        创建 YOLO 格式的边界框
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes.areas())
        [15000]

    注意：
        此类不负责边界框的归一化或反归一化。
    """

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        """使用指定格式的边界框数据初始化 Bboxes 类。

        参数：
            bboxes (np.ndarray): 边界框数组，形状为 (N, 4) 或 (4,)。
            format (str): 边界框格式，可选 'xyxy'、'xywh' 或 'ltwh'。
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format: str) -> None:
        """将边界框从一种格式转换为另一种格式。

        参数：
            format (str): 目标格式，可选 'xyxy'、'xywh' 或 'ltwh'。
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        """计算边界框面积。"""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # xyxy 格式
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # xywh 或 ltwh 格式
        )

    def mul(self, scale: int | tuple | list) -> None:
        """将边界框坐标乘以缩放因子。

        参数：
            scale (int | tuple | 列表): 四个坐标的缩放因子；如果是整数，则将同一因子应用于所有坐标。
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: int | tuple | list) -> None:
        """为边界框坐标添加偏移量。

        参数：
            offset (int | tuple | 列表): 四个坐标的偏移量；如果是整数，则将同一偏移量应用于所有坐标。
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        """返回边界框数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        """将 Bboxes 对象列表拼接为单个 Bboxes 对象。

        参数：
            boxes_list (列表[Bboxes]): 待拼接的 Bboxes 对象列表。
            axis (int, 可选): 拼接边界框所沿的轴。

        返回：
            (Bboxes): 包含拼接后边界框的新 Bboxes 对象。

        注意：
            输入应为 Bboxes 对象组成的列表或元组。
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
        """通过索引获取单个边界框或一组边界框。

        参数：
            索引 (int | slice | np.ndarray): 用于选择目标边界框的索引、切片或布尔数组。

        返回：
            (Bboxes): 包含所选边界框的新 Bboxes 对象。

        注意：
            使用布尔索引时，请确保提供的布尔数组长度与边界框数量相同。
        """
        b = self.bboxes[index]
        if b.ndim == 1:
            b = b.reshape(1, -1)
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b, format=self.format)


class Instances:
    """用于存储图像中检测对象的边界框、分割段和关键点的容器。

    此类为处理不同类型的对象标注提供统一接口，包括边界框、分割段和关键点，并支持缩放、归一化、裁剪和格式转换等操作。

    属性：
        _bboxes (Bboxes): 用于处理边界框操作的内部对象。
        keypoints (np.ndarray): 关键点，形状为 (N, 17, 3)，格式为 (x, y, visible)。
        normalized (bool): 指示边界框坐标是否已归一化的标志。
        segments (np.ndarray): 重采样后的分割段数组，形状为 (N, M, 2)。

    方法：
        convert_bbox: 转换边界框格式。
        scale: 按给定因子缩放坐标。
        denormalize: 将归一化坐标转换为绝对坐标。
        normalize: 将绝对坐标转换为归一化坐标。
        add_padding: 为坐标添加填充。
        flipud: 垂直翻转坐标。
        fliplr: 水平翻转坐标。
        clip: 将坐标裁剪到图像边界内。
        remove_zero_area_boxes: 移除面积为零的边界框。
        update: 更新实例变量。
        concatenate: 拼接多个 Instances 对象。

    示例：
        创建包含边界框和分割段的实例
        >>> instances = Instances(
        ...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
        ...     segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
        ...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        ... )
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray = None,
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """使用边界框、分割段和关键点初始化 Instances 对象。

        参数：
            bboxes (np.ndarray): 边界框，形状为 (N, 4)。
            segments (np.ndarray, 可选): 分割段。
            keypoints (np.ndarray, 可选): 关键点，形状为 (N, 17, 3)，格式为 (x, y, visible)。
            bbox_format (str): 边界框格式。
            normalized (bool): 坐标是否已归一化。
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments if segments is not None else np.zeros((0, 0, 2), dtype=np.float32)

    def convert_bbox(self, format: str) -> None:
        """转换边界框格式。

        参数：
            format (str): 目标格式，可选 'xyxy'、'xywh' 或 'ltwh'。
        """
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """计算边界框面积。"""
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """按给定因子缩放坐标。

        参数：
            scale_w (float): 宽度缩放因子。
            scale_h (float): 高度缩放因子。
            bbox_only (bool, 可选): 是否只缩放边界框。
        """
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        """将归一化坐标转换为绝对坐标。

        参数：
            w (int): 图像宽度。
            h (int): 图像高度。
        """
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        """将绝对坐标转换为归一化坐标。

        参数：
            w (int): 图像宽度。
            h (int): 图像高度。
        """
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """为坐标添加填充。

        参数：
            padw (int): 水平填充宽度。
            padh (int): 垂直填充高度。
        """
        assert not self.normalized, "添加填充时必须使用绝对坐标。"
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
        """通过索引获取单个实例或一组实例。

        参数：
            索引 (int | slice | np.ndarray): 用于选择目标实例的索引、切片或布尔数组。

        返回：
            (Instances): 包含所选边界框、分割段和关键点（如果存在）的新 Instances 对象。
        """
        index = [index] if isinstance(index, (int, np.integer)) else index
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        """垂直翻转坐标。

        参数：
            h (int): 图像高度。
        """
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """水平翻转坐标。

        参数：
            w (int): 图像宽度。
        """
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int, preserve_obb: bool = False) -> None:
        """将坐标裁剪到图像边界内。

        参数：
            w (int): 图像宽度。
            h (int): 图像高度。
            preserve_obb (bool): 裁剪分割段时是否保持有向边界框方向。
        """
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        if preserve_obb and len(self.segments):
            segment_mins, segment_maxs = self.segments.min(1), self.segments.max(1)
            inside = ((segment_mins >= 0) & (segment_maxs <= (w, h))).all(1)
            if not inside.all():
                canvas = np.array(((0, 0), (w, 0), (w, h), (0, h)), dtype=np.float32)
                bboxes, segments = self.bboxes.copy(), self.segments.copy()
                for i in np.flatnonzero(~inside):
                    segment = self.segments[i]
                    visible_area, visible = cv2.intersectConvexConvex(
                        cv2.convexHull(segment.astype(np.float32)), canvas
                    )
                    if visible is None or visible_area <= 0:
                        bboxes[i], segments[i] = 0, 0
                        continue
                    visible = visible.reshape(-1, 2)
                    bboxes[i] = np.array((*visible.min(0), *visible.max(0)), dtype=segment.dtype)

                    # 将可见多边形拟合到完整边界框坐标中，避免边缘裁剪改变其角度。
                    (_, _), (box_w, box_h), angle = cv2.minAreaRect(segment.astype(np.float32))
                    if box_w < box_h:
                        angle += 90
                    angle = np.deg2rad(angle)
                    cos, sin = np.cos(angle), np.sin(angle)
                    basis = np.array(((cos, -sin), (sin, cos)), dtype=np.float32)
                    aligned = visible @ basis
                    (u1, v1), (u2, v2) = aligned.min(0), aligned.max(0)
                    corners = np.array(((u2, v2), (u2, v1), (u1, v1), (u1, v2)), dtype=np.float32)
                    segments[i] = resample_segments([corners @ basis.T], n=len(segment))[0]
                self.update(bboxes, segments)
        else:
            self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
            self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
            self.segments[..., 0] = self.segments[..., 0].clip(0, w)
            self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        if self.keypoints is not None:
            # 将越界关键点的可见性设为零
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self) -> np.ndarray:
        """移除面积为零的边界框；裁剪后部分边界框的宽度或高度可能变为零。

        返回：
            (np.ndarray): 指示哪些边界框被保留的布尔数组。
        """
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes: np.ndarray, segments: np.ndarray = None, keypoints: np.ndarray = None):
        """更新实例变量。

        参数：
            bboxes (np.ndarray): 新的边界框。
            segments (np.ndarray, 可选): 新的分割段。
            keypoints (np.ndarray, 可选): 新的关键点。
        """
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        """返回实例数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
        """将 Instances 对象列表拼接为单个 Instances 对象。

        参数：
            instances_list (列表[Instances]): 待拼接的 Instances 对象列表。
            axis (int, 可选): 数组拼接所沿的轴。

        返回：
            (Instances): 包含拼接后边界框、分割段和关键点（如果存在）的新 Instances 对象。

        注意：
            列表中的 `Instances` 对象应具有相同属性，例如边界框格式、是否包含关键点以及坐标是否归一化。
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        seg_len = [b.segments.shape[1] for b in instances_list]
        if len(frozenset(seg_len)) > 1:  # 分割段长度不一致时重新采样
            max_len = max(seg_len)
            cat_segments = np.concatenate(
                [
                    resample_segments(list(b.segments), max_len)
                    if len(b.segments)
                    else np.zeros((0, max_len, 2), dtype=np.float32)  # re-generating empty 分割段
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self) -> np.ndarray:
        """返回 bounding 边界框."""
        return self._bboxes.bboxes

    def __repr__(self) -> str:
        """返回 Instances 对象的字符串表示。"""
        # 将私有名称映射为公开名称，并包含直接属性
        attr_map = {"_bboxes": "bboxes"}
        parts = []
        for key, value in self.__dict__.items():
            name = attr_map.get(key, key)
            if name == "bboxes":
                value = self.bboxes  # 使用该属性
            if value is not None:
                parts.append(f"{name}={value!r}")
        return "Instances({})".format("\n".join(parts))
