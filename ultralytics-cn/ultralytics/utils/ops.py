# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import NOT_MACOS14, TORCH_VERSION
from ultralytics.utils.checks import check_version
from ultralytics.utils.torch_utils import get_torch_device_backend


class Profile(contextlib.ContextDecorator):
    """用于测量代码执行时间的 Ultralytics Profile 类。

    可作为 @Profile() 装饰器或 'with Profile():' 上下文管理器使用，并支持加速器同步以获得准确计时。

    属性：
        t (float): 累计时间，单位为秒。
        device (torch.device): 模型推理使用的设备。
        accelerator (模块): 用于计时同步的 PyTorch 设备模块。

    示例：
        作为上下文管理器使用，用于统计代码执行时间：
        >>> with Profile() as dt:
        ...     pass  # slow operation here
        >>> str(dt).startswith("Elapsed time is ")
        True

        作为装饰器使用，用于统计函数执行时间：
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """初始化 Profile 类。

        参数：
            t (float): Initial accumulated time in seconds.
            device (torch.device, 可选): 模型推理使用的设备，用于启用加速器同步。
        """
        self.t = t
        self.device = device
        device_type = getattr(device, "type", str(device).split(":")[0] if device else None)
        self.accelerator = get_torch_device_backend(device_type) if device_type in {"cuda", "npu", "xpu"} else None

    def __enter__(self):
        """开始计时。"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """停止计时。"""
        self.dt = self.time() - self.start  # 时间增量
        self.t += self.dt  # 累加时间增量

    def __str__(self):
        """返回表示累计耗时的易读字符串。"""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """获取当前时间，并在适用时执行加速器同步。"""
        if self.accelerator is not None:
            self.accelerator.synchronize(self.device)
        return time.perf_counter()


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    """将分割线坐标转换为边界框坐标。

    通过查找裁剪到图像范围内的多边形 x、y 最小和最大坐标，将单个分割标签转换为边界框标签，
    因此跨越图像边界的分割线仍保留可见范围。已经位于图像内部的分割线会直接返回，无需裁剪。

    参数：
        segment (np.ndarray): 分割线坐标，形状为 (N, 2)，N 为点数。
        width (int): 图像宽度，单位为像素。
        height (int): 图像高度，单位为像素。

    返回：
        (np.ndarray): xyxy 格式的边界框坐标 [x1, y1, x2, y2]。
    """
    if not len(segment):
        return np.zeros(4, dtype=segment.dtype)
    x, y = segment[:, 0], segment[:, 1]
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:  # fully inside 图像
        return np.array([xmin, ymin, xmax, ymax], dtype=segment.dtype)
    axes = np.array((0, 0, 1, 1))
    bounds = np.array((0, width, 0, height), dtype=segment.dtype)
    lims = np.array((height, height, width, width), dtype=segment.dtype)  # (高度, 宽度)[axis] per boundary
    start, delta = segment, np.roll(segment, -1, axis=0) - segment
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (bounds - start[:, axes]) / delta[:, axes]
        inter = start[:, None, :] + t[:, :, None] * delta[:, None, :]
    other = inter[:, np.arange(4), 1 - axes]
    corners = np.array(((0, 0), (width, 0), (0, height), (width, height)), dtype=segment.dtype)
    contour = segment.astype(np.float32)
    points = np.concatenate(
        (
            segment[(x >= 0) & (y >= 0) & (x <= width) & (y <= height)],
            inter[(t >= 0) & (t <= 1) & (other >= 0) & (other <= lims)],
            corners[[cv2.pointPolygonTest(contour, tuple(map(float, p)), False) >= 0 for p in corners]],
        )
    )
    return (
        np.array([*points.min(0), *points.max(0)], dtype=segment.dtype)
        if len(points)
        else np.zeros(4, dtype=segment.dtype)
    )


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor | np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
    padding: bool = True,
    xywh: bool = False,
) -> torch.Tensor | np.ndarray:
    """将边界框从一种图像尺寸缩放到另一种图像尺寸。

    将边界框从 img1_shape 缩放到 img0_shape，同时处理填充和宽高比变化。
    同时支持 xyxy 和 xywh 边界框格式。

    参数：
        img1_shape (tuple[int, int]): 源图像尺寸 (高度, 宽度)。
        boxes (torch.Tensor | np.ndarray): 要缩放的边界框，形状为 (N, 4)。
        img0_shape (tuple[int, int]): 目标图像尺寸 (高度, 宽度)。
        ratio_pad (tuple, 可选): 缩放比例和填充值，格式为 ((ratio_h, ratio_w), (pad_w, pad_h))。
        padding (bool): 边界框是否来自带有填充的 YOLO 风格增强图像。
        xywh (bool): 边界框格式是否为 xywh；为 False 时使用 xyxy 格式。

    返回：
        (torch.Tensor | np.ndarray): 缩放后的边界框，格式与输入相同。
    """
    if ratio_pad is None:  # 根据 img0_shape 计算缩放参数
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 缩放比例 = 旧尺寸 / 新尺寸
        gain_y = gain_x = gain
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    else:
        gain_y, gain_x = ratio_pad[0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x 填充
        boxes[..., 1] -= pad_y  # y 填充
        if not xywh:
            boxes[..., 2] -= pad_x  # x 填充
            boxes[..., 3] -= pad_y  # y 填充
    boxes[..., 0] /= gain_x
    boxes[..., 1] /= gain_y
    boxes[..., 2] /= gain_x
    boxes[..., 3] /= gain_y
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """返回大于等于 x 且可被给定除数整除的最小数值。

    参数：
        x (int): 要调整为可整除的数值。
        divisor (int | torch.Tensor): 除数。

    返回：
        (int): 大于等于 x 且可被除数整除的最小数值。
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # 转换为整数
    return math.ceil(x / divisor) * divisor


def clip_boxes(boxes, shape):
    """将边界框裁剪到图像边界内。

    参数：
        boxes (torch.Tensor | np.ndarray): 要裁剪的边界框。
        shape (tuple): 图像尺寸，可为 HWC 或 HW 格式。

    返回：
        (torch.Tensor | np.ndarray): 裁剪后的边界框。
    """
    h, w = shape[:2]  # 同时支持 HWC 和 HW 尺寸
    if isinstance(boxes, torch.Tensor):  # 逐项裁剪更快
        if NOT_MACOS14 and not (boxes.device.type == "mps" and check_version(TORCH_VERSION, "<2.5.0")):
            boxes[..., 0].clamp_(0, w)  # x1
            boxes[..., 1].clamp_(0, h)  # y1
            boxes[..., 2].clamp_(0, w)  # x2
            boxes[..., 3].clamp_(0, h)  # y2
        else:  # macOS 14 或 torch<2.5 存在 MPS 跨步原地操作错误
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # NumPy 数组（分组裁剪更快）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """将线段坐标裁剪到图像边界内。

    参数：
        coords (torch.Tensor | np.ndarray): 要裁剪的线段坐标。
        shape (tuple): 图像尺寸，可为 HWC 或 HW 格式。

    返回：
        (torch.Tensor | np.ndarray): 裁剪后的坐标。
    """
    h, w = shape[:2]  # 同时支持 HWC 和 HW 尺寸
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14 and not (coords.device.type == "mps" and check_version(TORCH_VERSION, "<2.5.0")):
            coords[..., 0].clamp_(0, w)  # x
            coords[..., 1].clamp_(0, h)  # y
        else:  # macOS 14 或 torch<2.5 存在 MPS 跨步原地操作错误
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:  # np.数组
        coords[..., 0] = coords[..., 0].clip(0, w)  # x
        coords[..., 1] = coords[..., 1].clip(0, h)  # y
    return coords


def xyxy2xywh(x):
    """将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, 宽度, 高度) 格式。

    其中 (x1, y1) 是左上角，(x2, y2) 是右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2)。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 (x, y, 宽度, 高度)。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比克隆或复制更快
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x 中心坐标
    y[..., 1] = (y1 + y2) / 2  # y 中心坐标
    y[..., 2] = x2 - x1  # 宽度
    y[..., 3] = y2 - y1  # 高度
    return y


def xywh2xyxy(x):
    """将边界框坐标从 (x, y, 宽度, 高度) 格式转换为 (x1, y1, x2, y2) 格式。

    其中 (x1, y1) 是左上角，(x2, y2) 是右下角。注意：按两个通道一组执行运算比逐通道运算更快。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x, y, 宽度, 高度)。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 (x1, y1, x2, y2)。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比克隆或复制更快
    xy = x[..., :2]  # 中心坐标
    wh = x[..., 2:] / 2  # 宽度和高度的一半
    y[..., :2] = xy - wh  # 左上角坐标
    y[..., 2:] = xy + wh  # 右下角坐标
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """将归一化边界框坐标转换为像素坐标。

    参数：
        x (np.ndarray | torch.Tensor): 归一化边界框坐标，格式为 (x, y, w, h)。
        w (int): 图像宽度，单位为像素。
        h (int): 图像高度，单位为像素。
        padw (int): 宽度方向的填充，单位为像素。
        padh (int): 高度方向的填充，单位为像素。

    返回：
        (np.ndarray | torch.Tensor): 边界框坐标，格式为 (x1, y1, x2, y2)。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比克隆或复制更快
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    y[..., 0] = w * (xc - half_w) + padw  # 左上角 x
    y[..., 1] = h * (yc - half_h) + padh  # 左上角 y
    y[..., 2] = w * (xc + half_w) + padw  # 右下角 x
    y[..., 3] = h * (yc + half_h) + padh  # 右下角 y
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """将边界框坐标从 (x1, y1, x2, y2) 格式转换为归一化的 (x, y, 宽度, 高度) 格式。

    x、y、宽度和高度均按照图像尺寸进行归一化。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2)。
        w (int): 图像宽度，单位为像素。
        h (int): 图像高度，单位为像素。
        clip (bool): 是否将边界框裁剪到图像边界内。
        eps (float): 边界框宽度和高度的最小值。

    返回：
        (np.ndarray | torch.Tensor): 归一化后的边界框坐标，格式为 (x, y, 宽度, 高度)。
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比克隆或复制更快
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = ((x1 + x2) / 2) / w  # x 中心坐标
    y[..., 1] = ((y1 + y2) / 2) / h  # y 中心坐标
    y[..., 2] = (x2 - x1) / w  # 宽度
    y[..., 3] = (y2 - y1) / h  # 高度
    return y


def xywh2ltwh(x):
    """将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h]，其中 x1、y1 是左上角坐标。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 xywh。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 ltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上角 x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上角 y
    return y


def xyxy2ltwh(x):
    """将边界框从 [x1, y1, x2, y2] 格式转换为 [x1, y1, w, h] 格式。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 xyxy。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 ltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def ltwh2xywh(x):
    """将边界框从 [x1, y1, w, h] 格式转换为 [x, y, w, h] 格式，其中 xy1 表示左上角，xy 表示中心点。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 xywh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # 中心点 x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # 中心点 y
    return y


def xyxyxyxy2xywhr(x):
    """将批量旋转边界框（OBB）从 [xy1, xy2, xy3, xy4] 格式转换为 [xywh, rotation] 格式。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框角点，形状为 (N, 8) 或 (N, 4, 2)，格式为 [xy1, xy2, xy3, xy4]。
            也接受相同两种布局 (N, 2P) 或 (N, P, 2) 的多于四个点的多边形，并将其转换为最小面积矩形。

    返回：
        (np.ndarray | torch.Tensor): 转换后的数据，格式为 [cx, cy, w, h, rotation]，形状为 (N, 5)。
            结果使用规范化参数表示：w 是较长边，rotation 是范围为 [-pi/4, 3pi/4) 的弧度角。
            因此，输入 w < h 的边界框返回时会交换 w 和 h，并将角度按 pi/2 调整（模 pi）。
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    rboxes = []
    for pts in points:
        # 注意：使用 cv2.minAreaRect 获取准确的 xywhr，
        # 尤其适用于数据加载器增强操作裁切了部分目标的情况。
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts.reshape(-1, 2))
        # 将角度转换为弧度，并规范化到 [-pi/4, 3pi/4)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    rboxes = np.asarray(rboxes).reshape(-1, 5)  # reshape 会在输入为空时保留 (0, 5) 形状
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else rboxes


def xywhr2xyxyxyxy(x):
    """将批量旋转边界框（OBB）从 [xywh, rotation] 格式转换为 [xy1, xy2, xy3, xy4] 格式。

    参数：
        x (np.ndarray | torch.Tensor): 输入边界框，格式为 [cx, cy, w, h, rotation]，形状为 (N, 5) 或 (B, N, 5)。
            rotation 使用弧度表示，函数不会检查或规范化其范围；边界框也不会被规范化。
            因此，将 (N, 4, 2) 角点通过 xyxyxyxy2xywhr 转换回来时，返回的是同一矩形的规范化表示，而不是这些原始值。

    返回：
        (np.ndarray | torch.Tensor): 转换后的角点，形状为 (N, 4, 2) 或 (B, N, 4, 2)。
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """将边界框从 [x1, y1, w, h] 格式转换为 [x1, y1, x2, y2] 格式，其中 xy1 是左上角，xy2 是右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标。

    返回：
        (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 xyxy。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # x2
    y[..., 3] = x[..., 3] + x[..., 1]  # y2
    return y


def segments2boxes(segments):
    """将分割段坐标转换为 xywh 格式的边界框标签。

    参数：
        segments (列表): 分割段列表，每个分割段由 [x, y] 坐标点组成。

    返回：
        (np.ndarray): xywh 格式的边界框坐标。
    """
    boxes = []
    for s in segments:
        x, y = s.T  # 分割段的 xy 坐标
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes).reshape(-1, 4))  # xywh


def resample_segments(segments, n: int = 1000):
    """使用线性插值将每条分割线重采样为 n 个点。

    参数：
        segments (列表): 形状为 (N, 2) 的数组列表，其中 N 是每个分割段的点数。
        n (int): 每个分割段重采样后的点数。

    返回：
        (列表): 重采样后的分割段，每个分割段包含 n 个点。
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # 分割段的 xy 坐标
    return segments


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """将掩码裁剪到边界框区域。

    参数：
        masks (torch.Tensor): 形状为 (N, H, W) 的掩码。
        boxes (torch.Tensor): 形状为 (N, 4) 的边界框坐标，使用 xyxy 像素格式。

    返回：
        (torch.Tensor): Cropped 掩码.
    """
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # 每个 形状 (n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # columns (1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # rows (1,h,1)
    # 分别原地应用列掩码和行掩码。由于边界框区域可分离，
    # 这样无需构造完整的 (n, h, w) 布尔网格，也不需要针对每个掩码执行 Python 循环。
    masks *= (r >= x1) * (r < x2)  # 将边界框外的列置零
    masks *= (c >= y1) * (c < y2)  # 将边界框外的行置零
    return masks


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """使用掩码头输出将掩码应用到边界框。

    参数：
        protos (torch.Tensor): 掩码原型，形状为 (mask_dim, mask_h, mask_w)。
        masks_in (torch.Tensor): 掩码系数，形状为 (N, mask_dim)，其中 N 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): 边界框，形状为 (N, 4)，其中 N 是 NMS 后的掩码数量。
        shape (tuple): 输入图像尺寸，格式为 (高度, 宽度)。
        upsample (bool): 是否将掩码上采样到原始图像尺寸。

    返回：
        (torch.Tensor): 形状为 [n, h, w] 的二值掩码张量，其中 n 是 NMS 后的掩码数量。
            当 upsample=True 时，h 和 w 与输入图像尺寸一致；否则使用掩码原型的分辨率。
    """
    c, mh, mw = protos.shape  # CHW
    if masks_in.shape[0] == 0:  # 没有检测结果：下面的 F.interpolate 不接受空的 (N=0) 批次
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.uint8, device=masks_in.device)
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # NHW

    if upsample:
        # 先上采样到图像分辨率再裁剪；先裁剪会使双线性插值边缘扩散到边界框外（#24272）
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]  # NHW
    else:
        width_ratio = mw / shape[1]
        height_ratio = mh / shape[0]
        ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device)
        bboxes = bboxes * ratios  # 将边界框缩放到原型分辨率
    # 先二值化再裁剪，使 crop_mask 在 uint8 而不是 float32 上运行，与 process_mask_native 保持一致
    return crop_mask(masks.gt_(0.0).byte(), bboxes)


def process_mask_native(protos, masks_in, bboxes, shape):
    """使用掩码头输出和原生上采样将掩码应用到边界框。

    参数：
        protos (torch.Tensor): 掩码原型，形状为 (mask_dim, mask_h, mask_w)。
        masks_in (torch.Tensor): 掩码系数，形状为 (N, mask_dim)，其中 N 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): 边界框，形状为 (N, 4)，其中 N 是 NMS 后的掩码数量。
        shape (tuple): 输入图像尺寸，格式为 (高度, 宽度)。

    返回：
        (torch.Tensor): 形状为 (N, H, W) 的二值掩码张量。
    """
    c, mh, mw = protos.shape  # CHW
    h, w = shape
    if masks_in.shape[0] == 0:  # no detections: 返回 a well-formed empty 掩码 stack
        return torch.zeros((0, h, w), dtype=torch.uint8, device=masks_in.device)
    coeffs = masks_in @ protos.float().view(c, -1)  # (N, mh*mw) prototype-resolution 掩码 logits
    # 一次性上采样全部 N 个掩码会分配 N*H*W 大小的浮点中间结果；在包含大量检测的大图像上约占 9 GB，可能导致工作进程内存溢出。
    # 按像素预算分块上采样，并立即将每个分块二值化为 uint8，保持较小的浮点中间结果，最后裁剪拼接后的 uint8 堆栈。
    step = max(1, 32_000_000 // (h * w))
    masks = [
        scale_masks(coeffs[i : i + step].view(-1, mh, mw)[None], shape)[0].gt_(0.0).byte()
        for i in range(0, coeffs.shape[0], step)
    ]
    return crop_mask(torch.cat(masks), bboxes)


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    """将分割掩码缩放到目标尺寸。

    参数：
        masks (torch.Tensor): 形状为 (N, C, H, W) 的掩码。
        shape (tuple[int, int]): 目标尺寸，格式为 (高度, 宽度)。
        ratio_pad (tuple, 可选): 缩放比例和填充值，格式为 ((ratio_h, ratio_w), (pad_w, pad_h))。
        padding (bool): 掩码是否来自带有填充的 YOLO 风格增强图像。
        mode (str): 插值模式，例如 logits 使用 'bilinear'，整数类别图使用 'nearest'。

    返回：
        (torch.Tensor): Rescaled 掩码.
    """
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks
    if masks.shape[1] == 0:  # 空掩码堆栈：F.interpolate 不接受长度为 0 的通道维度
        return masks.new_zeros((*masks.shape[:2], im0_h, im0_w), dtype=torch.float32)

    if ratio_pad is None:  # 根据 im0_shape 计算缩放参数
        gain = min(im1_h / im0_h, im1_w / im0_w)  # 缩放比例 = 旧尺寸 / 新尺寸
        pad_w, pad_h = (im1_w - round(im0_w * gain)), (im1_h - round(im0_h * gain))  # 宽高方向的填充
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(masks[..., top:bottom, left:right].float(), shape, mode=mode)  # NCHW 掩码


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """将分割线坐标从 img1_shape 缩放到 img0_shape。

    参数：
        img1_shape (tuple): 源图像尺寸，可为 HWC 或 HW 格式。
        coords (torch.Tensor): 要缩放的坐标，形状为 (N, 2)。
        img0_shape (tuple): 目标图像尺寸，可为 HWC 或 HW 格式。
        ratio_pad (tuple, 可选): 缩放比例和填充值，格式为 ((ratio_h, ratio_w), (pad_w, pad_h))。
        normalize (bool): 是否将坐标归一化到 [0, 1] 范围。
        padding (bool): 坐标是否来自带有填充的 YOLO 风格增强图像。

    返回：
        (torch.Tensor): 缩放后的坐标。
    """
    img0_h, img0_w = img0_shape[:2]  # 同时支持 HWC 和 HW 尺寸
    if ratio_pad is None:  # 根据 img0_shape 计算缩放参数
        img1_h, img1_w = img1_shape[:2]  # 同时支持 HWC 和 HW 尺寸
        gain = min(img1_h / img0_h, img1_w / img0_w)  # 缩放比例 = 旧尺寸 / 新尺寸
        gain_y = gain_x = gain
        pad = round((img1_w - round(img0_w * gain)) / 2 - 0.1), round((img1_h - round(img0_h * gain)) / 2 - 0.1)
    else:
        gain_y, gain_x = ratio_pad[0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x 填充
        coords[..., 1] -= pad[1]  # y 填充
    coords[..., 0] /= gain_x
    coords[..., 1] /= gain_y
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w  # 宽度
        coords[..., 1] /= img0_h  # 高度
    return coords


def regularize_rboxes(rboxes):
    """将旋转边界框的角度规范化到 [0, pi/2) 范围。

    参数：
        rboxes (torch.Tensor): 输入的旋转边界框，采用 xywhr 格式，形状为 (N, 5)。

    返回：
        (torch.Tensor): Regularized rotated 边界框.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # 当 t >= pi/2 且不是中心对称的相反边时交换宽高
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized 边界框


def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]:
    """使用轮廓检测将掩码转换为分割段。

    参数：
        masks (np.ndarray | torch.Tensor): 形状为 (N, H, W) 的二值掩码。
        strategy (str): 分割策略，可选 'all'（全部轮廓）或 'largest'（最大轮廓）。

    返回：
        (列表): float32 数组格式的分割段列表。
    """
    from ultralytics.data.converter import merge_multi_segment

    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # 合并并拼接所有分割段
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # 选择最大的分割段
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # 未找到分割段
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """将 FP32 torch 张量批次转换为 NumPy uint8 数组，并将布局从 BCHW 改为 BHWC。

    参数：
        batch (torch.Tensor): 输入张量批次，形状为 (批次, 通道, 高度, 宽度)，数据类型为 torch.float32。

    返回：
        (np.ndarray): 输出的 NumPy 数组批次，形状为 (批次, 高度, 宽度, 通道)，数据类型为 uint8。
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).byte().cpu().numpy()


def clean_str(s):
    """通过将特殊字符替换为 '_' 清理字符串。

    参数：
        s (str): 需要替换特殊字符的字符串。

    返回：
        (str): 已将特殊字符替换为下划线的字符串。
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)


def empty_like(x):
    """创建与输入具有相同形状和数据类型的空 torch.Tensor 或 np.ndarray。"""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)


_assignment_solver = None  # 首次调用时确定：优先使用已安装的 SciPy 求解器，否则使用 NumPy 回退实现


def linear_sum_assignment(cost_matrix):
    """求解矩形线性和分配问题（最小代价的一对一匹配）。

    安装 SciPy 时使用 `scipy.optimize.linear_sum_assignment`（更快的编译版 C++ 求解器），否则使用等价的纯 NumPy
    实现。两者都采用改进的 Jonker-Volgenant 最短增广路径算法（Crouse 2016）。这样既不将 SciPy 加入 Ultralytics
    的必需依赖，又能在其存在时保留更高速度。SciPy 采用延迟导入，因此不会拖慢 `import ultralytics`。
    对于矩形矩阵，只匹配 min(行数, 列数) 个元素。

    NumPy 回退实现支持使用 `+inf` 表示禁止分配；当不存在可行分配时会抛出 `ValueError("cost matrix is infeasible")`。
    调用方必须先处理 `NaN` 和 `-inf`。两个后端在代价完全相同的并列情况下可能返回不同的分配结果，但总成本相同。

    NumPy 回退实现已通过约 6,900 个随机测试用例与 SciPy 进行验证，最优成本完全一致（涵盖空矩阵、高矩阵、宽矩阵、
    并列值、负值、IoU 和 RT-DETR 风格矩阵、通过取负实现的 `maximize` 以及 torch 张量输入），另有约 2,000 个独立的
    暴力全局最优检查。SciPy 的编译内循环更快，但在调用点规模下（较小的维度等于目标数量），回退实现通常远低于 1 毫秒：

        成本矩阵       NumPy   SciPy
        300 x 20      0.2ms   0.02ms
        300 x 80      0.6ms   0.1ms
        300 x 300     28ms    1.5ms

    参数：
        cost_matrix (np.ndarray | torch.Tensor): 形状为 (N, M) 的成本矩阵；`+inf` 表示禁止分配。

    返回：
        row_ind (np.ndarray): 最优分配的行索引，按升序排列，长度为 min(N, M)。
        col_ind (np.ndarray): 与 row_ind 中每一行匹配的列索引。

    示例：
        >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
        >>> row_ind, col_ind = linear_sum_assignment(cost)
        >>> float(cost[row_ind, col_ind].sum())
        5.0
    """
    global _assignment_solver
    if _assignment_solver is None:  # 仅首次确定后端，后续调用复用该后端
        try:
            from scipy.optimize import linear_sum_assignment as solver  # 安装 SciPy 时使用更快的编译版 C++ 求解器

            _assignment_solver = solver
        except ImportError:
            _assignment_solver = _linear_sum_assignment_numpy
    return _assignment_solver(np.asarray(cost_matrix, dtype=np.float64))


def _linear_sum_assignment_numpy(a):
    """使用 NumPy 求解矩形线性和分配问题（无 SciPy 的 Jonker-Volgenant 回退实现）。

    参数：
        a (np.ndarray): Float64 cost matrix of 形状 (N, M); `+inf` forbids assignments.

    返回：
        row_ind (np.ndarray): 最优分配的行索引，按升序排列，长度为 min(N, M)。
        col_ind (np.ndarray): Column 索引 matched to 每个 row in row_ind.
    """
    n, m = a.shape
    if n == 0 or m == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    transposed = n > m
    if transposed:
        a, n, m = a.T, m, n  # ensure rows <= columns
    u, v = np.zeros(n + 1), np.zeros(m + 1)  # 行和列的对偶势
    p, way = np.zeros(m + 1, np.intp), np.zeros(m + 1, np.intp)  # 列到行的匹配和路径指针
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
        while True:  # 从第 i 行扩展最短增广路径
            used[j0] = True
            i0 = p[j0]
            cur = a[i0 - 1] - u[i0] - v[1:]
            improve = (~used[1:]) & (cur < minv[1:])
            minv[1:][improve], way[1:][improve] = cur[improve], j0
            candidates = np.where(used[1:], np.inf, minv[1:])
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            if delta == np.inf:
                raise ValueError("cost matrix is infeasible")
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:  # 沿路径进行增广
            p[j0] = p[way[j0]]
            j0 = way[j0]
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    row_ind, col_ind = (cols, rows) if transposed else (rows, cols)
    order = np.argsort(row_ind, kind="stable")  # match scipy's row-sorted 输出
    return row_ind[order].astype(np.intp), col_ind[order].astype(np.intp)
