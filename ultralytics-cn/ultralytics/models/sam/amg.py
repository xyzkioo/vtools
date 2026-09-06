# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections.abc import Generator
from itertools import product
from typing import Any

import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: list[int], orig_box: list[int], atol: float = 20.0
) -> torch.Tensor:
    """根据指定容差判断边界框是否靠近裁剪图像区域的边缘。

    参数：
        boxes (torch.Tensor): XYXY 格式的边界框。
        crop_box (列表[int]): 裁剪边界框坐标，格式为 [x0, y0, x1, y1]。
        orig_box (列表[int]): 原始图像边界框坐标，格式为 [x0, y0, x1, y1]。
        atol (float, 可选): 判断边缘接近程度的绝对容差。

    返回：
        (torch.Tensor): 表示哪些边界框靠近裁剪边缘的布尔张量。

    示例：
        >>> boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]])
        >>> crop_box = [0, 0, 200, 200]
        >>> orig_box = [0, 0, 300, 300]
        >>> near_edge = is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0)
    """
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def batch_iterator(batch_size: int, *args) -> Generator[list[Any]]:
    """以指定批次大小从输入参数中生成数据批次，以提高处理效率。

    此函数接收批次大小和任意数量的可迭代对象，然后从这些可迭代对象中生成元素批次。
    所有输入可迭代对象的长度必须相同。

    参数：
        batch_size (int): 要生成的每个批次大小。
        *args (Any): 要分批处理的可变长度输入可迭代对象，所有对象长度必须相同。

    Yields:
        (列表[Any]): 从每个输入可迭代对象中生成的分批元素列表。

    示例：
        >>> data = [1, 2, 3, 4, 5]
        >>> labels = ["a", "b", "c", "d", "e"]
        >>> for batch in batch_iterator(2, data, labels):
        ...     print(batch)
        [[1, 2], ['a', 'b']]
        [[3, 4], ['c', 'd']]
        [[5], ['e']]
    """
    assert args and all(len(a) == len(args[0]) for a in args), "Batched iteration must have same-size inputs."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """计算一个掩码批次的稳定性分数。

    稳定性分数是二值掩码之间的 IoU，这些二值掩码分别由高阈值和低阈值对预测掩码 logits 进行阈值化得到。

    参数：
        masks (torch.Tensor): 预测掩码 logits 批次。
        mask_threshold (float): 创建二值掩码的阈值。
        threshold_offset (float): 用于生成高、低阈值二值掩码的偏移量。

    返回：
        (torch.Tensor): 批次中每个掩码的稳定性分数。

    示例：
        >>> masks = torch.rand(10, 256, 256)  # Batch of 10 masks
        >>> mask_threshold = 0.5
        >>> threshold_offset = 0.1
        >>> stability_scores = calculate_stability_score(masks, mask_threshold, threshold_offset)

    注意：
        - 一个掩码始终包含在另一个掩码内。
        - 通过避免不必要地转换为 torch.int64 来节省内存。
    """
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """为图像分割任务生成 [0,1]x[0,1] 范围内均匀分布的二维点网格。"""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> list[np.ndarray]:
    """为多个裁剪层生成具有不同尺度和密度的点网格。"""
    return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers + 1)]


def generate_crop_boxes(
    im_size: tuple[int, ...], n_layers: int, overlap_ratio: float
) -> tuple[list[list[int]], list[int]]:
    """为多尺度图像处理生成不同大小的裁剪边界框，并设置分层重叠区域。

    参数：
        im_size (tuple[int, ...]): 输入图像的高度和宽度。
        n_layers (int): 要生成裁剪边界框的层数。
        overlap_ratio (float): 相邻裁剪边界框之间的重叠比例。

    返回：
        crop_boxes (列表[列表[int]]): [x0, y0, x1, y1] 格式的裁剪边界框列表。
        layer_idxs (列表[int]): 与每个裁剪边界框对应的层索引列表。

    示例：
        >>> im_size = (800, 1200)  # Height, width
        >>> n_layers = 3
        >>> overlap_ratio = 0.25
        >>> crop_boxes, layer_idxs = generate_crop_boxes(im_size, n_layers, overlap_ratio)
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # 原始图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """根据原始长度、裁剪数量和重叠比例计算每个裁剪区域的长度。"""
        return math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops)

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # 裁剪框采用 XYWH 格式计算
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: list[int]) -> torch.Tensor:
    """将裁剪边界框偏移量加回坐标，从而恢复边界框在原图中的位置。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # 检查边界框是否包含通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: list[int]) -> torch.Tensor:
    """将裁剪边界框偏移量加回点坐标，从而恢复点在原图中的位置。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # 检查点是否包含通道维度
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(masks: torch.Tensor, crop_box: list[int], orig_h: int, orig_w: int) -> torch.Tensor:
    """通过填充掩码恢复到原始图像尺寸，并处理坐标变换。"""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # 对掩码执行坐标变换
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> tuple[np.ndarray, bool]:
    """根据面积阈值和模式移除掩码中的小型断开区域或孔洞。

    参数：
        mask (np.ndarray): 要处理的二值掩码。
        area_thresh (float): 小于此面积阈值的区域将被移除。
        mode (str): 处理模式，可选 'holes'（填充小孔洞）或 'islands'（移除小型断开区域）。

    返回：
        processed_mask (np.ndarray): 移除小区域后的二值掩码。
        modified (bool): 是否修改了任意区域。

    示例：
        >>> mask = np.zeros((100, 100), dtype=np.bool_)
        >>> mask[40:60, 40:60] = True  # Create a square
        >>> mask[45:55, 45:55] = False  # Create a hole
        >>> processed_mask, modified = remove_small_regions(mask, 50, "holes")
    """
    import cv2  # type: ignore

    assert mode in {"holes", "islands"}, f"Provided mode {mode} is invalid"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0, *small_regions]
    if not correct_holes:
        # 如果所有区域都低于阈值，则保留最大的区域
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """计算包围二值掩码的 XYXY 格式边界框。

    参数：
        masks (torch.Tensor): 形状为 (B, H, W) 或 (B, C, H, W) 的二值掩码。

    返回：
        (torch.Tensor): XYXY 格式的边界框，形状为 (B, 4) 或 (B, C, 4)。

    注意：
        - 空掩码将返回全零边界框。
        - 输出会保留输入张量的维度。
    """
    # torch.max 在空输入上会报错，因此此时直接跳过
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # 将形状统一为 CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    # 获取顶部和底部边缘
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左侧和右侧边缘
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩码为空，右边缘会位于左边缘左侧。
    # 将这些边界框替换为 [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # 恢复为原始形状
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
