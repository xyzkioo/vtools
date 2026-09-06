# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults


class Heatmap(ObjectCounter):
    """根据对象跟踪结果在实时视频流中绘制热力图的类。

    此类扩展 ObjectCounter，生成并可视化视频流中的对象运动热力图，利用跟踪对象的位置随时间累积热力图效果。

    属性：
        initialized (bool): 表示热力图是否已初始化的标志。
        colormap (int): 用于可视化热力图的 OpenCV 颜色映射。
        heatmap (np.ndarray): 保存累计热力图数据的数组。
        annotator (SolutionAnnotator): 用于在图像上绘制标注的对象。

    方法：
        heatmap_effect: 计算并更新给定边界框的热力图效果。
        process: 为每个视频帧生成并应用热力图效果。

    示例：
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolo26n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """根据对象跟踪结果初始化 Heatmap 类，用于实时视频流热力图生成。

        参数：
            **kwargs (Any): 传递给父类 ObjectCounter 的关键字参数。
        """
        super().__init__(**kwargs)

        self.initialized = False  # 热力图初始化标志
        if self.region is not None:  # 检查用户是否提供区域坐标
            self.initialize_region()

        # 保存颜色映射
        self.colormap = self.CFG["colormap"]
        self.heatmap = None

    def heatmap_effect(self, box: torch.Tensor) -> None:
        """高效计算热力图区域和效果位置，以应用颜色映射。

        参数：
            box (torch.Tensor): 边界框坐标 `[x0, y0, x1, y1]`，或形状为 `(4, 2)` 的 OBB 角点。
        """
        x0, y0, x1, y1 = map(int, self.get_enclosing_box(box))
        h, w = self.heatmap.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)  # 将 OBB 角点裁剪到图像边界内
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # 创建感兴趣区域（ROI）的网格，以便向量化计算距离
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # 计算各点到中心的平方距离
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # 创建半径范围内点的掩码
        within_radius = dist_squared <= radius_squared

        # 通过一次向量化操作，仅更新边界框内的值
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def process(self, im0: np.ndarray) -> SolutionResults:
        """使用 Ultralytics 跟踪功能为每个视频帧生成热力图。

        参数：
            im0 (np.ndarray): 要处理的输入图像数组。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、进入区域的对象数 `in_count`、离开区域的对象数
                `out_count`、按类别统计的对象数量 `classwise_count` 以及跟踪对象总数 `total_tracks`。
        """
        if not self.initialized:
            self.heatmap = np.zeros(im0.shape[:2], dtype=np.float32)
            self.initialized = True  # 仅初始化一次热力图

        self.extract_tracks(im0)  # 提取跟踪结果
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        if self.region is not None:
            self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        # 遍历边界框、跟踪 ID 和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # 为边界框应用热力图效果
            self.heatmap_effect(box)

            if self.region is not None:
                self.store_tracking_history(track_id, box)  # 保存跟踪历史
                # 获取上一位置（如果可用）
                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # 统计对象数量

        plot_im = self.annotator.result()
        if self.region is not None:
            self.display_counts(plot_im)  # 在帧上显示计数

        # 对热力图归一化、应用颜色映射，并与原始图像合并
        if self.heatmap.any():
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)
            plot_im = cv2.addWeighted(plot_im, 0.5, colored_heatmap, 0.5, 0)

        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
