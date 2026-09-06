# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class DistanceCalculation(BaseSolution):
    """根据跟踪结果计算实时视频流中两个对象之间距离的类。

    此类扩展 BaseSolution，使用 YOLO 目标检测和跟踪功能，在视频流中选择对象并计算它们之间的距离。

    属性：
        left_mouse_count (int): 鼠标左键点击计数器。
        selected_boxes (dict[int, Any]): 以跟踪 ID 为键保存所选边界框的字典。
        centroids (列表[列表[int]]): 保存所选边界框中心点的列表。

    方法：
        mouse_event_for_distance: 处理视频流中选择对象的鼠标事件。
        process: 处理视频帧并计算所选对象之间的距离。

    示例：
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = distance_calc.process(frame)
        >>> cv2.imshow("Distance Calculation", results.plot_im)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 DistanceCalculation 类，用于测量视频流中对象之间的距离。"""
        super().__init__(**kwargs)

        # 鼠标事件信息
        self.left_mouse_count = 0
        self.selected_boxes: dict[int, list[float]] = {}
        self.centroids: list[list[int]] = []  # 保存所选对象的中心点

    def mouse_event_for_distance(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """处理实时视频流中的鼠标事件，以选择区域并计算距离。

        参数：
            event (int): 鼠标事件类型（例如 cv2.EVENT_MOUSEMOVE、cv2.EVENT_LBUTTONDOWN）。
            x (int): 鼠标指针的 X 坐标。
            y (int): 鼠标指针的 Y 坐标。
            flags (int): 与事件关联的标志（例如 cv2.EVENT_FLAG_CTRLKEY、cv2.EVENT_FLAG_SHIFTKEY）。
            param (Any): 传递给函数的其他参数。

        示例：
            >>> # 假设 dc 是 DistanceCalculation 的实例
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    x0, y0, x1, y1 = self.get_enclosing_box(box)
                    if x0 < x < x1 and y0 < y < y1 and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def process(self, im0) -> SolutionResults:
        """处理视频帧，并计算两个所选边界框之间的距离。

        此方法从输入帧中提取跟踪结果，为边界框添加标注，并在用户选择了两个对象时计算它们之间的距离。

        参数：
            im0 (np.ndarray): 要处理的输入图像帧。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、表示跟踪对象总数的 `total_tracks`，以及以像素为单位
                表示所选对象之间距离的 `pixels_distance`。

        示例：
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = dc.process(frame)
            >>> print(f"Distance: {results.pixels_distance:.2f} pixels")
        """
        self.extract_tracks(im0)  # 提取跟踪结果
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        pixels_distance = 0
        # 遍历边界框、跟踪 ID 和类别索引
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, color=colors(int(cls), True), label=self.adjust_box_label(cls, conf, track_id))

            # 如果所选边界框仍在跟踪，则更新它
            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes:
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # 计算所选边界框的中心点
            self.centroids.extend(
                [
                    [int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)]
                    for box in map(self.get_enclosing_box, self.selected_boxes.values())
                ]
            )
            # 计算中心点之间的欧氏距离
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []  # 为下一帧重置中心点
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出
        if self.CFG.get("show") and self.env_check:
            cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        # 返回包含处理后图像和计算指标的 SolutionResults
        return SolutionResults(plot_im=plot_im, pixels_distance=pixels_distance, total_tracks=len(self.track_ids))
