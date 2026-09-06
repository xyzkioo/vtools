# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class TrackZone(BaseSolution):
    """管理视频流中基于区域的对象跟踪的类。

    此类扩展 BaseSolution，在多边形区域定义的指定范围内跟踪对象，并排除区域外对象。

    属性：
        region (np.ndarray): 用点的凸包表示的跟踪多边形区域。
        line_width (int): 绘制边界框和区域边界所使用的线宽。
        names (dict[int, str]): 类别索引到类别名称的映射。
        boxes (列表[np.ndarray]): 跟踪对象的边界框。
        track_ids (列表[int]): 每个跟踪对象的唯一标识符。
        clss (列表[int]): 跟踪对象的类别索引。

    方法：
        process: 处理视频的每一帧并应用基于区域的跟踪。
        extract_tracks: 从输入帧中提取跟踪信息。
        display_output: 显示处理后的输出。

    示例：
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = tracker.process(frame)
        >>> cv2.imshow("Tracked Frame", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 TrackZone 类，用于跟踪视频流指定区域内的对象。

        参数：
            **kwargs (Any): 传递给父类的其他关键字参数。
        """
        super().__init__(**kwargs)
        default_region = [(75, 75), (565, 75), (565, 285), (75, 285)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))
        self.mask = None

    def process(self, im0: np.ndarray) -> SolutionResults:
        """处理输入帧，跟踪指定区域内的对象。

        此方法初始化标注器，为指定区域创建掩码，仅从掩码区域提取跟踪结果并更新跟踪信息，区域外对象会被忽略。

        参数：
            im0 (np.ndarray): 要处理的输入图像或帧。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im` 和 `total_tracks`，后者表示指定区域内跟踪对象的总数。

        示例：
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = tracker.process(frame)
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        if self.mask is None:  # 为区域创建掩码
            self.mask = np.zeros_like(im0[:, :, 0])
            cv2.fillPoly(self.mask, [self.region], 255)
        masked_frame = cv2.bitwise_and(im0, im0, mask=self.mask)
        self.extract_tracks(masked_frame)

        # 绘制区域边界
        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # 遍历边界框、跟踪 ID 和类别索引列表，并绘制边界框
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, track_id=track_id), color=colors(track_id, True)
            )

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
