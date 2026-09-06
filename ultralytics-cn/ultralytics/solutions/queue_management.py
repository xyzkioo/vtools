# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class QueueManager(BaseSolution):
    """根据对象跟踪结果管理实时视频流中的队列计数。

    此类扩展 BaseSolution，提供在视频帧指定区域内跟踪和统计对象的功能。

    属性：
        counts (int): 队列中当前对象数量。
        rect_color (tuple[int, int, int]): 绘制队列区域矩形的 BGR 颜色元组。
        region_length (int): 定义队列区域的点数量。
        track_line (列表[tuple[int, int]]): 跟踪线坐标列表。
        track_history (dict[int, 列表[tuple[int, int]]]): 保存每个对象跟踪历史的字典。

    方法：
        initialize_region: 初始化队列区域。
        process: 处理单帧图像以管理队列。
        extract_tracks: 从当前帧提取对象跟踪结果。
        store_tracking_history: 保存对象的跟踪历史。
        display_output: 显示处理后的输出。

    示例：
        >>> cap = cv2.VideoCapture("path/to/video.mp4")
        >>> queue_manager = QueueManager(region=[(20, 400), (1080, 400), (1080, 360), (20, 360)])
        >>> while cap.isOpened():
        ...     success, im0 = cap.read()
        ...     if not success:
        ...         break
        ...     results = queue_manager.process(im0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用视频流中跟踪和统计对象所需的参数初始化 QueueManager。"""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # 队列计数信息
        self.rect_color = (255, 255, 255)  # 可视化矩形颜色
        self.region_length = len(self.region)  # 保存区域长度供后续使用

    def process(self, im0) -> SolutionResults:
        """处理单帧视频图像以管理队列。

        参数：
            im0 (np.ndarray): 要处理的输入图像，通常是视频流中的一帧。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、队列中的对象数量 `queue_count` 和跟踪对象总数 `total_tracks`。

        示例：
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = queue_manager.process(frame)
        """
        self.counts = 0  # 每帧重置计数
        self.extract_tracks(im0)  # 从当前帧提取跟踪结果
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)  # 绘制区域

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 绘制边界框并处理计数区域
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # 保存跟踪历史

            # 缓存频繁访问的属性
            track_history = self.track_history.get(track_id, [])

            # 保存对象上一位置，并检查对象是否位于计数区域内
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # 显示队列计数
        annotator.queue_counts_display(
            f"Queue Counts : {self.counts}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回包含处理后数据的 SolutionResults 对象
        return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
