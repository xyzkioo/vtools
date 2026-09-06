# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """根据跟踪结果管理实时视频流中对象计数的类。

    此类扩展 BaseSolution，提供统计视频流中进出指定区域对象数量的功能，同时支持多边形区域和线性区域计数。

    属性：
        in_count (int): 向内移动对象的计数。
        out_count (int): 向外移动对象的计数。
        counted_ids (set[int]): 已计数对象的 ID 集合。
        classwise_count (dict[str, dict[str, int]]): 按对象类别统计数量的字典。
        region_initialized (bool): 表示计数区域是否已初始化的标志。
        show_in (bool): 是否显示向内计数。
        show_out (bool): 是否显示向外计数。
        margin (int): 用于正确显示计数的背景矩形边距。

    方法：
        count_objects: 根据跟踪结果统计多边形或线性区域内的对象。
        display_counts: 在帧上显示对象计数。
        process: 处理输入数据并更新计数。

    示例：
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = counter.process(frame)
        >>> print(f"Inward count: {results.in_count}, Outward count: {results.out_count}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 ObjectCounter 类，用于实时视频流中的对象计数。"""
        super().__init__(**kwargs)

        self.in_count = 0  # 向内移动对象的计数
        self.out_count = 0  # 向外移动对象的计数
        self.counted_ids = set()  # 已计数对象的 ID
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # 按类别统计数量
        self.region_initialized = False  # 区域初始化标志

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2  # 调整背景矩形边距以正确显示计数

    def count_objects(
        self,
        current_centroid: tuple[float, float],
        track_id: int,
        prev_position: tuple[float, float] | None,
        cls: int,
    ) -> None:
        """根据跟踪结果统计多边形或线性区域内的对象。

        参数：
            current_centroid (tuple[float, float]): 当前帧中的中心点坐标 `(x, y)`。
            track_id (int): 跟踪对象的唯一标识符。
            prev_position (tuple[float, float], 可选): 跟踪对象上一帧的位置坐标 `(x, y)`。
            cls (int): 用于更新分类计数的类别索引。

        示例：
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id_num = 1
            >>> previous_position = (120, 220)
            >>> class_to_count = 0  # 在 COCO 模型中，类别 0 表示人
            >>> counter.count_objects((140, 240), track_id_num, previous_position, class_to_count)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:  # 线性区域（由线段定义）
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                # 判断区域方向（垂直或水平）
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # 垂直区域：比较 x 坐标判断方向
                    if current_centroid[0] > prev_position[0]:  # 向右移动
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:  # 向左移动
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                # 水平区域：比较 y 坐标判断方向
                elif current_centroid[1] > prev_position[1]:  # 向下移动
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # 向上移动
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.add(track_id)

        # 移动过快的对象可能跨过区域而没有中心点落在区域内，因此也要统计穿越线段的情况；
        # 但只从区域外开始统计，因为已经在区域内的跟踪对象自进入后一直执行包含检查。
        elif len(self.region) > 2 and (
            self.r_s.contains(self.Point(current_centroid))
            or (
                not self.r_s.contains(self.Point(prev_position))
                and self.r_s.crosses(self.LineString([prev_position, current_centroid]))
            )
        ):
            # 根据对象近期跟踪轨迹中的主运动轴判断方向，而不是根据区域形状判断；约 5 帧的基线
            # 比单帧差分更能抵抗跟踪抖动。基线取区域外最近历史点中最早的一个，避免对象在区域内生成时
            # 未计数的第一帧污染进入向量（快速离开后重新进入的情况）。
            window = self.track_history[track_id][-5:] or [prev_position]
            baseline = next((p for p in window if not self.r_s.contains(self.Point(p))), window[0])
            dx = current_centroid[0] - baseline[0]
            dy = current_centroid[1] - baseline[1]
            moving_in = dx > 0 if abs(dx) > abs(dy) else dy > 0  # 向右或向下移动
            if moving_in:
                self.in_count += 1
                self.classwise_count[self.names[cls]]["IN"] += 1
            else:  # 向左或向上移动
                self.out_count += 1
                self.classwise_count[self.names[cls]]["OUT"] += 1
            self.counted_ids.add(track_id)

    def forget_tracks(self, track_ids: list[int]) -> None:
        """从 `counted_ids` 中移除已结束的 ID，避免全天候视频流中的集合无限增长（参见 BaseSolution）。"""
        super().forget_tracks(track_ids)
        self.counted_ids.difference_update(track_ids)

    def display_counts(self, plot_im) -> None:
        """在输入图像或帧上显示对象计数。

        参数：
            plot_im (np.ndarray): 要显示计数的图像或帧。

        示例：
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if (value["IN"] != 0 and self.show_in) or (value["OUT"] != 0 and self.show_out)
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0) -> SolutionResults:
        """处理输入数据（帧或对象跟踪结果）并更新对象计数。

        此方法初始化计数区域、提取跟踪结果、绘制边界框和区域、更新对象计数，并在输入图像上显示结果。

        参数：
            im0 (np.ndarray): 要处理的输入图像或帧。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、进入区域对象数 `in_count`、离开区域对象数 `out_count`、
                按类别统计的对象数量 `classwise_count` 和跟踪对象总数 `total_tracks`。

        示例：
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = counter.process(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  # 提取跟踪结果
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # 绘制区域

        # 遍历边界框、跟踪 ID 和类别索引
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 绘制边界框并处理计数区域
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # 保存跟踪历史

            # 保存跟踪对象上一位置，用于对象计数
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # 对象 counting

        plot_im = self.annotator.result()
        self.display_counts(plot_im)  # 在帧上显示计数
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
