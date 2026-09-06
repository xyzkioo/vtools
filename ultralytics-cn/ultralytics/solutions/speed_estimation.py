# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque
from math import sqrt
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class SpeedEstimator(BaseSolution):
    """根据跟踪结果估计实时视频流中对象速度的类。

    此类继承 BaseSolution，利用视频流中的跟踪数据估计对象速度。速度根据一段时间内的像素位移计算，
    再通过可配置的米/像素比例转换为现实世界中的速度单位。

    属性：
        fps (float): 用于时间计算的视频帧率。
        frame_count (int): 用于记录时间信息的全局帧计数器。
        trk_frame_ids (dict): 将跟踪 ID 映射到其起始帧索引。
        spd (dict): 速度锁定后各对象的最终速度，单位为 km/h。
        trk_hist (dict): 将跟踪 ID 映射到位置历史队列。
        locked_ids (set): 速度已经确定的跟踪 ID 集合。
        max_hist (int): 计算速度前所需保留的最大帧历史长度。
        meter_per_pixel (float): 场景比例，即一个像素代表的现实世界米数。
        max_speed (int): 允许的对象最大速度，超过该值时会被截断。

    方法：
        process: 根据跟踪数据处理输入帧并估计对象速度。
        store_tracking_history: 保存对象的跟踪历史。
        extract_tracks: 从当前帧提取跟踪结果。
        display_output: 显示带标注的输出。

    示例：
        初始化速度估计器并处理一帧图像。
        >>> estimator = SpeedEstimator(meter_per_pixel=0.04, max_speed=120)
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("Speed Estimation", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用速度估计参数和数据结构初始化 SpeedEstimator 对象。

        参数：
            **kwargs (Any): 传递给父类的其他关键字参数。
        """
        super().__init__(**kwargs)

        self.fps = self.CFG["fps"]  # 用于时间计算的视频帧率
        self.frame_count = 0  # 全局帧计数器
        self.trk_frame_ids = {}  # 跟踪 ID → 首帧索引
        self.spd = {}  # 速度锁定后各对象的最终速度（km/h）
        self.trk_hist = {}  # 跟踪 ID 到位置历史队列的映射
        self.locked_ids = set()  # 速度已经确定的跟踪 ID
        self.max_hist = self.CFG["max_hist"]  # 计算速度前所需的帧历史长度
        self.meter_per_pixel = self.CFG["meter_per_pixel"]  # 场景比例，取决于摄像头参数
        self.max_speed = self.CFG["max_speed"]  # 速度上限

    def forget_tracks(self, track_ids):
        """从速度记录中清理已退出的 ID，避免全天候视频流运行时记录无限增长（参见 BaseSolution）。"""
        super().forget_tracks(track_ids)
        for track_id in track_ids:
            self.trk_hist.pop(track_id, None)
            self.trk_frame_ids.pop(track_id, None)
            self.spd.pop(track_id, None)
            self.locked_ids.discard(track_id)

    def process(self, im0) -> SolutionResults:
        """根据跟踪数据处理输入帧并估计对象速度。

        参数：
            im0 (np.ndarray): 要处理的输入图像，形状为 (H, W, C)，格式为 OpenCV BGR。

        返回：
            (SolutionResults): 包含处理后图像 `plot_im` 和 `total_tracks`（跟踪对象数量）。

        示例：
            Process a frame for speed estimation
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, _, _ in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:  # 发现新跟踪对象时初始化历史记录
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
                self.trk_frame_ids[track_id] = self.frame_count

            if track_id not in self.locked_ids:  # 在速度锁定前持续更新历史记录
                trk_hist = self.trk_hist[track_id]
                trk_hist.append(self.track_line[-1])

                # 收集到足够历史记录后计算并锁定速度
                if len(trk_hist) == self.max_hist:
                    p0, p1 = trk_hist[0], trk_hist[-1]  # 跟踪轨迹的起点和终点
                    dt = (self.frame_count - self.trk_frame_ids[track_id]) / self.fps  # 时间，单位为秒
                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]  # 像素位移
                        pixel_distance = sqrt(dx * dx + dy * dy)  # 计算像素距离
                        meters = pixel_distance * self.meter_per_pixel  # 转换为米
                        self.spd[track_id] = int(
                            min((meters / dt) * 3.6, self.max_speed)
                        )  # 转换为 km/h 并保存最终速度
                        self.locked_ids.add(track_id)  # 防止后续更新
                        self.trk_hist.pop(track_id, None)  # 释放内存
                        self.trk_frame_ids.pop(track_id, None)  # 删除起始帧记录

            if track_id in self.spd:
                speed_label = f"{self.spd[track_id]} km/h"
                annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边界框

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类方法显示输出

        # 返回包含处理后图像和跟踪摘要的结果
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
