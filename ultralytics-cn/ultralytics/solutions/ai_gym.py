# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """根据人体姿态管理实时视频流中健身动作的类。

    此类扩展 BaseSolution，使用 YOLO 姿态估计模型监控健身动作，并根据预设的上下姿态角度阈值跟踪和计数动作重复次数。

    属性：
        states (dict[int, dict[str, float | int | str]]): 每个跟踪对象的角度、重复次数和动作阶段。
        up_angle (float): 判定动作处于“向上”位置的角度阈值。
        down_angle (float): 判定动作处于“向下”位置的角度阈值。
        kpts (列表[int]): 用于计算角度的关键点索引。

    方法：
        process: 处理视频帧、检测姿态、计算角度并统计重复次数。

    示例：
        >>> gym = AIGym(model="yolo26n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用姿态估计和预设角度初始化 AIGym，以监控健身动作。

        参数：
            **kwargs (Any): 传递给父类构造函数的关键字参数，包括：
                - model (str): 模型名称或路径，默认为 "yolo26n-pose.pt"。
        """
        kwargs["model"] = kwargs.get("model", "yolo26n-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})  # 计数、角度和阶段字典

        # 从 CFG 中提取配置，后续重复使用
        self.up_angle = float(self.CFG["up_angle"])  # 判定向上姿态的预设角度
        self.down_angle = float(self.CFG["down_angle"])  # 判定向下姿态的预设角度
        self.kpts = self.CFG["kpts"]  # 用户选择的健身动作关键点

    def forget_tracks(self, track_ids):
        """从健身状态中移除已结束的 ID，避免全天候视频流中的状态无限增长（参见 BaseSolution）。"""
        super().forget_tracks(track_ids)
        for track_id in track_ids:
            self.states.pop(track_id, None)

    def process(self, im0) -> SolutionResults:
        """使用 Ultralytics YOLO 姿态模型监控健身动作。

        此函数处理输入图像，跟踪并分析人体姿态以监控健身动作。它使用 YOLO 姿态模型检测关键点、估计角度，
        并根据预设角度阈值统计重复次数。

        参数：
            im0 (np.ndarray): 要处理的输入图像。

        返回：
            (SolutionResults): 包含处理后图像 `plot_im`、'workout_count'（每个当前跟踪对象已完成重复次数的列表）、
                'workout_stage'（当前阶段列表）、'workout_angle'（当前角度列表）以及 'total_tracks'（当前跟踪对象总数）。
                各跟踪对象列表与当前可见轨迹保持对齐，因此 ``len(结果.workout_count) == 结果.total_tracks``。

        示例：
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化 annotator

        self.extract_tracks(im0)  # 提取轨迹（边界框、类别和掩码）

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data.cpu().numpy()  # one host transfer, avoids per-keypoint GPU sync

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]  # get state details
                # 获取关键点并估计角度
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # 根据角度阈值判断动作阶段并执行计数逻辑
                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

        # 显示角度、计数和阶段文本
                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],  # 要显示的角度文本
                        count_text=state["count"],  # 健身动作计数文本
                        stage_text=state["stage"],  # 动作阶段文本
                        center_kpt=k[int(self.kpts[1])],  # 用于显示的中心关键点
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 如果环境支持显示，则显示输出图像

        # 返回仅与当前跟踪对象对齐的 SolutionResults。
        # self.states 以 track_id 为键，可能包含已经离开画面的对象；遍历 self.track_ids 可使每个跟踪对象的列表
        # 与 total_tracks 保持同步，从而始终满足 len(workout_count) == total_tracks。
        return SolutionResults(
            plot_im=plot_im,
            workout_count=[self.states[tid]["count"] for tid in self.track_ids],
            workout_stage=[self.states[tid]["stage"] for tid in self.track_ids],
            workout_angle=[self.states[tid]["angle"] for tid in self.track_ids],
            total_tracks=len(self.track_ids),
        )
