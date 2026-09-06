# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.solutions.config import SolutionConfig
from ultralytics.utils import ASSETS_URL, LOGGER, ops
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator


class BaseSolution:
    """管理 Ultralytics 解决方案的基类。

    此类为各种 Ultralytics 解决方案提供模型加载、对象跟踪和区域初始化等核心功能，
    是实现对象计数、姿态估计和数据分析等计算机视觉方案的基础。

    属性：
        LineString: 使用 shapely 创建线几何对象的类。
        Polygon: 使用 shapely 创建多边形几何对象的类。
        Point: 使用 shapely 创建点几何对象的类。
        prep: shapely 提供的预准备几何函数，用于优化空间操作。
        CFG (dict[str, Any]): 从 YAML 文件加载并通过 kwargs 更新的配置字典。
        LOGGER: 解决方案专用日志记录器实例。
        annotator: 用于在图像上绘制内容的 Annotator 实例。
        tracks: 最近一次推理得到的 YOLO 跟踪结果。
        track_data: 从跟踪结果中提取的边界框或 OBB 数据。
        boxes (列表): 跟踪结果中的边界框坐标。
        clss (列表[int]): 跟踪结果中的类别索引。
        track_ids (列表[int]): 跟踪结果中的跟踪 ID。
        confs (列表[float]): 跟踪结果中的置信度分数。
        track_line: 用于保存跟踪历史的当前跟踪线。
        masks: 跟踪结果中的分割掩码。
        r_s: 用于空间操作的区域或线几何对象。
        frame_no (int): 用于日志记录的当前帧编号。
        region (列表[tuple[int, int]]): 定义感兴趣区域的坐标元组列表。
        line_width (int): 可视化使用的线宽。
        model (YOLO): 已加载的 YOLO 模型实例。
        names (dict[int, str]): 类别索引到类别名称的映射。
        classes (列表[int]): 要跟踪的类别索引列表。
        show_conf (bool): 是否在标注中显示置信度分数。
        show_labels (bool): 是否在标注中显示类别标签。
        device (str): 模型推理设备。
        track_add_args (dict[str, Any]): 跟踪配置的其他参数。
        env_check (bool): 表示当前环境是否支持图像显示的标志。
        track_history (defaultdict): 保存每个对象跟踪历史的字典。
        profilers (tuple): 用于性能监控的分析器实例。

    方法：
        adjust_box_label: 生成边界框的格式化标签。
        extract_tracks: 执行对象跟踪并从输入图像中提取跟踪结果。
        store_tracking_history: 保存给定跟踪 ID 和边界框对应的对象跟踪历史。
        initialize_region: 根据配置初始化计数区域和计数线。
        display_output: 显示处理结果，包括视频帧或保存的结果。
        process: 由每个解决方案子类实现的处理方法。

    示例：
        >>> solution = BaseSolution(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        """使用配置设置和 YOLO 模型初始化 BaseSolution 类。

        参数：
            is_cli (bool): 设置为 True 时启用 CLI 模式。
            **kwargs (Any): 覆盖默认值的其他配置参数。
        """
        self.CFG = vars(SolutionConfig().update(**kwargs))
        self.LOGGER = LOGGER  # 保存日志对象，供多个解决方案类使用

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # 初始化标注器
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1  # 仅用于日志记录

        self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")
        self.region = self.CFG["region"]  # 保存区域数据，供其他类使用
        self.line_width = self.CFG["line_width"]

        # 加载模型并保存其他信息（类别、show_conf、show_label）
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo26n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.show_conf = self.CFG["show_conf"]
        self.show_labels = self.CFG["show_labels"]
        self.device = self.CFG["device"]

        self.track_add_args = {  # 跟踪器高级配置的其他参数
            k: self.CFG[k] for k in ("iou", "conf", "device", "max_det", "quantize", "tracker", "imgsz")
        }  # verbose 必须传给 track 方法；在 YOLO 中设置为 False 仍会记录跟踪信息。

        if is_cli and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            self.LOGGER.warning(f"source not provided. using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{d_s}")  # 从 ultralytics 资源下载源文件
            self.CFG["source"] = d_s  # 设置默认源

        # 初始化环境和区域设置
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

        self.profilers = (
            ops.Profile(device=self.device),  # track
            ops.Profile(device=self.device),  # solution
        )

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        """生成边界框的格式化标签。

        此方法使用类别索引和置信度分数构造边界框标签；如果提供跟踪 ID，则将其包含在标签中。
        标签格式根据 `self.show_conf` 和 `self.show_labels` 中定义的显示设置自动调整。

        参数：
            cls (int): 检测对象的类别索引。
            conf (float): 检测结果的置信度分数。
            track_id (int, 可选): 跟踪对象的唯一标识符。

        返回：
            (str | None): `self.show_labels` 为 True 时返回格式化标签，否则返回 None。
        """
        name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None

    def extract_tracks(self, im0: np.ndarray) -> None:
        """在输入图像或帧上执行对象跟踪并提取跟踪结果。

        参数：
            im0 (np.ndarray): 输入图像或视频帧。

        示例：
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        is_obb = self.tracks.obb is not None
        self.track_data = self.tracks.obb if is_obb else self.tracks.boxes  # 提取 OBB 或目标检测跟踪结果

        if self.track_data and self.track_data.is_track:
            self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.LOGGER.warning("No tracks found.")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []
        self.forget_tracks([track.track_id for track in self.model.predictor.trackers[0].removed_stracks_frame])

    def forget_tracks(self, track_ids: list[int]) -> None:
        """移除活动跟踪器已经结束的 ID 记录。"""
        for track_id in track_ids:
            self.track_history.pop(track_id, None)

    def store_tracking_history(self, track_id: int, box) -> None:
        """保存对象的跟踪历史。

        此方法将对象边界框的中心点追加到跟踪线，以更新给定对象的跟踪历史，并将历史点数限制为最多 30 个。

        参数：
            track_id (int): 跟踪对象的唯一标识符。
            box (列表[float]): 对象边界框坐标，格式为 `[x1, y1, x2, y2]`。

        示例：
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # 保存跟踪历史
        self.track_line = self.track_history[track_id]
        self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    @staticmethod
    def get_enclosing_box(box: torch.Tensor | list[float]) -> torch.Tensor | list[float]:
        """返回包含 `extract_tracks` 提取边界框的轴对齐边界框 `[x1, y1, x2, y2]`。

        OBB 模型的框是形状为 `(4, 2)` 的 xyxyxyxy 角点，而检测模型的边界框已经是轴对齐的 `[x1, y1, x2, y2]`。
        此方法将两种格式统一为 `[x1, y1, x2, y2]`，供图像切片或边界框中心点等需要轴对齐坐标的功能使用。

        参数：
            box (torch.Tensor | 列表[float]): `[x1, y1, x2, y2]` 格式的边界框，或形状为 `(4, 2)` 的 OBB 角点。

        返回：
            (torch.Tensor | 列表[float]): `[x1, y1, x2, y2]` 格式的轴对齐边界框。

        示例：
            >>> import torch
            >>> BaseSolution.get_enclosing_box(torch.tensor([[2.0, 1.0], [4.0, 3.0], [2.0, 5.0], [0.0, 3.0]]))
            张量([0., 1., 4., 5.])
        """
        return torch.cat([box.amin(0), box.amax(0)]) if isinstance(box, torch.Tensor) and box.numel() > 4 else box

    def initialize_region(self) -> None:
        """根据配置设置初始化计数区域或计数线。"""
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # 区域或计数线

    def display_output(self, plot_im: np.ndarray) -> None:
        """显示处理结果，包括显示视频帧、打印计数或保存结果。

        此方法负责可视化对象检测和跟踪流程的输出，显示带标注的处理帧，并允许用户关闭显示窗口。

        参数：
            plot_im (np.ndarray): 已处理并标注的图像或视频帧。

        示例：
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        注意：
            - 只有当 'show' 配置为 True 且当前环境支持图像显示时，此方法才会显示输出。
            - 按下 'q' 键可以关闭显示窗口。
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # 关闭当前帧窗口
                return

    def process(self, *args: Any, **kwargs: Any):
        """处理方法，应由每个 Solution 子类实现。"""

    def __call__(self, *args: Any, **kwargs: Any):
        """允许以函数方式调用实例，并传入灵活参数。"""
        with self.profilers[1]:
            result = self.process(*args, **kwargs)  # 调用子类专用处理方法
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # 解决方案耗时 = 处理耗时 - 跟踪耗时
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)  # 仅用于日志记录
            # 如果预测器可用，则使用模型输入尺寸（反映 imgsz）
            if hasattr(self.model, "predictor") and self.model.predictor and hasattr(self.model.predictor, "imgsz"):
                input_h, input_w = self.model.predictor.imgsz
            else:
                input_h, input_w = result.plot_im.shape[:2]
            LOGGER.info(
                f"{self.frame_no}: {input_h}x{input_w} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape "
                f"(1, {getattr(self.model, 'channels', 3)}, {input_h}, {input_w})\n"
            )
        return result


class SolutionAnnotator(Annotator):
    """用于可视化和分析计算机视觉任务的专用标注器类。

    此类扩展基础 Annotator 类，为 Ultralytics 解决方案增加绘制区域、中心点、跟踪轨迹和视觉标注的方法，
    为目标检测、跟踪、姿态估计和数据分析等计算机视觉应用提供完整的可视化能力。

    属性：
        im (np.ndarray): 要标注的图像。
        line_width (int): 标注线条的宽度。
        font_size (int): 标注文本的字体大小。
        font (str): 用于文本渲染的字体文件路径。
        pil (bool): 是否使用 PIL 渲染文本。
        example (str): 用于检测 PIL 渲染所需非 ASCII 标签的示例文本。

    方法：
        draw_region: 使用指定点、颜色和线宽绘制区域。
        queue_counts_display: 在指定区域显示队列计数。
        display_analytics: 显示停车场管理的总体统计信息。
        estimate_pose_angle: 计算对象姿态中三个点之间的角度。
        draw_specific_kpts: 在图像上绘制指定关键点。
        plot_workout_information: 在图像上绘制带标签的文本框。
        plot_angle_and_count_and_stage: 可视化健身动作监控的角度、步数和阶段。
        plot_distance_and_line: 显示中心点之间的距离并用线连接中心点。
        display_objects_labels: 使用对象类别标签标注边界框。
        sweep_annotator: 可视化垂直扫描线和可选标签。
        visioneye: 将对象中心点映射并连接到视觉“眼睛”点。
        adaptive_label: 在边界框中心绘制带矩形或圆形背景的标签。

    示例：
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """使用图像初始化用于标注的 SolutionAnnotator 类。

        参数：
            im (np.ndarray): 要标注的图像。
            line_width (int, 可选): 图像绘制线宽。
            font_size (int, 可选): 标注文本字体大小。
            font (str): 字体文件路径。
            pil (bool): 是否使用 PIL 渲染文本。
            example (str): 用于检测 PIL 渲染所需非 ASCII 标签的示例文本。
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(
        self,
        reg_pts: list[tuple[int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5,
    ):
        """在图像上绘制区域或线段。

        参数：
            reg_pts (列表[tuple[int, int]], 可选): 区域点（线段使用 2 个点，区域使用 4 个或更多点）。
            color (tuple[int, int, int]): 区域使用的 BGR 颜色值（OpenCV 格式）。
            thickness (int): 区域绘制线宽。
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # 在角点处绘制小圆
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 表示填充圆形

    def queue_counts_display(
        self,
        label: str,
        points: list[tuple[int, int]] | None = None,
        region_color: tuple[int, int, int] = (255, 255, 255),
        txt_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """在以给定点为中心的图像区域中显示队列计数，并支持自定义字体大小和颜色。

        参数：
            label (str): 队列计数标签。
            points (列表[tuple[int, int]], 可选): 用于计算文本显示中心点的区域点。
            region_color (tuple[int, int, int]): 队列区域 BGR 颜色（OpenCV 格式）。
            txt_color (tuple[int, int, int]): 文本 BGR 颜色（OpenCV 格式）。
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # 绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
    ):
        """显示解决方案的总体统计信息（例如停车场管理和对象计数）。

        参数：
            im0 (np.ndarray): 推理图像。
            text (dict[str, Any]): 标签字典。
            txt_color (tuple[int, int, int]): 文本颜色（BGR，OpenCV 格式）。
            bg_color (tuple[int, int, int]): 背景颜色（BGR，OpenCV 格式）。
            margin (int): 文本与矩形之间的间距，用于改善显示效果。
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        """将类似关键点的对象转换为 `(x, y)` 浮点数元组。"""
        if hasattr(point, "detach"):  # torch.Tensor
            point = point.detach()
        if hasattr(point, "cpu"):  # torch.Tensor
            point = point.cpu()
        if hasattr(point, "numpy"):  # torch.Tensor
            point = point.numpy()
        if hasattr(point, "tolist"):  # numpy / torch
            point = point.tolist()
        return float(point[0]), float(point[1])

    @staticmethod
    @lru_cache(maxsize=256)
    def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """计算健身动作监控中三个点之间的角度（带缓存）。"""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle if angle <= 180.0 else (360 - angle)

    @staticmethod
    def estimate_pose_angle(a: Any, b: Any, c: Any) -> float:
        """计算健身动作监控中三个点之间的角度。

        参数：
            a (Any): 第一个点的坐标（例如列表、元组、NumPy 数组或 torch 张量）。
            b (Any): 第二个点（顶点）的坐标。
            c (Any): 第三个点的坐标。

        返回：
            (float): 三个点之间的角度，单位为度。
        """
        a_xy, b_xy, c_xy = (
            SolutionAnnotator._point_xy(a),
            SolutionAnnotator._point_xy(b),
            SolutionAnnotator._point_xy(c),
        )
        return SolutionAnnotator._estimate_pose_angle_cached(a_xy, b_xy, c_xy)

    def draw_specific_kpts(
        self,
        keypoints: list[list[float]],
        indices: list[int] | None = None,
        radius: int = 2,
        conf_thresh: float = 0.25,
    ) -> np.ndarray:
        """绘制用于健身动作计数的指定关键点。

        参数：
            keypoints (列表[列表[float]]): 要绘制的关键点数据，每个关键点格式为 `[x, y, 置信度]`。
            indices (列表[int], 可选): 要绘制的关键点索引，绘制顺序与此列表一致。
            radius (int): 关键点半径。
            conf_thresh (float): 关键点置信度阈值。

        返回：
            (np.ndarray): 绘制关键点后的图像。

        注意：
            关键点格式为 `[x, y]` 或 `[x, y, 置信度]`。
            此方法会原地修改 self.im。
        """
        indices = indices or [2, 5, 7]
        n = len(keypoints)
        points = [
            (int(keypoints[j][0]), int(keypoints[j][1]))
            for j in indices
            if 0 <= j < n and (float(keypoints[j][2]) if len(keypoints[j]) > 2 else 1.0) >= conf_thresh
        ]

        # 绘制相邻点之间的连线
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # 绘制关键点圆形标记
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(
        self,
        display_text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> int:
        """在图像上绘制带背景的健身动作文本。

        参数：
            display_text (str): 要显示的文本。
            position (tuple[int, int]): 文本在图像上的坐标 `(x, y)`。
            color (tuple[int, int, int]): 文本背景颜色。
            txt_color (tuple[int, int, int]): 文本前景颜色。

        返回：
            (int): 文本高度。
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

        # 绘制背景矩形
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # 绘制文本
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self,
        angle_text: str,
        count_text: str,
        stage_text: str,
        center_kpt: list[int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """绘制健身动作监控中的姿态角度、计数值和动作阶段。

        参数：
            angle_text (str): 健身动作监控的角度值。
            count_text (str): 健身动作监控的计数值。
            stage_text (str): 健身动作监控的阶段判断。
            center_kpt (列表[int]): 健身动作监控的中心姿态关键点。
            color (tuple[int, int, int]): 文本背景颜色。
            txt_color (tuple[int, int, int]): 文本前景颜色。
        """
        # 格式化文本
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # 绘制角度、计数和阶段文本
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(
        self,
        pixels_distance: float,
        centroids: list[tuple[int, int]],
        line_color: tuple[int, int, int] = (104, 31, 17),
        centroid_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """在图像帧上绘制两个中心点之间的距离和连接线。

        参数：
            pixels_distance (float): 两个边界框中心点之间的像素距离。
            centroids (列表[tuple[int, int]]): 两个边界框中心点坐标。
            line_color (tuple[int, int, int]): 距离连线颜色。
            centroid_color (tuple[int, int, int]): 边界框中心点颜色。
        """
        # 获取文本尺寸
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # 使用 10 像素边距确定矩形角点并绘制矩形
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # 使用 10 像素边距计算文本位置并绘制文本
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(
        self,
        im0: np.ndarray,
        text: str,
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        x_center: float,
        y_center: float,
        margin: int,
    ):
        """在停车场管理应用中显示边界框标签。

        参数：
            im0 (np.ndarray): 推理图像。
            text (str): 对象或类别名称。
            txt_color (tuple[int, int, int]): 文本前景颜色。
            bg_color (tuple[int, int, int]): 文本背景颜色。
            x_center (float): 边界框中心点的 x 坐标。
            y_center (float): 边界框中心点的 y 坐标。
            margin (int): 文本与矩形之间的间距，用于改善显示效果。
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),  # 确保颜色值为整数
            -1,
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),  # 确保颜色值为整数
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def sweep_annotator(
        self,
        line_x: int = 0,
        line_y: int = 0,
        label: str | None = None,
        color: tuple[int, int, int] = (221, 0, 186),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """绘制扫描标注线和可选标签。

        参数：
            line_x (int): 扫描线的 x 坐标。
            line_y (int): 扫描线的 y 坐标终点。
            label (str, 可选): 绘制在扫描线中央的文本标签。为 None 时不绘制标签。
            color (tuple[int, int, int]): 扫描线和标签背景的 BGR 颜色（OpenCV 格式）。
            txt_color (tuple[int, int, int]): 标签文本的 BGR 颜色（OpenCV 格式）。
        """
        # 绘制扫描线
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # 如果提供标签，则绘制标签
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def visioneye(
        self,
        box: list[float],
        center_point: tuple[int, int],
        color: tuple[int, int, int] = (235, 219, 11),
        pin_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """执行人眼视觉映射并绘图。

        参数：
            box (列表[float]): 边界框坐标，格式为 [x1, y1, x2, y2]。
            center_point (tuple[int, int]): 视觉映射中心点。
            color (tuple[int, int, int]): 对象中心点和连线颜色。
            pin_color (tuple[int, int, int]): 视觉映射点颜色。
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def adaptive_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        shape: str = "rect",
        margin: int = 5,
    ):
        """在给定边界框中心绘制带矩形或圆形背景的标签。

        参数：
            box (tuple[float, float, float, float]): 边界框坐标 (x1, y1, x2, y2)。
            label (str): 要显示的文本标签。
            color (tuple[int, int, int]): 矩形或圆形背景颜色（B、G、R）。
            txt_color (tuple[int, int, int]): 文本颜色（B、G、R）。
            shape (str): 标签形状，可选值为 "circle" 或 "rect"。
            margin (int): 文本与背景边界之间的间距。
        """
        if shape == "circle" and len(label) > 3:
            LOGGER.warning(f"标签长度为 {len(label)}，圆形标注只会使用前 3 个字符。")
            label = label[:3]

        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # 计算边界框中心点
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]  # 获取文本尺寸
        text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2  # 计算文本左上角位置

        if shape == "circle":
            cv2.circle(
                self.im,
                (x_center, y_center),
                int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,  # 计算半径
                color,
                -1,
            )
        else:
            cv2.rectangle(
                self.im,
                (text_x - margin, text_y - text_size[1] - margin),  # 计算矩形坐标
                (text_x + text_size[0] + margin, text_y + margin),  # 计算矩形坐标
                color,
                -1,
            )

        # 在背景矩形或圆形上绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),  # 绘制文本的左上角位置
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )


class SolutionResults:
    """封装 Ultralytics 解决方案运行结果的类。

    此类用于保存和管理解决方案流程生成的各种输出，包括计数值、角度、健身动作阶段以及其他分析数据，
    并为对象计数、姿态估计和跟踪分析等计算机视觉解决方案提供结构化的结果访问方式。

    属性：
        plot_im (np.ndarray): 包含计数、模糊效果或其他解决方案处理结果的图像。
        in_count (int): 视频流中进入区域的对象总数。
        out_count (int): 视频流中离开区域的对象总数。
        classwise_count (dict[str, int]): 按类别统计对象数量的字典。
        queue_count (int): 队列或等待区域中的对象数量。
        workout_count (列表[int]): 每个跟踪对象的健身动作重复次数。
        workout_angle (列表[float]): 当前跟踪对象的健身动作角度。
        workout_stage (列表[str]): 当前跟踪对象的健身动作阶段。
        pixels_distance (float): 两个点或对象之间计算得到的像素距离。
        available_slots (int): 监控区域中的可用车位数量。
        filled_slots (int): 监控区域中的已占用车位数量。
        email_sent (bool): 是否已发送电子邮件通知的标志。
        total_tracks (int): 被跟踪对象的总数。
        region_counts (dict[str, int]): 特定区域内的对象数量。
        speed_dict (dict[str, float]): 保存被跟踪对象速度信息的字典。
        total_crop_objects (int): 使用 ObjectCropper 类裁剪得到的对象总数。
        speed (dict[str, float]): 跟踪和解决方案处理的性能计时信息。
    """

    def __init__(self, **kwargs):
        """使用默认值或用户指定的值初始化 SolutionResults 对象。

        参数：
            **kwargs (Any): 用于覆盖默认属性值的可选参数。
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = []
        self.workout_angle = []
        self.workout_stage = []
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}  # 用于保存速度估计结果
        self.total_crop_objects = 0
        self.speed = {}

        # 使用用户定义的值覆盖默认值
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """返回 SolutionResults 对象的格式化字符串表示。

        返回：
            (str): 列出非空属性的字符串表示。
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # 显式排除 `plot_im`
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
