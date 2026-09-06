# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2


@dataclass
class SolutionConfig:
    """管理 Ultralytics Vision AI 解决方案配置参数的类。

    SolutionConfig 是所有 Ultralytics 解决方案模块的集中配置容器，参见
    https://docs.ultralytics.com/solutions#solutions。它使用 Python `dataclass`，使参数定义清晰、类型安全且易于维护。

    属性：
        source (str, 可选): 输入源路径（视频、RTSP 等），仅可与 Solutions CLI 一起使用。
        model (str, 可选): 用于推理的 Ultralytics YOLO 模型路径。
        classes (list[int], 可选): 用于筛选检测结果的类别索引列表。
        show_conf (bool): 是否在可视化输出中显示置信度分数。
        show_labels (bool): 是否在可视化输出中显示类别标签。
        show_boxes (bool): 是否在可视化输出中显示边界框。
        region (list[tuple[int, int]], 可选): 用于对象计数的多边形区域或线段。
        colormap (int, 可选): 可视化叠加使用的 OpenCV 颜色映射常量。
        show_in (bool): 是否显示进入区域的对象数量。
        show_out (bool): 是否显示离开区域的对象数量。
        up_angle (float): 基于姿态的健身动作监控中使用的上方角度阈值。
        down_angle (int): 基于姿态的健身动作监控中使用的下方角度阈值。
        kpts (list[int]): 要监控的关键点索引，例如姿态分析所需的关键点。
        analytics_type (str): 要执行的分析类型（"line"、"area"、"bar"、"pie" 等）。
        figsize (tuple[float, float], 可选): 分析图表 Matplotlib 图形尺寸（宽度、高度）。
        blur_ratio (float): 视频帧中的对象模糊比例（0.0 到 1.0）。
        vision_point (tuple[int, int]): 方向跟踪或透视绘制的参考点。
        crop_dir (str): 保存裁剪检测图像的目录路径。
        json_file (str, 可选): 包含停车区域数据的 JSON 文件路径。
        line_width (int): 可视化显示线宽，例如边界框、关键点和计数的线宽。
        records (int): 用于发送邮件提醒的检测记录数量。
        fps (float): 速度估计计算使用的帧率。
        max_hist (int): 每个跟踪对象为速度估计保存的历史点或状态最大数量。
        meter_per_pixel (float): 真实世界测量比例，用于速度或距离计算。
        max_speed (int): 可视化提醒或约束中使用的最大速度限制。
        show (bool): 是否在屏幕上显示可视化输出。
        iou (float): 用于筛选检测结果的交并比阈值。
        conf (float): 保留预测结果的置信度阈值。
        device (str, 可选): 执行推理的设备，例如 'cpu' 或 CUDA GPU 的 '0'。
        max_det (int): 每个视频帧允许的最大检测数。
        quantize (int | str | None): 推理精度，例如 16（FP16），用于替代已弃用的 half 标志。
        tracker (str): 跟踪配置 YAML 文件路径，例如 'botsort.yaml'。
        verbose (bool): 是否启用详细日志输出，用于调试或诊断。
        data (str): 相似度搜索使用的图像目录路径。

    方法：
        update: 使用用户定义的关键字参数更新配置，并在键无效时抛出错误。

    示例：
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """

    source: str | None = None
    model: str | None = None
    classes: list[int] | None = None
    show_conf: bool = True
    show_labels: bool = True
    show_boxes: bool = True
    region: list[tuple[int, int]] | None = None
    colormap: int | None = cv2.COLORMAP_DEEPGREEN
    show_in: bool = True
    show_out: bool = True
    up_angle: float = 145.0
    down_angle: int = 90
    kpts: list[int] = field(default_factory=lambda: [6, 8, 10])
    analytics_type: str = "line"
    figsize: tuple[float, float] | None = (12.8, 7.2)
    blur_ratio: float = 0.5
    vision_point: tuple[int, int] = (20, 20)
    crop_dir: str = "cropped-detections"
    json_file: str | None = None
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: str | None = None
    max_det: int = 300
    quantize: int | str | None = None
    imgsz: int = 640
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs: Any):
        """使用关键字参数提供的新值更新配置参数。"""
        if "half" in kwargs:  # 已弃用的别名，转发到 quantize
            from ultralytics.utils import deprecation_warn

            deprecation_warn("half", "quantize")
            kwargs["quantize"] = 16 if kwargs.pop("half") else None
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                url = "https://docs.ultralytics.com/solutions#solutions-arguments"
                raise ValueError(f"{key} is not a valid solution argument, see {url}")

        return self
