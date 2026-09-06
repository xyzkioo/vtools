# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """在视频流的用户自定义区域内实时统计目标数量的类。

    此类继承自 `BaseSolution`，用于在视频帧中定义多边形区域、跟踪目标，并统计经过每个区域的目标数量。
    适用于需要在指定区域计数的场景，例如监控区域或分段区域。

    属性：
        region_template (dict): 创建新计数区域的模板，包含名称、多边形坐标和显示颜色等默认属性。
        counting_regions (列表): 存储所有已定义区域的列表；每个元素基于 `region_template`，并包含区域名称、坐标和颜色等设置。
        region_counts (dict): 存储每个命名区域中目标数量的字典。

    方法：
        add_region: 使用指定属性添加新的计数区域。
        process: 处理视频帧并统计每个区域中的目标数量。
        initialize_regions: 初始化用于统计目标的区域，也支持多个区域。

    示例：
        初始化 RegionCounter 并添加计数区域
        >>> counter = RegionCounter()
        >>> counter.add_region("Zone1", [(100, 100), (200, 100), (200, 200), (100, 200)], (255, 0, 0), (255, 255, 255))
        >>> results = counter.process(frame)
        >>> print(f"Total tracks: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 RegionCounter，用于在用户自定义区域内实时统计目标数量。"""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "prepared_polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        """根据给定模板和指定属性，向计数列表添加新区域。

        参数：
            name (str): 分配给新区域的名称。
            polygon_points (列表[tuple]): 定义区域多边形的 `(x, y)` 坐标列表。
            region_color (tuple[int, int, int]): 区域可视化使用的 BGR 颜色。
            text_color (tuple[int, int, int]): 区域内文本使用的 BGR 颜色。

        返回：
            (dict[str, Any]): 区域信息，包括名称、多边形和显示颜色。
        """
        if len(polygon_points) < 3:
            raise ValueError(
                f"RegionCounter requires regions with at least 3 points to form a polygon, "
                f"but got {len(polygon_points)} for '{name}'."
            )
        polygon = self.Polygon(polygon_points)
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": polygon,
                "prepared_polygon": self.prep(polygon),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """仅根据 `self.region` 初始化一次区域。"""
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # 确保 self.region 已初始化为字典结构
            self.region = {"Region#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            self.add_region(name, pts, colors(i, True), (255, 255, 255))

    def process(self, im0: np.ndarray) -> SolutionResults:
        """处理输入帧，检测并统计每个定义区域内的对象。

        参数：
            im0 (np.ndarray): 要处理的输入图像帧，对象和区域会在其上标注。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、跟踪对象总数 `total_tracks` 以及各区域对象数量 `region_counts`。
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.region_counts = {region["name"]: 0 for region in self.counting_regions}

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            x0, y0, x1, y1 = self.get_enclosing_box(box)
            center = self.Point(((x0 + x1) / 2, (y0 + y1) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # 显示区域计数
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
            region["counts"] = 0  # 为下一帧重置
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
