# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imshow


class ParkingPtsSelection:
    """使用基于 Tkinter 的界面在图像上选择和管理停车区域点的类。

    此类用于上传图像、选择停车区域的顶点，并将所选顶点保存到 JSON 文件。它使用 Tkinter 构建图形用户界面。

    属性：
        tk (模块): 用于图形界面操作的 Tkinter 模块。
        filedialog (模块): 用于文件选择操作的 Tkinter filedialog 模块。
        messagebox (模块): 用于显示消息框的 Tkinter messagebox 模块。
        master (tk.Tk): Tkinter 主窗口。
        canvas (tk.Canvas): 用于显示图像和绘制边界框的画布控件。
        image (PIL.Image.Image): 上传的图像。
        canvas_image (ImageTk.PhotoImage): 显示在画布上的图像。
        rg_data (列表[列表[tuple[int, int]]]): 边界框列表，每个边界框由 4 个点定义。
        current_box (列表[tuple[int, int]]): 临时保存当前边界框的顶点。
        imgw (int): 上传图像的原始宽度。
        imgh (int): 上传图像的原始高度。
        canvas_max_width (int): 画布的最大宽度。
        canvas_max_height (int): 画布的最大高度。

    方法：
        initialize_properties: 初始化图像、画布、边界框和尺寸等属性。
        upload_image: 上传图像并显示在画布上，同时调整尺寸以适应指定范围。
        on_canvas_click: 处理鼠标点击，在画布上添加边界框顶点。
        draw_box: 使用给定坐标在画布上绘制边界框。
        remove_last_bounding_box: 从列表中移除最后一个边界框并重绘画布。
        redraw_canvas: 使用图像和所有边界框重绘画布。
        save_to_json: 将所选停车区域顶点及其缩放坐标保存到 JSON 文件。

    示例：
        >>> parking_selector = ParkingPtsSelection()
        >>> # 使用 GUI 上传图像、选择停车区域并保存数据
    """

    def __init__(self) -> None:
        """初始化 ParkingPtsSelection 类，并设置停车区域点选择所需的界面和属性。"""
        try:  # 检查是否安装了 tkinter
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:  # 显示错误和建议
            import platform

            install_cmd = {
                "Linux": "sudo apt install python3-tk (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | "
                "sudo pacman -S tk (Arch)",
                "Windows": "reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation",
                "Darwin": "reinstall Python from https://www.python.org/downloads/macos/ or `brew install python-tk`",
            }.get(platform.system(), "Unknown OS. Check your Python installation.")

            LOGGER.warning(f" Tkinter is not configured or supported. Potential fix: {install_cmd}")
            return

        if not check_imshow(warn=True):
            return

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()  # 主应用窗口的引用
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        self.canvas = self.tk.Canvas(self.master, bg="white")  # 用于显示图像的画布控件
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None  # 保存已加载图像的变量
        self.canvas_image = None  # 画布上显示的图像引用
        self.canvas_max_width = None  # 画布允许的最大宽度
        self.canvas_max_height = None  # 画布允许的最大高度
        self.rg_data = None  # 区域标注管理数据
        self.current_box = None  # 保存当前选中的边界框
        self.imgh = None  # 当前图像的高度
        self.imgw = None  # 当前图像的宽度

        # 创建包含按钮的按钮区域
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last Bounding Box", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self) -> None:
        """初始化图像、画布、边界框和尺寸等属性。"""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self) -> None:
        """将图像上传并显示到画布上，并调整大小以适应指定尺寸。"""
        from PIL import Image, ImageTk  # 局部导入，因为 ImageTk 需要 tkinter 包

        file = self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file:
            LOGGER.info("No image selected.")
            return

        self.image = Image.open(file)
        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event) -> None:
        """处理鼠标点击，在画布上添加边界框点。"""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box: list[tuple[int, int]]) -> None:
        """使用给定坐标在画布上绘制边界框。"""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self) -> None:
        """从列表中移除最后一个边界框，并重绘画布。"""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        """使用图像和所有边界框重绘画布。"""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self) -> None:
        """将所选停车区域点及其缩放后的坐标保存到 JSON 文件。"""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        from io import StringIO  # 函数级导入，因为仅在保存坐标时需要

        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())
        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """使用 YOLO 模型实时监控和可视化停车位占用情况与可用情况。

    此类扩展 BaseSolution，提供停车场管理功能，包括检测已占用车位、可视化停车区域以及显示占用统计信息。

    属性：
        json_file (str): 包含停车区域详情的 JSON 文件路径。
        json (列表[dict]): 加载后的停车区域信息 JSON 数据。
        pr_info (dict[str, int]): 保存停车信息（已占用和可用车位）的字典。
        arc (tuple[int, int, int]): 可用区域可视化使用的 BGR 颜色元组。
        occ (tuple[int, int, int]): 已占用区域可视化使用的 BGR 颜色元组。
        dc (tuple[int, int, int]): 检测对象中心点可视化使用的 BGR 颜色元组。

    方法：
        process: 处理输入图像，执行停车场管理和可视化。

    示例：
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolo26n.pt", json_file="parking_regions.json")
        >>> print(f"Occupied spaces: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"Available spaces: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用 YOLO 模型和可视化设置初始化停车场管理系统。"""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # 加载停车区域 JSON 数据
        if not self.json_file:
            LOGGER.warning("ParkingManagement requires `json_file` with parking region coordinates.")
            raise ValueError("❌ JSON file path cannot be empty.")

        with open(self.json_file, encoding="utf-8") as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # 停车信息字典

        self.arc = (0, 0, 255)  # 可用区域颜色
        self.occ = (0, 255, 0)  # 已占用区域颜色
        self.dc = (255, 0, 189)  # 每个边界框的中心点颜色

    def process(self, im0: np.ndarray) -> SolutionResults:
        """处理输入图像，执行停车场管理和可视化。

        此函数分析输入图像、提取跟踪结果，并确定 JSON 文件中定义的停车区域占用状态。
        它会在图像上标注已占用和可用停车位，并更新停车信息。

        参数：
            im0 (np.ndarray): 用于推理的输入图像。

        返回：
            (SolutionResults): 包含处理后的图像 `plot_im`、已占用车位数量 `filled_slots`、可用车位数量
                `available_slots` 以及跟踪对象总数 `total_tracks`。

        示例：
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> results = parking_manager.process(image)
        """
        self.extract_tracks(im0)  # 从 im0 提取轨迹
        available_slots, occupied_slots = len(self.json), 0
        annotator = SolutionAnnotator(im0, self.line_width)  # 初始化 annotator

        for region in self.json:
            # 将点转换为正确数据类型的 NumPy 数组，并正确调整形状
            region_polygon = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            region_occupied = False
            for box, cls in zip(self.boxes, self.clss):
                x0, y0, x1, y1 = self.get_enclosing_box(box)
                xc, yc = int((x0 + x1) / 2), int((y0 + y1) / 2)
                inside_distance = cv2.pointPolygonTest(region_polygon, (xc, yc), False)
                if inside_distance >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    region_occupied = True
                    break
            if region_occupied:
                occupied_slots += 1
                available_slots -= 1
        # 绘制区域
            cv2.polylines(
                im0, [region_polygon], isClosed=True, color=self.occ if region_occupied else self.arc, thickness=2
            )

        self.pr_info["Occupancy"], self.pr_info["Available"] = occupied_slots, available_slots

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类方法显示输出

        # 返回 SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=self.pr_info["Occupancy"],
            available_slots=self.pr_info["Available"],
            total_tracks=len(self.track_ids),
        )
