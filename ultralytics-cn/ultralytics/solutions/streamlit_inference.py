# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
from typing import Any

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []  # Torch 模块 __path__._path 问题：https://github.com/datalab-to/marker/issues/442


class Inference:
    """执行对象检测、图像分类、图像分割和姿态估计推理的类。

    此类提供加载模型、配置设置、上传视频文件以及使用 Streamlit 和 Ultralytics YOLO 模型执行实时推理的功能。

    属性：
        st (模块): 用于创建用户界面的 Streamlit 模块。
        temp_dict (dict): 保存模型路径和其他配置的临时字典。
        model_path (str): 已加载模型的路径。
        model (YOLO): YOLO 模型实例。
        source (str): 选定的视频源（摄像头或视频文件）。
        enable_trk (bool): 是否启用跟踪。
        conf (float): 检测置信度阈值。
        iou (float): 非极大值抑制使用的 IoU 阈值。
        org_frame (Any): 用于显示原始帧的容器。
        ann_frame (Any): 用于显示标注帧的容器。
        vid_file_name (str | int): 上传视频文件的名称或摄像头索引。
        selected_ind (列表[int]): 选定的检测类别索引。

    方法：
        web_ui: 使用自定义 HTML 元素设置 Streamlit 网页界面。
        sidebar: 配置 Streamlit 侧边栏中的模型和推理设置。
        source_upload: 通过 Streamlit 界面处理视频文件上传。
        configure: 配置模型并加载用于推理的选定类别。
        inference: 执行实时对象检测推理。

    示例：
        使用自定义模型创建 Inference 实例
        >>> inf = Inference(model="path/to/model.pt")
        >>> inf.inference()

        使用默认设置创建 Inference 实例
        >>> inf = Inference()
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 Inference 类，检查 Streamlit 依赖并设置模型路径。

        参数：
            **kwargs (Any): 用于模型配置的其他关键字参数。
        """
        check_requirements("streamlit>=1.29.0")  # 使用局部导入以加快 ultralytics 包的加载速度
        import streamlit as st

        self.st = st  # Streamlit 模块引用
        self.source = None  # 视频源选择（摄像头或视频文件）
        self.img_file_names = []  # 图像文件名称列表
        self.enable_trk = False  # 是否启用对象跟踪
        self.conf = 0.25  # 检测置信度阈值
        self.iou = 0.45  # 交并比（IoU）非极大值抑制阈值
        self.org_frame = None  # 原始帧显示容器
        self.ann_frame = None  # 标注帧显示容器
        self.vid_file_name = None  # 视频文件名称或摄像头索引
        self.selected_ind: list[int] = []  # 选定的检测类别索引列表
        self.model = None  # YOLO 模型实例

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # 模型文件路径
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]
        self.imgsz = self.temp_dict.get("imgsz", 640)

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """使用自定义 HTML 元素设置 Streamlit 网页界面。"""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # 隐藏主菜单样式

        # Streamlit 应用主标题
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Streamlit 应用副标题
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam, videos, and images
        with the power of Ultralytics YOLO! 🚀</h5></div>"""

        # 设置 HTML 页面配置并添加自定义 HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """配置 Streamlit 侧边栏中的模型和推理设置。"""
        with self.st.sidebar:  # 添加 Ultralytics 标志
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # 添加垂直设置菜单元素
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )  # 添加输入源下拉菜单
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"  # 启用对象跟踪
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # 置信度滑块
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # NMS 阈值滑块

        if self.source != "image":  # 仅为视频或摄像头创建列
            col1, col2 = self.st.columns(2)  # 创建两列以显示视频帧
            self.org_frame = col1.empty()  # 原始帧容器
            self.ann_frame = col2.empty()  # 标注帧容器

    def source_upload(self) -> None:
        """通过 Streamlit 界面处理视频文件上传。"""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # 使用局部导入

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO 对象
                with open("ultralytics.mp4", "wb") as out:  # 以二进制方式打开临时文件
                    out.write(g.read())  # 将字节写入文件
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # 使用索引为 0 的摄像头
        elif self.source == "image":
            import tempfile  # 使用局部导入

            if imgfiles := self.st.sidebar.file_uploader(
                "Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:  # 将每个上传图像保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        """配置模型并加载用于推理的选定类别。"""
        # 添加模型选择下拉菜单，按标准尺寸和任务组合提供模型
        available_models = [
            f"{size}{task}".replace("yolo", "YOLO")
            for size in ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
            for task in ("", "-seg", "-sem", "-depth", "-pose", "-obb", "-cls")
            if f"{size}{task}" in GITHUB_ASSETS_STEMS
        ]
        if self.model_path:  # 将用户提供的自定义模型插入可用模型列表
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            if selected_model.endswith((".pt", ".onnx", ".torchscript", ".mlpackage", ".engine")) or any(
                fmt in selected_model for fmt in ("openvino_model", "rknn_model")
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"  # 函数调用未提供模型时默认使用 .pt
            self.model = YOLO(model_path)  # 加载 YOLO 模型
            class_names = list(self.model.names.values())  # 将类别名称字典转换为列表
        self.st.success("Model loaded successfully!")

        # 使用类别名称进行多选，并获取选定类别的索引
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # 确保 selected_ind 为列表
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        """对上传的图像执行推理。"""
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)  # 加载并显示原始图像
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="Predicted Image")
                try:  # 清理临时文件
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass  # 文件不存在，忽略此异常
            else:
                self.st.error("Could not load the uploaded image.")

    def inference(self) -> None:
        """在视频或摄像头输入上执行实时对象检测推理。"""
        self.web_ui()  # 初始化网页界面
        self.sidebar()  # 创建侧边栏
        self.source_upload()  # 上传视频源
        self.configure()  # 配置应用

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file to perform inference.")
                return

            stop_button = self.st.sidebar.button("Stop")  # 停止推理的按钮
            cap = cv2.VideoCapture(self.vid_file_name)  # 捕获视频
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                    break

                # 使用模型处理当前帧
                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, imgsz=self.imgsz, persist=True
                    )
                else:
                    results = self.model(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, imgsz=self.imgsz
                    )

                annotated_frame = results[0].plot()  # 在帧上添加标注

                if stop_button:
                    cap.release()  # 释放视频捕获对象
                    self.st.stop()  # 停止 Streamlit 应用

                self.org_frame.image(frame, channels="BGR", caption="Original Frame")  # 显示原始帧
                self.ann_frame.image(annotated_frame, channels="BGR", caption="Predicted Frame")  # 显示处理后帧

            cap.release()  # 释放视频捕获对象
        cv2.destroyAllWindows()  # 销毁所有 OpenCV 窗口


if __name__ == "__main__":
    import sys  # 导入 sys 模块以访问命令行参数

    # 检查命令行参数中是否提供了模型名称
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # 如果提供则将第一个参数作为模型名称
    # 创建 Inference 类实例并执行推理
    Inference(model=model).inference()
