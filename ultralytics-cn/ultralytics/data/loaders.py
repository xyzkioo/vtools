# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import time
import urllib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    """表示预测输入源不同类型的类。

    此类使用 dataclass 定义布尔标志，用于区分 YOLO 模型预测时可能使用的不同输入源类型。

    属性：
        stream (bool): 输入源是否为视频流。
        screenshot (bool): 输入源是否为截图。
        from_img (bool): 输入源是否为内存中的图像（PIL/NumPy）或图像列表。
        tensor (bool): 输入源是否为张量。

    示例：
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """用于加载多种视频流的流式加载器。

    支持 RTSP、RTMP、HTTP 和 TCP 流。此类可以同时加载和处理多个视频流，适用于实时视频分析任务。

    属性：
        sources (列表[str]): 视频流的源输入路径或 URL。
        vid_stride (int): 视频帧率步长。
        buffer (bool): 是否缓冲输入流。
        running (bool): 表示流式线程是否正在运行的标志。
        mode (str): 设置为 'stream'，表示实时捕获。
        imgs (列表[列表[np.ndarray]]): 每个视频流对应的图像帧列表。
        fps (列表[float]): 每个视频流对应的 FPS 列表。
        frames (列表[int]): 每个视频流对应的总帧数列表。
        threads (列表[Thread]): 每个视频流对应的线程列表。
        shape (列表[tuple[int, int, int]]): 每个视频流对应的图像形状列表。
        caps (列表[cv2.VideoCapture]): 每个视频流对应的 cv2.VideoCapture 对象列表。
        bs (int): 用于处理的批量大小。
        cv2_flag (int): 读取图像时使用的 OpenCV 标志（灰度或彩色/BGR）。

    方法：
        update: 在线程中读取视频流帧。
        close: 关闭流式加载器并释放资源。
        __iter__: 返回该类的迭代器对象。
        __next__: 返回源路径、变换后的图像和原始图像，供后续处理。
        __len__: 返回 sources 对象的长度。

    示例：
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # 处理图像
        ...     pass
        >>> stream_loader.close()

    注意：
        - 此类使用多线程高效地同时加载多个视频流的帧。
        - 此类会自动处理 YouTube 链接，将其转换为可用的最佳视频流 URL。
        - 此类实现了缓冲区系统，用于管理帧的存储和读取。
    """

    def __init__(self, sources: str = "file.streams", vid_stride: int = 1, buffer: bool = False, channels: int = 3):
        """初始化支持多种流类型的多视频源流式加载器。

        参数：
            sources (str): streams 文件路径或单个视频流 URL。
            vid_stride (int): 视频帧率步长。
            buffer (bool): 是否缓冲输入流。
            channels (int): 图像通道数（1 表示灰度，3 表示彩色）。
        """
        torch.backends.cudnn.benchmark = True  # 固定图像尺寸推理时速度更快
        self.buffer = buffer  # 输入流缓冲区
        self.running = True  # 线程运行标志
        self.mode = "stream"
        self.vid_stride = vid_stride  # 视频帧率步长
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # 灰度或彩色（BGR）

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # 每秒帧数
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # 视频捕获对象
        self.imgs = [[] for _ in range(n)]  # 图像
        self.shape = [[] for _ in range(n)]  # 图像形状
        self.sources = [ops.clean_str(x).replace(os.sep, "_") for x in sources]  # 清理源名称，供后续使用
        try:
            for i, s in enumerate(sources):  # 索引、源
                # 启动线程读取视频流帧
                st = f"{i + 1}/{n}: {s}... "
                if urllib.parse.urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # YouTube 视频
                    # YouTube 格式，例如 'https://www.youtube.com/watch?v=Jsn8D3aC840' 或 'https://youtu.be/Jsn8D3aC840'
                    s = get_best_youtube_url(s)
                s = int(s) if s.isnumeric() else s  # 例如 s = '0' 表示本地摄像头
                if s == 0 and (IS_COLAB or IS_KAGGLE):
                    raise NotImplementedError(
                        "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                        "Try running 'source=0' in a local environment."
                    )
                self.caps[i] = cv2.VideoCapture(s)  # 保存视频捕获对象
                if not self.caps[i].isOpened():
                    raise ConnectionError(f"{st}Failed to open {s}")
                w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # 注意：可能返回 0 或 nan
                self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                    "inf"
                )  # 无限流回退值
                self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS 回退值

                success, im = self.caps[i].read()  # 确保读取第一帧
                if not success or im is None:
                    raise ConnectionError(f"{st}Failed to read images from {s}")
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
                self.imgs[i].append(im)
                self.shape[i] = im.shape
                self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
                LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
        except Exception:
            self.close()  # 重新抛出异常前，释放已打开的捕获对象并停止已启动的线程
            raise
        LOGGER.info("")  # 换行

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        """在线程中读取视频流帧，并更新图像缓冲区。"""
        n, f = 0, self.frames[i]  # 当前帧数量、总帧数
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # 保持不超过 30 张图像的缓冲区
                n += 1
                cap.grab()  # .read() 等于先调用 .grab()，再调用 .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success or im is None:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # 信号丢失时重新打开视频流
                    elif self.cv2_flag == cv2.IMREAD_GRAYSCALE:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # 等待缓冲区变空

    def close(self):
        """终止流式加载器，停止线程并释放视频捕获资源。"""
        self.running = False  # 线程停止标志
        for thread in self.threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)  # 添加超时时间
        for cap in self.caps:  # 遍历保存的视频捕获对象
            if cap is None:
                continue
            try:
                cap.release()  # 释放视频捕获对象
            except Exception as e:
                LOGGER.warning(f"Could not release VideoCapture object: {e}")

    def __iter__(self):
        """返回迭代器对象，并重置帧计数器。"""
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回多个视频流的下一批帧，供后续处理。"""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # 等待每个缓冲区都有可用帧
            while not x:
                if not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"Waiting for stream {i}")

            # 获取并移除图像缓冲区中的第一帧
            if self.buffer:
                images.append(x.pop(0))

            # 获取最后一帧，并清空图像缓冲区中的其他帧
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self) -> int:
        """返回 LoadStreams 对象中视频流的数量。"""
        return self.bs


class LoadScreenshots:
    """用于捕获和处理屏幕图像的 Ultralytics 截图数据加载器。

    此类负责加载截图并将其交给 YOLO 处理，适用于 `yolo predict source=screen` 场景。

    属性：
        screen (int): 要捕获的屏幕编号。
        left (int): 截图区域左边界坐标。
        top (int): 截图区域上边界坐标。
        width (int): 截图区域宽度。
        height (int): 截图区域高度。
        mode (str): 当前模式，设置为表示实时捕获的 `'stream'`。
        frame (int): 已捕获帧的计数器。
        sct (mss.mss): 来自 `mss` 库的屏幕捕获对象。
        bs (int): 批次大小，固定为 1。
        fps (int): 每秒帧数，固定为 30。
        monitor (dict[str, int]): 屏幕监视区域配置。
        cv2_flag (int): OpenCV 图像读取标志（灰度或彩色/BGR）。

    方法：
        __iter__: 返回迭代器对象。
        __next__: 捕获并返回下一张截图。

    示例：
        >>> loader = LoadScreenshots("0 100 100 640 480")  # 屏幕 0，左上角坐标为 (100,100)，尺寸为 640x480
        >>> for sources, imgs, info in loader:
        ...     print(f"捕获帧的尺寸：{imgs[0].shape}")
    """

    def __init__(self, source: str, channels: int = 3):
        """使用指定的屏幕和区域参数初始化截图捕获器。

        参数：
            source (str): 屏幕捕获源字符串，格式为 `"screen_num left top width height"`。
            channels (int): 图像通道数，1 表示灰度图，3 表示彩色图。
        """
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # 默认捕获屏幕 0 的完整区域
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # 灰度或彩色（BGR）

        # 解析监视区域配置
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """返回截图捕获器的迭代器对象。"""
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """使用 mss 库捕获并返回下一张截图的 NumPy 数组。"""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # 从 BGRA 转为 BGR
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im0
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        self.frame += 1
        return [str(self.screen)], [im0], [s]  # 屏幕编号、图像、说明文本


class LoadImagesAndVideos:
    """用于为 YOLO 对象检测加载和处理图像与视频的类。

    此类负责从多种来源加载并预处理图像和视频数据，包括单个图像文件、视频文件以及图像和视频路径列表。

    属性：
        files (列表[str]): 图像和视频文件路径列表。
        nf (int): 文件总数（图像和视频）。
        video_flag (列表[bool]): 文件类型标志，True 表示视频，False 表示图像。
        mode (str): 当前模式，可为 `'image'` 或 `'video'`。
        vid_stride (int): 视频帧采样步长。
        bs (int): 批次大小。
        cap (cv2.VideoCapture): OpenCV 视频捕获对象。
        frame (int): 当前视频帧计数器。
        frames (int): 当前视频的总帧数。
        count (int): 迭代计数器，在 __iter__() 中初始化为 0。
        ni (int): 图像数量。
        cv2_flag (int): OpenCV 图像读取标志（灰度或彩色/BGR）。

    方法：
        __init__: 初始化 LoadImagesAndVideos 对象。
        __iter__: 返回 VideoStream 或 ImageFolder 的迭代器对象。
        __next__: 返回下一批图像或视频帧，以及对应的路径和元数据。
        _new_video: 为指定路径创建新的视频捕获对象。
        __len__: 返回对象中的批次数量。

    示例：
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # 处理一批图像或视频帧
        ...     pass

    注意：
        - 支持多种图像格式，包括 HEIC。
        - 支持读取本地文件和目录。
        - 支持从包含图像和视频路径的文本文件中读取数据。
    """

    def __init__(self, path: str | Path | list, batch: int = 1, vid_stride: int = 1, channels: int = 3):
        """初始化图像和视频数据加载器，支持多种输入格式。

        参数：
            path (str | Path | 列表): 图像或视频路径、目录，或路径列表。
            batch (int): 处理时的批次大小。
            vid_stride (int): 视频帧采样步长。
            channels (int): 图像通道数，1 表示灰度图，3 表示彩色图。
        """
        parent = None
        if isinstance(path, str) and Path(path).suffix in {".txt", ".csv"}:  # 从 txt/csv 文件读取源路径
            parent, content = Path(path).parent, Path(path).read_text()
            path = content.splitlines() if Path(path).suffix == ".txt" else content.split(",")  # 源路径列表
            path = [p.strip() for p in path]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # 不要使用 .resolve()，详见 https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # 使用 glob 模式匹配
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(glob.escape(a), "*.*"))))  # 读取目录中的文件
            elif os.path.isfile(a):
                files.append(a)  # 文件（绝对路径或相对于当前工作目录的路径）
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # 文件（相对于 *.txt 文件所在目录）
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # 将文件划分为图像或视频
        images, videos = [], []
        for f in files:
            suffix = f.rpartition(".")[-1].lower()  # 获取不含点号的小写文件扩展名
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # 文件数量
        self.ni = ni  # 图像数量
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"  # 没有图像时默认使用视频模式
        self.vid_stride = vid_stride  # 视频帧采样步长
        self.bs = batch
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # 灰度或彩色（BGR）
        if any(videos):
            self._new_video(videos[0])  # 初始化第一个视频
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """遍历图像和视频文件，依次返回源路径、图像和元数据。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批图像或视频帧，以及对应的路径和元数据。"""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # 文件列表结束
                if imgs:
                    return paths, imgs, info  # 返回最后一个不完整批次
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # 视频结束或读取失败

                if success:
                    success, im0 = self.cap.retrieve()
                    if success and im0 is not None:
                        if self.cv2_flag == cv2.IMREAD_GRAYSCALE:
                            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None]
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # 视频结束
                            self.count += 1
                            self.cap.release()
                else:
                    # 当前视频结束或打开失败时切换到下一个文件
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                # 处理图像文件
                self.mode = "image"
                im0 = imread(path, flags=self.cv2_flag)  # BGR 格式
                if im0 is None:
                    LOGGER.warning(f"Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # 移动到下一个文件
                if self.count >= self.ni and imgs:  # 图像列表结束，仅返回非空批次
                    break

        return paths, imgs, info

    def _new_video(self, path: str):
        """为指定路径创建新的视频捕获对象，并初始化视频相关属性。"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self) -> int:
        """返回数据集中的批次数量。"""
        return math.ceil(self.nf / self.bs)  # 批次数量


class LoadPilAndNumpy:
    """从 PIL 图像和 NumPy 数组加载图像，以便进行批处理。

    此类负责加载并预处理 PIL 和 NumPy 格式的图像数据，执行基本验证和格式转换，确保图像符合后续处理的要求。

    属性：
        paths (列表[str]): 图像路径列表，或自动生成的文件名列表。
        im0 (列表[np.ndarray]): 以 NumPy 数组形式保存的图像列表。
        mode (str): 当前处理模式，设置为 `'image'`。
        bs (int): 批次大小，等于 `im0` 的长度。

    方法：
        _single_check: 验证单张图像，并将其转换为 NumPy 数组。

    示例：
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        已加载 2 张图像
    """

    def __init__(self, im0: Image.Image | np.ndarray | list, channels: int = 3):
        """初始化 PIL 和 NumPy 图像加载器，并将输入转换为统一格式。

        参数：
            im0 (PIL.Image.Image | np.ndarray | 列表): 单张图像，或 PIL/NumPy 格式的图像列表。
            channels (int): 图像通道数，1 表示灰度图，3 表示彩色图。
        """
        if not isinstance(im0, list):
            im0 = [im0]
        if not im0:  # 空批次会在 Predictor.preprocess 的 np.stack 内部触发无明确提示的错误
            raise FileNotFoundError("No images found in source, predict requires at least one image.")
        # 当 Image.filename 返回空路径时，使用 `image{i}.jpg` 作为文件名。
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im, channels) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)
        self.count = 0

    @staticmethod
    def _single_check(im: Image.Image | np.ndarray, channels: int = 3) -> np.ndarray:
        """验证单张图像，并统一其通道数量。

        注意：
            - PIL 输入会转换为 NumPy 数组，彩色图像按 OpenCV 兼容的 BGR 顺序返回。
            - NumPy 彩色输入默认已经采用 OpenCV 兼容的 BGR 顺序。
        """
        if not isinstance(im, (Image.Image, np.ndarray)):
            raise TypeError(f"Expected PIL/np.ndarray image type, but got {type(im)}")
        pil = isinstance(im, Image.Image)
        if pil:
            flag = "L" if channels == 1 else "RGB"
            im = np.asarray(im.convert(flag))
            im = im[..., None] if flag == "L" else im[..., ::-1]
        im = np.atleast_3d(im)
        # 两种输入路径都在此处完成验证：零维度会导致 LetterBox 除零，批量数组会错误地将形状[2] 当作通道数。
        # 使用异常而不是 assert，这样在 `python -O` 模式下仍会保留检查。
        if im.ndim != 3 or not all(im.shape):
            raise ValueError(f"Expected a single (H, W, C) image, but got array of shape {im.shape}")
        if pil:
            return np.ascontiguousarray(im)
        c = im.shape[2]
        if c == channels:
            return im
        if c == 2:  # 灰度通道 + Alpha 通道
            im, c = im[..., :1], 1
        if c == 1:
            return np.repeat(im, channels, axis=2)
        if channels == 1:
            return cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY if c == 4 else cv2.COLOR_BGR2GRAY)[..., None]
        return np.ascontiguousarray(im[..., :3])

    def __len__(self) -> int:
        """返回 `im0` 属性的长度，即已加载图像的数量。"""
        return len(self.im0)

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批图像、路径和元数据，供后续处理。"""
        if self.count == 1:  # 这是批量推理，因此只迭代一次
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """遍历 PIL/NumPy 图像，依次返回路径、原始图像和元数据。"""
        self.count = 0
        return self


class LoadTensor:
    """用于为对象检测任务加载和处理张量数据的类。

    此类负责加载并预处理来自 PyTorch 张量的图像数据，为后续对象检测流程做好准备。

    属性：
        im0 (torch.Tensor): 包含图像的输入张量，形状为 (B, C, H, W)。
        bs (int): 批次大小，根据 `im0` 的形状推断。
        mode (str): 当前处理模式，设置为 `'image'`。
        paths (列表[str]): 图像路径列表，或自动生成的文件名列表。

    方法：
        _single_check: 验证并格式化输入张量。

    示例：
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"Processed {len(images)} images")
    """

    def __init__(self, im0: torch.Tensor) -> None:
        """初始化 LoadTensor 对象，以处理 torch.Tensor 图像数据。

        参数：
            im0 (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        """
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [f"image{i}.jpg" for i in range(self.bs)]
        self.count = 0

    @staticmethod
    def _single_check(im: torch.Tensor, stride: int = 32) -> torch.Tensor:
        """验证并格式化图像张量，确保形状正确且数值已归一化。"""
        s = (
            f"torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if not all(im.shape) or im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)  # 零维度张量会在后续调用 im.max() 时触发错误
        if im.max() > 1.0 + (torch.finfo(im.dtype).eps if im.is_floating_point() else 0):
            LOGGER.warning(
                f"torch.Tensor 输入应归一化到 0.0-1.0，但当前最大值为 {im.max()}。正在将输入除以 255。"
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """返回用于遍历张量图像数据的迭代器对象。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], torch.Tensor, list[str]]:
        """返回下一批张量图像和元数据，供后续处理。"""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self) -> int:
        """返回张量输入的批次大小。"""
        return self.bs


def autocast_list(source: list[Any]) -> list[Image.Image | np.ndarray]:
    """将源列表转换为 NumPy 数组或 PIL 图像列表，供 Ultralytics 预测使用。"""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # 文件名或 URI
            if str(im).startswith("http"):  # requests 支持 HTTP 308 重定向，而 3.11 以前的 urllib 不支持
                import requests  # 仅在此处导入，避免不必要的慢速导入

                im = BytesIO(requests.get(im).content)
            im = Image.open(im)
            filename = im.filename
            im = ImageOps.exif_transpose(im)
            im.filename = filename
            files.append(im)
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL 图像或 NumPy 图像
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                f"See https://docs.ultralytics.com/modes/predict for supported source types."
            )

    return files


def get_best_youtube_url(url: str, method: str = "pytube") -> str | None:
    """从指定的 YouTube 视频中获取质量最高的 MP4 视频流 URL。

        参数：
        url (str): YouTube 视频的 URL。
        method (str): 提取视频信息所使用的方法，可选值为 `"pytube"`、`"pafy"` 和 `"yt-dlp"`。

    返回：
        (str | None): 质量最高的 MP4 视频流 URL；如果找不到合适的视频流，则返回 None。

    示例：
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    注意：
        - 根据所选方法，需要额外安装 pytubefix、pafy 或 yt-dlp 库。
        - 如果存在，函数会优先选择分辨率至少为 1080p 的视频流。
        - 使用 `"yt-dlp"` 方法时，会查找包含视频编码、无音频且扩展名为 *.mp4 的格式。
    """
    if method == "pytube":
        # 从 pytube 切换到 pytubefix，以解决 https://github.com/pytube/pytube/issues/1954 中的问题
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)  # 按分辨率从高到低排序
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:  # 检查分辨率是否至少为 1080p
                return stream.url

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy

        return pafy.new(url).getbestvideo(preftype="mp4").url

    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # 提取视频信息
        for f in reversed(info_dict.get("formats", [])):  # 通常最佳格式位于列表末尾，因此反向遍历
            # 查找包含视频编码、无音频、扩展名为 *.mp4 且尺寸至少为 1920x1080 的格式
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


# 定义常量
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)
