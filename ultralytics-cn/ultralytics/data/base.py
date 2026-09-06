# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """用于加载和处理图像数据的基础数据集类。

    此类提供加载图像、缓存图像以及为目标检测任务准备训练和推理数据的核心功能。

    属性：
        img_path (str | 列表[str]): 包含图像的文件夹路径。
        imgsz (int): 调整图像尺寸时使用的目标尺寸。
        augment (bool): 是否应用数据增强。
        single_cls (bool): 是否将所有目标视为同一类别。
        prefix (str): 日志消息的打印前缀。
        fraction (float): 使用的数据集比例。
        channels (int): 图像通道数（灰度图为 1，彩色图为 3）。OpenCV 加载的彩色图像通道顺序为 BGR。
        cv2_flag (int): 读取图像时使用的 OpenCV 标志。
        im_files (列表[str]): 图像文件路径列表。
        labels (列表[dict]): 标签数据字典列表。
        ni (int): 数据集中的图像数量。
        rect (bool): 是否使用矩形训练。
        batch_size (int): 批次大小。
        stride (int): 模型使用的步长。
        pad (float): 填充值。
        buffer (列表): 马赛克图像缓冲区。
        max_buffer_length (int): 缓冲区最大尺寸。
        ims (列表): 已加载图像列表。
        im_hw0 (列表): 原始图像尺寸 (h, w) 列表。
        im_hw (列表): 调整后图像尺寸 (h, w) 列表。
        npy_files (列表[Path]): numpy 文件路径列表。
        cache (str | None): 缓存设置（'ram'、'disk' 或 None 表示不缓存）。
        transforms (callable): 图像变换函数。
        batch_shapes (np.ndarray): 矩形训练使用的批次形状。
        batch (np.ndarray): 每张图像对应的批次索引。

    方法：
        get_img_files: 从指定路径读取图像文件。
        update_labels: 更新标签，仅保留指定类别。
        load_image: 从数据集中加载图像。
        cache_images: 将图像缓存到内存或磁盘。
        cache_images_to_disk: 将图像保存为 *.npy 文件以便快速加载。
        check_cache_disk: 检查图像缓存需求与可用磁盘空间。
        check_cache_ram: 检查图像缓存需求与可用内存。
        set_rectangle: 按宽高比对图像排序，并设置矩形训练的批次形状。
        get_image_and_label: 获取并返回数据集中的标签信息。
        update_labels_info: 由子类实现的自定义标签格式方法。
        build_transforms: 由子类实现的变换流水线构建方法。
        get_labels: 由子类实现的标签获取方法。
    """

    class _ImageCache:
        """将图像存储在连续数组中，以保留工作进程之间的写时复制共享。"""

        def __init__(self, images: list[np.ndarray]):
            """将图像及其布局打包为连续的 NumPy 数组。"""
            self.shapes = np.array([im.shape for im in images])
            self.dtypes = np.array([im.dtype.str for im in images])
            self.offsets = np.concatenate(([0], np.cumsum([im.nbytes for im in images])))
            self.buffer = np.empty(self.offsets[-1], dtype=np.uint8)
            for i, im in enumerate(images):
                self.buffer[self.offsets[i] : self.offsets[i + 1]] = im.reshape(-1).view(np.uint8)
                images[i] = None

        def __getitem__(self, i: int) -> np.ndarray:
            """根据索引返回图像视图。"""
            i = range(len(self.shapes))[i]
            return self.buffer[self.offsets[i] : self.offsets[i + 1]].view(self.dtypes[i]).reshape(self.shapes[i])

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """使用给定配置和选项初始化 BaseDataset。

        参数：
            img_path (str | 列表[str]): 包含图像的文件夹路径，或图像路径列表。
            imgsz (int): 调整图像尺寸时使用的目标尺寸。
            cache (bool | str): 训练期间将图像缓存到内存或磁盘。
            augment (bool): 为 True 时应用数据增强。
            hyp (dict[str, Any]): 用于数据增强的超参数。
            prefix (str): 日志消息的打印前缀。
            rect (bool): 为 True 时使用矩形训练。
            batch_size (int): 批次大小。
            stride (int): 模型使用的步长。
            pad (float): 填充值。
            single_cls (bool): 为 True 时使用单类别训练。
            classes (列表[int], 可选): 要包含的类别列表。
            fraction (float): 要使用的数据集比例。
            channels (int): 图像通道数（灰度图为 1，彩色图为 3）。使用 OpenCV 加载的彩色图像采用 BGR 通道顺序。
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # 单类别过滤和类别筛选
        self.ni = len(self.labels)  # 图像数量
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # 用于马赛克图像的缓冲线程
        self.buffer = []  # 缓冲区大小等于批次大小
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 缓存图像（选项可以是 cache = True、False、None、"ram" 或 "disk"）
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                        "cache='ram' 可能产生非确定性的训练结果。"
                        "如果磁盘空间允许，可使用 cache='disk' 作为确定性替代方案。"
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # 变换
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """从指定路径读取图像文件。

        参数：
            img_path (str | 列表[str]): 图像目录或文件的路径，或由这些路径组成的列表。

        返回：
            (列表[str]): 图像文件路径列表。

        异常：
            FileNotFoundError: 找不到图像或指定路径不存在时抛出。
        """
        try:
            f = []  # 图像 文件
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # 目录
                    f += glob.glob(str(Path(glob.escape(p)) / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib 路径处理方式
                elif p.is_file():  # 文件
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # 将本地路径转换为全局路径
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # 从本地路径转换为全局路径（pathlib）
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib 路径处理方式
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # 保留部分数据集
        check_file_speeds(im_files, prefix=self.prefix)  # 检查图像读取速度
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """更新标签，仅保留指定类别。

        参数：
            include_class (列表[int], 可选): 要包含的类别列表。为 None 时包含所有类别。
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i].get("keypoints")
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:] = 0

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """从数据集索引 'i' 加载图像。

        参数：
            i (int): 要加载的图像索引。
            rect_mode (bool): 是否使用矩形缩放（将长边缩放到 imgsz）。
            resize_short (bool): 是否在保持宽高比的同时将短边缩放到 imgsz。为 True 时覆盖 rect_mode。

        返回：
            im (np.ndarray): 加载后的图像 NumPy 数组。
            hw_original (tuple[int, int]): 原始图像尺寸，格式为 (高度, 宽度)。
            hw_resized (tuple[int, int]): 缩放后图像尺寸，格式为 (高度, 宽度)。

        异常：
            FileNotFoundError: 找不到图像文件时抛出。
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # 未缓存到内存
            if fn.exists():  # 加载 npy
                try:
                    im = np.load(fn)
                    npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                    if npy_channels != self.channels:
                        LOGGER.warning(
                            f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, expected {self.channels}"
                        )
                        Path(fn).unlink(missing_ok=True)
                        im = imread(f, flags=self.cv2_flag)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read 图像
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # 保持宽高比，将长边缩放到 imgsz
                if resize_short:  # 保持宽高比，将短边缩放到 imgsz
                    r = self.imgsz / min(h0, w0)  # ratio
                    if r != 1:  # 尺寸不相等时调整大小
                        w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # 尺寸不相等时调整大小
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # 通过拉伸图像调整为正方形 imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # 训练时使用数据增强则添加到缓冲区
            if self.augment and self.cache != "ram":
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # 防止缓冲区为空
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """将图像缓存到内存或磁盘，以加快训练速度。"""
        b, gb = 0, 1 << 30  # 缓存图像的字节数，每 GB 对应的字节数
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()
        if self.cache == "ram":
            self.ims = self._ImageCache(self.ims)

    def cache_images_to_disk(self, i: int) -> None:
        """将图像保存为 *.npy 文件，以加快加载速度。"""
        f = self.npy_files[i]
        if not f.exists():
            try:
                np.save(f.as_posix(), imread(self.im_files[i], flags=self.cv2_flag), allow_pickle=False)
            except Exception as e:
                f.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ Failed to cache image {f}: {e}")

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """检查磁盘空间是否足够缓存图像。

        参数：
            safety_margin (float): Safety margin factor for disk space calculation.

        返回：
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # 缓存图像的字节数，每 GB 对应的字节数
        n = min(self.ni, 30)  # 根据最多 30 张随机图像估算
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # 将数据集缓存到磁盘所需的字节数
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 1.0) -> bool:
        """检查内存是否足够缓存图像。

        参数：
            safety_margin (float): Safety margin factor for RAM calculation.

        返回：
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # 缓存图像的字节数，每 GB 对应的字节数
        n = min(self.ni, 30)  # 根据最多 30 张随机图像估算
        for _ in range(n):
            b += self.load_image(random.randrange(self.ni))[0].nbytes
        mem_required = b * self.ni / n * (1 + safety_margin)  # 将数据集缓存到 RAM 所需的 GB 数
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """按宽高比对图像排序，并为矩形训练设置批次形状。"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch 索引
        nb = bi[-1] + 1  # 批次数量

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # 设置 训练 图像 shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # 图像的批次索引

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回给定索引对应的变换后标签信息。"""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """获取并返回数据集中的标签信息。

        参数：
            索引 (int): Index of 图像 to retrieve.

        返回：
            (dict[str, Any]): 包含图像和元数据的标签字典。
        """
        label = deepcopy(self.labels[index])  # 需要 deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # 形状仅用于矩形训练，删除它
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # 用于评估
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """返回数据集标签列表的长度。"""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """在此处自定义标签格式。"""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """用户可以在此处自定义数据增强。

        示例：
            >>> if self.augment:
            ...     # 训练变换
            ...     return Compose([])
            >>> else:
            ...    # 验证变换
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """用户可以在此处自定义自己的格式。

        示例：
            确保输出是包含以下键的字典：
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # 格式：(height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # 或 False
            ...     bbox_format="xyxy",  # 或 xywh、ltwh
            ... )
        """
        raise NotImplementedError
