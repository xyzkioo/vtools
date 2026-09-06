# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import random
import subprocess
import time
import zipfile
from pathlib import Path
from tarfile import is_tarfile
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    ASSETS_URL,
    DATASETS_DIR,
    LOGGER,
    ROOT,
    SETTINGS_FILE,
    YAML,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii, normalize_platform_uri
from ultralytics.utils.downloads import download, safe_download
from ultralytics.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {
    "avif",
    "bmp",
    "dng",
    "heic",
    "heif",
    "jp2",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
}
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # videos
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

DEPTH_PNG_SCALE = 1000  # uint16 默认表示毫米；零值无效


def save_depth_png(path: str | Path, depth: np.ndarray, scale: float = DEPTH_PNG_SCALE) -> None:
    """将以米为单位的深度保存为缩放后的 uint16 PNG，零值保留给无效像素。"""
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be a positive finite number")
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    if valid.any():
        scaled = np.rint(depth[valid] * scale)
        if scaled.max() > np.iinfo(np.uint16).max:
            raise ValueError(f"Depth map exceeds the {np.iinfo(np.uint16).max / scale:g} meter PNG limit")
        encoded[valid] = np.maximum(scaled, 1).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise OSError(f"Failed to save depth map to {path}")


def load_depth(path: str | Path, scale: float = DEPTH_PNG_SCALE) -> np.ndarray:
    """从缩放后的 uint16 PNG 或以米为单位的浮点 NPY 文件加载深度。"""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
        if depth.ndim != 2 or depth.dtype.kind != "f":
            raise ValueError(f"Depth map {path} must be a 2D floating-point NPY array")
        return np.nan_to_num(depth.astype(np.float32, copy=False), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be a positive finite number")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in {"I", "I;16"}:
            raise ValueError(f"Depth PNG {path} must be a 2D uint16 map")
        encoded = np.asarray(image)
    depth = encoded.astype(np.float32)
    depth /= scale
    return depth


def img2label_paths(img_paths: list[str], label_dir: str = "labels", suffix: str = ".txt") -> list[str]:
    """将图像路径转换为标签路径：把 `images` 目录替换为标签目录，并将扩展名替换为 `.txt`。"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}{label_dir}{os.sep}"  # /images/ 和标签目录的路径片段
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + f"{suffix}" for x in img_paths]


def check_file_speeds(
    files: list[str], threshold_ms: float = 10, threshold_mb: float = 50, max_files: int = 5, prefix: str = ""
):
    """检查数据集文件访问速度并提供性能反馈。

    此函数通过测量 ping（stat 调用）耗时和读取速度来测试数据集文件的访问速度，从给定列表中最多抽取
    `max_files` 个文件，并在访问时间超过阈值时发出警告。

    参数：
        files (列表[str]): 要检查访问速度的文件路径列表。
        threshold_ms (float, 可选): ping 耗时警告阈值，单位为毫秒。
        threshold_mb (float, 可选): 读取速度警告阈值，单位为 MB/s。
        max_files (int, 可选): 要检查的最大文件数。
        prefix (str, 可选): 添加到日志消息前的前缀字符串。

    示例：
        >>> from pathlib import Path
        >>> image_files = list(Path("dataset/images").glob("*.jpg"))
        >>> check_file_speeds(image_files, threshold_ms=15)
    """
    if not files:
        LOGGER.warning(f"{prefix}Image speed checks: No files to check")
        return

    # 抽样文件（最多 5 个）
    files = random.sample(files, min(max_files, len(files)))

    # 测试 ping（stat 耗时）
    ping_times = []
    file_sizes = []
    read_speeds = []

    for f in files:
        try:
            # 测量 ping（stat 调用）
            start = time.perf_counter()
            file_size = os.stat(f).st_size
            ping_times.append((time.perf_counter() - start) * 1000)  # ms
            file_sizes.append(file_size)

            # 测量读取速度
            start = time.perf_counter()
            with open(f, "rb") as file_obj:
                _ = file_obj.read()
            read_time = time.perf_counter() - start
            if read_time > 0:  # 避免除零
                read_speeds.append(file_size / (1 << 20) / read_time)  # MB/s
        except Exception:
            pass

    if not ping_times:
        LOGGER.warning(f"{prefix}Image speed checks: failed to access files")
        return

    # 计算带不确定性的统计数据
    avg_ping = np.mean(ping_times)
    std_ping = np.std(ping_times, ddof=1) if len(ping_times) > 1 else 0
    size_msg = f", size: {np.mean(file_sizes) / (1 << 10):.1f} KB"
    ping_msg = f"ping: {avg_ping:.1f}±{std_ping:.1f} ms"

    if read_speeds:
        avg_speed = np.mean(read_speeds)
        std_speed = np.std(read_speeds, ddof=1) if len(read_speeds) > 1 else 0
        speed_msg = f", read: {avg_speed:.1f}±{std_speed:.1f} MB/s"
    else:
        avg_speed = float("inf")
        speed_msg = ""

    if avg_ping < threshold_ms and avg_speed > threshold_mb:
        LOGGER.info(f"{prefix}Fast image access ✅ ({ping_msg}{speed_msg}{size_msg})")
    else:
        LOGGER.warning(
            f"{prefix}Slow image access detected ({ping_msg}{speed_msg}{size_msg}). "
            f"Use local storage instead of remote/mounted storage for better performance. "
            f"See https://docs.ultralytics.com/guides/model-training-tips"
        )


def get_hash(paths: list[str]) -> str:
    """根据文件或目录路径列表返回单个哈希值。"""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # 对文件大小计算哈希
    h.update("".join(paths).encode())  # 对路径计算哈希
    return h.hexdigest()  # 返回哈希值


def exif_size(img: Image.Image) -> tuple[int, int]:
    """返回经过 EXIF 校正的 PIL 图像尺寸。"""
    s = img.size  # （宽度，高度）
    if img.format == "JPEG":  # 仅支持 JPEG 图像
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # EXIF 方向标签的键为 274
                if rotation in {6, 8}:  # 旋转 270 度或 90 度
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def check_image(im_file: str) -> tuple[str, tuple[int, int]]:
    """检查图像文件的完整性，并在发现损坏的 JPEG 时进行修复。

    参数：
        im_file (str): 要检查的图像文件路径。

    返回：
        (str): 描述所执行修复操作的消息；图像有效时返回空字符串。
        (tuple[int, int]): 图像尺寸，格式为像素高度和宽度 `(高度, 宽度)`。

    异常：
        AssertionError: 图像任一维度小于 10 像素或格式无效时抛出。
    """
    msg = ""
    im = Image.open(im_file)
    im.verify()  # PIL 完整性检查
    shape = exif_size(im)  # 图像尺寸
    shape = (shape[1], shape[0])  # hw
    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
    assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
    if im.format.lower() in {"jpg", "jpeg"}:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # 损坏的 JPEG
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                msg = f"{im_file}: corrupt JPEG restored and saved"
    return msg, shape


def verify_image(args: tuple) -> tuple:
    """检查单张图像。"""
    (im_file, cls), prefix = args
    # 数量（找到、损坏）和消息
    nf, nc, msg = 0, 0, ""
    try:
        msg = check_image(im_file)[0]
        msg = f"{prefix}{msg}" if msg else ""
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_depth(args: tuple) -> tuple:
    """检查图像及其配对深度图是否存在且可读取。"""
    im_file, depth_file, prefix, scale = args
    # 数量（找到、缺失、损坏）和消息
    nf, nm, nc, msg = 0, 0, 0, ""
    try:
        msg, shape = check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""
        if not os.path.isfile(depth_file):
            nm = 1
            msg = f"{prefix}{im_file}: ignoring image with missing depth map {depth_file}"
            return None, None, nf, nm, nc, msg
        if Path(depth_file).suffix.lower() == ".npy":
            depth = np.load(depth_file, mmap_mode="r", allow_pickle=False)
            assert depth.ndim == 2 and depth.dtype.kind == "f", "depth NPY must be 2D and floating-point"
            depth_shape = depth.shape
        else:
            assert (
                isinstance(scale, (int, float)) and not isinstance(scale, bool) and np.isfinite(scale) and scale > 0
            ), "depth_scale must be a positive finite number"
            with Image.open(depth_file) as depth:
                assert depth.format == "PNG" and depth.mode in {"I", "I;16"}, (
                    f"depth map {depth_file} must be an integer grayscale PNG"
                )
                depth_shape = (depth.height, depth.width)
                depth.verify()
        assert abs(np.log((depth_shape[1] / depth_shape[0]) / (shape[1] / shape[0]))) <= 0.02, (
            f"depth map shape {depth_shape} does not match image shape {shape}"
        )
        nf = 1
        return im_file, shape, nf, nm, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/depth: {e}"
    return None, None, nf, nm, nc, msg


def verify_image_mask(args: tuple) -> tuple:
    """检查图像及其语义掩码是否存在、可读取且尺寸匹配。"""
    im_file, mask_file, prefix, check_bit_depth = args
    # 数量（找到、缺失、损坏）和消息
    nf, nm, nc, msg = 0, 0, 0, ""
    try:
        msg, shape = check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""
        if not os.path.isfile(mask_file):
            for ext in IMG_FORMATS:  # 检查其他后缀
                alt_mask_file = mask_file.rsplit(".", 1)[0] + f".{ext}"
                if os.path.isfile(alt_mask_file):
                    mask_file = alt_mask_file
                    break
        if os.path.isfile(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"mask file {mask_file} is unreadable"
            assert mask.shape[:2] == shape, f"mask size {mask.shape[:2]} does not match image size {shape}"
            is_1bit = False
            if check_bit_depth:
                with Image.open(mask_file) as im:
                    is_1bit = im.mode == "1"
            nf = 1
        else:
            nm = 1
            msg = f"{prefix}{im_file}: ignoring image with missing mask {mask_file}"
            return None, None, None, None, nm, nf, nc, msg
        return im_file, mask_file, shape, is_1bit, nm, nf, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/mask: {e}"
    return None, None, None, None, nm, nf, nc, msg


def verify_image_label(args: tuple) -> list:
    """检查单个图像与标签对。"""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    # 数量（缺失、找到、空白、损坏）、消息、分割段和关键点
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # 检查图像
        msg, shape = check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""

        # 检查标签
        if os.path.isfile(lb_file):
            nf = 1  # 找到标签
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # 是否为分割标注
                    assert not any(len(x) == 5 for x in lb), "labels mix segment and detection rows"
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # （cls, xy1...）
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # （cls, xywh）
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                # 检查坐标点，允许 1% 误差
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"

                # 检查所有标签
                max_cls = 0 if single_cls else lb[:, 0].max()  # 最大标签编号
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # 检查重复行
                    lb = lb[i]  # 移除重复项
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def visualize_image_annotations(image_path: str, txt_path: str, label_map: dict[int, str]):
    """在图像上可视化 YOLO 标注（边界框和类别标签）。

    此函数读取图像及其 YOLO 格式的标注文件，为检测到的对象绘制边界框，并使用对应类别名称进行标注。
    边界框颜色根据类别 ID 分配，文字颜色则根据背景亮度动态调整，以保证可读性。

    参数：
        image_path (str): 待标注图像的路径，文件必须可由 PIL 读取。
        txt_path (str): YOLO 格式标注文件的路径，文件中每行应对应一个对象。
        label_map (dict[int, str]): 将类别 ID（整数）映射到类别标签（字符串）的字典。

    示例：
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # 应包含所有已标注类别
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    _, ax = plt.subplots(1)  # 绘制图像和标注
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, False))  # 获取并归一化供 Matplotlib 使用的 RGB 颜色
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # 创建矩形
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # 亮度计算公式
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()


def polygon2mask(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    """将多边形列表转换为指定图像尺寸的二值掩码。

    参数：
        imgsz (tuple[int, int]): 图像尺寸，格式为 `(高度, 宽度)`。
        polygons (列表[np.ndarray]): 多边形列表。每个多边形是一维坐标数组，长度为 M，且 M % 2 = 0（x、y 交替排列）。
        color (int, 可选): 在掩码中填充多边形时使用的颜色值。
        downsample_ratio (int, 可选): 掩码的下采样比例。

    返回：
        (np.ndarray): 指定图像尺寸的二值掩码，其中已填充给定多边形。
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # 注意：先填充多边形再缩放，是为了在 downsample_ratio=1 时保持相同的损失计算方式
    return cv2.resize(mask, (nw, nh))


def polygons2masks(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int = 1
) -> np.ndarray:
    """将多边形列表转换为指定图像尺寸的一组二值掩码。

    参数：
        imgsz (tuple[int, int]): 图像尺寸，格式为 `(高度, 宽度)`。
        polygons (列表[np.ndarray]): 多边形列表。每个多边形都是可重塑为 `(-1, 2)` 的坐标数组，表示 `(x, y)` 点对。
        color (int): 在掩码中填充多边形时使用的颜色值。
        downsample_ratio (int, 可选): 每个掩码的下采样比例。

    返回：
        (np.ndarray): 指定图像尺寸的一组二值掩码，其中已填充给定多边形。
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(
    imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """返回下采样后的重叠掩码和按面积排序的索引。"""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for segment in segments:
        mask = polygon2mask(
            imgsz,
            [segment.reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    # 使用运行最大值：旧的 `masks + mask` 求和在重叠实例超过 128 个时会达到 2 * i + 1 并溢出 uint8
    for i in range(len(segments)):
        np.maximum(masks, ms[i] * (i + 1), out=masks)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """查找并返回 Detect、Segment 或 Pose 数据集对应的 YAML 文件。

    此函数首先在给定目录的根目录中搜索 YAML 文件；如果未找到，则执行递归搜索。
    如果存在与给定路径主文件名相同的 YAML 文件，则优先返回该文件。

    参数：
        path (Path): 要搜索 YAML 文件的目录路径。

    返回：
        (Path): 找到的 YAML 文件路径。
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # 先搜索根目录，再递归搜索
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # 优先选择主文件名匹配的 YAML 文件
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def convert_ndjson_to_yolo_if_needed(data: str | Path) -> str | Path:
    """在需要时将 NDJSON 数据集或 Platform 数据集 URI 转换为 YOLO 格式。"""
    data = normalize_platform_uri(data)  # 接受 Platform 网页 URL
    data_str = str(data)
    if clean_url(data_str).endswith(".ndjson") or (data_str.startswith("ul://") and "/datasets/" in data_str):
        import asyncio

        from ultralytics.data.converter import convert_ndjson_to_yolo

        return asyncio.run(convert_ndjson_to_yolo(data))
    return data


def check_det_dataset(dataset: str, autodownload: bool = True, split: str = "") -> dict[str, Any]:
    """在本地找不到数据集时下载、验证并按需解压数据集。

    此函数检查指定数据集是否可用；如果找不到，可选择下载并解压数据集。随后读取并解析配套 YAML 数据，
    确保满足关键要求，并解析与数据集相关的路径。

    参数：
        dataset (str): 数据集或数据集描述文件（例如 YAML 文件）的路径。
        autodownload (bool, 可选): 找不到数据集时是否自动下载。
        split (str, 可选): 调用方所需的数据集划分。

    返回：
        (dict[str, Any]): 解析后的数据集信息和路径。
    """
    dataset = str(dataset)
    if "://" not in dataset and not Path(dataset).exists() and Path(dataset).suffix not in {".yaml", ".yml"}:
        # 允许只提供数据集名称，例如将 'coco8' 转为 'coco8.yaml'
        dataset = next((f"{dataset}{x}" for x in (".yaml", ".yml") if check_file(f"{dataset}{x}", hard=False)), dataset)
    file = Path(check_file(dataset))
    if file.is_dir():
        file = find_dataset_yaml(file)

    # 下载（可选）
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # 读取 YAML
    data = YAML.load(file, append_filename=True)  # 字典

    # 检查
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.warning("renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # 将 validation 键替换为 val 键
    if split and not data.get(split):
        raise FileNotFoundError(f"{dataset} '{split}:' images not found ❌")
    # `names` 与 None 比较，而不是检查键是否存在：单独的 `names:` 会解析为 None，下面调用 len(None) 会报错。
    # `nc` 仍检查键是否存在，使没有值的 `nc:` 能继续触发“必须是整数”的错误。
    if data.get("names") is None and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "nc" in data and not isinstance(data["nc"], int):
        try:
            nc = float(data["nc"])  # 接受类似整数的值，例如 '10' 或 10.0，但不接受 1.9 或占位符
            if nc != int(nc):
                raise ValueError
            data["nc"] = int(nc)
        except (TypeError, ValueError):
            raise SyntaxError(emojis(f"{dataset} 'nc: {data['nc']}' must be an integer ❌."))
    if data.get("names") is not None and data.get("nc") is not None and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if data.get("names") is None:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])
    data["channels"] = data.get("channels", 3)  # 获取图像通道数，默认为 3

    # 解析路径
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # 数据集根目录
    if not path.exists() and not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()  # 相对于 DATASETS_DIR 的路径

    # 设置路径
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # 添加根路径前缀
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # 解析 YAML
    val, s = (data.get(x) for x in (split or "val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 验证集路径
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # 移除 URL 身份验证信息后的数据集名称
            LOGGER.info("")
            m = f"Dataset '{name}' images not found, missing path '{next(x for x in val if not x.exists())}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # 成功
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # Bash 脚本
                LOGGER.info(f"Running {s} ...")
                subprocess.run(s.split(), check=True)
            else:  # Python 脚本
                exec(s, {"yaml": data})  # noqa: S102
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # 下载字体

    return data  # 字典


def check_cls_dataset(dataset: str | Path, split: str = "") -> dict[str, Any]:
    """检查 ImageNet 等图像分类数据集。

    此函数接受数据集名称，并尝试获取对应的数据集信息。如果本地找不到数据集，则尝试从互联网下载并保存到本地。

    参数：
        dataset (str | Path): 数据集名称。
        split (str, 可选): 数据集划分，可选值为 `'val'`、`'test'` 或空字符串。

    返回：
        (dict[str, Any]): 包含以下键的字典：

            - 'train' (Path)：包含训练集的数据目录路径。
            - 'val' (Path)：包含验证集的数据目录路径。
            - 'test' (Path)：包含测试集的数据目录路径。
            - 'nc' (int)：数据集中的类别数量。
            - 'names' (dict[int, str])：数据集类别名称字典。
    """
    if split and split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid classification dataset split '{split}'. Use 'train', 'val', or 'test'.")

    # 下载（如果直接传入 dataset=https://file.zip，则可选）
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif str(dataset).endswith((".zip", ".tar", ".gz")):
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        if data_dir.suffix != "":
            raise ValueError(
                f'Classification datasets must be a directory (data="path/to/dir") not a file (data="{dataset}"), '
                "See https://docs.ultralytics.com/datasets/classify"
            )
        LOGGER.info("")
        LOGGER.warning(f"Dataset not found, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(["bash", str(ROOT / "data/scripts/get_imagenet.sh")], check=True)
        else:
            download(f"{ASSETS_URL}/{dataset}.zip", dir=data_dir.parent)
        LOGGER.info(f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n")
    train_set = data_dir / "train"
    if not train_set.is_dir():
        LOGGER.warning(f"Dataset 'split=train' not found at {train_set}")
        if image_files := [f for f in data_dir.rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]:
            from ultralytics.data.split import split_classify_dataset

            LOGGER.info(f"Found {len(image_files)} images in subdirectories. Attempting to split...")
            data_dir = split_classify_dataset(data_dir, train_ratio=0.8)
            train_set = data_dir / "train"
        else:
            raise FileNotFoundError(f"No images found in {data_dir} or its subdirectories.")
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else data_dir / "valid"
        if (data_dir / "valid").exists()
        else None
    )  # 数据/test or 数据/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # 数据/val or 数据/test
    if split == "val" and not val_set:
        LOGGER.warning("Dataset 'split=val' not found, using 'split=test' instead.")
        val_set = test_set
    elif split == "test" and not test_set:
        LOGGER.warning("Dataset 'split=test' not found, using 'split=val' instead.")
        test_set = val_set

    if (ndjson_names := data_dir / ".ndjson.yaml").is_file():
        names = YAML.load(ndjson_names)["names"]
    else:
        names = dict(enumerate(sorted(x.name for x in (data_dir / "train").iterdir() if x.is_dir())))
    nc = len(names)

    # 输出到控制台
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)  # 文件数量
            nd = len({file.parent for file in files})  # 目录数量
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(f"{dataset} '{k}:' no training images found")
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes (no images found)")
            elif nd != nc and not ndjson_names.is_file():
                LOGGER.error(f"{prefix} found {nf} images in {nd} classes (requires {nc} classes, not {nd})")
            else:
                class_count = f"{nd}/{nc}" if ndjson_names.is_file() else nd
                LOGGER.info(f"{prefix} found {nf} images in {class_count} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names, "channels": 3}


def compress_one_image(f: str, f_new: str | None = None, max_dim: int = 1920, quality: int = 50):
    """使用 Python Imaging Library（PIL）或 OpenCV 压缩单张图像文件，在保持宽高比和质量的同时缩小尺寸。
    如果输入图像小于最大尺寸，则不会调整大小。

    参数：
        f (str): 输入图像文件路径。
        f_new (str, 可选): 输出图像文件路径；未指定时覆盖输入文件。
        max_dim (int, 可选): 输出图像宽度或高度允许的最大尺寸。
        quality (int, 可选): 图像压缩质量百分比。

    示例：
        >>> from pathlib import Path
        >>> from ultralytics.data.utils import compress_one_image
        >>> for f in Path("path/to/dataset").rglob("*.jpg"):
        >>>    compress_one_image(f)
    """
    try:  # 使用 PIL
        Image.MAX_IMAGE_PIXELS = None  # 修复 DecompressionBombError，允许处理超过约 1.789 亿像素的图像
        im = Image.open(f)
        if im.mode in {"RGBA", "LA"}:  # 必要时转换为 RGB（用于 JPEG）
            im = im.convert("RGB")
        r = max_dim / max(im.height, im.width)  # 缩放比例
        if r < 1.0:  # 图像过大
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # 保存
    except Exception as e:  # 使用 OpenCV
        LOGGER.warning(f"Image compression PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # 缩放比例
        if r < 1.0:  # 图像过大
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def load_dataset_cache_file(path: Path) -> dict:
    """从路径加载 Ultralytics 的 `*.cache` 字典。"""
    import gc

    gc.disable()  # 减少 pickle 加载时间
    cache = np.load(str(path), allow_pickle=True).item()  # 加载字典
    gc.enable()
    return cache


def save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str):
    """将 Ultralytics 数据集 `*.cache` 字典 x 保存到路径。"""
    x["version"] = version  # 添加缓存版本
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # 如果存在则删除 *.cache 文件
        try:
            with open(str(path), "wb") as file:  # 此处使用上下文管理器可修复 Windows 异步 np.save 问题
                np.save(file, x)
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            Path(path).unlink(missing_ok=True)  # 删除未完整写入的文件
            LOGGER.warning(f"{prefix}WARNING ⚠️ Failed to save cache to {path}: {e}")
    else:
        LOGGER.warning(f"{prefix}Cache directory {path.parent} is not writable, cache not saved.")


def add_polygon_background(data: dict) -> dict:
    """为没有 `masks_dir` 的多边形语义数据集设置背景类别。

    - nc > 1：在 id=nc 处追加 `background` 类别，并将 `data['nc']` 增加到 nc+1；多边形的 cls 值保持为前景类别 ID。
    - nc == 1：保持 nc=1（二值分割）。无论标签 cls 值如何，多边形栅格化都会得到 `{0=背景, 1=前景}` 掩码。
    """
    if data.get("masks_dir") or data.get("_polygon_bg_added"):
        return data
    nc = int(data.get("nc") or len(data.get("names") or {}))
    if nc == 1:  # 二值分割：背景=0、前景=1（隐式）；模型在单个输出通道上使用 BCE
        data["bg_class_idx"] = 0
    else:
        names = dict(data.get("names") or {})
        names[nc] = "background"
        data["bg_class_idx"] = nc
        data["nc"] = nc + 1
        data["names"] = names
    data["_polygon_bg_added"] = True
    return data
