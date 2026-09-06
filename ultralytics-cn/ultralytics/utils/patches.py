# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""用于更新或扩展现有函数功能的猴子补丁。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# OpenCV 多语言文件名支持函数 -----------------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """从文件读取图像，并支持包含多语言字符的文件名。

    参数：
        filename (str | Path): 要读取的文件路径。
        flags (int, 可选): 可取 cv2.IMREAD_* 中的值，用于控制图像的读取方式。

    返回：
        (np.ndarray | None): 读取到的图像数组；读取失败时返回 None。

    示例：
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    filename = str(filename)
    try:
        file_bytes = np.fromfile(filename, np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # 处理多帧 TIFF 和彩色图像
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        # 对 OpenCV imdecode 可能不支持的格式（AVIF、HEIC、HEIF）使用备用方案
        if im is None and filename.lower().endswith((".avif", ".heic", ".heif")):
            im = _imread_pil(filename, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im  # 始终确保图像具有 3 个维度


# PIL 补丁 ------------------------------------------------------------------------------------------------------------
_image_open = Image.open  # copy to avoid recursion errors
_pil_plugins_registered = False


def image_open(filename, *args, **kwargs):
    """使用 PIL 打开图像，并在首次失败时按需注册 HEIF 插件。

    此猴子补丁通过 pi-heif（轻量级、仅支持解码）为 PIL.Image.open 增加 HEIC/HEIF 支持，只有实际需要时才导入该包，
    从而避免约 800 毫秒的启动开销。Pillow 12 及更高版本原生支持 AVIF，无需额外插件。

    参数：
        filename (str): 图像文件路径。
        *args (Any): 传递给 PIL.Image.open 的其他位置参数。
        **kwargs (Any): 传递给 PIL.Image.open 的其他关键字参数。

    返回：
        (PIL.Image.Image): 打开的 PIL 图像。
    """
    global _pil_plugins_registered
    if _pil_plugins_registered:
        return _image_open(filename, *args, **kwargs)
    try:
        return _image_open(filename, *args, **kwargs)
    except Exception:
        from ultralytics.utils.checks import check_requirements

        check_requirements("pi-heif")
        from pi_heif import register_heif_opener

        register_heif_opener()
        _pil_plugins_registered = True
        return _image_open(filename, *args, **kwargs)


Image.open = image_open  # apply patch


def _imread_pil(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """使用 PIL 读取图像，作为 OpenCV 不支持格式的备用方案。

    参数：
        filename (str): 要读取的文件路径。
        flags (int, 可选): OpenCV 的 imread 标志，用于确定是否转换为灰度图。

    返回：
        (np.ndarray | None): 读取到的 BGR 格式图像数组；读取失败时返回 None。
    """
    try:
        with Image.open(filename) as img:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(img.convert("L"))
            return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imread_unicode(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像并支持包含多语言字符的文件名，同时保持 cv2.imread 的原生行为。

    此函数用于在 Windows 上替换 cv2.imread。与 `imread` 不同，它不会扩展灰度图维度，也不处理 TIFF、AVIF 或 HEIC
    格式的备用读取。

    参数：
        filename (str | Path): 要读取的文件路径。
        flags (int, 可选): 可取 cv2.IMREAD_* 中的值。

    返回：
        (np.ndarray | None): 读取到的图像数组；读取失败时返回 None。
    """
    try:
        return cv2.imdecode(np.fromfile(filename, np.uint8), flags)
    except (FileNotFoundError, OSError):
        return None


def imwrite(filename: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    """将图像写入文件，并支持包含多语言字符的文件名。

    参数：
        filename (str): 要写入的文件路径。
        img (np.ndarray): 要写入的图像。
        params (列表[int], 可选): 图像编码的其他参数。

    返回：
        (bool): 文件写入成功返回 True，否则返回 False。

    示例：
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建黑色图像
        >>> success = imwrite("output.jpg", img)  # 将图像写入文件
        >>> print(success)
        True
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray) -> None:
    """在指定窗口中显示图像，并支持包含多语言字符的窗口名称。

    此函数封装了 OpenCV 的 imshow，用于在命名窗口中显示图像。它会先对多语言窗口名称进行适当编码，
    以确保与 OpenCV 兼容。

    参数：
        winname (str): 显示图像的窗口名称。如果已存在同名窗口，图像将在该窗口中显示。
        mat (np.ndarray): 要显示的图像，应为表示图像的有效 NumPy 数组。

    示例：
        >>> import numpy as np
        >>> img = np.zeros((300, 300, 3), dtype=np.uint8)  # 创建黑色图像
        >>> img[:100, :100] = [255, 0, 0]  # 添加蓝色方块
        >>> imshow("Example Window", img)  # 显示图像
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch 函数 ---------------------------------------------------------------------------------------------------------
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """使用更新后的参数加载 PyTorch 模型，以避免警告。

    此函数封装 torch.load，并为 PyTorch 1.13.0 及更高版本增加 `weights_only` 参数，以避免警告。

    参数：
        *args (Any): 要传递给 torch.load 的可变长度位置参数列表。
        **kwargs (Any): 要传递给 torch.load 的任意关键字参数。

    返回：
        (Any): 加载得到的 PyTorch 对象。

    注意：
        对于 PyTorch 1.13 及更高版本，如果未提供该参数，此函数会自动设置 `weights_only=False`，以避免弃用警告。
    """
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return torch.load(*args, **kwargs)


def torch_save(*args, **kwargs):
    """保存 PyTorch 对象，并通过重试机制提高稳定性。

    此函数封装 torch.save。保存失败时最多重试 3 次，并采用指数退避，这些失败可能由设备刷新延迟或杀毒软件扫描造成。

    参数：
        *args (Any): 要传递给 torch.save 的位置参数。
        **kwargs (Any): 要传递给 torch.save 的关键字参数。

    示例：
        >>> model = torch.nn.Linear(10, 1)
        >>> torch_save(model.state_dict(), "model.pt")
    """
    for i in range(4):  # 最多重试 3 次
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError:  # 无法保存，可能正在等待设备刷新或杀毒软件扫描
            if i == 3:
                raise
            time.sleep((2**i) / 2)  # 指数退避：0.5 秒、1.0 秒、2.0 秒


@contextmanager
def arange_patch(dynamic: bool = False, quantize: int | str | None = None, fmt: str = ""):
    """解决 ONNX 中 torch.arange 与 FP16 不兼容的问题。

    https://github.com/pytorch/pytorch/issues/148041.
    """
    if dynamic and quantize == 16 and fmt == "onnx":
        func = torch.arange

        def arange(*args, dtype=None, **kwargs):
            """封装 torch.arange，在创建张量后转换 dtype，而不是直接传入该参数。"""
            return func(*args, **kwargs).to(dtype)  # 转换为目标 dtype，而不是直接传入 dtype

        torch.arange = arange  # patch
        yield
        torch.arange = func  # unpatch
    else:
        yield


@contextmanager
def onnx_export_patch():
    """解决 PyTorch 2.9 及更高版本启用 Dynamo 时的 ONNX 导出问题。"""
    from ultralytics.utils.torch_utils import TORCH_2_9

    if TORCH_2_9:
        func = torch.onnx.export

        def torch_export(*args, **kwargs):
            """禁用 Dynamo，将模型导出为 ONNX 格式，以确保兼容性。"""
            return func(*args, **kwargs, dynamo=False)

        torch.onnx.export = torch_export  # patch
        yield
        torch.onnx.export = func  # unpatch
    else:
        yield


@contextmanager
def override_configs(args, overrides: dict[str, Any] | None = None):
    """临时覆盖 args 中配置项的上下文管理器。

    参数：
        args (IterableSimpleNamespace): 原始配置参数。
        overrides (dict[str, Any] | None): 要应用的覆盖配置字典。

    Yields:
        (IterableSimpleNamespace): 已应用覆盖配置的参数对象。
    """
    if overrides:
        original_args = copy(args)
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            yield args
        finally:
            args.__dict__.update(original_args.__dict__)
    else:
        yield args
