# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import TORCH_1_10


def _gaussian_filter1d(y, sigma: int = 3, truncate: float = 4.0) -> np.ndarray:
    """使用高斯核平滑一维数组（scipy.ndimage.gaussian_filter1d 的 NumPy 替代实现）。

    参数：
        y (np.ndarray): 要平滑的一维输入数组。
        sigma (int): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the kernel at this many standard deviations.

    返回：
        (np.ndarray): 平滑后的一维数组，长度与输入相同。
    """
    y = np.asarray(y, dtype=float)
    radius = int(truncate * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    # scipy 的“reflect”边界模式等价于 NumPy 的“symmetric”。
    return np.convolve(np.pad(y, radius, mode="symmetric"), kernel, mode="valid")


class Colors:
    """用于可视化和绘图的 Ultralytics 调色板。

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to RGB
    values and accessing predefined color schemes for object detection and pose estimation.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        Ultralytics 品牌颜色参见 [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand)。
        Please use the official Ultralytics colors for all marketing materials.

    属性：
        palette (list[tuple]): List of RGB color tuples for general use.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array for pose estimation with dtype np.uint8.

    示例：
        >>> from ultralytics.utils.plotting import Colors
        >>> colors = Colors()
        >>> colors(5, True)  # 返回 BGR 格式：(221, 111, 255)
        >>> colors(5, False)  # 返回 RGB 格式：(255, 111, 221)
    """

    def __init__(self):
        """根据固定的十六进制颜色代码列表初始化 Ultralytics 调色板。"""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: int | torch.Tensor, bgr: bool = False) -> tuple:
        """根据索引返回调色板中的颜色。

        参数：
            i (int | torch.Tensor): 颜色索引。
            bgr (bool, 可选): 是否返回 BGR 格式，而不是 RGB 格式。

        返回：
            (tuple): RGB or BGR color tuple.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """将十六进制颜色代码转换为 RGB 值（即 PIL 默认顺序）。"""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 为 'from utils.plots import colors' 创建实例


# Spectral_r 锚框（RGB，从远到近）已预先写入 LUT，因此 colorize_depth 无需导入 matplotlib。
_SPECTRAL_R_ANCHORS = np.array(
    [
        [94, 79, 162],
        [51, 135, 188],
        [102, 194, 165],
        [170, 220, 164],
        [230, 245, 152],
        [255, 254, 190],
        [254, 224, 139],
        [253, 173, 96],
        [244, 109, 67],
        [212, 61, 79],
        [158, 1, 66],
    ],
    dtype=np.float32,
)


def _spectral_lut() -> np.ndarray:
    """通过线性插值锚点，为 cv2.applyColorMap 构建 256x1x3 的 BGR uint8 Spectral_r 查找表。"""
    xs = np.linspace(0.0, 10.0, 256)
    i = np.clip(xs.astype(int), 0, 9)
    f = (xs - i)[:, None]
    rgb = _SPECTRAL_R_ANCHORS[i] * (1.0 - f) + _SPECTRAL_R_ANCHORS[i + 1] * f
    return rgb.round().astype(np.uint8)[:, ::-1].reshape(256, 1, 3)  # 按 cv2 约定将 RGB 转为 BGR


_SPECTRAL_LUT = _spectral_lut()
_DEPTH_CMAPS = {"inferno": cv2.COLORMAP_INFERNO, "jet": cv2.COLORMAP_JET, "spectral": None}


def colorize_depth(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    mode: str = "disparity",
) -> np.ndarray:
    """将 (H, W) 的度量深度数组映射为 BGR uint8 彩色图像，无效（<= 0）像素显示为黑色。

    参数：
        depth (np.ndarray): 以米为单位的 (H, W) 深度数组。
        vmin (float, 可选): 颜色范围下界；默认为有效像素最小值（度量模式）或第 2 个视差百分位数（视差模式）。
        vmax (float, 可选): 颜色范围上界；默认为有效像素最大值（度量模式）或第 98 个视差百分位数（视差模式）。
        cmap (str): 颜色映射，可选 ``"inferno"``、``"jet"`` 或 ``"spectral"``（matplotlib Spectral_r，近处偏暖）。
        mode (str): ``"metric"`` 对深度进行线性归一化；``"disparity"`` 在第 2 到第 98 个百分位数之间
            对逆深度（1/d）进行归一化，以模拟 DepthAnything 的视觉效果，使近处对象偏暖并降低远处异常值的影响。

    返回：
        (np.ndarray): 形状为 (H, W, 3) 的 BGR uint8 彩色深度图。
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    v = np.where(valid, 1.0 / np.where(valid, d, 1.0), 0.0) if mode == "disparity" else d
    if vmin is None or vmax is None:
        pool = v[valid]
        if mode == "disparity":
            lo, hi = np.percentile(pool, (2, 98)) if pool.size else (0.0, 1.0)
        else:
            lo, hi = (float(pool.min()), float(pool.max())) if pool.size else (0.0, 1.0)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6
    dn = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (dn * 255).astype(np.uint8)
    lut = _SPECTRAL_LUT if cmap == "spectral" else None
    color = cv2.applyColorMap(idx, lut) if lut is not None else cv2.applyColorMap(idx, _DEPTH_CMAPS[cmap])  # BGR
    color[~valid] = 0
    return color


class Annotator:
    """用于训练/验证拼图、JPG 图像和预测结果标注的 Ultralytics 标注器。

    Tensor 图像必须是连续的 HWC BGR uint8 格式。

    属性：
        im (Image.Image | np.ndarray | torch.Tensor): 要进行标注的图像。
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (int): 绘制线条的宽度。
        skeleton (列表[列表[int]]): 关键点的骨架连接结构。
        limb_color (np.ndarray): Color palette for limbs.
        kpt_color (np.ndarray): 关键点颜色调色板。
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.

    示例：
        >>> from ultralytics.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """使用图像、线宽以及关键点和肢体的颜色调色板初始化 Annotator 类。"""
        non_ascii = not is_ascii(example)  # 非拉丁标签，例如中文、阿拉伯文或西里尔文
        input_is_pil = isinstance(im, Image.Image)
        input_is_tensor = isinstance(im, torch.Tensor)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if input_is_tensor:
            assert im.ndim == 3 and im.shape[2] == 3 and im.dtype == torch.uint8, (
                f"Expected HWC uint8 tensor image with 3 channels, but got shape {tuple(im.shape)} and dtype {im.dtype}."
            )
            if self.pil or im.device.type == "cpu":
                im, input_is_tensor = im.cpu().numpy(), False
        if not input_is_pil:
            if im.shape[2] == 1:  # handle grayscale
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 2:  # 处理双通道图像
                im = np.ascontiguousarray(np.dstack((im, np.zeros_like(im[..., :1]))))
            elif im.shape[2] > 3:  # multispectral
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)  # 保持 BGR，因为颜色调色板为 BGR
            if self.im.mode not in {"RGB", "RGBA"}:  # multispectral
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # 兼容弃用接口：将 w, h = getsize(string) 改为 _, _, w, h = getbox(string)。
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # 文本宽度和高度
        else:  # use cv2
            assert im.is_contiguous() if input_is_tensor else im.data.contiguous, (
                "Image not contiguous. Apply contiguous() or np.ascontiguousarray(im) to Annotator input images."
            )
            self.im = im if input_is_tensor or im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # 字体缩放比例
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """根据背景颜色指定文本颜色。

        参数：
            color (tuple, 可选): 文本矩形框的背景颜色。
            txt_color (tuple, 可选): 文本的备用颜色。

        返回：
            (tuple): Text color for label.

        示例：
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # 返回 (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """在图像上绘制带指定标签的边界框。

        参数：
            box (tuple): 边界框坐标 (x1, y1, x2, y2)。
            label (str, 可选): 要显示的文本标签。
            color (tuple, 可选): 矩形框的背景颜色。
            txt_color (tuple, 可选): 文本颜色。

        示例：
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        self._to_numpy()
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, (torch.Tensor, np.ndarray)):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # 形状为 (n, 2) 的多个点
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)  # 文本宽度和高度
                outside = p1[1] >= h  # 标签是否能放在边界框外
                if p1[0] > self.im.size[0] - w:  # 尺寸为 (w, h)，检查标签是否超出图像右侧
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text([边界框[0], 边界框[1]], label, fill=txt_color, font=self.font, 锚框='ls')  # 适用于 PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # 文本宽度和高度
                h += 3  # 增加像素以填充文本
                outside = p1[1] >= h  # 标签是否能放在边界框外
                if p1[0] > self.im.shape[1] - w:  # 形状为 (h, w)，检查标签是否超出图像右侧
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, alpha: float = 0.5):
        """在图像上绘制掩码。

        参数：
            masks (torch.Tensor | np.ndarray): 形状为 [n, h, w] 的预测掩码。
            colors (列表[列表[int]]): 预测掩码的 BGR 颜色，格式为 [[b, g, r] * n]，与 `self.im` 一致。
            alpha (float, 可选): 掩码透明度：0.0 表示完全透明，1.0 表示完全不透明。
        """
        if self.pil:
            # 先转换为 NumPy 数组。
            self.im = np.asarray(self.im).copy()
        if isinstance(masks, np.ndarray):
            self._to_numpy()
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        elif len(masks):
            # 使用 scale_masks 正确移除填充并上采样，先将布尔值转换为浮点数。
            tensor_image = isinstance(self.im, torch.Tensor)
            device = self.im.device if tensor_image else masks.device
            masks = ops.scale_masks(masks[None].to(device).float(), self.im.shape[:2])[0] > 0.5
            colors = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0  # 形状(n,3)
            colors = colors[:, None, None] * alpha  # 形状(n,1,1,3)，预先乘以透明度
            masks = masks.unsqueeze(3)  # 形状(n,h,w,1)
            mcs = torch.empty((*masks.shape[1:3], 3), device=device, dtype=torch.float32)  # 形状(h,w,3)
            inv_alpha_masks = torch.empty((*masks.shape[1:3], 1), device=device, dtype=torch.float32)  # 形状(h,w,1)
            # 按行分带处理，避免 (n,h,w,*) 中间张量覆盖完整高度。
            bands = max(1, masks.numel() * 12 // 2**23)  # 下游每个掩码元素占 12 字节，每个分带约 8 MB
            for m, mcs_band, inv_band in zip(masks.chunk(bands, 1), mcs.chunk(bands), inv_alpha_masks.chunk(bands)):
                torch.amax(m * colors, 0, out=mcs_band)
                torch.prod(1 - m * alpha, 0, out=inv_band)
            im = (self.im if tensor_image else torch.from_numpy(self.im)).to(device).float() / 255.0
            im = ((im * inv_alpha_masks + mcs) * 255).byte()
            self.im[:] = im if tensor_image else im.cpu().numpy()
        if self.pil:
            # 将 im 转回 PIL，并更新绘图对象。
            self.fromarray(self.im)

    def semantic_mask(self, mask, alpha: float = 0.5, ignore_index: int = 255):
        """在图像上绘制语义分割掩码。

        参数：
            mask (np.ndarray): 形状为 [h, w]、包含整数类别索引的语义掩码。
            alpha (float, 可选): 掩码透明度：0.0 表示完全透明，1.0 表示完全不透明。
            ignore_index (int, 可选): 要忽略的类别索引（例如表示空白或忽略区域的 255）。
        """
        self._to_numpy()
        if self.pil:
            # 先转换为 NumPy 数组。
            self.im = np.asarray(self.im).copy()
        ids = np.unique(mask)  # 存在的类别 ID，按升序排列
        palette = np.array([(0, 0, 0) if i == ignore_index else colors(int(i), True) for i in ids], self.im.dtype)
        overlay = palette[np.searchsorted(ids, mask)] if len(ids) else np.zeros_like(self.im)
        self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        if self.pil:
            # 将 im 转回 PIL，并更新绘图对象。
            self.fromarray(self.im)

    def depth_map(
        self,
        depth: np.ndarray,
        alpha: float = 0.6,
        cmap: str = "jet",
        mode: str = "disparity",
    ) -> None:
        """渲染彩色深度图，并将其叠加到图像上。

        参数：
            depth (np.ndarray): (H, W) depth in meters.
            alpha (float): Blend factor for the heatmap overlay.
            cmap (str): Colormap, one of "inferno", "jet", "spectral". See `colorize_depth`.
            mode (str): "metric" or "disparity" normalization. See `colorize_depth`.
        """
        self._to_numpy()
        if self.pil:
            self.im = np.asarray(self.im).copy()
        heat = colorize_depth(depth, cmap=cmap, mode=mode)  # BGR，与 Annotator 缓冲区约定一致
        if heat.shape[:2] != self.im.shape[:2]:
            heat = cv2.resize(heat, (self.im.shape[1], self.im.shape[0]))
        self.im = cv2.addWeighted(self.im, 1 - alpha, heat, alpha, 0)
        if self.pil:
            self.fromarray(self.im)

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """在图像上绘制关键点。

        参数：
            kpts (torch.Tensor): 关键点，形状为 [17, 3]（x、y、置信度）。
            shape (tuple, 可选): 图像形状 (h, w)。
            radius (int, 可选): 关键点半径。
            kpt_line (bool, 可选): 是否绘制关键点之间的连线。
            conf_thres (float, 可选): 置信度阈值。
            kpt_color (tuple, 可选): 关键点颜色。

        注意：
            - `kpt_line=True` 当前仅支持人体姿态绘图。
            - 原地修改 self.im。
            - 如果 self.pil 为 True，则将图像转换为 NumPy 数组，再转换回 PIL 图像。
        """
        radius = radius if radius is not None else self.lw
        self._to_numpy()
        if self.pil:
            # 先转换为 NumPy 数组。
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # 当前 `kpt_line=True` 仅支持人体姿态绘图
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if len(k) == 3:
                if k[2] < conf_thres:
                    continue
            elif x_coord == 0 and y_coord == 0:  # 没有置信度通道时，(0, 0) 表示缺失的关键点
                continue
            cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                elif not (kpts[sk[0] - 1, :2].any() and kpts[sk[1] - 1, :2].any()):  # (0, 0) marks a missing keypoint
                    continue
                if min(pos1 + pos2) < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # 将 im 转回 PIL，并更新绘图对象。
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """向图像添加矩形（仅 PIL 模式）。"""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text: str, txt_color: tuple = (255, 255, 255), anchor: str = "top", box_color: tuple = ()):
        """使用 PIL 或 cv2 向图像添加文本。

        参数：
            xy (列表[int]): 放置文本的左上角坐标。
            text (str): Text to be drawn.
            txt_color (tuple, 可选): 文本颜色。
            anchor (str, 可选): 文本锚点位置（``'top'`` 或 ``'bottom'``）。
            box_color (tuple, 可选): 文本框背景颜色，可包含透明度。
        """
        self._to_numpy()
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":  # 从字体底部开始计算 y
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    # 为每条线绘制矩形。
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3  # 增加像素以填充文本
                outside = xy[1] >= h  # 标签是否能放在边界框外
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """使用 NumPy 数组或 PIL 图像更新 `self.im`。"""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def _to_numpy(self):
        """仅在 CPU 绘图操作需要时将张量图像移动到 CPU。"""
        if isinstance(self.im, torch.Tensor):
            self.im = self.im.cpu().numpy()

    def result(self, pil=False):
        """将标注后的图像作为数组或 PIL 图像返回。"""
        self._to_numpy()
        im = np.asarray(self.im)  # self.im 为 BGR
        return Image.fromarray(im[..., ::-1]) if pil else im

    def show(self, title: str | None = None):
        """显示标注后的图像。"""
        im = Image.fromarray(self.result()[..., ::-1])  # 将 BGR NumPy 数组转换为 RGB PIL 图像
        if IS_COLAB or IS_KAGGLE:  # 不能使用 IS_JUPYTER，因为它适用于所有 IPython 环境
            try:
                display(im)  # noqa - display() 函数仅在 IPython 环境中可用
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """将标注后的图像保存到 filename。"""
        cv2.imwrite(filename, self.result())

    @staticmethod
    def get_bbox_dimension(bbox: tuple | list):
        """计算边界框的尺寸和面积。

        参数：
            bbox (tuple | 列表): 格式为 (x_min, y_min, x_max, y_max) 的边界框坐标。

        返回：
            width (float): 边界框宽度。
            height (float): 边界框高度。
            area (float): 边界框包围的面积。

        示例：
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """绘制训练标签，包括类别直方图和边界框统计信息。

    参数：
        boxes (np.ndarray): 格式为 [x, y, 宽度, 高度] 的边界框坐标。
        cls (np.ndarray): 类别索引。
        names (dict, 可选): 将类别索引映射到类别名称的字典。
        save_dir (Path, 可选): 保存绘图的目录。
        on_plot (Callable, 可选): 绘图保存后调用的函数。
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars
    from matplotlib.colors import LinearSegmentedColormap

    # 绘制数据集标签
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # 类别数量
    boxes = boxes[:1000000]  # 最多处理 100 万个边界框
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])

    # Matplotlib 标签
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for class_id, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in (0, 1, 2, 3):
        for s in ("top", "right", "left", "bottom"):
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """将图像裁剪保存为 {file}，裁剪尺寸按 {gain} 倍放大并添加 {pad} 像素填充；可保存和/或返回裁剪结果。

    此函数接收一个边界框和一张图像，然后根据边界框裁剪图像的一部分并保存。
    还可以选择将裁剪区域调整为正方形，并通过 gain 和填充参数调整边界框。

    参数：
        xyxy (torch.Tensor | 列表): 表示 xyxy 格式边界框的张量或列表。
        im (np.ndarray): 输入图像。
        file (Path, 可选): 保存裁剪图像的路径。
        gain (float, 可选): 增大边界框尺寸的乘数。
        pad (int, 可选): 添加到边界框宽度和高度上的像素数量。
        square (bool, 可选): 为 True 时，将边界框转换为正方形。
        BGR (bool, 可选): 为 True 时以 BGR 格式返回图像，否则以 RGB 格式返回。
        save (bool, 可选): 为 True 时，将裁剪图像保存到磁盘。

    返回：
        (np.ndarray): 裁剪后的图像。

    示例：
        >>> from ultralytics.utils.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if isinstance(xyxy, np.ndarray):
        xyxy = torch.from_numpy(xyxy)
    elif not isinstance(xyxy, torch.Tensor):  # 也可以是列表
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # 边界框
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 尝试将矩形调整为正方形
    b[:, 2:] = b[:, 2:] * gain + pad  # 边界框 wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1  # 灰度图像
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR or grayscale else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # 保存 BGR；参见 https://github.com/ultralytics/yolov5/issues/7007 的色度抽样问题
        im_save = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(im_save).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(
    labels: dict[str, Any],
    images: torch.Tensor | np.ndarray | None = None,
    paths: list[str] | None = None,
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    on_plot: Callable | None = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
    show_labels: bool = True,
    show_conf: bool = True,
) -> np.ndarray | None:
    """绘制带标签、边界框、掩码和关键点的图像网格。

    参数：
        labels (dict[str, Any]): 包含检测数据的字典，键包括 'cls'、'bboxes'、'conf'、'masks'、
            '关键点', 'batch_idx', 'img'.
        images (torch.Tensor | np.ndarray): 要绘制的图像批次，形状为 (batch_size, 通道, 高度, 宽度)。
        paths (列表[str] | None): 批次中每张图像对应的文件路径列表。
        fname (str): 绘图网格的输出文件名。
        names (dict[int, str] | None): 将类别索引映射到类别名称的字典。
        on_plot (Callable | None): Callback function to be called after saving the plot.
        max_size (int): 输出图像网格的最大尺寸。
        max_subplots (int): 图像网格中子图的最大数量。
        save (bool): 是否将绘制的图像网格保存到文件。
        conf_thres (float): 显示检测结果所需的置信度阈值。
        show_labels (bool): 是否显示类别标签。
        show_conf (bool): 是否显示置信度值。

    返回：
        (np.ndarray | None): 当 save 为 False 时返回绘制的图像网格，否则返回 None。

    注意：
        此函数同时支持张量和 NumPy 数组输入，并会自动将张量转换为 NumPy 数组进行处理。

        Channel Support:
        - 1 通道：灰度图像
        - 2 通道：添加全零的第三通道
        - 3 通道：直接使用（标准 RGB）
        - 4 个及以上通道：裁剪为前 3 个通道
    """
    images = np.zeros((0, 3, 640, 640), dtype=np.float32) if images is None else images
    for k in ("cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images", "semantic_mask", "depth"):
        if k not in labels:
            continue
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)  # 当形状为 (n, 1) 时去除多余维度
        if isinstance(labels[k], torch.Tensor):
            labels[k] = labels[k].cpu().numpy()

    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
    confs = labels.get("conf", None)
    masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
    kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
    semantic_masks = labels.get("semantic_mask", np.zeros(0, dtype=np.int64))
    depth_maps = labels.get("depth", np.zeros(0, dtype=np.float32))
    images = labels.get("img", images)  # 默认使用输入图像

    if len(images) and isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # 处理双通道和多通道图像。
    c = images.shape[1]
    if c == 2:
        zero = np.zeros_like(images[:, :1])
        images = np.concatenate((images, zero), axis=1)  # 为双通道图像填充黑色通道
    elif c > 3:
        images = images[:, :3]  # 将多光谱图像裁剪为前 3 个通道

    bs, _, h, w = images.shape  # 批次大小, _, 高度, 宽度
    bs = min(bs, max_subplots)  # 限制绘图图像数量
    ns = np.ceil(bs**0.5)  # 子图数量（尽量排列为正方形）
    if np.max(images[0]) <= 1:
        images *= 255  # 取消归一化（可选）

    # 构建图像网格
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # 调整尺寸（可选）
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # 字体尺寸
    fs = max(fs, 18)  # 确保字体尺寸足够大，便于阅读
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None
            conf = confs[idx] if confs is not None else None  # 检查是否存在置信度（标签或预测）

            if len(bboxes):
                boxes = bboxes[idx]
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # 缩放到像素坐标
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # 图像缩放时需要对绝对坐标进行缩放
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        conf_text = f"{conf[j]:.1f}" if conf is not None else ""
                        label = f"{c}" if show_labels else ""
                        label += f" {conf_text}".strip() if show_conf else ""
                        annotator.box_label(box, label, color=color)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                    annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))

            # 绘制关键点
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # 缩放到像素坐标
                        kpts_[..., 1] *= h
                    elif scale < 1:  # 图像缩放时需要对绝对坐标进行缩放
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # 绘制掩码
            if len(masks):
                if idx.shape[0] == masks.shape[0] and masks.max() <= 1:  # overlap_mask=False
                    image_masks = masks[idx]
                else:  # overlap_mask=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                    image_masks = (image_masks == index).astype(np.float32)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)

        # 绘制语义掩码
        if len(semantic_masks) and i < len(semantic_masks):
            mask = semantic_masks[i]
            mh, mw = mask.shape
            if mh != h or mw != w:
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            im = np.asarray(annotator.im).copy()
            sub_annotator = Annotator(np.ascontiguousarray(im[y : y + h, x : x + w]), line_width=1, pil=False)
            sub_annotator.semantic_mask(mask, alpha=0.4)
            im[y : y + h, x : x + w] = sub_annotator.im
            annotator.fromarray(im)

        # 绘制深度图
        if len(depth_maps) and i < len(depth_maps):
            d = depth_maps[i]
            if d.ndim == 3:
                d = d.squeeze(0)
            dh, dw = d.shape
            if dh != h or dw != w:
                d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            im = np.asarray(annotator.im).copy()
            # 拼图使用 RGB，而 Annotator 约定使用 BGR 缓冲区，因此先将图块转为 BGR 进行叠加，
            # 再转回 RGB 写入拼图。
            sub_bgr = cv2.cvtColor(np.ascontiguousarray(im[y : y + h, x : x + w]), cv2.COLOR_RGB2BGR)
            sub_annotator = Annotator(sub_bgr, line_width=1, pil=False)
            sub_annotator.depth_map(d, alpha=0.6)
            im[y : y + h, x : x + w] = cv2.cvtColor(sub_annotator.im, cv2.COLOR_BGR2RGB)
            annotator.fromarray(im)

    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # 保存结果
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """从结果 CSV 文件绘制训练结果。此函数支持实例分割、语义分割、姿态估计和分类等多种数据。
    绘图结果将以 ``results.png`` 文件保存到 CSV 所在目录。

    参数：
        file (str, 可选): 包含训练结果的 CSV 文件路径。
        dir (str, 可选): 未提供 ``file`` 时，CSV 文件所在的目录。
        on_plot (Callable, 可选): 绘图完成后执行的回调函数，接收输出文件名作为参数。

    示例：
        >>> from ultralytics.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv")
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
    import polars as pl

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    fig, ax = None, None
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(2, len(columns) // 2, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            for i, j in enumerate(columns):
                y = data.select(j).to_numpy().flatten().astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # 实际结果
                ax[i].plot(x, _gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # 平滑曲线
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")
    if ax is not None:
        ax[1].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()
        if on_plot:
            on_plot(fname)


@plt_settings()
def plot_multitrain_results(scores: dict, key: str = "fitness", save_dir=Path()):
    """将多数据集训练运行中的各数据集指标绘制为柱状图，并显示跨数据集均值。

    参数：
        scores (dict): 数据集名称到标量指标值的映射。
        key (str): 要绘制的指标名称，同时用作 y 轴标签。
        save_dir (str | Path): 保存 ``multitrain_results.png`` 图像的目录。

    返回：
        (Path): 保存后的图像路径。

    示例：
        >>> from ultralytics.utils.plotting import plot_multitrain_results
        >>> plot_multitrain_results({"coco8": 0.61, "dota8": 0.48}, key="metrics/mAP50-95(B)")
    """
    import matplotlib.pyplot as plt  # 在函数作用域导入，以加快 ``import ultralytics`` 的速度

    mean = sum(scores.values()) / len(scores)
    fig, ax = plt.subplots(figsize=(max(6.0, len(scores) * 0.45), 5), tight_layout=True)
    ax.bar(range(len(scores)), list(scores.values()), color="#042AFF")
    ax.axhline(mean, color="orange", linestyle="--", label=f"mean = {mean:.3f}")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(list(scores), rotation=90)
    ax.set_ylabel(key)
    ax.set_title(f"MultiTrainer results across {len(scores)} datasets")
    ax.legend()
    fname = Path(save_dir) / "multitrain_results.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def plt_color_scatter(v, f, bins: int = 20, cmap: str = "viridis", alpha: float = 0.8, edgecolors: str = "none"):
    """绘制散点图，并根据二维直方图为点着色。

    参数：
        v (数组): x 轴数据。
        f (数组): y 轴数据。
        bins (int, 可选): 直方图的分箱数量。
        cmap (str, 可选): 散点图使用的颜色映射。
        alpha (float, 可选): 散点图的透明度。
        edgecolors (str, 可选): 散点图的边缘颜色。

    示例：
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    import matplotlib.pyplot as plt  # 在函数作用域导入，以加快 ``import ultralytics`` 的速度

    # 计算二维直方图及其对应颜色。
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            np.clip(np.digitize(v[i], xedges, right=False) - 1, 0, hist.shape[0] - 1),
            np.clip(np.digitize(f[i], yedges, right=False) - 1, 0, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # 散点图
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_depth_panels(
    imgs: torch.Tensor,
    preds: list[torch.Tensor],
    fname: str | Path,
    gt: torch.Tensor | None = None,
    titles: list[str] | None = None,
    max_images: int = 4,
) -> None:
    """写入深度面板网格：每行一张图像，列为 RGB、GT（如果提供）以及 ``preds`` 中每个预测结果。

    每行的所有深度列共用 GT 有效像素范围，因此 GT 与任意预测之间的尺度误差会直接表现为颜色不匹配。
    各面板会调整到 RGB 图像尺寸，因此来自不同检测头步长的预测结果无需提前插值。

    参数：
        imgs (torch.Tensor): 形状为 (B, 3, H, W)、取值范围为 [0, 1] 的浮点图像张量。
        preds (列表): 形状为 (B, 1, H, W) 或 (B, H, W) 的预测深度张量列表，每个张量对应一列。
        fname (str | Path): 输出图像路径。
        gt (torch.Tensor, 可选): 形状为 (B, 1, H, W) 或 (B, H, W)、单位为米的真实深度；像素值 <= 0
            视为无效并绘制为黑色。用于生成 GT 列并设置共享颜色范围。
        titles (列表, 可选): 列标题列表，绘制在 24 像素高的标题栏中。为 None 时不显示标题栏。
        max_images (int): 最大行数。
    """
    preds = [p.unsqueeze(1) if p.ndim == 3 else p for p in preds]
    h, w = imgs.shape[-2:]
    rows = []
    for i in range(min(imgs.shape[0], max_images)):
        rgb = (imgs[i].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]

        if gt is not None:
            g = gt[i, 0] if gt.ndim == 4 else gt[i]
            gv = g[g > 0]
            vmin = float(gv.min()) if gv.numel() else 0.0
            vmax = float(gv.max()) if gv.numel() else 1.0
            d = g.detach().float().cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g, np.float32)
            panels.append(
                cv2.resize(colorize_depth(d, vmin, vmax, mode="metric"), (w, h), interpolation=cv2.INTER_NEAREST)
            )
        else:
            # 没有 GT：根据每个预测自身的有效范围进行缩放。
            vmin = vmax = None

        for p in preds:
            d = p[i, 0] if p.ndim == 4 else p[i]
            d = d.detach().float().cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d, np.float32)
            lo, hi = vmin, vmax
            if lo is None or hi is None:
                dv = d[d > 0]
                lo, hi = (float(dv.min()), float(dv.max())) if dv.size else (0.0, 1.0)
            panels.append(cv2.resize(colorize_depth(d, lo, hi, mode="metric"), (w, h), interpolation=cv2.INTER_NEAREST))

        rows.append(np.hstack(panels))
    grid = np.vstack(rows)
    if titles:
        strip = np.full((24, grid.shape[1], 3), 255, dtype=np.uint8)
        for j, t in enumerate(titles):
            cv2.putText(strip, str(t), (j * w + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        grid = np.vstack([strip, grid])
    cv2.imwrite(str(fname), grid)


@plt_settings()
def plot_tune_results(results_file: str = "tune_results.ndjson", exclude_zero_fitness_points: bool = True):
    """绘制调优 NDJSON 文件中保存的演化结果。

    参数：
        results_file (str, 可选): 包含调优结果的 NDJSON 文件路径。
        exclude_zero_fitness_points (bool, 可选): 是否在调优图中排除适应度为零的点。

    示例：
        >>> plot_tune_results("path/to/tune_results.ndjson")
    """
    import json

    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    def _save_one_file(file):
        """将一个 matplotlib 绘图保存到文件。"""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"Saved {file}")

    results_file = Path(results_file)
    with open(results_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return

    keys = list(records[0].get("hyperparameters", {}))
    x = np.array(
        [[r.get("fitness", 0.0)] + [r.get("hyperparameters", {}).get(k, np.nan) for k in keys] for r in records],
        dtype=float,
    )
    len(x)
    all_fitness = x[:, 0]  # fitness
    zero_mask = slice(None)
    if exclude_zero_fitness_points:
        zero_mask = all_fitness > 0  # exclude zero-fitness points
        x, all_fitness = x[zero_mask], all_fitness[zero_mask]
    if len(all_fitness) == 0:
        LOGGER.warning("No valid fitness values to plot (all iterations may have failed)")
        return
    fitness = all_fitness.copy()
    # 仅对下界执行迭代式 sigma 拒绝。
    for _ in range(3):  # max 3 iterations
        mean, std = fitness.mean(), fitness.std()
        lower_bound = mean - 3 * std
        mask = fitness >= lower_bound
        if mask.all():  # 没有更多异常值
            break
        x, fitness = x[mask], fitness[mask]
    j = np.argmax(fitness)  # 最大适应度的索引
    n = math.ceil(len(keys) ** 0.5)  # 绘图中的列数和行数
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + 1]
        mu = v[j]  # 最佳单项结果
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # 限制为 40 个字符
        plt.tick_params(axis="both", labelsize=8)  # 将坐标轴标签字体大小设为 8
        if i % n != 0:
            plt.yticks([])
    _save_one_file(results_file.with_name("tune_scatter_plots.png"))

    # 适应度与迭代次数的关系
    x = range(1, len(all_fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    for dataset in sorted({k for r in records for k in r.get("datasets", {})}):
        y = np.array([r.get("datasets", {}).get(dataset, {}).get("fitness", np.nan) for r in records], dtype=float)
        if exclude_zero_fitness_points and not isinstance(zero_mask, slice):
            y = y[zero_mask]
        plt.plot(x, y, "o", markersize=5, alpha=0.8, label=dataset)
    plt.plot(x, _gaussian_filter1d(all_fitness, sigma=3), ":", color="0.35", label="smoothed mean", linewidth=2)
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    _save_one_file(results_file.with_name("tune_fitness.png"))


def class_activation_map(
    model,
    im: torch.Tensor,
    paths: list[str],
    save_dir: Path,
    *args,
    conf: float = 0.25,
    classes=None,
    topk: int = 16,
    **kwargs,
) -> Any:
    """运行推理，并为批次中的每张图像保存类别激活热力图。

    LayerCAM 根据预测类别分数沿正梯度方向的大小，为每个检测头输入位置分配权重。
    每个预测结果和检测头层级会先独立归一化，再取逐元素最大值，避免较强的预测结果或层级掩盖较弱结果。

    参数：
        model (torch.nn.Module): 封装 PyTorch 模型的 AutoBackend。
        im (torch.Tensor): 形状为 (B, 3, H, W) 的预处理图像。
        paths (列表[str]): 批次中每张图像的源路径，用于命名保存的叠加图。
        save_dir (Path): 保存叠加图的目录。
        *args (Any): 传递给模型前向传播的其他位置参数。
        conf (float): 预测结果参与计算所需通过的分数阈值。对于没有预测结果达到阈值的图像，
            会回退到其最佳预测结果，以便仍可检查接近正确的结果。
        classes (int | 列表[int], 可选): 仅允许这些类别 ID 参与计算，与预测中的 ``classes`` 过滤器一致。
        topk (int): 每张图像最多解释的预测结果数量，每个结果都会消耗一次反向传播。
        **kwargs (Any): 传递给模型前向传播的其他关键字参数。

    返回：
        (Any): 从自动求导图中分离后的模型预测结果。
    """
    acts, scores = [], []

    def pre_hook(module, inputs):
        """捕获进入检测头的特征图，避免 WorldDetect 等检测头原地覆盖这些特征。"""
        x = inputs[0]
        acts.extend(a for a in (x if isinstance(x, (list, tuple)) else [x]) if a.ndim == 4)

    def hook(module, inputs, output):
        """捕获离开检测头的类别 logits。"""
        raw = output[1] if isinstance(output, tuple) else output  # 返回（预测结果，raw）的检测头保留 raw 输出
        if isinstance(raw, dict):  # Detect and subclasses, end2end heads predict from their one2one branch
            s = raw.get("one2one", raw)["scores"]  # (B, nc, 锚框)
        elif isinstance(raw, tuple):  # RTDETRDecoder, raw = (dec_bboxes, dec_scores, ...)
            s = raw[1][-1].transpose(1, 2)  # 最后一个解码器层，形状为 (B, nc, queries)
        else:  # Classify (B, nc), SemanticSegment (B, nc, h, w), Depth (B, 1, h, w)
            s = raw
        scores.append(s.reshape(*s.shape[:2], -1))  # 类别 logits，形状为 (B, nc, predictions)

    head = model.model.model[-1]  # AutoBackend -> PyTorch 模型 -> 检测头
    head.shape = head.shapes = None  # 重建锚框缓存，其中的推理张量会破坏自动求导图
    handles = [head.register_forward_pre_hook(pre_hook), head.register_forward_hook(hook)]
    # smart_inference_mode() 在 torch 1.10 及更高版本中将调用方包装在 inference_mode 中，
    # 在更低版本中包装在 no_grad 中；只有前者需要退出，之后 autograd 才能记录计算。
    with torch.inference_mode(False) if TORCH_1_10 else contextlib.nullcontext(), torch.enable_grad():
        try:
            im = im.clone().requires_grad_(True)  # 模型参数 requires_grad=False，因此在此处建立计算图
            preds = model(im, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        s = torch.cat(scores, 2)  # (B, nc, 预测结果) 类别 logits
        if classes is not None:
            cls = torch.as_tensor(classes, dtype=torch.long, device=s.device).flatten()
            cls = cls[(cls >= 0) & (cls < s.shape[1])]  # 丢弃超出模型输出通道范围的类别 ID
            if len(cls):
                s = s[:, cls]  # 仅保留请求类别的热力图
        s = s.amax(1)  # (B, 预测结果)，每个预测结果的最佳类别 logit
        keep = (s.sigmoid() >= conf) | (s == s.amax(1, keepdim=True))  # 没有结果超过阈值时至少保留最佳预测
        n = min(int(keep.sum(1).amax()), topk)  # 要解释的预测结果数量，每个结果执行一次反向传播
        if int(keep.sum(1).amax()) > n:
            LOGGER.warning(f"Explaining the {n} strongest predictions per image out of {int(keep.sum(1).amax())}.")
        rank = torch.arange(n, device=s.device) % keep.sum(1, keepdim=True).clamp(min=1)  # 对较短的图像重复索引
        order = s.masked_fill(~keep, float("-inf")).argsort(1, descending=True).gather(1, rank)  # (B, n)
        cam = None
        for k in range(n):
            levels = []
            grads = torch.autograd.grad(s.gather(1, order[:, k : k + 1]).sum(), acts, retain_graph=k < n - 1)
            for a, g in zip(acts, grads):
                c = (g.float().clamp(min=0) * a.float()).sum(1, keepdim=True)  # LayerCAM, per-position weighting
                c = c.clamp(min=0)  # 激活值可能为负数，仅保留支持该预测的证据
                c = torch.nn.functional.interpolate(c, im.shape[2:], mode="bilinear", align_corners=False)
                levels.append(c / c.amax((2, 3), keepdim=True).clamp(min=1e-7))
            # 某一预测层的峰值可能远高于其他层，直接求和会把其他层保留的更广泛证据压缩成微弱背景。
            level = torch.stack(levels).amax(0)
            cam = level if cam is None else torch.maximum(cam, level)

    cam = (cam.squeeze(1) * 255).byte().cpu().numpy()  # (B, H, W), maps are already scaled to [0, 1]
    ims = im.detach()[:, :3].float()
    lo, hi = ims.amin((2, 3), keepdim=True), ims.amax((2, 3), keepdim=True)  # classify inputs are mean-std normalized
    ims = ((ims - lo) / (hi - lo).clamp(min=1e-7) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]  # to BGR
    save_dir.mkdir(parents=True, exist_ok=True)
    for c, img, p in zip(cam, ims, paths):
        f = increment_path(save_dir / f"{Path(p).stem}_cam.jpg")
        img = np.ascontiguousarray(img if img.shape[2] == 3 else img[..., :1].repeat(3, 2))  # 灰度图转为 BGR
        heatmap = cv2.addWeighted(cv2.applyColorMap(c, cv2.COLORMAP_JET), 0.5, img, 0.5, 0)
        cv2.imwrite(str(f), heatmap)
        LOGGER.info(f"Saving {f}... (LayerCAM)")

    def detach(x):
        """将嵌套模型输出中的张量从 autograd 计算图中分离。"""
        if isinstance(x, torch.Tensor):
            return x.detach()
        if isinstance(x, dict):
            return {k: detach(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(detach(v) for v in x)
        return x

    return detach(preds)
