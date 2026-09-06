# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote

import cv2
import numpy as np
import torch

from ultralytics import __version__
from ultralytics.utils.git import GitRepo
from ultralytics.utils.patches import imread as imread  # 重新导出以保持向后兼容
from ultralytics.utils.patches import imread_unicode, imshow, imwrite, torch_save  # 补丁函数
from ultralytics.utils.tqdm import TQDM  # noqa


def env_bool(name: str, default: bool = False) -> bool:
    """解析布尔环境变量，并接受常见的真值字符串。

    接受 `"1"`、`"true"`、`"yes"`、`"on"`、`"y"` 和 `"t"`（不区分大小写并去除空白）作为 True；
    其他已设置的值均为 False。仅当环境变量未设置时返回默认值，设置为空字符串时不会返回默认值。

    参数：
        name (str): 环境变量名称。
        default (bool): 环境变量未设置时返回的值。

    返回：
        (bool): 解析后的布尔值。

    示例：
        >>> env_bool("YOLO_UNSET_EXAMPLE_VAR", True)  # 环境变量未设置时返回默认值
        True
    """
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


# PyTorch 多 GPU DDP 常量，仅在真实 DDP 工作进程（即 WORLD_SIZE > 1）中可信（#16446）
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
RANK = int(os.getenv("RANK", "-1")) if WORLD_SIZE > 1 else -1
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1")) if WORLD_SIZE > 1 else -1

# 其他常量
ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # 默认图像
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # assets GitHub URL
# 可配置的 Platform 调试 URL（例如 ULTRALYTICS_PLATFORM_URL=http://localhost:3000）
PLATFORM_URL = os.getenv("ULTRALYTICS_PLATFORM_URL", "https://platform.ultralytics.com").rstrip("/")
PLATFORM_API_URL = os.getenv("PLATFORM_API_URL", f"{PLATFORM_URL}/api/webhooks")
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # YOLO 多进程线程数量
AUTOINSTALL = env_bool("YOLO_AUTOINSTALL", True)  # 全局自动安装模式
VERBOSE = env_bool("YOLO_VERBOSE", True)  # 全局详细输出模式
SAFE_LOAD = env_bool("ULTRALYTICS_SAFE_LOAD")  # 可选启用 weights_only 模型加载
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__)  # Normalize torch.__version__ (PyTorch>1.9 返回 TorchVersion 对象)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
IS_VSCODE = os.environ.get("TERM_PROGRAM") == "vscode"
RKNN_CHIPS = frozenset(
    {
        "rk3588",
        "rk3576",
        "rk3566",
        "rk3568",
        "rk3562",
        "rv1103",
        "rv1106",
        "rv1103b",
        "rv1106b",
        "rk2118",
        "rv1126b",
    }
)  # 可用于导出的 Rockchip 处理器
QNN_HTP_TARGETS = {
    "68": ("htp_arch", "68"),  # Snapdragon 888
    "69": ("htp_arch", "69"),  # Snapdragon 8 Gen 1
    "73": ("htp_arch", "73"),  # Snapdragon 8 Gen 2 / X Elite
    "75": ("htp_arch", "75"),  # Snapdragon 8 Gen 3
    "79": ("soc_model", "69"),  # Snapdragon 8 Elite (SM8750)
    "81": ("htp_arch", "81"),  # Snapdragon 8 Elite Gen 5
    "iq-8275": ("soc_model", "82"),  # Dragonwing IQ-8275
    "qcs8275": ("soc_model", "82"),
}  # Qualcomm Hexagon HTP 目标及其 ONNX Runtime QNN provider 选项
HELP_MSG = """
    Examples for running Ultralytics:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.yaml")  # 从头构建新模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用模型
        results = model.train(data="coco8.yaml", epochs=3)  # 训练模型
        results = model.val()  # 在验证集上评估模型性能
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
        success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    3. Use the command line interface (CLI):

        Ultralytics 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, semantic, classify, pose, obb, depth]
                    MODE (required) is one of [train, val, predict, export, track, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"

        - 使用初始学习率 0.01 训练检测模型 10 个周期
            yolo detect train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01

        - 使用预训练分割模型以图像尺寸 320 预测 YouTube 视频：
            yolo segment predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - 使用批次大小 1、图像尺寸 640 验证预训练检测模型：
            yolo detect val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640

        - 将 YOLO26n 分类模型以 224×128 的图像尺寸导出为 ONNX 格式（无需指定 TASK）
            yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128

        - 运行特殊命令：
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# 设置和环境变量
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # 使用短格式 g，精度为 5 位
cv2.setNumThreads(0)  # 防止 OpenCV 多线程（与 PyTorch DataLoader 不兼容）
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # 抑制 "NNPACK.cpp could not initialize NNPACK" 警告
os.environ["KINETO_LOG_LEVEL"] = "5"  # 计算 FLOPs 时抑制 PyTorch profiler 的详细输出

# 集中抑制警告
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")  # PyTorch deprecation
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")  # matplotlib>=3.7.2
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")  # mobileclip timm.层 deprecation
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # ONNX/TorchScript export tracer warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")  # ONNX 形状 warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")  # CoreML np.bool deprecation
logging.getLogger("coremltools").setLevel(logging.ERROR)  # 抑制非 macOS 上的原生二进制加载失败信息

# 预编译类型元组，用于加速 isinstance() 检查
FLOAT_OR_INT = (float, int)
STR_OR_PATH = (str, Path)


class DataExportMixin:
    """用于将验证指标或预测结果导出为多种格式的混入类。

    此类提供工具，可将分类、目标检测、分割或姿态估计任务的性能指标（例如 mAP、精确率、召回率）或预测结果
    导出为 Polars DataFrame、CSV 和 JSON 等格式。

    方法：
        to_df: 将摘要转换为 Polars DataFrame。
        to_csv: 将结果导出为 CSV 字符串。
        to_json: 将结果导出为 JSON 字符串。
        tojson: `to_json()` 的弃用别名。

    示例：
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """根据预测结果摘要或验证指标创建 Polars DataFrame。

        参数：
            normalize (bool, 可选): 是否归一化数值，以便进行比较。
            decimals (int, 可选): 浮点数保留的小数位数。

        返回：
            (polars.DataFrame): 包含摘要数据的 Polars DataFrame。
        """
        import polars as pl  # scope for faster 'import ultralytics'

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """将结果或指标导出为 CSV 字符串。

        参数：
            normalize (bool, 可选): 是否归一化数值。
            decimals (int, 可选): 小数精度。

        返回：
            (str): CSV 内容字符串。
        """
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)

        try:
            return df.write_csv()
        except Exception:
            # 将剩余复杂类型转换为最简字符串
            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """将结果导出为 JSON 格式。

        参数：
            normalize (bool, 可选): 是否归一化数值。
            decimals (int, 可选): 小数精度。

        返回：
            (str): JSON 格式的结果字符串。
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:
    """用于创建可将属性转换为字符串表示的对象的简单基类。

    此类为创建易于打印或转换为字符串的对象提供基础，并显示所有不可调用属性，适用于调试和检查对象状态。

    方法：
        __str__: 返回对象的可读字符串表示。
        __repr__: 返回对象的机器可读字符串表示。
        __getattr__: 提供包含有用信息的自定义属性访问错误消息。

    示例：
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> text = str(obj)  # "<module>.MyClass object with attributes:" followed by "x: 10" and "y: 'hello'"

    注意：
        - 此类用于被继承，为检查对象属性提供便利方式。
        - 字符串表示中包含对象的模块和类名称。
        - 可调用属性以及以下划线开头的属性不会出现在字符串表示中。
    """

    def __str__(self):
        """返回对象的可读字符串表示。"""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # 对子类仅显示模块和类名称
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """返回对象的机器可读字符串表示。"""
        return self.__str__()

    def __getattr__(self, attr):
        """提供包含有用信息的自定义属性访问错误消息。"""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """可迭代的 SimpleNamespace 类，为属性访问和迭代提供增强功能。

    此类扩展 SimpleNamespace，增加了迭代、字符串表示和属性访问方法，适合作为便捷的配置参数容器。

    方法：
        __iter__: 返回命名空间属性的键值对迭代器。
        __str__: 返回对象的可读字符串表示。
        __getattr__: 提供包含有用信息的自定义属性访问错误消息。
        get: 获取指定键的值；键不存在时返回默认值。

    示例：
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    注意：
        与标准字典相比，此类特别适合以更易访问且可迭代的形式存储配置参数。
    """

    def __iter__(self):
        """返回命名空间属性的键值对迭代器。"""
        return iter(vars(self).items())

    def __str__(self):
        """返回对象的可读字符串表示。"""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """提供包含有用信息的自定义属性访问错误消息。"""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """如果指定键存在则返回其值，否则返回默认值。"""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """临时设置绘图函数 rc 参数和后端的装饰器。

    参数：
        rcparams (dict, 可选): 要设置的 rc 参数字典。
        backend (str, 可选): 要使用的后端名称。

    返回：
        (Callable): 已设置临时 rc 参数和后端的装饰函数。

    示例：
        >>> @plt_settings({"font.size": 12})
        ... def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """将临时 rc 参数和后端应用到函数的装饰器。"""

        def wrapper(*args, **kwargs):
            """设置 rc 参数和后端，调用原函数，然后恢复原设置。"""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            # 为非拉丁文本（中文、阿拉伯文等）优先添加 Arial Unicode；缺少时由 Matplotlib 使用后备字体
            if "font.sans-serif" not in rcparams and not wrapper._fonts_registered:
                from matplotlib import font_manager

                # 将 Ultralytics 配置目录中的字体（例如 Arial.Unicode.ttf）注册到 Matplotlib
                known = {f.fname for f in font_manager.fontManager.ttflist}
                for f in USER_CONFIG_DIR.glob("*.ttf"):
                    if str(f) not in known:
                        font_manager.fontManager.addfont(str(f))
                wrapper._fonts_registered = True
            rc = (
                rcparams
                if "font.sans-serif" in rcparams
                else {**rcparams, "font.sans-serif": ["Arial Unicode MS", *plt.rcParams.get("font.sans-serif", [])]}
            )

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # 自 3.8 起，切换后端时自动关闭图形的行为已弃用
                plt.switch_backend(backend)

            # 使用指定后端绘图，并始终恢复原始后端
            try:
                with plt.rc_context(rc):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        wrapper._fonts_registered = False
        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """使用 UTF-8 编码和可配置的详细程度设置日志记录。

    此函数为 Ultralytics 库配置日志，根据详细输出标志和当前进程秩设置合适的日志级别与格式化器，
    并处理 Windows 环境中默认编码可能不是 UTF-8 的特殊情况。

    参数：
        name (str): 日志记录器名称。
        verbose (bool): 为 True 时将日志级别设为 INFO，否则设为 ERROR。

    返回：
        (logging.Logger): 配置完成的日志记录器对象。

    示例：
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("这是一条信息日志")

    注意：
        - 在 Windows 上，此函数会尽可能重新配置 stdout，使其使用 UTF-8 编码。
        - 如果无法重新配置，则退回到可处理非 UTF-8 环境的自定义格式化器。
        - 此函数使用适当的格式化器和级别设置 StreamHandler。
        - 记录器的 propagate 标志设为 False，以避免日志在父记录器中重复输出。
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # 多 GPU 训练中的进程秩

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """根据日志级别为日志记录添加前缀。"""
            # 根据日志级别添加前缀
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"

            # 根据平台处理消息中的表情符号
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # 处理 Windows UTF-8 编码问题
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            # 尽可能重新配置 stdout，使其使用 UTF-8 编码
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # reconfigure 不可用时，将 stdout 包装为 TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # 使用适当的格式化器和级别创建并配置 StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.set_name("ultralytics.utils.set_logging")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # 设置日志记录器
    logger = logging.getLogger(name)
    for h in [h for h in logger.handlers if h.name == stream_handler.name]:
        logger.removeHandler(h)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# 设置日志记录器
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # 全局定义，供 train.py、val.py、predict.py 等使用
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """返回适配当前平台、可安全处理表情符号的字符串版本。"""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """确保函数或方法以线程安全方式执行的装饰器类。

    此类可作为装饰器使用，确保被装饰函数即使被多个线程调用，也只有一个线程能同时执行该函数。

    属性：
        lock (threading.Lock): 用于管理被装饰函数访问权限的锁对象。

    示例：
        >>> from ultralytics.utils import ThreadingLocked
        >>> @ThreadingLocked()
        ... def my_function():
        ...    # Your code here
    """

    def __init__(self):
        """使用线程锁初始化装饰器类。"""
        self.lock = threading.Lock()

    def __call__(self, f):
        """以线程安全方式执行函数或方法。"""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """将线程安全机制应用到被装饰函数或方法。"""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class YAML:
    """高效文件操作的 YAML 工具类，并自动检测 C 实现。

    此类使用 PyYAML 可用的最快实现（尽可能使用基于 C 的实现）优化 YAML 加载和保存操作。
    它采用单例模式和延迟初始化，无需显式实例化即可直接调用类方法，并自动处理文件路径创建、验证和字符编码问题。

    此实现通过以下方式优先保证性能：
        - 可用时自动选择基于 C 的加载器和转储器
        - 使用单例模式复用同一实例
        - 使用延迟初始化，将导入开销推迟到实际需要时
        - 使用回退机制处理有问题的 YAML 内容

    属性：
        _instance: 内部单例实例存储。
        yaml: PyYAML 模块引用。
        SafeLoader: 可用的最佳 YAML 加载器（如果可用则为 CSafeLoader）。
        SafeDumper: 可用的最佳 YAML 转储器（如果可用则为 CSafeDumper）。

    示例：
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """首次使用时初始化单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """使用最佳 YAML 实现进行初始化（可用时使用基于 C 的实现）。"""
        import yaml

        self.yaml = yaml
        # 可用时使用基于 C 的实现以提升性能
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """将 Python 对象保存为 YAML 文件。

        参数：
            file (str | Path): 要保存的 YAML 文件路径。
            data (dict | None): 要保存的字典或兼容对象。
            header (str): 可选的文件头字符串，将添加到文件开头。
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # 必要时创建父目录
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        # 将不可序列化对象转换为字符串
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # 写入 YAML 文件
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """将 YAML 文件加载为 Python 对象，并提供稳健的错误处理。

        参数：
            file (str | Path): YAML 文件路径。
            append_filename (bool): 是否将文件名添加到返回字典中。

        返回：
            (dict): 加载后的 YAML 内容。
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

        # 读取文件内容
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # 尝试加载 YAML；遇到异常字符时使用回退方案
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader)
        except Exception as e:
            # 移除异常字符后重试
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            try:
                data = instance.yaml.load(s, Loader=instance.SafeLoader)
            except Exception:
                raise ValueError(
                    f"YAML syntax error in '{file}': {e}\nVerify YAML with https://ray.run/tools/yaml-formatter"
                ) from None

        if data is None:  # 空文件、仅包含注释的文件或显式的 'null'
            data = {}
        elif not isinstance(data, dict):  # 拒绝非映射 YAML（标量或列表），并提供清晰错误信息
            raise ValueError(
                f"'{file}' is not a valid YAML mapping. Verify YAML with https://ray.run/tools/yaml-formatter"
            )

        # 检查用户误写的 None 字符串（YAML 中应使用 'null'）
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        """将 YAML 文件或对象格式化后输出到控制台。

        参数：
            yaml_file (str | Path | dict): 要输出的 YAML 文件路径或字典。
        """
        instance = cls._get_instance()

        # 提供路径时加载文件
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

        # 基于 C 的实现中使用 -1 表示不限制宽度
        dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

        LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# 默认 配置
DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """读取系统中的设备型号信息。

    返回：
        (str): 小写的平台版本字符串，用于识别 Jetson 或 Raspberry Pi 等设备型号。
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """检查操作系统是否为 Ubuntu。

    返回：
        (bool): 操作系统为 Ubuntu 时返回 True，否则返回 False。
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: list[str] | str | None = None) -> list[bool] | bool:
    """检查操作系统是否为 Debian。

    参数：
        codenames (列表[str] | None | str): 要检查的 Debian 代号（例如 'buster'、'bullseye'）。为 None 时仅检查是否为 Debian。

    返回：
        (列表[bool] | bool): 表示操作系统是否匹配每个 Debian 代号的布尔值列表；未提供代号时返回单个布尔值。
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    """检查当前脚本是否运行在 Google Colab 笔记本中。

    返回：
        (bool): 运行在 Colab 笔记本中时返回 True，否则返回 False。
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """检查当前脚本是否运行在 Kaggle 内核中。

    返回：
        (bool): 运行在 Kaggle 内核中时返回 True，否则返回 False。
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """检查当前脚本是否运行在 Jupyter Notebook 中。

    返回：
        (bool): 运行在 Jupyter Notebook 中时返回 True，否则返回 False。

    注意：
        - 此方法仅对 Colab 和 Kaggle 有效，无法可靠检测 JupyterLab 和 Paperspace 等其他环境。
        - 当手动安装 IPython 包时，globals() 中的 "get_ipython" 方法可能产生误报。
    """
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    """检查当前脚本是否运行在 RunPod 容器中。

    返回：
        (bool): 运行在 RunPod 中时返回 True，否则返回 False。
    """
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    """判断当前脚本是否运行在 Docker 容器中。

    返回：
        (bool): 运行在 Docker 容器中时返回 True，否则返回 False。
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def is_raspberrypi() -> bool:
    """判断 Python 环境是否运行在 Raspberry Pi 上。

    返回：
        (bool): 运行在 Raspberry Pi 上时返回 True，否则返回 False。
    """
    return "rpi" in DEVICE_MODEL


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """判断 Python 环境是否运行在 NVIDIA Jetson 设备上。

    参数：
        jetpack (int | None): 如果指定，则检查特定的 JetPack 版本（4、5、6）。

    返回：
        (bool): 运行在 NVIDIA Jetson 设备上时返回 True，否则返回 False。
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = Path("/etc/nv_tegra_release").read_text()
            version_map = {4: "R32", 5: "R35", 6: "R36", 7: "R38"}  # JetPack 到 L4T 主版本的映射
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


def is_dgx() -> bool:
    """检查当前脚本是否运行在 DGX（NVIDIA Data Center GPU）、DGX-Ready 或 DGX Spark 系统中。

    返回：
        (bool): 运行在 DGX、DGX-Ready 或 DGX Spark 系统中时返回 True，否则返回 False。
    """
    try:
        with open("/etc/dgx-release") as f:
            return "DGX" in f.read()
    except FileNotFoundError:
        return False


def is_online() -> bool:
    """使用 DNS（IPv4/IPv6）解析快速检查网络连接（Cloudflare + Google）。

    返回：
        (bool): 连接成功时返回 True，否则返回 False。
    """
    if env_bool("YOLO_OFFLINE"):
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    """判断给定路径中的文件是否属于 pip 包。

    参数：
        filepath (str): 要检查的文件路径。

    返回：
        (bool): 文件属于 pip 包时返回 True，否则返回 False。
    """
    import importlib.util

    # 获取模块规格
    spec = importlib.util.find_spec(filepath)

    # 返回规格和来源均不为 None 的结果，表示它是一个包
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: str | Path) -> bool:
    """检查目录是否可写。

    参数：
        dir_path (str | Path): 目录路径。

    返回：
        (bool): 目录可写时返回 True，否则返回 False。
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """判断 pytest 当前是否正在运行。

    返回：
        (bool): pytest 正在运行时返回 True，否则返回 False。
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """判断当前环境是否为 GitHub Actions 运行器。

    返回：
        (bool): 当前环境为 GitHub Actions 运行器时返回 True，否则返回 False。
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_default_args(func):
    """返回函数的默认参数字典。

    参数：
        func (callable): 要检查的函数。

    返回：
        (dict): 参数名称到参数默认值的字典。
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """如果操作系统为 Ubuntu，则获取 Ubuntu 版本。

    返回：
        (str): Ubuntu 版本；如果不是 Ubuntu 系统则返回 None。
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    """返回可写配置目录，优先使用 YOLO_CONFIG_DIR，并根据操作系统选择路径。

    参数：
        sub_dir (str): 要创建的子目录名称。

    返回：
        (Path): 用户配置目录路径。
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    if p.exists():  # 已创建，直接使用
        return p
    if is_dir_writeable(p.parent):  # 可能时创建目录
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Docker、GCP/AWS 等仅有 /tmp 可写时使用的回退路径
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt

    # 最后回退到当前工作目录
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


# 定义常量（下面的代码需要使用）
DEVICE_MODEL = read_device_model()  # is_jetson() 和 is_raspberrypi() 依赖此常量
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
IS_DEBIAN, IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE = is_debian([None, "bookworm", "trixie"])
IS_UBUNTU = is_ubuntu()
GIT = GitRepo()
USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics 设置目录
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    r"""根据给定的颜色和样式参数，使用 ANSI 转义码为字符串添加颜色。

    此函数有两种调用方式：
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    第二种形式默认应用 'blue' 和 'bold' 样式。

    参数：
        *input (str | Path): 字符串序列，前 n-1 个字符串为颜色和样式参数，最后一个字符串为要着色的内容。

    返回：
        (str): 使用指定颜色和样式的 ANSI 转义码包装后的输入字符串。

    示例：
        >>> colorstr("blue", "bold", "hello world")
        '\x1b[34m\x1b[1mhello world\x1b[0m'

    注意：
        支持的颜色和样式：
        - 基本颜色：'black'、'red'、'green'、'yellow'、'blue'、'magenta'、'cyan'、'white'
        - 亮色：'bright_black'、'bright_red'、'bright_green'、'bright_yellow'、
          'bright_blue'、'bright_magenta'、'bright_cyan'、'bright_white'
        - 其他样式：'end'、'bold'、'underline'

    参考：
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # 颜色参数和字符串
    colors = {
        "black": "\033[30m",  # 基本颜色
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # 亮色
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # 其他样式
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """移除字符串中的 ANSI 转义码，使其恢复为无颜色文本。

    参数：
        input_string (str): 要移除颜色和样式的字符串。

    返回：
        (str): 移除所有 ANSI 转义码后的新字符串。

    示例：
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        'hello world'
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """用于平稳处理异常的 Ultralytics TryExcept 类。

    此类可作为装饰器或上下文管理器捕获异常，并按需打印警告消息。即使发生异常，也可以让代码继续执行，
    适用于非关键操作。

    属性：
        msg (str): 发生异常时要显示的可选消息。
        verbose (bool): 是否打印异常消息。

    示例：
        作为装饰器：
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        ... def func():
        ...     # Function logic here
        ...     pass

        作为上下文管理器：
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        ...     # Code block here
        ...     pass
    """

    def __init__(self, msg="", verbose=True):
        """使用可选消息和详细输出设置初始化 TryExcept 类。"""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """进入 TryExcept 上下文时执行并初始化实例。"""

    def __exit__(self, exc_type, value, traceback):
        """定义退出 with 代码块时的行为，并在需要时打印错误消息。"""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """使用指数退避重试函数执行的 Retry 类。

    此装饰器可在函数发生异常时重试，最多重试指定次数，并在每次重试之间逐步增加等待时间。
    它适用于处理网络操作或其他不稳定流程中的临时故障。

    属性：
        times (int): 最大重试次数。
        delay (int): 重试之间的初始等待时间，单位为秒。

    示例：
        作为装饰器使用：
        >>> @Retry(times=3, delay=2)
        ... def test_func():
        ...     # 替换为可能抛出异常的函数逻辑
        ...     return True
    """

    def __init__(self, times=3, delay=2, verbose=True):
        """使用指定的重试次数和延迟初始化 Retry 类。"""
        self.times = times
        self.delay = delay
        self.verbose = verbose  # 调用方自己的日志会反馈到重试流程时设为 False
        self._attempts = 0

    def __call__(self, func):
        """实现带指数退避的 Retry 装饰器。"""

        def wrapped_func(*args, **kwargs):
            """对被装饰函数或方法执行重试。"""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    if self.verbose:
                        LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise
                    time.sleep(self.delay * (2**self._attempts))  # 指数退避延迟

        return wrapped_func


def threaded(func):
    """默认在多线程中运行目标函数，并返回线程对象或函数结果。

    此装饰器支持在独立线程中或同步执行目标函数。默认在线程中运行，也可以通过 `threaded=False` 关键字参数控制；
    该参数会在调用函数前从 kwargs 中移除。

    参数：
        func (callable): 可能在独立线程中执行的函数。

    返回：
        (callable): 包装函数；它返回线程对象或直接返回函数结果。线程不是守护线程，因此解释器关闭时会先等待线程结束，
            避免模块清理期间发生竞争。

    示例：
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # 在线程后台运行
        >>> result = process_data(my_data, threaded=False)  # 同步运行并返回函数结果
    """

    def wrapper(*args, **kwargs):
        """根据 `threaded` 关键字参数多线程运行函数，并返回线程对象或函数结果。"""
        if kwargs.pop("threaded", True):  # 在线程中运行
            # 使用非守护线程，使解释器关闭时先等待线程结束；守护线程可能在 C 调用中途被终止，导致进程异常退出。
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=False)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    """初始化 Sentry SDK，用于错误跟踪和报告。

    仅当已安装 sentry_sdk 包且设置中的 sync=True 时使用。运行 `yolo settings` 可查看和更新设置。

    发送错误所需条件（必须全部满足，否则不会报告错误）：
        - 已安装 sentry_sdk 包
        - YOLO 设置中的 sync=True
        - pytest 未运行
        - 运行于 pip 安装的包中
        - 运行于非 git 目录
        - 进程秩为 -1 或 0
        - 处于联网环境
        - 使用 CLI 运行包（通过主 CLI 命令名称为 'yolo' 判断）
    """
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or not ONLINE
        or not IS_PIP_PACKAGE
        or GIT.is_repo
    ):
        return
    # 未安装 sentry_sdk 包时直接返回，不使用 Sentry
    try:
        import sentry_sdk
    except ImportError:
        return

    def before_send(event, hint):
        """根据特定异常类型和消息，在将事件发送到 Sentry 前修改事件。

        参数：
            event (dict): 包含错误信息的事件字典。
            hint (dict): 包含错误附加信息的字典。

        返回：
            (dict | None): 修改后的事件；不应发送到 Sentry 时返回 None。
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # 不发送事件

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 anonymized UUID hash


class JSONDict(dict):
    """为内容提供 JSON 持久化功能的类字典对象。

    此类扩展内置字典，在内容修改时自动保存到 JSON 文件。它使用锁确保操作线程安全，并处理 Path 对象的 JSON 序列化。

    属性：
        file_path (Path): 用于持久化的 JSON 文件路径。
        lock (threading.Lock): 确保操作线程安全的锁对象。

    方法：
        _load: 从 JSON 文件将数据加载到字典。
        _save: 将字典当前状态保存到 JSON 文件。
        __setitem__: 保存键值对并持久化到磁盘。
        __delitem__: 移除项目并更新持久化存储。
        update: 更新字典并持久化修改。
        clear: 清空所有条目并更新持久化存储。

    示例：
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        值
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: str | Path = "data.json"):
        """使用指定的文件路径初始化 JSONDict 对象，以进行 JSON 持久化。"""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """从 JSON 文件将数据加载到字典。"""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    # 使用基类字典的 update，避免读取时触发持久化
                    super().update(json.load(f))
        except json.JSONDecodeError:
            LOGGER.warning(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            LOGGER.error(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        """将字典当前状态保存到 JSON 文件。"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            LOGGER.error(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        """处理 Path 对象的 JSON 序列化。"""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        """保存键值对并持久化到磁盘。"""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """移除项目并更新持久化存储。"""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """返回字典的格式化 JSON 字符串表示。"""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """更新字典并持久化修改。"""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """清空所有条目并更新持久化存储。"""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """用于管理和持久化 Ultralytics 设置的 SettingsManager 类。

    此类扩展 JSONDict，为设置提供 JSON 持久化，确保操作线程安全并提供默认值。初始化时会验证设置，
    并提供更新或重置设置的方法。设置包括数据集、权重和运行结果目录，以及各种集成开关。

    属性：
        file (Path): 用于持久化的 JSON 文件路径。
        version (str): 设置架构版本。
        defaults (dict): 包含默认设置的字典。
        help_msg (str): 指导用户查看和更新设置的帮助消息。

    方法：
        _validate_settings: 验证当前设置，并在需要时重置设置。
        update: 验证键和值类型后更新设置。
        reset: 将设置重置为默认值并保存。

    示例：
        初始化 and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.7"):
        """使用默认设置初始化 SettingsManager，并加载用户设置。"""
        import hashlib
        import uuid

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (root.parent if GIT.root and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # 设置架构版本
            "datasets_dir": str(datasets_root / "datasets"),  # 数据集目录
            "weights_dir": str(root / "weights"),  # 模型权重目录
            "runs_dir": str(root / "runs"),  # 实验运行目录
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # 匿名 UUID 的 SHA-256 哈希
            "sync": True,  # 启用同步
            "api_key": "",  # Ultralytics API 密钥
            "openai_api_key": "",  # OpenAI API 密钥
            "clearml": True,  # ClearML 集成
            "comet": True,  # Comet 集成
            "dvc": True,  # DVC 集成
            "mlflow": True,  # MLflow 集成
            "neptune": True,  # Neptune 集成
            "raytune": True,  # Ray Tune 集成
            "tensorboard": False,  # TensorBoard 日志
            "wandb": False,  # Weights & Biases 日志
            "vscode_msg": True,  # VS Code 消息
            "openvino_msg": True,  # Intel CPU 上 OpenVINO 导出消息
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart#ultralytics-settings."
        )

        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # 检查文件是否不存在或为空
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """验证设置，并将有效值迁移到当前架构。"""
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "Ultralytics settings updated to the latest schema. Existing values were preserved where possible. "
                f"{self.help_msg}"
            )
            valid = {k: v for k, v in self.items() if k in self.defaults and isinstance(v, type(self.defaults[k]))}
            if not re.fullmatch(r"ul_[0-9a-f]{40}", valid.get("api_key", "")):
                if valid.get("api_key"):
                    LOGGER.warning(
                        f"Legacy API key removed. Get a Platform API key from {PLATFORM_URL}/settings?tab=api-keys "
                        "and run 'yolo login API_KEY'."
                    )
                valid["api_key"] = ""  # 丢弃无法通过 Platform 身份验证的旧密钥
            valid["settings_version"] = self.version
            self.clear()
            self.update({**self.defaults, **valid})

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def __setitem__(self, key, value):
        """更新一个键值对。"""
        self.update({key: value})

    def update(self, *args, **kwargs):
        """验证键和值类型后更新设置。"""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        """将设置重置为默认值并保存。"""
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    """使用弃用参数时发出弃用警告，并建议使用更新后的参数。"""
    msg = f"'{arg}' is deprecated and will be removed in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)


def clean_url(url):
    """移除 URL 中的身份验证信息，例如将 `https://example.com/path/file.txt?auth` 转为 `https://example.com/path/file.txt`。"""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib 会将 :// 转为 :/，as_posix() 用于 Windows
    return unquote(url).split("?", 1)[0]  # 将 '%2F' 转为 '/'，并去除认证查询字符串


def url2file(url):
    """将 URL 转换为文件名，例如将 `https://example.com/path/file.txt?auth` 转为 `file.txt`。"""
    return Path(clean_url(url)).name or "download"


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """如果尚未安装 Ultralytics-Snippets，则显示安装 VS Code 扩展的提示。"""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # 该文件记录已卸载的扩展，而源目录仍可能保留
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"


# 在 utils 初始化时运行以下代码 ------------------------------------------------------------------------------------

# 检查首次安装步骤
PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  # 初始化设置
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # global datasets 目录
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # global 权重 目录
RUNS_DIR = Path(SETTINGS["runs_dir"])  # global runs 目录
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()

# 应用猴子补丁
torch.save = torch_save
if WINDOWS:
    # 为图像路径中的非 ASCII 和非 UTF 字符应用 cv2 补丁
    cv2.imread, cv2.imwrite, cv2.imshow = imread_unicode, imwrite, imshow
