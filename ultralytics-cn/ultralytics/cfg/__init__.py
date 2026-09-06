# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ultralytics import __version__
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    FLOAT_OR_INT,
    IS_VSCODE,
    LOGGER,
    PLATFORM_URL,
    RANK,
    ROOT,
    SETTINGS,
    SETTINGS_FILE,
    STR_OR_PATH,
    TESTS_RUNNING,
    YAML,
    IterableSimpleNamespace,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
)

# 定义有效的解决方案
SOLUTION_MAP = {
    "count": "ObjectCounter",
    "crop": "ObjectCropper",
    "blur": "ObjectBlurrer",
    "workout": "AIGym",
    "heatmap": "Heatmap",
    "isegment": "InstanceSegmentation",
    "visioneye": "VisionEye",
    "speed": "SpeedEstimator",
    "queue": "QueueManager",
    "analytics": "Analytics",
    "inference": "Inference",
    "trackzone": "TrackZone",
    "region": "RegionCounter",
    "security": "SecurityAlarm",
    "parking": "ParkingManagement",
    "help": None,
}

# 定义有效的任务和模式，顺序与文档及 Ultralytics Platform 中的出现顺序一致
MODES = ("train", "val", "predict", "export", "track", "benchmark")
TASKS = ("detect", "segment", "semantic", "depth", "classify", "pose", "obb")
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "semantic": "cityscapes8.yaml",
    "depth": "depth8.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2CALIBRATIONDATA = {
    "detect": "coco128.yaml",
    "segment": "coco128-seg.yaml",
    "semantic": "cityscapes8.yaml",
    "depth": "depth8.yaml",
    "classify": "imagenet100",
    "pose": "coco8-pose.yaml",
    "obb": "dota128.yaml",
}
TASK2MODEL = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
    "semantic": "yolo26n-sem.pt",
    "depth": "yolo26n-depth.pt",
    "classify": "yolo26n-cls.pt",
    "pose": "yolo26n-pose.pt",
    "obb": "yolo26n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "semantic": "metrics/mIoU",
    "depth": "metrics/delta1",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}

ARGV = sys.argv or ["", ""]  # sys.argv 有时为空列表
_YOLO_CLI_COMMAND = [sys.executable, "-m", "ultralytics.cfg.__init__"]
SOLUTIONS_HELP_MSG = f"""
    收到的参数：{["yolo", *ARGV[1:]]!s}。以下是 Ultralytics 'yolo solutions' 的用法概览：

        yolo solutions SOLUTION ARGS

        其中 SOLUTION（可选）可以是 {list(SOLUTION_MAP.keys())[:-1]} 中的任意一个值
              ARGS（可选）是任意数量的自定义 'arg=value' 参数，例如 'show_in=True'，用于覆盖默认值
                  详见 https://docs.ultralytics.com/usage/cfg

    1. 调用目标计数解决方案
        yolo solutions count source="path/to/video.mp4" region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"

    2. 调用热力图解决方案
        yolo solutions heatmap colormap=cv2.COLORMAP_PARULA model=yolo26n.pt

    3. 调用队列管理解决方案
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" model=yolo26n.pt

    4. 调用俯卧撑训练监测解决方案
        yolo solutions workout model=yolo26n-pose.pt kpts=[6, 8, 10]

    5. 生成分析图表
        yolo solutions analytics analytics_type="pie"

    6. 跟踪指定区域内的目标
        yolo solutions trackzone source="path/to/video.mp4" region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"

    7. 统计指定区域内的目标数量
        yolo solutions region source="path/to/video.mp4" region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"

    8. 运行安全告警监测（邮件提醒需要使用 Python API）
        yolo solutions security source="path/to/video.mp4"

    9. 监测停车位占用情况（请先通过 Python 的 ParkingPtsSelection 创建 JSON 标注）
        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json"

    10. 使用 Streamlit 运行实时摄像头推理界面
        yolo streamlit-predict
    """
CLI_HELP_MSG = f"""
    收到的参数：{["yolo", *ARGV[1:]]!s}。Ultralytics 'yolo' 命令使用以下语法：

        yolo TASK MODE ARGS

        其中 TASK（可选）可以是 {list(TASKS)} 中的任意一个值
                MODE（必需）可以是 {list(MODES)} 中的任意一个值
                ARGS（可选）是任意数量的自定义 'arg=value' 参数，例如 'imgsz=320'，用于覆盖默认值。
                    所有 ARGS 详见 https://docs.ultralytics.com/usage/cfg，或运行 'yolo cfg' 查看

    1. 使用初始 learning_rate 为 0.01 的设置训练检测模型 10 个周期
        yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01

    2. 使用预训练分割模型，以图像尺寸 320 对 YouTube 视频进行预测：
        yolo predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. 使用批量大小 1、图像尺寸 640 验证预训练检测模型：
        yolo val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640

    4. 将 YOLO26n 分类模型以图像尺寸 224×128 导出为 ONNX 格式（无需指定 TASK）
        yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics 解决方案用法
        yolo solutions count or any of {list(SOLUTION_MAP.keys())[1:-1]} source="path/to/video.mp4"

    6. 运行特殊命令：
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo login API_KEY
        yolo logout
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    Docs: https://docs.ultralytics.com
    Platform: https://platform.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# 量化别名：将所有可接受的 `quantize` 值映射为规范形式。常用形式是整数位宽 8（INT8）、16（FP16）和 32（FP32）；
# 同时也接受 w<权重>a<激活值> 形式，并将其归并为相同的整数值，但没有对应位宽简写的混合精度方案除外：
# 'w8a16' 表示 INT8 权重和 FP16 激活值，'w8a32' 表示 INT8 权重和 FP32 激活值（即 LiteRT 动态量化/仅权重量化 INT8）。
QUANTIZE_ALIASES = {
    "8": 8,
    "16": 16,
    "32": 32,
    "int8": 8,
    "fp16": 16,
    "fp32": 32,
    "w8a8": 8,
    "w16a16": 16,
    "w32a32": 32,
    "w8a16": "w8a16",
    "w8a32": "w8a32",
}
QUANTIZE_DOCS_URL = "https://docs.ultralytics.com/modes/export#quantization-options"
QUANTIZE_VALID_VALUES = "8, 16, 32, 'int8', 'fp16', 'fp32', 'w8a8', 'w16a16', 'w8a16', or 'w8a32'"

# 定义用于参数类型检查的键
CFG_FLOAT_KEYS = frozenset(
    {  # 整数或浮点数参数，例如 x=2 和 x=2.0
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "pose",
        "kobj",
        "rle",
        "angle",
        "dlog",
        "dgrad",
        "dis",
        "degrees",
        "shear",
        "time",
        "workspace",
        "batch",
    }
)
CFG_FRACTION_KEYS = frozenset(
    {  # 分数型浮点参数取值范围为 [0.0, 1.0]，但数据集 fraction 使用 (0.0, 1.0]
        "dropout",
        "lr0",
        "lrf",
        "cls_pw",
        "momentum",
        "weight_decay",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
        "conf",
        "iou",
        "fraction",
        "multi_scale",
        "dlam",
    }
)
CFG_INT_KEYS = frozenset(
    {  # 仅允许整数的参数
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
    }
)
CFG_INT_MIN = {  # 用作除数、尺寸或随机种子的整数参数的最小有效值
    "nbs": 1,
    "max_det": 1,
    "mask_ratio": 1,
    "vid_stride": 1,
    "seed": 0,
}
CFG_BOOL_KEYS = frozenset(
    {  # 仅允许布尔值的参数
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "channels_last",
        "end2end",
        "cls_remap",
    }
)
CFG_STR_KEYS = frozenset({"optimizer", "split", "copy_paste_mode", "auto_augment"})


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    """将配置对象转换为字典。

    参数：
        cfg (str | Path | dict | SimpleNamespace): 要转换的配置对象，可以是文件路径、字符串、字典或 SimpleNamespace 对象。

    返回：
        (dict): 字典格式的配置对象。

    示例：
        将 YAML 文件路径转换为字典：
        >>> config_dict = cfg2dict("config.yaml")

        将 SimpleNamespace 转换为字典：
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        直接传递已有字典：
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    注意：
        - 如果 cfg 是路径或字符串，则按 YAML 加载并转换为字典。
        - 如果 cfg 是 SimpleNamespace 对象，则使用 vars() 将其转换为字典。
        - 如果 cfg 已经是字典，则原样返回。
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)  # 加载字典
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # 转换为字典
    return cfg


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace:
    """从文件或字典加载并合并配置数据，并应用可选的覆盖项。

    参数：
        cfg (str | Path | dict | SimpleNamespace): 配置数据源，可以是文件路径、字典或 SimpleNamespace 对象。
        overrides (dict | None): 用于覆盖基础配置的键值对字典。

    返回：
        (SimpleNamespace): 包含合并后配置参数的命名空间。

    示例：
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # 加载默认配置
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch": 16})

    注意：
        - 如果同时提供 `cfg` 和 `overrides`，则优先使用 `overrides` 中的值。
        - 特殊处理可确保配置的一致性和正确性，例如将数字形式的 `project` 和 `name` 转为字符串，并验证配置键和值。
        - 此函数会检查配置数据的类型和值。
    """
    cfg = cfg2dict(cfg)

    # 合并覆盖项
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # 合并 cfg 和 overrides 字典（优先使用 overrides）

    # 对数字形式的 project/name 做特殊处理
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # 将模型名赋给 'name' 参数
        cfg["name"] = Path(str(cfg.get("model") or "")).stem
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    # 类型和值检查
    check_cfg(cfg)

    # 返回实例
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg: dict, hard: bool = True) -> None:
    """检查 Ultralytics 库配置参数的类型和值。

    此函数验证配置参数的类型和值，在必要时进行转换以确保正确性，并检查 `CFG_FLOAT_KEYS`、
    `CFG_FRACTION_KEYS`、`CFG_INT_KEYS` 和 `CFG_BOOL_KEYS` 等全局变量定义的键类型。

    参数：
        cfg (dict): 要验证的配置字典。
        hard (bool): 为 True 时对无效类型和值抛出异常；为 False 时尝试进行转换。

    示例：
        >>> config = {
        ...     "epochs": 50,  # 有效整数
        ...     "lr0": 0.01,  # 有效浮点数
        ...     "momentum": 0.937,  # 有效浮点数
        ...     "save": "true",  # 无效布尔值
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 0.937, 'save': True}

    注意：
        - 此函数会原地修改输入字典。
        - None 值会被忽略，因为它们可能来自可选参数。
        - fraction 键使用 [0.0, 1.0]，但数据集 fraction 使用 (0.0, 1.0]。
    """
    typed_keys = CFG_FLOAT_KEYS | CFG_FRACTION_KEYS | CFG_INT_KEYS | CFG_BOOL_KEYS | CFG_STR_KEYS | {"scale", "compile"}
    for k, v in cfg.items():
        if v is None and DEFAULT_CFG_DICT.get(k) is not None and k in typed_keys and k != "auto_augment":
            raise TypeError(f"'{k}=None' is invalid. '{k}' must not be None.")
        if v is not None:  # None 值可能来自可选参数
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k == "scale":
                if isinstance(v, (list, tuple)):
                    if len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
                        if hard:
                            raise TypeError(
                                f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int, float, or a tuple/list of two floats (i.e. '{k}=(0.5, 2.0)')"
                            )
                        continue
                    continue
                elif not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0) or (k == "fraction" and v == 0.0):
                    raise ValueError(f"'{k}={v}' is invalid. Use (0.0, 1.0] for fraction; [0.0, 1.0] otherwise.")
            elif k in CFG_INT_KEYS:
                if not isinstance(v, int):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                        )
                    cfg[k] = v = int(v)
                if k in CFG_INT_MIN and v < CFG_INT_MIN[k]:
                    raise ValueError(f"'{k}={v}' is an invalid value. '{k}' must be >= {CFG_INT_MIN[k]}.")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)
            elif k in CFG_STR_KEYS and not isinstance(v, str):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a str.")
                cfg[k] = str(v)
            elif k == "compile" and not isinstance(v, (bool, str)):  # False 表示关闭，True 表示使用默认值，也可以传入模式字符串
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool or str (i.e. '{k}=True' or '{k}=max-autotune')"
                    )
                cfg[k] = bool(v)
            elif k == "quantize":  # 将 8/16/32 或 w 形式规范化为对应方案（未设置时保持 None，表示 FP32）
                scheme = QUANTIZE_ALIASES.get(str(v).lower())
                if scheme is None:
                    if hard:
                        raise ValueError(
                            f"'{k}={v}' is invalid. Valid '{k}' values are {QUANTIZE_VALID_VALUES}. "
                            f"See {QUANTIZE_DOCS_URL}"
                        )
                else:
                    cfg[k] = scheme


def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    """根据参数或默认设置，返回用于保存输出结果的目录路径。

    参数：
        args (SimpleNamespace): 包含 'project'、'name'、'task'、'mode' 和 'save_dir' 等配置的命名空间对象。
        name (str | None): 输出目录的可选名称。如果未提供，则使用 'args.name' 或 'args.mode'。

    返回：
        (Path): 用于保存输出结果的目录路径。

    示例：
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", name="exp", task="detect", mode="train", exist_ok=True)
        >>> get_save_dir(args).parts[-3:]
        ('detect', 'my_project', 'exp')
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or ""
        if not Path(project).is_absolute():
            base = ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else Path(SETTINGS["runs_dir"])
            worker = os.environ.get("PYTEST_XDIST_WORKER")
            if worker and TESTS_RUNNING:  # 为并行运行的 pytest-xdist 工作进程隔离目录
                base = base / worker
            project = base / args.task / project
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir).resolve()  # 解析为完整路径，以便在控制台显示


def _handle_deprecation(custom: dict) -> dict:
    """处理已弃用的配置键，将其映射为当前配置键，并发出弃用警告。

    参数：
        custom (dict): 可能包含已弃用配置键的配置字典。

    返回：
        (dict): 替换已弃用配置键后的配置字典。

    示例：
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        {'show_boxes': True, 'show_labels': False, 'line_width': 2}

    注意：
        此函数会原地修改输入字典，将已弃用的配置键替换为当前配置键。
        必要时还会转换配置值，例如反转 'hide_labels' 和 'hide_conf' 的布尔值。
    """
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    # 将已弃用的精度标志转发到统一的 `quantize` 方案（int8 优先于 half）。先将值解析为布尔值：带引号的字符串
    # 'False' 会禁用该选项，而不带值的 CLI 标志（空字符串）会启用该选项；显式传入 false 会映射为 None，
    # 以清除继承的 quantize 设置。显式传入的 `quantize=` 始终优先于旧版精度标志。
    int8 = custom.pop("int8", None)
    half = custom.pop("half", None)
    if (int8 is not None or half is not None) and "quantize" not in custom:
        int8_on = int8 is not None and str(int8).strip().lower() not in {"none", "false", "0"}
        half_on = half is not None and str(half).strip().lower() not in {"none", "false", "0"}
        custom["quantize"] = 8 if int8_on else 16 if half_on else None  # False/0 清除精度设置并恢复为 FP32
        deprecation_warn("int8" if int8 is not None else "half", "quantize")

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom


def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None:
    """检查自定义配置字典与基础配置字典的一致性，处理已弃用的配置键，并为不匹配的键提供错误信息。

    参数：
        base (dict): 包含有效配置键的基础配置字典。
        custom (dict): 待检查一致性的自定义配置字典。
        e (Exception | None): 可选的异常实例，由调用方传入。
        allowed_custom_keys (set | None): 允许出现在自定义字典中的额外配置键集合。

    异常：
        SyntaxError: 自定义配置与基础配置之间存在不匹配的键时抛出。

    示例：
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SyntaxError:
        ...     print("Mismatched keys found")
        Mismatched keys found

    注意：
        - 根据与有效键的相似度，为不匹配的键提供可能的修正建议。
        - 自动将自定义配置中的已弃用配置键替换为当前配置键。
        - 为每个不匹配的键输出详细错误信息，帮助用户修正配置。
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    # 允许将 'augmentations' 作为自定义 Albumentations 变换的有效参数
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations", "save_dir"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # 获取相似的键列表
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: list[str]) -> list[str]:
    """合并字符串列表中独立的 '=' 参数，并拼接被方括号拆开的参数片段。

    此函数处理以下情况：
        1. ['arg', '=', 'val'] 合并为 ['arg=val']
        2. ['arg=', 'val'] 合并为 ['arg=val']
        3. ['arg', '=val'] 合并为 ['arg=val']
        4. 拼接方括号中的片段，例如 ['imgsz=[3,', '640,', '640]'] 合并为 ['imgsz=[3,640,640]']

    参数：
        args (列表[str]): 字符串列表，其中每个元素表示一个参数或参数片段。

    返回：
        (列表[str]): 合并独立 '=' 两侧参数并拼接方括号片段后的字符串列表。

    示例：
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_equals_args(args)
        ['arg1=值', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # 处理等号的合并
        if arg == "=" and 0 < i < len(args) - 1:  # 合并 ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # 合并 ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # 合并 ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # 处理方括号片段的拼接
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # 添加剩余的当前字符串
    if current:
        new_args.append(current)

    return new_args


def handle_yolo_login(args: list[str]) -> None:
    """使用 API 密钥登录 Ultralytics Platform，或删除已保存的密钥。"""
    if args[0] == "logout":
        SETTINGS["api_key"] = ""
        LOGGER.info("Logged out ✅. To log in again, use 'yolo login API_KEY'.")
        return

    api_key_url = f"{PLATFORM_URL}/settings?tab=api-keys"
    if len(args) < 2:
        LOGGER.info(f"Get an API key from {api_key_url} and then run 'yolo login API_KEY'.")
        return

    import requests  # 延迟导入，因为该模块加载较慢

    try:
        response = requests.get(
            f"{PLATFORM_URL}/api/settings",
            headers={"Authorization": f"Bearer {args[1]}"},
            timeout=30,
        )
        if response.status_code == 200:
            SETTINGS["api_key"] = args[1]
            LOGGER.info("New authentication successful ✅")
        elif response.status_code == 401:
            LOGGER.warning("Invalid API key")
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.warning(f"Authentication request failed, check your connection: {e}")


def handle_yolo_settings(args: list[str]) -> None:
    """处理 YOLO settings 命令行界面（CLI）命令。

    此函数处理 YOLO settings CLI 命令，例如重置设置或更新单个设置项。
    执行与 YOLO 设置管理相关的命令行参数时应调用此函数。

    参数：
        args (list[str]): 用于管理 YOLO 设置的命令行参数列表。

    示例：
        >>> handle_yolo_settings(["reset"])  # 重置 YOLO 设置
        >>> handle_yolo_settings(["runs_dir=path/to/dir"])  # 更新指定设置

    注意：
        - 如果未提供参数，则显示当前设置。
        - 'reset' 命令会删除现有设置文件，并创建新的默认设置。
        - 其他参数会被视为键值对，用于更新指定设置。
        - 此函数会检查传入设置与现有设置的一致性。
        - 处理完成后会显示更新后的设置。
        - 关于 YOLO 设置管理的更多信息，请访问：
          https://docs.ultralytics.com/quickstart#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart#ultralytics-settings"  # 帮助文档地址
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # 删除设置文件
                SETTINGS.reset()  # 创建新的默认设置
                LOGGER.info("Settings reset successfully")  # 提示用户设置已重置
            else:  # 保存新的设置
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for k, v in new.items():
                    LOGGER.info(f"✅ Updated '{k}={v}'")

        LOGGER.info(SETTINGS)  # 打印当前设置
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"settings error: '{e}'. Please see {url} for help.")


def handle_yolo_solutions(args: list[str]) -> None:
    """解析 YOLO solutions 参数，并运行指定的计算机视觉解决方案流程。

    参数：
        args (列表[str]): 用于配置和运行 Ultralytics YOLO 解决方案的命令行参数。

    示例：
        使用默认设置运行人数统计解决方案：
        >>> handle_yolo_solutions(["count"])

        使用自定义配置运行分析功能：
        >>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video.mp4"])

        使用自定义配置运行推理，需要 Streamlit 1.29.0 或更高版本。
        >>> handle_yolo_solutions(["inference", "model=yolo26n.pt"])

    注意：
        - 参数可以使用 'key=value' 格式提供，也可以使用布尔标志。
        - 可用的解决方案定义在 SOLUTION_MAP 中，并对应相应的类和方法。
        - 如果提供了无效的解决方案，则默认使用 'count' 解决方案。
        - 输出视频会保存到 'runs/solutions/exp' 目录。
        - 对于 'analytics' 解决方案，会跟踪帧编号以生成分析图表。
        - 按下 'q' 键可以中断视频处理。
        - 函数会按顺序处理视频帧，并将输出保存为 .avi 格式。
        - 如果未指定 source，则下载并使用默认示例视频。
        - 推理解决方案会通过 'streamlit run' 命令启动。
        - Streamlit 应用文件位于 Ultralytics 软件包目录中。
    """
    from ultralytics.solutions.config import SolutionConfig

    full_args_dict = vars(SolutionConfig())  # 参数字典
    overrides = {}

    # 检查字典的一致性
    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)  # 检查字典一致性

    # 获取解决方案名称
    if not args:
        LOGGER.warning("No solution name provided. i.e `yolo solutions count`. Defaulting to 'count'.")
        args = ["count"]
    if args[0] == "help":
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return  # 'help' 情况直接返回
    elif args[0] in SOLUTION_MAP:
        solution_name = args.pop(0)  # 直接提取解决方案名称
    else:
        LOGGER.warning(
            f"❌ '{args[0]}' is not a valid solution. 💡 Defaulting to 'count'.\n"
            f"🚀 Available solutions: {', '.join(list(SOLUTION_MAP.keys())[:-1])}\n"
        )
        solution_name = "count"  # 无效解决方案的默认值

    if solution_name == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 Loading Ultralytics live inference app...")
        subprocess.run(
            [  # 使用自定义 Streamlit 参数运行子进程
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo26n.pt"),
            ],
            check=False,
        )
    else:
        import cv2  # 仅 cap 和 vw 功能需要使用

        from ultralytics import solutions

        solution = getattr(solutions, SOLUTION_MAP[solution_name])(is_cli=True, **overrides)  # 类，例如 ObjectCounter

        cap = cv2.VideoCapture(solution.CFG["source"])  # 读取视频文件
        if solution_name != "crop":
            # 获取视频文件的宽度、高度和帧率，创建保存目录并初始化视频写入器
            w, h, fps = (
                int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
            )
            if solution_name == "analytics":  # 分析图表的输出尺寸固定为 w=1280、h=720
                w, h = 1280, 720
            save_dir = get_save_dir(SimpleNamespace(task="solutions", name="exp", exist_ok=False, project=None))
            save_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录，例如 runs/solutions/exp
            vw = cv2.VideoWriter(str(save_dir / f"{solution_name}.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:  # 处理视频帧
            f_n = 0  # 帧数量，分析图表需要使用
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = solution(frame, f_n := f_n + 1) if solution_name == "analytics" else solution(frame)
                if solution_name != "crop":
                    vw.write(results.plot_im)
                if solution.CFG["show"] and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()


def parse_key_value_pair(pair: str = "key=value") -> tuple:
    """将键值对字符串解析为独立的键和值。

    参数：
        pair (str): 包含键值对的字符串，格式为 "key=value"。

    返回：
        key (str): 解析得到的键。
        value (str): 解析得到的值。

    异常：
        AssertionError: 值缺失或为空时抛出。

    示例：
        >>> key, value = parse_key_value_pair("model=yolo26n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo26n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    注意：
        - 此函数按第一个 '=' 字符拆分输入字符串。
        - 会删除键和值首尾的空白字符。
        - 如果删除空白后值为空，则抛出断言错误。
    """
    k, v = pair.split("=", 1)  # 按第一个 '=' 符号拆分
    k, v = k.strip(), v.strip()  # 删除首尾空白
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v: str) -> Any:
    """将值的字符串表示转换为合适的 Python 类型。

    此函数会尝试将给定字符串转换为最合适的 Python 对象，支持转换为 None、bool、int、float
    以及其他可以安全求值的类型。

    参数：
        v (str): 待转换值的字符串表示。

    返回：
        (Any): 转换后的值。类型可以是 None、bool、int、float；如果无法转换，则返回原始字符串。

    示例：
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> print(smart_value("None"))
        None
        >>> smart_value("some_string")
        'some_string'

    注意：
        - 此函数会以不区分大小写的方式比较布尔值和 None 值。
        - 对于其他类型，函数会尝试使用 Python 的 ast.literal_eval() 进行安全求值。
        - 如果无法完成转换，则返回原始字符串。
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            name, _, attr = v.rpartition(".")
            if (module := sys.modules.get(name)) and attr.isupper():
                value = getattr(module, attr, None)
                if isinstance(value, (int, float)):
                    return value
            return v


def entrypoint(debug: str = "") -> None:
    """解析并执行命令行参数的 Ultralytics 入口函数。

    此函数是 Ultralytics CLI 的主要入口，负责解析命令行参数并执行相应任务，
    例如训练、验证、预测和导出模型等。

    参数：
        debug (str): 用于调试的、以空格分隔的命令行参数字符串。

    示例：
        使用初始 learning_rate 为 0.01 的设置训练检测模型 10 个周期：
        >>> entrypoint("train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01")

        使用预训练分割模型，以图像尺寸 320 对 YouTube 视频进行预测：
        >>> entrypoint("predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        使用批量大小 1、图像尺寸 640 验证预训练检测模型：
        >>> entrypoint("val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640")

    注意：
        - 如果未传入参数，则显示用法帮助信息。
        - 所有可用命令及其参数请参阅帮助信息，以及 Ultralytics 文档：https://docs.ultralytics.com。
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # 未传入参数
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "login": lambda: handle_yolo_login(args),
        "logout": lambda: handle_yolo_login(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # 定义特殊命令的常见误用形式，例如 -h、-help 和 --help
    special.update({k[0]: v for k, v in special.items()})  # 单字符形式
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # 去掉末尾 s 的形式
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # 基础覆盖项，例如 imgsz=320
    for a in merge_equals_args(args):  # 合并 '=' 符号两侧的空格
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # 传入了自定义 YAML 配置文件
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # 默认布尔参数自动设为 True，例如 'yolo show' 会设置 show=True
        elif a in {"half", "int8"}:
            overrides[a] = True  # 不带值的已弃用精度标志，之后由 _handle_deprecation 转发到 quantize
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # 检查配置键
    check_dict_alignment(full_args_dict, overrides)

    # 模式
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{CLI_HELP_MSG}")

    # 任务
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # 模型
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo26n.pt"
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # 推测模型架构
        from ultralytics import RTDETR

        model = RTDETR(model)  # 无需传入 task 参数
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
        if "yoloe" in stem or "world" in stem:
            cls_list = overrides.pop("classes", DEFAULT_CFG.classes)
            if cls_list is not None and isinstance(cls_list, str):
                model.set_classes([c.strip() for c in cls_list.split(",")])  # "person, bus" -> ['person', 'bus']
    # 更新任务
    if task != model.task:
        if task:
            LOGGER.warning(
                f"conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # 模式
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")

    # 在 Python 中运行命令
    getattr(model, mode)(**overrides)  # 使用模型的默认参数

    # 显示帮助信息
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    # 推荐 VS Code 扩展
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


# 特殊模式 ------------------------------------------------------------------------------------------------------------
def copy_default_cfg() -> None:
    """复制默认配置文件，并创建一个名称末尾带有 '_copy' 的新配置文件。

    此函数复制现有的默认配置文件（DEFAULT_CFG_PATH），并将副本保存到当前工作目录，
    文件名末尾添加 '_copy'。这为用户基于默认设置创建自定义配置文件提供了便利。

    示例：
        >>> copy_default_cfg()

    注意：
        - 新配置文件会创建在当前工作目录中。
        - 复制完成后，函数会输出新文件的位置，以及展示如何使用该配置文件的 YOLO 命令示例。
        - 对于希望修改默认配置、但不想改动原始文件的用户，此函数非常实用。
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"使用此自定义配置文件的 YOLO 命令示例：\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # 示例：entrypoint(debug='yolo predict model=yolo26n.pt')
    entrypoint(debug="")
