# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import functools
import glob
import inspect
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    AUTOINSTALL,
    GIT,
    IS_COLAB,
    IS_DOCKER,
    IS_JETSON,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PLATFORM_API_URL,
    PLATFORM_URL,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    env_bool,
    is_github_action_running,
    url2file,
)

REMOTE_FILE_PREFIXES = ("https://", "http://", "rtsp://", "rtmp://", "tcp://", "ul://", "gs://")


def normalize_platform_uri(uri):
    """将 Ultralytics Platform 网页 URL 重写为 ul:// URI，以便直接加载为数据或模型。

    参数：
        uri (str | Path): 资源标识符，例如以 "/user/datasets/slug" 结尾的 Ultralytics Platform 网页 URL。

    返回：
        (str | Path): 对于 Platform 网页 URL 返回 "ul://user/datasets/slug"，否则返回未修改的输入值。
    """
    s = str(uri)
    return f"ul://{s[len(PLATFORM_URL) + 1 :].strip('/')}" if s.startswith(f"{PLATFORM_URL}/") else uri


def resolve_platform_uri(uri, hard=True):
    """通过 Ultralytics Platform 身份验证，将 ul:// URI 解析为签名 URL。

    格式：
        ul://username/datasets/slug  -> 返回指向 NDJSON 文件的签名 URL
        ul://username/project/model  -> 返回指向 .pt 文件的签名 URL

    参数：
        uri (str): 以 "ul://" 开头的 Platform URI。
        hard (bool): 解析失败时是否抛出异常。

    返回：
        (str | None): 成功时返回签名 URL；未找到资源且 hard=False 时返回 None。

    异常：
        ValueError: API 密钥或 URI 无效时抛出。
        PermissionError: 访问被拒绝时抛出。
        RuntimeError: 资源尚未就绪或 Platform 返回其他错误时抛出。
        FileNotFoundError: hard=True 且资源不存在时抛出。
        ConnectionError: hard=True 且请求失败时抛出。
    """
    import requests  # 延迟导入，因为该模块加载较慢

    # 延迟导入：SettingsManager 会导入 torch_utils，而 torch_utils 会在 SETTINGS 赋值前导入 checks。
    from ultralytics.utils import SETTINGS

    path = str(uri)[5:]
    parts = path.split("/")
    api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not api_key:
        raise ValueError(f"ULTRALYTICS_API_KEY required for '{uri}'. Get a key at {PLATFORM_URL}/settings")

    if len(parts) == 3 and parts[1] == "datasets":
        username, _, slug = parts
        endpoint = f"datasets/{username}/{slug}/export"
    elif len(parts) == 3:
        username, project, model = parts
        endpoint = f"models/{username}/{project}/{model}/download"
    else:
        raise ValueError(f"Invalid Platform URI: {uri}. Use ul://user/datasets/name or ul://user/project/model")

    url = f"{PLATFORM_API_URL}/{endpoint}"
    # 连接超时时间较短以便快速重试；读取超时时间较长，以等待服务器端生成结果。
    timeout = (10, 3600) if "/datasets/" in url else (10, 90)
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        for attempt in range(5):
            try:
                # GET 可以保留 Platform 的错误响应内容，而 HEAD 不会保留。
                response = requests.get(url, headers=headers, allow_redirects=False, timeout=timeout)
                if response.status_code in {408, 429} or response.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"HTTP {response.status_code}", response=response)
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.HTTPError,
            ) as error:
                if attempt >= 4:
                    raise
                delay = 2 * (2**attempt)  # 2s, 4s, 8s, 16s backoff
                LOGGER.warning(f"Retry {attempt + 1}/5 for {uri} in {delay}s: {error}")
                time.sleep(delay)
    except Exception as error:
        if hard:
            raise ConnectionError(f"Failed to resolve {uri}: {error}") from error
        LOGGER.warning(f"Failed to resolve {uri}: {error}")
        return None

    if 300 <= response.status_code < 400 and "location" in response.headers:
        return response.headers["location"]

    # 只返回 Platform 提供的受限长度 JSON 错误信息，避免代理或 WAF 页面泄露 Authorization 请求头。
    try:
        detail = str(response.json().get("error", "")).strip()
    except (AttributeError, TypeError, ValueError):
        detail = ""
    detail = f" {detail[:500]}" if detail else ""
    if response.status_code == 401:
        raise ValueError(f"Invalid ULTRALYTICS_API_KEY for '{uri}'.{detail}")
    if response.status_code == 403:
        raise PermissionError(f"Access denied for '{uri}'. Check dataset/model visibility settings.{detail}")
    if response.status_code == 404:
        if hard:
            raise FileNotFoundError(f"Not found on Platform: {uri}.{detail}")
        LOGGER.warning(f"Not found on Platform: {uri}.{detail}")
        return None
    if response.status_code == 409:
        raise RuntimeError(f"Resource not ready: {uri}. Dataset may still be processing.{detail}")
    raise RuntimeError(f"Platform error for '{uri}' (HTTP {response.status_code}).{detail or f' {response.reason}'}")


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """解析 requirements.txt 文件，忽略以 '#' 开头的行以及 '#' 之后的所有文本。

    参数：
        file_path (Path): requirements.txt 文件的路径。
        package (str, 可选): 用于替代 requirements.txt 文件的 Python 软件包名称。

    返回：
        requirements (列表[SimpleNamespace]): 解析后的依赖列表，每项都是包含 `name` 和 `specifier` 属性的
            SimpleNamespace 对象。

    示例：
        >>> from ultralytics.utils.checks import parse_requirements
        >>> parse_requirements(package="ultralytics")
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.partition("#")[0].strip()  # 忽略行内注释
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def get_distribution_name(import_name: str) -> str:
    """获取指定导入名称对应的 pip 发行包名称（例如 'cv2' -> 'opencv-python-headless'）。"""
    for dist in metadata.distributions():
        top_level = (dist.read_text("top_level.txt") or "").split()
        if import_name in top_level:
            return dist.metadata["Name"]
    return import_name


@functools.lru_cache
def parse_version(version="0.0.0") -> tuple:
    """将版本字符串转换为由发行版本段组成的整数元组，并忽略前缀和后缀。

    此函数不遵循 PEP 440：会删除预发布、开发版、后发布和本地版本后缀，因此 '1.0rc1'、'1.0.post1' 和
    '1.0+cu118' 都会与 '1.0' 比较为相同。如果需要精确的预发布版本排序，请使用 `packaging` 库。

    参数：
        version (str): 版本字符串，例如 '2.0.1+cpu'、'4.13.0.92' 或 'v2.1'。

    返回：
        (tuple): 表示发行版本段的整数元组，长度至少为 3，例如 (2, 0, 1)。
    """
    try:
        nums = [int(x) for x in re.search(r"\d+(?:\.\d+)*", version).group(0).split(".")]
        return tuple(nums + [0] * (3 - len(nums)))  # 保留所有发行版本段，忽略 'v' 前缀和 '+cu118'/'rc1' 后缀
    except Exception as e:
        LOGGER.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """检查字符串是否仅由 ASCII 字符组成。

    参数：
        s (str | 列表 | tuple | dict): 待检查的输入（所有输入都会转换为字符串后检查）。

    返回：
        (bool): 字符串仅由 ASCII 字符组成时返回 True，否则返回 False。
    """
    return str(s).isascii()


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """检查图像尺寸是否在每个维度上都是给定步长的倍数。

    如果图像尺寸不是步长的倍数，则将其调整为大于或等于给定 floor 值的最近步长倍数。

    参数：
        imgsz (int | 列表[int]): 图像尺寸。
        stride (int): 步长值。
        min_dim (int): 最小维度数量。
        max_dim (int): 最大维度数量。
        floor (int): 图像尺寸允许的最小值。

    返回：
        (列表[int] | int): 调整后的图像尺寸。
    """
    # 如果步长是张量，则将其转换为整数
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # 如果图像尺寸是整数，则将其转换为列表
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # 例如 '640' 或 '[640,640]'
        try:
            imgsz = [int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"'imgsz={imgsz}' is not a valid image size. "
                f"Valid imgsz values are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
            ) from None
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # 应用 max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # 将图像尺寸调整为步长的倍数
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # 如果图像尺寸发生调整，则打印警告信息
    if sz != imgsz:
        LOGGER.warning(f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # 如有需要，补充缺失的维度
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


@functools.lru_cache
def check_uv():
    """检查 uv 软件包管理器是否已安装且可以正常运行。"""
    try:
        return subprocess.run(["uv", "-V"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """检查当前版本是否满足指定版本或版本范围。

    参数：
        current (str): 当前版本，或用于获取版本的包名称。
        required (str): 要求的版本或版本范围（采用 pip 格式）。
        name (str): 警告消息中使用的名称。
        hard (bool): 为 True 时，如果不满足要求则抛出 ModuleNotFoundError。
        verbose (bool): 为 True 时，如果不满足要求则打印警告信息。
        msg (str): verbose 为 True 时要显示的附加消息。

    返回：
        (bool): 满足版本要求时返回 True，否则返回 False。

    示例：
        检查当前版本是否恰好为 22.04：
        >>> check_version(current="22.04", required="==22.04")

        检查当前版本是否大于或等于 22.04：
        >>> check_version(current="22.10", required="22.04")  # 未指定运算符时默认使用 '>='

        检查当前版本是否小于或等于 22.04：
        >>> check_version(current="22.04", required="<=22.04")

        检查当前版本是否位于 20.04（包含）和 22.04（不包含）之间：
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # 如果 current 是 '' 或 None
        LOGGER.warning(f"invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current 是包名称而不是版本字符串，例如 current='ultralytics'
        try:
            name = current  # 将包名称赋给 name 参数
            current = metadata.version(current)  # 根据包名称获取版本字符串
        except metadata.PackageNotFoundError as e:
            if re.fullmatch(
                r"v\d+(\.\d+)*([-_.]?(a|b|c|rc|alpha|beta|pre|preview)[-_.]?\d*)?"
                r"([-_.]?(post|rev|r)[-_.]?\d*)?([-_.]?dev[-_.]?\d*)?(\+[\w.-]+)?",
                current,
                re.IGNORECASE,
            ):
                pass
            elif hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False

    if not required:  # 如果 required 是 '' 或 None
        return True

    if "sys_platform" in required and (  # 例如 required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # 将 '>=22.04' 拆分为 ('>=', '22.04')
        if not op:
            op = ">="  # 未传入运算符时默认使用 >=
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        n = max(len(c), len(v))  # 补齐到相同长度，使 '!=4.13.0.90' 这样的四段版本能够精确比较
        cn, vn = c + (0,) * (n - len(c)), v + (0,) * (n - len(v))
        if (
            (op == "==" and cn != vn)
            or (op == "!=" and cn == vn)
            or (op == ">=" and not (cn >= vn))
            or (op == "<=" and not (cn <= vn))
            or (op == ">" and not (cn > vn))
            or (op == "<" and not (cn < vn))
        ):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # 确认版本要求已满足
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """获取 PyPI 软件包的最新版本，但不下载或安装该软件包。

    参数：
        package_name (str): 要查询最新版本的软件包名称。

    返回：
        (str | None): 软件包的最新版本；如果无法获取则返回 None。
    """
    import requests  # 延迟导入，因为该模块加载较慢

    try:
        requests.packages.urllib3.disable_warnings()  # 禁用 InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


def check_pip_update_available():
    """检查 PyPI 上是否有更新版本的 ultralytics 软件包。

    返回：
        (bool): 有可用更新时返回 True，否则返回 False。
    """
    if ONLINE and IS_PIP_PACKAGE:
        try:
            from ultralytics import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}"):  # 检查当前版本是否低于最新版本
                LOGGER.info(
                    f"New https://pypi.org/project/ultralytics/{latest} available 😃 "
                    f"Update with 'pip install -U ultralytics'"
                )
                return True
        except Exception:
            pass
    return False


@ThreadingLocked()
@functools.lru_cache
def check_font(font="Arial.ttf"):
    """在本地查找字体；如果不存在，则下载到用户配置目录。

    参数：
        font (str): 字体路径或名称。

    返回：
        (Path | str): 解析后的字体文件路径。
    """
    from matplotlib import font_manager  # 限定导入范围，以加快 'import ultralytics'

    # 检查 USER_CONFIG_DIR
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # 检查系统字体
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # 如果不存在，则下载到 USER_CONFIG_DIR
    url = f"{ASSETS_URL}/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0", hard: bool = True, verbose: bool = False) -> bool:
    """检查当前 Python 版本是否满足最低版本要求。

    参数：
        minimum (str): Python 要求的最低版本。
        hard (bool): 为 True 时，如果不满足要求则抛出 ModuleNotFoundError。
        verbose (bool): 为 True 时，如果不满足要求则打印警告信息。

    返回：
        (bool): 已安装的 Python 版本是否满足最低要求。
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose)


@TryExcept()
def check_apt_requirements(requirements):
    """检查 apt 软件包是否已安装，并安装缺失的软件包。

    参数：
        requirements (列表[str]): 要检查并安装的 apt 软件包名称列表。
    """
    prefix = colorstr("red", "bold", "apt requirements:")
    # 检查缺失的软件包
    missing_packages = []
    for package in requirements:
        try:
            # 使用 dpkg -l 检查软件包是否已安装
            result = subprocess.run(["dpkg", "-l", package], capture_output=True, text=True, check=False)
            # 检查软件包是否已安装（查找 "ii" 状态）
            if result.returncode != 0 or not any(
                line.startswith("ii") and package in line for line in result.stdout.splitlines()
            ):
                missing_packages.append(package)
        except Exception:
            # 检查失败时，认为软件包未安装
            missing_packages.append(package)

    # 如果存在缺失的软件包，则进行安装
    if missing_packages:
        LOGGER.info(
            f"{prefix} Ultralytics requirement{'s' * (len(missing_packages) > 1)} {missing_packages} not found, attempting AutoUpdate..."
        )
        # 可选：先更新软件包列表
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "update"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # 构建并运行安装命令
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "install", "-y"] + missing_packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        LOGGER.info(f"{prefix} AutoUpdate success ✅")
        LOGGER.warning(f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n")


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds="", constrain=()):
    """检查已安装的依赖是否满足 Ultralytics YOLO 模型要求，并在必要时尝试自动更新。

    参数：
        requirements (Path | str | list[str|tuple] | tuple[str]): requirements.txt 文件路径、单个软件包依赖字符串、
            软件包依赖字符串列表，或包含可替代软件包字符串和元组的列表。
        exclude (tuple): 不参与检查的软件包名称元组。
        install (bool): 为 True 时，尝试自动更新不满足要求的软件包。
        cmds (str): 自动更新时传给 pip install 命令的附加参数。
        constrain (tuple | list): 始终追加到安装命令中的额外版本约束，即使依赖已经满足也会追加，
            以防止安装过程中解析器升级这些软件包。

    示例：
        >>> from ultralytics.utils.checks import check_requirements

        检查 requirements.txt 文件
        >>> check_requirements("path/to/requirements.txt")

        检查单个软件包
        >>> check_requirements("ultralytics>=8.3.200", cmds="--index-url https://download.pytorch.org/whl/cpu")

        检查多个软件包
        >>> check_requirements(["numpy", "ultralytics"])

        检查可替代的软件包
        >>> check_requirements([("onnxruntime", "onnxruntime-gpu"), "numpy"])
    """
    prefix = colorstr("red", "bold", "requirements:")

    if env_bool("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"):
        LOGGER.info(f"{prefix} ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS detected, skipping requirements check.")
        return True

    if isinstance(requirements, Path):  # requirements.txt 文件
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        candidates = r if isinstance(r, (list, tuple)) else [r]
        satisfied = False

        for candidate in candidates:
            r_stripped = candidate.rpartition("/")[-1].replace(".git", "")  # 将 git+https://org/repo.git 转换为 'repo'
            match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
            name, required = match[1], match[2].strip() if match[2] else ""
            try:
                if check_version(metadata.version(name), required):
                    satisfied = True
                    break
            except (AssertionError, metadata.PackageNotFoundError):
                continue

        if not satisfied:
            pkg = candidates[0]
            if "git+" in pkg:  # 删除 git URL 中的版本约束，以便传给 pip
                url, sep, marker = pkg.partition(";")
                pkg = re.sub(r"[<>!=~]+.*$", "", url) + sep + marker
            pkgs.append(pkg)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands, use_uv):
        """如果 uv 可用则使用 uv 安装软件包，否则回退到 pip。"""
        if use_uv:
            # 使用 --python 明确指定当前解释器（虚拟环境或系统解释器）
            # 即使未设置 VIRTUAL_ENV 环境变量，也能确保安装到正确的解释器环境
            return subprocess.check_output(
                [
                    "uv",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--python",
                    sys.executable,
                    *packages,
                    *shlex.split(commands),
                    "--index-strategy=unsafe-best-match",
                    "--break-system-packages",
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages, *shlex.split(commands)],
            stderr=subprocess.STDOUT,
            text=True,
        )

    if pkgs:
        packages = [*pkgs, *constrain]
        if install and AUTOINSTALL:  # 检查环境变量
            # 注意：uv 在 arm64 macOS 和 Raspberry Pi 运行环境中可能失败
            n = len(pkgs)  # 待更新的软件包数量
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                use_uv = not ARM64 and check_uv()  # uv 在 ARM64 环境中可能失败
                LOGGER.info(attempt_install(packages, cmds, use_uv=use_uv))
                dt = time.time() - t
                LOGGER.info(f"{prefix} AutoUpdate success ✅ {dt:.1f}s")
                LOGGER.warning(
                    f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                msg = f"{prefix} ❌ {e}"
                if hasattr(e, "output") and e.output:
                    msg += f"\n{e.output}"
                LOGGER.warning(msg)
                return False
        else:
            return False

    return True


def check_executorch_requirements():
    """检查并安装 ExecuTorch 依赖，包括平台专用依赖。"""
    # BUG：arm64 Docker 中构建 executorch 需要 packaging>=22.0 https://github.com/pypa/setuptools/issues/4483
    if LINUX and ARM64 and IS_DOCKER:
        check_requirements("packaging>=22.0")

    check_requirements("executorch", cmds=f"torch=={TORCH_VERSION.split('+')[0]}")


def check_tensorrt(min_version: str = "7.0.0"):
    """检查并安装 TensorRT 依赖，包括平台专用依赖。

    参数：
        min_version (str): 支持的 TensorRT 最低版本（默认值为 "7.0.0"）。
    """
    if LINUX:
        cuda_version = torch.version.cuda.split(".")[0]
        check_requirements(f"tensorrt-cu{cuda_version}>={min_version},!=10.2.0")


def check_torchvision():
    """检查已安装的 PyTorch 和 Torchvision 版本，确保二者兼容。

    此函数会检查已安装的 PyTorch 和 Torchvision 版本，并根据以下兼容性表在版本不兼容时发出警告：
    https://github.com/pytorch/vision#installation
    """
    compatibility_table = {
        "2.10": ["0.25"],
        "2.9": ["0.24"],
        "2.8": ["0.23"],
        "2.7": ["0.22"],
        "2.6": ["0.21"],
        "2.5": ["0.20"],
        "2.4": ["0.19"],
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }

    # 检查主版本号和次版本号
    v_torch = ".".join(TORCH_VERSION.split("+", 1)[0].split(".")[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+", 1)[0].split(".")[:2])
        if all(v_torchvision != v for v in compatible_versions):
            LOGGER.warning(
                f"torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n"
                f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                "'pip install -U torch torchvision' to update both.\n"
                "For a full compatibility table see https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolo26n.pt", suffix=".pt", msg=""):
    """检查文件是否具有可接受的后缀。

    参数：
        file (str | 列表[str]): 待检查的文件或文件列表。
        suffix (str | tuple): 可接受的后缀或后缀元组。
        msg (str): 出错时显示的附加消息。
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = {suffix}
        for f in file if isinstance(file, (list, tuple)) else [file]:
            if s := clean_url(f).rpartition(".")[-1].lower().strip():  # 文件后缀
                assert f".{s}" in suffix, f"{msg}{f} acceptable suffix is {suffix}, not .{s}"


def check_yolov5u_filename(file: str, verbose: bool = True) -> str:
    """将旧版 YOLOv5 文件名替换为更新后的 YOLOv5u 文件名。

    参数：
        file (str): 待检查并可能更新的文件名。
        verbose (bool): 是否打印替换信息。

    返回：
        (str): 更新后的文件名。
    """
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5."
                )
    return file


def check_model_file_from_stem(model: str = "yolo11n") -> str | Path:
    """根据有效的模型主干名称返回模型文件名。

    参数：
        model (str): 待检查的模型主干名称。

    返回：
        (str | Path): 带有适当后缀的模型文件名。
    """
    path = Path(model)
    if not path.suffix and path.stem in downloads.GITHUB_ASSETS_STEMS:
        return path.with_suffix(".pt")  # 添加后缀，例如 yolo26n -> yolo26n.pt
    return model


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """搜索或下载文件（如果需要），检查后缀（如果提供），并返回文件路径。

    参数：
        file (str): 文件名称或路径、URL、Platform URI（ul://）或 GCS 路径（gs://）。
        suffix (str | tuple): 用于验证文件的可接受后缀或后缀元组。
        download (bool): 文件在本地不存在时是否下载。
        download_dir (str): 文件下载目录。
        hard (bool): 文件未找到时是否抛出异常。

    返回：
        (str | 列表): 文件路径；如果未找到则返回空列表。
    """
    file = normalize_platform_uri(file)  # 接受 Platform 网页 URL（重写为 ul://）
    check_suffix(file, suffix)  # 可选
    file = str(file).strip()  # 转换为字符串并删除空白
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # Windows Python<3.10 需要检查 '://'
        or file.lower().startswith("grpc://")
    ):  # 文件存在，或这是 gRPC Triton 图像源
        return file
    elif download and file.lower().startswith("ul://"):  # Ultralytics Platform URI
        url = resolve_platform_uri(file, hard=hard)  # 转换为签名 HTTPS URL
        if url is None:
            return []  # 未找到时软失败（与文件搜索行为一致）
        # 使用 URI 路径构建唯一目录结构：ul://user/project/model -> user/project/model/filename.pt
        uri_path = Path(file[5:])  # 移除 "ul://" 前缀
        if uri_path.is_absolute() or ".." in uri_path.parts:
            raise ValueError(f"Unsafe Ultralytics Platform URI path: {file}")
        local_file = Path(download_dir) / uri_path / url2file(url)
        # 始终重新下载 NDJSON 数据集（成本较低，可确保更新后使用最新数据）
        if local_file.suffix == ".ndjson":
            local_file.unlink(missing_ok=True)
        if local_file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {local_file}")
        else:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            downloads.safe_download(url=url, file=local_file, unzip=False)
        return str(local_file)
    elif download and file.lower().startswith(REMOTE_FILE_PREFIXES):  # 下载远程文件
        if file.startswith("gs://"):
            file = "https://storage.googleapis.com/" + file[5:]  # 将 gs:// 转换为公开 HTTPS URL
        url = file  # 注意：Pathlib 会将 :// 转换为 :/
        file = Path(download_dir) / url2file(file)  # 将 '%2F' 转换为 '/'，并拆分身份验证查询字符串
        if file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # 文件已存在
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:  # 搜索
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # 查找文件
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # 返回 文件


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """搜索或下载 YAML 文件（如果需要），检查后缀并返回路径。

    参数：
        file (str | Path): 文件名称或路径。
        suffix (tuple): 可接受的 YAML 文件后缀元组。
        hard (bool): 文件未找到或找到多个文件时是否抛出异常。

    返回：
        (str): YAML 文件路径。
    """
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir: Path | str, path: Path | str) -> bool:
    """检查解析后的路径是否位于指定目录下，以防止路径穿越。

    参数：
        basedir (Path | str): 目标目录。
        path (Path | str): 待检查的路径。

    返回：
        (bool): 路径安全时返回 True，否则返回 False。
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


@functools.lru_cache
def check_imshow(warn=False):
    """检查当前环境是否支持图像显示。

    参数：
        warn (bool): 如果环境不支持图像显示，是否发出警告。

    返回：
        (bool): 环境支持图像显示时返回 True，否则返回 False。
    """
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))  # 显示一个 8 像素大小的小图像
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_yolo(verbose=True, device=""):
    """打印易于阅读的 YOLO 软件和硬件摘要。

    参数：
        verbose (bool): 是否打印详细信息。
        device (str | torch.device): YOLO 使用的设备。
    """
    import psutil  # 延迟导入，因为该模块加载较慢

    from ultralytics.utils.torch_utils import select_device

    if IS_COLAB:
        shutil.rmtree("sample_data", ignore_errors=True)  # 删除 Colab 的 /sample_data 目录

    if verbose:
        # 系统信息
        gib = 1 << 30  # 每 GiB 包含的字节数
        ram = psutil.virtual_memory().total
        total, _used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        try:
            from IPython import display

            display.clear_output()  # 如果在 Notebook 中运行，则清除显示内容
        except ImportError:
            pass
    else:
        s = ""

    if GIT.is_repo:
        check_multiple_install()  # 如果使用本地克隆仓库，则检查是否存在冲突安装

    select_device(device=device, newline=False)
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():
    """收集并打印相关系统信息，包括操作系统、Python、内存、CPU 和 CUDA。

    返回：
        (dict): 包含系统信息的字典。
    """
    import psutil  # 延迟导入，因为该模块加载较慢

    from ultralytics.utils import ENVIRONMENT  # 限定导入范围，以避免循环导入
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    gib = 1 << 30  # 每 GiB 包含的字节数
    cuda = torch.cuda.is_available()
    check_yolo()
    total, _, free = shutil.disk_usage("/")

    info_dict = {
        "OS": platform.platform(),
        "Environment": ENVIRONMENT,
        "Python": PYTHON_VERSION,
        "Install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
        "Path": str(ROOT),
        "RAM": f"{psutil.virtual_memory().total / gib:.2f} GB",
        "Disk": f"{(total - free) / gib:.1f}/{total / gib:.1f} GB",
        "CPU": get_cpu_info(),
        "CPU count": os.cpu_count(),
        "GPU": get_gpu_info(index=0) if cuda else None,
        "GPU count": torch.cuda.device_count() if cuda else None,
        "CUDA": torch.version.cuda if cuda else None,
    }
    LOGGER.info("\n" + "\n".join(f"{k:<23}{v}" for k, v in info_dict.items()) + "\n")

    package_info = {}
    for r in parse_requirements(package=get_distribution_name("ultralytics")):
        try:
            current = metadata.version(r.name)
            is_met = "✅ " if check_version(current, str(r.specifier), name=r.name) else "❌ "
        except metadata.PackageNotFoundError:
            current = "(not installed)"
            is_met = "❌ "
        package_info[r.name] = f"{is_met}{current}{r.specifier}"
        LOGGER.info(f"{r.name:<23}{package_info[r.name]}")

    info_dict["Package Info"] = package_info

    if is_github_action_running():
        github_info = {
            "RUNNER_OS": os.getenv("RUNNER_OS"),
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW"),
            "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "GITHUB_REPOSITORY_OWNER": os.getenv("GITHUB_REPOSITORY_OWNER"),
        }
        LOGGER.info("\n" + "\n".join(f"{k}: {v}" for k, v in github_info.items()))
        info_dict["GitHub Info"] = github_info

    return info_dict


def check_amp(model):
    """检查 YOLO 模型的 PyTorch 自动混合精度（AMP）功能。

    如果检查失败，说明当前系统中的 AMP 存在异常，可能导致 NaN 损失或 mAP 为零的结果，
    因此训练期间会禁用 AMP。

    参数：
        model (torch.nn.Module): YOLO 模型实例。

    返回：
        (bool): YOLO 模型的 AMP 功能正常时返回 True，否则返回 False。

    示例：
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils.checks import check_amp
        >>> model = YOLO("yolo26n.pt").model.cuda()
        >>> check_amp(model)
    """
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device  # 获取模型所在设备
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False  # AMP 仅用于加速器设备
    elif device.type == "cuda":
        # 存在 AMP 问题的 GPU
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False

    def amp_allclose(m, im):
        """比较 FP32 与 AMP 结果是否接近。"""
        batch = [im] * 8
        imgsz = max(256, int(model.stride.max() * 4))  # 最大步长为 P5-32 和 P6-64
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # FP32 推理
        with autocast(enabled=True, device=device.type):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # AMP 推理
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # 绝对容差为 0.5

    im = ASSETS / "bus.jpg"  # 待检查的图像
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO
        from ultralytics.utils import WEIGHTS_DIR

        amp_weights = WEIGHTS_DIR / "yolo26n.pt"
        if not amp_weights.is_file():
            LOGGER.info(f"{prefix}downloading yolo26n.pt for AMP checks (one-time, not used for training)...")
        assert amp_allclose(YOLO(amp_weights), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped. Offline and unable to download YOLO26n for AMP checks. {warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped. "
            f"Unable to load YOLO26n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.error(
            f"{prefix}checks failed. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True


def check_multiple_install():
    """检查是否存在多个 Ultralytics 安装。"""
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "ultralytics"], capture_output=True, text=True, check=False
        )
        install_msg = (
            f"Install your local copy in editable mode with 'pip install -e {ROOT.parent}' to avoid "
            "issues. See https://docs.ultralytics.com/quickstart"
        )
        if result.returncode != 0:
            if "not found" in result.stderr.lower():  # 软件包未通过 pip 安装，但当前从本地导入
                LOGGER.warning(f"Ultralytics not found via pip but importing from: {ROOT}. {install_msg}")
            return
        yolo_path = (
            Path(re.findall(r"location:\s+(.+)", result.stdout, flags=re.IGNORECASE)[-1]) / "ultralytics"
        ).resolve()
        if not yolo_path.samefile(ROOT.resolve()):
            LOGGER.warning(
                f"Multiple Ultralytics installations detected. The `yolo` command uses: {yolo_path}, "
                f"but current session imports from: {ROOT}. This may cause version conflicts. {install_msg}"
            )
    except Exception:
        return


def print_args(args: dict | None = None, show_file=True, show_func=False):
    """打印函数参数（可选的 args 字典）。

    参数：
        args (dict, 可选): 要打印的参数。
        show_file (bool): 是否显示文件名称。
        show_func (bool): 是否显示函数名称。
    """

    def strip_auth(v):
        """删除较长 URL 中可能包含的身份验证信息。"""
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # 上一层栈帧
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # 自动获取参数
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in sorted(args.items())))


def cuda_device_count() -> int:
    """获取当前环境中可用的 NVIDIA GPU 数量。

    返回：
        (int): 可用 NVIDIA GPU 的数量。
    """
    if IS_JETSON:
        # NVIDIA Jetson 不完全支持 nvidia-smi，因此改用 PyTorch
        return torch.cuda.device_count()
    else:
        try:
            # 运行 nvidia-smi 命令并获取输出
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
            )

            # 取第一行并删除首尾空白字符
            first_line = output.strip().split("\n", 1)[0]

            return int(first_line)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            # 如果命令失败、找不到 nvidia-smi 或输出不是整数，则认为没有可用 GPU
            return 0


def cuda_is_available() -> bool:
    """检查当前环境是否支持 CUDA。

    返回：
        (bool): 存在一个或多个 NVIDIA GPU 时返回 True，否则返回 False。
    """
    return cuda_device_count() > 0


def is_rockchip():
    """检查当前环境是否运行在 Rockchip SoC 上。

    返回：
        (bool): 运行在 Rockchip SoC 上时返回 True，否则返回 False。
    """
    if LINUX and ARM64:
        try:
            with open("/proc/device-tree/compatible") as f:
                dev_str = f.read()
                *_, soc = dev_str.split(",")
                if soc.replace("\x00", "").split("-", 1)[0] in RKNN_CHIPS:
                    return True
        except OSError:
            return False
    else:
        return False


def is_intel():
    """检查系统是否包含 Intel 硬件（CPU 或 GPU）。

    返回：
        (bool): 检测到 Intel 硬件时返回 True，否则返回 False。
    """
    from ultralytics.utils.torch_utils import get_cpu_info

    # 检查 CPU
    if "intel" in get_cpu_info().lower():
        return True

    # 通过 xpu-smi 检查 GPU
    try:
        result = subprocess.run(["xpu-smi", "discovery"], capture_output=True, text=True, timeout=5, check=False)
        return "intel" in result.stdout.lower()
    except Exception:  # 使用宽泛异常，以捕获所有 Intel GPU 异常类型
        return False


def is_sudo_available() -> bool:
    """检查当前环境是否可以使用 sudo 命令。

    返回：
        (bool): sudo 命令可用时返回 True，否则返回 False。
    """
    if WINDOWS:
        return False
    cmd = "sudo --version"
    return (
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
        == 0
    )


# 执行检查并定义常量
check_python("3.8", hard=False, verbose=True)  # 检查 Python 版本
check_torchvision()  # 检查 torch 与 torchvision 的兼容性

# 定义常量
IS_PYTHON_3_8 = PYTHON_VERSION.startswith("3.8")
IS_PYTHON_3_9 = PYTHON_VERSION.startswith("3.9")
IS_PYTHON_3_10 = PYTHON_VERSION.startswith("3.10")
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
IS_PYTHON_3_13 = PYTHON_VERSION.startswith("3.13")

IS_PYTHON_MINIMUM_3_9 = check_python("3.9", hard=False)
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_MINIMUM_3_12 = check_python("3.12", hard=False)
IS_PYTHON_MINIMUM_3_13 = check_python("3.13", hard=False)
