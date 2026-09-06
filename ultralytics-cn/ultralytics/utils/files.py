# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    """用于临时切换工作目录的上下文管理器和装饰器。

    此类允许通过上下文管理器或装饰器临时切换工作目录，并确保上下文或被装饰函数执行完毕后恢复原工作目录。

    属性：
        dir (Path | str): 要切换到的新目录。
        cwd (Path): 切换前的原始工作目录。

    方法：
        __enter__: 将当前目录切换到指定目录。
        __exit__: 退出上下文时恢复原工作目录。

    示例：
        作为上下文管理器使用：
        >>> with WorkingDirectory("/path/to/new/dir"):
        ...     # Perform operations in the new directory
        ...     pass

        作为装饰器使用：
        >>> @WorkingDirectory("/path/to/new/dir")
        ... def some_function():
        ...     # Perform operations in the new directory
        ...     pass
    """

    def __init__(self, new_dir: str | Path):
        """使用目标目录初始化 WorkingDirectory 上下文管理器。"""
        self.dir = new_dir  # 新目录
        self.cwd = Path.cwd().resolve()  # 当前目录

    def __enter__(self):
        """进入上下文时将当前工作目录切换到指定目录。"""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时恢复原始工作目录。"""
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path: str | Path):
    """处理路径名称中包含空格情况的上下文管理器。

    如果路径包含空格，则将空格替换为下划线，把文件或目录复制到新路径，执行上下文代码块，
    最后再将文件或目录复制回原位置。

    参数：
        path (str | Path): 可能包含空格的原始路径。

    Yields:
        (Path | str): 将空格替换为下划线后的临时路径。

    示例：
        >>> with spaces_in_path("/path/with spaces") as new_path:
        ...     # Your code here
        ...     pass
    """
    # 如果路径包含空格，则将空格替换为下划线
    if " " in str(path):
        string = isinstance(path, str)  # 输入类型
        path = Path(path)

        # 创建临时目录并构造新路径
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")

            # 复制文件或目录
            if path.is_dir():
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)

            try:
                # 返回临时路径
                yield str(tmp_path) if string else tmp_path

            finally:
                # 将文件或目录复制回原位置
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)  # 复制文件

    else:
        # 如果路径不包含空格，则直接返回原路径
        yield path


def increment_path(path: str | Path, exist_ok: bool = False, sep: str = "-", mkdir: bool = False) -> Path:
    """递增文件或目录路径，例如 runs/exp --> runs/exp{sep}2、runs/exp{sep}3 等。

    如果路径存在且 `exist_ok` 不为 True，则在路径末尾追加分隔符 `sep` 和编号来递增路径。
    如果路径是文件，则保留文件扩展名；如果路径是目录，则直接在目录名末尾追加编号。

    参数：
        path (str | Path): 要递增的路径。
        exist_ok (bool, 可选): 为 True 时不递增路径，直接按原样返回。
        sep (str, 可选): 路径与递增编号之间使用的分隔符。
        mkdir (bool, 可选): 目录不存在时是否创建目录。

    返回：
        (Path): 递增后的路径。

    示例：
        递增目录路径：
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     exp = Path(tmp) / "exp"
        ...     exp.mkdir()
        ...     increment_path(exp).name
        'exp-2'

        递增文件路径：
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     results = Path(tmp) / "results.txt"
        ...     results.touch()
        ...     increment_path(results).name
        'results-2.txt'
    """
    path = Path(path)  # 与操作系统无关
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # 方法一
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # 递增路径
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 创建目录

    return path


def file_age(path: str | Path = __file__) -> int:
    """返回指定文件自上次修改以来经过的天数。"""
    dt = datetime.now().astimezone() - datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone()  # 时间差
    return dt.days  # + dt.seconds / 86400  # 小数天数


def file_date(path: str | Path = __file__) -> str:
    """以 `YYYY-M-D` 格式返回文件的修改日期。"""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone()
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path: str | Path) -> float:
    """以 mebibyte（MiB）为单位返回文件或目录的大小。"""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # 字节转换为 MiB（1024 ** 2）
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir: str = ".") -> str:
    """返回指定目录中最近的 `last.pt` 文件路径，用于恢复训练。"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(model_names: tuple = ("yolo26n.pt",), source_dir: Path = Path("."), update_names: bool = False):
    """更新指定的 YOLO 模型，并将其重新保存到 `updated_models` 子目录。

    参数：
        model_names (tuple, 可选): 要更新的模型文件名。
        source_dir (Path, 可选): 包含模型和目标子目录的目录。
        update_names (bool, 可选): 根据数据集 YAML 更新模型类别名称。

    示例：
        更新指定的 YOLO 模型，并将其保存到 `updated_models` 子目录：
        >>> from ultralytics.utils.files import update_models
        >>> model_names = ("yolo26n.pt", "yolo11s.pt")
        >>> update_models(model_names, source_dir=Path("/models"), update_names=True)
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names
    from ultralytics.utils import LOGGER

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在

    for model_name in model_names:
        model_path = source_dir / model_name
        LOGGER.info(f"Loading model from {model_path}")

        # 加载模型
        model = YOLO(model_path)
        model.half()
        if update_names:  # 根据数据集 YAML 更新模型类别名称
            model.model.names = default_class_names("coco8.yaml")

        # 定义新的保存路径
        save_path = target_dir / model_name

        # 使用 model.save() 保存模型
        LOGGER.info(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path)
