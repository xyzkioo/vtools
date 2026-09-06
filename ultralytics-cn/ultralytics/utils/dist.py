# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


def find_free_network_port() -> int:
    """查找本地主机上的空闲端口。

    在单节点训练中，如果不需要连接真实主节点但必须设置 `MASTER_PORT` 环境变量，该函数很有用。

    返回：
        (int): 可用的网络端口号。

    注意：
        候选端口取自操作系统默认临时端口范围下方（Linux 为 32768，macOS 和 Windows 为 49152），因为端口会在
        此处释放，并在稍后由 DDP 子进程重新绑定。在此期间，临时端口可能被任意出站连接占用，从而在启动时造成
        EADDRINUSE 会合失败。
    """
    import random
    import socket

    # init_seeds() 会提前为全局随机数生成器设定种子，因此使用 SystemRandom，避免同一主机上的并发 DDP 启动
    # 得到相同的候选端口列表
    for port in random.SystemRandom().sample(range(10000, 32768), 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue  # 已被显式监听器占用，尝试下一个候选端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))  # 没有可用的非临时端口时，退回使用临时端口
        return s.getsockname()[1]


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """为多 GPU 训练生成 DDP（分布式数据并行）文件。

    此函数创建临时 Python 文件，以便在多个 GPU 之间执行分布式训练。该文件包含在分布式环境中初始化训练器所需的配置。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含训练配置和参数的训练器，必须是具有 `args` 属性的类实例。

    返回：
        (str): 生成的临时 DDP 文件路径。

    注意：
        生成的文件会保存到 USER_CONFIG_DIR/DDP 目录，并包含：
        - 训练器类导入语句
        - 训练器参数中的配置覆盖项
        - 训练初始化代码
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    content = f"""
# Ultralytics 多 GPU 训练临时文件（使用后应自动删除）
from pathlib import Path, PosixPath  # 用于处理以 Path 而不是字符串保存的模型参数
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # 处理额外的 'save_dir' 键
    trainer = {name}(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """生成分布式训练命令。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含分布式训练配置的训练器。

    返回：
        cmd (列表[str]): 用于执行分布式训练的命令。
        file (str): 为 DDP 训练创建的临时文件路径。
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    """删除分布式数据并行（DDP）训练期间创建的临时文件。

    此函数检查给定文件名是否包含训练器 ID，以判断它是否是 DDP 训练创建的临时文件；如果是，则将其删除。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 用于分布式训练的训练器。
        file (str): 可能需要删除的文件路径。

    示例：
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # 如果文件名包含临时文件后缀
        os.remove(file)
