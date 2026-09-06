# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements

# Axelera 导出会修改进程级全局状态（包括下方的 PROTOCOL_BUFFERS 环境变量和编译器生成的任意工作目录文件），
# 因此使用模块级锁串行执行同一进程内的并发导出。跨进程的平台工作进程各自持有独立的锁，互不竞争。
_AXELERA_EXPORT_LOCK = threading.Lock()


def torch2axelera(
    model: torch.nn.Module,
    output_dir: Path | str,
    calibration_dataset: torch.utils.data.DataLoader,
    transform_fn: Callable[[Any], np.ndarray],
    model_name: str = "model",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """将 YOLO 模型转换为 Axelera 格式。

    参数：
        model (torch.nn.Module): 用于量化的源 YOLO 模型。
        output_dir (Path | str): 保存导出 Axelera 模型的目录。
        calibration_dataset (torch.utils.数据.DataLoader): 用于量化的校准数据加载器。
        transform_fn (Callable[[Any], np.ndarray]): 校准预处理变换函数。
        model_name (str, 可选): 编译模型的名称。默认为 "model"。
        metadata (dict | None, 可选): 保存为 YAML 的可选元数据。默认为 None。
        prefix (str, 可选): 日志消息前缀。默认为 ""。

    返回：
        (str): 导出的 Axelera 模型目录路径。
    """
    # 在进程内串行执行：下面的步骤会修改进程级全局状态（protobuf 环境变量和编译器写入的任意工作目录文件），
    # 因此同一进程内的并发导出不能重叠。
    with _AXELERA_EXPORT_LOCK:
        prev_protobuf = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        try:
            try:
                from axelera import compiler
            except ImportError:
                check_requirements(
                    ["axelera-devkit==1.7.0", "numpy<=2.3.5"],
                    cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
                )
                from axelera import compiler

            check_requirements("omnimalloc==0.5.0")
            from axelera.compiler import CompilerConfig
            from axelera.compiler.config.model_specific import extract_ultralytics_metadata

            LOGGER.info(f"\n{prefix} starting export with Axelera compiler...")

            # 解析为绝对路径，确保下面的相对编译目录不会与其产生别名。
            output_dir = Path(output_dir).resolve()
            if output_dir.exists():
                shutil.rmtree(output_dir)

            axelera_model_metadata = extract_ultralytics_metadata(model)
            config = CompilerConfig(
                model_metadata=axelera_model_metadata,
                model_name=model_name,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
            )
            qmodel = compiler.quantize(
                model=model,
                calibration_dataset=calibration_dataset,
                config=config,
                transform_fn=transform_fn,
            )

            # Axelera 编译器在绝对输出路径下会生成无效产物，因此在本地相对目录中编译。
            # TemporaryDirectory 会在当前工作目录中生成唯一名称（连续导出同名模型时也不会冲突），
            # 并在退出时删除该目录，即使编译抛出异常也一样；传入相对目录名可避免它与绝对 output_dir
            # 产生别名，从而确保清理操作不会删除最终结果。
            with tempfile.TemporaryDirectory(prefix="axelera_compile_", dir=".") as compile_root:
                compile_dir = Path(Path(compile_root).name)
                compiler.compile(model=qmodel, config=config, output_dir=compile_dir)

                output_dir.mkdir(parents=True, exist_ok=True)
                for artifact in [f"{model_name}.axm", "compiler_config_final.toml"]:
                    for artifact_path in [compile_dir / artifact, Path(artifact)]:
                        if artifact_path.exists():
                            artifact_path.replace(output_dir / artifact_path.name)
                            break

                # 删除编译器生成的中间产物，仅保留已编译模型和配置文件。
                keep_suffixes = {".axm"}
                keep_names = {"compiler_config_final.toml", "metadata.yaml"}
                for f in output_dir.iterdir():
                    if f.is_file() and f.suffix not in keep_suffixes and f.name not in keep_names:
                        f.unlink()

                if metadata is not None:
                    YAML.save(output_dir / "metadata.yaml", metadata)

            return str(output_dir)
        finally:
            # 恢复原始的 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION 值。
            if prev_protobuf is None:
                os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
            else:
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = prev_protobuf
