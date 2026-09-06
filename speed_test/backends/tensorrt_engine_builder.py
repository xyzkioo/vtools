#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ONNX 构建为 TensorRT engine。

默认使用已安装的 TensorRT Python API，因此不要求额外安装 ``trtexec``。
``trtexec`` 保留为可选路径，便于需要复用 NVIDIA 命令行工具的环境。

TensorRT 11 使用强类型网络，不能再通过 ``--fp16``/``--bf16`` 命令行开关
强行改变 FP32 ONNX 的精度。FP16/BF16 应由 adapter 导出对应 dtype 的 ONNX；
本模块只负责解析、构建和安全保存 engine。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _major_version(version: Any) -> int:
    match = re.search(r"(\d+)", str(version))
    return int(match.group(1)) if match else 0


def _atomic_write(path: Path, payload: bytes) -> None:
    """先写同目录临时文件，构建成功后再替换目标 engine。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser_errors(parser: Any) -> str:
    count = int(getattr(parser, "num_errors", 0))
    messages: list[str] = []
    for index in range(count):
        try:
            messages.append(str(parser.get_error(index)))
        except Exception as error:  # noqa: BLE001 - 保留原始解析失败信息
            messages.append(f"<无法读取 TensorRT parser 错误 {index}: {error}>")
    return "\n".join(messages)


def _set_legacy_precision_flag(trt: Any, builder_config: Any, precision: str) -> None:
    """仅为 TensorRT 10 及更早版本设置旧式 precision flag。"""
    flag_name = {"fp16": "FP16", "bf16": "BF16"}.get(precision)
    if flag_name is None:
        return
    builder_flag = getattr(getattr(trt, "BuilderFlag", None), flag_name, None)
    if builder_flag is None:
        print(f"[TensorRT] 当前版本没有 BuilderFlag.{flag_name}，按 ONNX dtype 构建")
        return
    builder_config.set_flag(builder_flag)


def build_engine_with_python(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mb: int,
) -> Path:
    """使用 TensorRT Python API 解析 ONNX 并保存序列化 engine。"""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 文件不存在：{onnx_path}")
    if workspace_mb <= 0:
        raise ValueError("workspace_mb 必须大于 0")

    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError(
            "未找到 TensorRT Python bindings；请在当前环境执行 pip install tensorrt"
        ) from error

    version = str(getattr(trt, "__version__", "unknown"))
    major = _major_version(version)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network_flags = 0
    creation_flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    strongly_typed = getattr(creation_flags, "STRONGLY_TYPED", None)
    if major >= 11 and strongly_typed is not None:
        network_flags |= 1 << int(strongly_typed)
    network = builder.create_network(network_flags)
    if network is None:
        raise RuntimeError("TensorRT 无法创建 network")

    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(onnx_path))
    if not parsed:
        details = _parser_errors(parser)
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(f"TensorRT 无法解析 ONNX：{onnx_path}{suffix}")

    builder_config = builder.create_builder_config()
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    if hasattr(builder_config, "set_memory_pool_limit"):
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # TensorRT 8.x 兼容
        builder_config.max_workspace_size = workspace_bytes

    if major >= 11:
        if precision in {"fp16", "bf16"}:
            print(
                f"[TensorRT] TensorRT {version} 使用强类型网络；"
                f"{precision} 精度由 ONNX dtype 决定，不设置旧式 BuilderFlag"
            )
    else:
        _set_legacy_precision_flag(trt, builder_config, precision)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, builder_config)
    else:  # TensorRT 8.x 兼容
        engine = builder.build_engine(network, builder_config)
        serialized = None if engine is None else engine.serialize()
    if serialized is None:
        raise RuntimeError(
            f"TensorRT engine 构建失败：{onnx_path}。"
            "请检查 ONNX 算子、输入 dtype/shape 和显存空间。"
        )

    payload = bytes(serialized)
    if not payload:
        raise RuntimeError("TensorRT 返回了空 engine")
    _atomic_write(engine_path, payload)
    print(
        f"[TensorRT] Python API 构建完成：{engine_path} "
        f"({len(payload) / 1024 / 1024:.2f} MB, TensorRT {version})",
        flush=True,
    )
    return engine_path


def _trtexec_help(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{completed.stdout}\n{completed.stderr}"


def build_engine_with_trtexec(
    onnx_path: Path,
    engine_path: Path,
    trtexec: str,
    precision: str,
    workspace_mb: int,
) -> Path:
    """使用外部 trtexec 构建 engine；自动兼容 TensorRT 11 的精度参数变化。"""
    executable = shutil.which(str(trtexec)) or (
        str(trtexec) if Path(str(trtexec)).is_file() else None
    )
    if executable is None:
        raise FileNotFoundError(
            f"找不到 trtexec：{trtexec}。请改用 builder: python，"
            "或通过 --trtexec 指定完整路径。"
        )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 文件不存在：{onnx_path}")
    if workspace_mb <= 0:
        raise ValueError("workspace_mb 必须大于 0")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--skipInference",
    ]
    help_text = _trtexec_help(executable)
    precision_flag = {"fp16": "--fp16", "bf16": "--bf16"}.get(precision)
    if precision_flag and precision_flag in help_text:
        command.append(precision_flag)
    elif precision_flag:
        print(
            f"[TensorRT] trtexec 不支持 {precision_flag}；"
            "按 TensorRT 11 强类型规则使用 ONNX dtype",
            flush=True,
        )
    print("[TensorRT] 构建 engine：" + " ".join(str(item) for item in command), flush=True)
    completed = subprocess.run(command, check=False, text=True)

    # TensorRT 8.x 使用旧的 --workspace 参数；对旧版 trtexec 自动回退。
    if completed.returncode != 0 and not engine_path.is_file():
        legacy_command = [
            (f"--workspace={workspace_mb}" if item.startswith("--memPoolSize=workspace:") else item)
            for item in command
        ]
        print("[TensorRT] 回退旧版 workspace 参数", flush=True)
        completed = subprocess.run(legacy_command, check=False, text=True)
    if completed.returncode != 0 or not engine_path.is_file():
        raise RuntimeError(f"trtexec 构建失败，退出码={completed.returncode}：{engine_path}")
    return engine_path


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    *,
    builder: str = "python",
    trtexec: str = "trtexec",
    precision: str = "fp32",
    workspace_mb: int = 2048,
) -> Path:
    """按配置选择 Python API、trtexec 或自动回退。"""
    mode = str(builder).strip().lower()
    if mode not in {"python", "trtexec", "auto"}:
        raise ValueError("tensorrt.builder 只支持 python、trtexec 或 auto")
    if mode == "python":
        return build_engine_with_python(onnx_path, engine_path, precision, workspace_mb)
    if mode == "trtexec":
        return build_engine_with_trtexec(onnx_path, engine_path, trtexec, precision, workspace_mb)

    executable = shutil.which(str(trtexec)) or (
        str(trtexec) if Path(str(trtexec)).is_file() else None
    )
    if executable is not None:
        return build_engine_with_trtexec(onnx_path, engine_path, executable, precision, workspace_mb)
    print("[TensorRT] 未找到 trtexec，builder: auto 自动改用 Python API", flush=True)
    return build_engine_with_python(onnx_path, engine_path, precision, workspace_mb)
