# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import functools
import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ultralytics import __version__
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    colorstr,
)
from ultralytics.utils.checks import check_version
from ultralytics.utils.cpu import CPUInfo
from ultralytics.utils.patches import torch_load

# 版本检查（所有标志默认表示版本 >= min_version）
TORCH_1_9 = check_version(TORCH_VERSION, "1.9.0")
TORCH_1_10 = check_version(TORCH_VERSION, "1.10.0")
TORCH_1_11 = check_version(TORCH_VERSION, "1.11.0")
TORCH_1_13 = check_version(TORCH_VERSION, "1.13.0")
TORCH_2_0 = check_version(TORCH_VERSION, "2.0.0")
TORCH_2_1 = check_version(TORCH_VERSION, "2.1.0")
TORCH_2_3 = check_version(TORCH_VERSION, "2.3.0")
TORCH_2_4 = check_version(TORCH_VERSION, "2.4.0")
TORCH_2_5 = check_version(TORCH_VERSION, "2.5.0")
TORCH_2_7 = check_version(TORCH_VERSION, "2.7.0")
TORCH_2_8 = check_version(TORCH_VERSION, "2.8.0")
TORCH_2_9 = check_version(TORCH_VERSION, "2.9.0")
TORCH_2_10 = check_version(TORCH_VERSION, "2.10.0")
TORCH_2_12 = check_version(TORCH_VERSION, "2.12.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(TORCH_VERSION, "==2.4.0"):  # reject version 2.4.0 on Windows
    LOGGER.warning(
        "Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve "
        "https://github.com/ultralytics/ultralytics/issues/15049"
    )


def get_torch_device_backend(device: torch.device | str):
    """返回负责所选设备后端的 PyTorch 模块。"""
    device_type = getattr(device, "type", str(device).split(":")[0])
    return torch.get_device_module(device_type) if hasattr(torch, "get_device_module") else getattr(torch, device_type)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """确保分布式训练中的所有进程等待本地主进程（rank 0）先完成任务。"""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[torch.cuda.current_device()]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()]) if use_ids else dist.barrier()


def smart_inference_mode(mode=True):
    """启用或禁用 torch 推理模式，同时兼容最低支持的 torch 版本。"""

    def decorate(fn):
        """根据 torch 版本应用适用的推理模式装饰器。"""
        if not mode:
            return torch.inference_mode(False)(torch.no_grad()(fn)) if TORCH_1_9 else torch.no_grad()(fn)
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # 已处于 inference_mode，直接透传
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """根据 PyTorch 版本和 AMP 设置获取适用的 autocast 上下文管理器。

    此函数返回一个适用于自动混合精度（AMP）训练的上下文管理器，兼容新旧 PyTorch 版本，
    并处理不同 PyTorch 版本之间 autocast API 的差异。

    参数：
        enabled (bool): 是否启用自动混合精度。
        device (str, 可选): autocast 使用的设备类型，例如 "cuda" 或 "npu"。

    返回：
        (torch.amp.autocast): 适用的 autocast 上下文管理器。

    示例：
        >>> with autocast(enabled=True):
        ...     # 在此处执行混合精度操作
        ...     pass

    注意：
        - 对于 PyTorch 1.13 及更高版本，使用 `torch.amp.autocast`。
        - 对于更早版本，使用特定后端的 AMP 上下文。
    """
    if device == "npu":
        import torch_npu

        return torch_npu.npu.amp.autocast(enabled=enabled)
    if TORCH_1_13:
        if device == "mps" and not TORCH_2_5:  # MPS autocast added in torch 2.5.0, errors on older versions
            device, enabled = "cpu", False
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


@functools.lru_cache
def get_cpu_info():
    """返回系统 CPU 信息字符串，例如 'Apple M2'。"""
    return CPUInfo.name()


@functools.lru_cache
def get_gpu_info(index):
    """返回系统 GPU 信息字符串，例如 'Tesla T4, 15102MiB'。"""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def parse_device(device: str | int | list | tuple | torch.device = "") -> str:
    """将任意形式的设备请求解析为规范设备字符串。

    参数：
        device (str | int | 列表 | tuple | torch.device, 可选): 设备请求，例如 'cuda:0'、'0,1'、[0, 1]、'cpu'、
            'mps' 或 '-1'（自动选择空闲 GPU；两个设备可使用 '-1,-1'）。

    返回：
        (str): Canonical device string, e.g. '', 'cpu', 'mps', '0', or '0,1'.

    示例：
        >>> parse_device("cuda:0")
        '0'

        >>> parse_device([0, 1])
        '0,1'

    注意：
        每个 '-1' 都会替换为空闲 GPU 索引。若请求的 ID 超出 torch 设备数量、但与外部 CUDA_VISIBLE_DEVICES
        限制下可见的物理 GPU ID 匹配，则将其转换为对应的 torch 索引；例如 CUDA_VISIBLE_DEVICES='3' 时，
        '3' 会转换为 '0'。范围内的 ID 始终视为 torch 索引，以保证重复解析结果稳定。返回的索引相对于当前限制，
        因此在同一环境中保存的字符串（例如恢复检查点参数）始终指向相同的物理 GPU。
    """
    if isinstance(device, torch.device):
        if device.type == "cuda" and device.index is None:
            return ""  # 无索引的 torch.device('cuda') 表示当前 CUDA 设备，即默认的 '' 请求
        if device.type in {"npu", "xpu"}:
            return device.type if device.index is None else f"{device.type}:{device.index}"
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # 转为字符串：'cuda:0' -> '0'，'(0, 1)' -> '0,1'
    if device == "cuda":
        device = "0"
    for backend in ("npu", "xpu"):
        if device.startswith(backend):
            indices = device[len(backend) :].lstrip(":").replace(f"{backend}:", "")
            indices = ",".join(str(int(x)) if x.isdigit() else x for x in indices.split(",") if x)
            return f"{backend}:{indices}" if indices else backend
    device = ",".join(str(int(x)) if x.isdigit() else x for x in device.split(",") if x)  # "0,,01" -> "0,1"
    # 将可见物理设备 ID 规范化为与请求 ID 相同的形式，并截断到 torch 设备数量，
    # 这与 CUDA 的 atoi 式解析以及遇到首个无效 CVD 条目时停止的行为一致。
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    visible = [str(int(x)) if x.isdigit() else x for x in cvd.split(",") if x][: torch.cuda.device_count()]
    indices = [x for x in device.split(",") if x.isdigit()]  # 请求的 ID，排除 '-1' 和非数字标记
    if indices and all(x in visible for x in indices) and any(int(x) >= torch.cuda.device_count() for x in indices):
        # 超出 torch 设备数量的 ID 只能是外部 CUDA_VISIBLE_DEVICES 限制下的物理 GPU ID，
        # 因此将其转换为 torch 索引；范围内的 ID 已经是 torch 索引，重复解析时保持稳定。
        device = ",".join(str(visible.index(x)) if x.isdigit() else x for x in device.split(","))
    if "-1" in device:
        from ultralytics.utils.autodevice import GPUInfo

        # 将每个 -1 替换为空闲 GPU，或将其移除；GPUInfo 只在外部可见 GPU 中搜索 NVML 物理 ID，
        # 再在 CUDA_VISIBLE_DEVICES 限制下将结果转换回 torch 索引。
        parts = device.split(",")
        candidates = [int(x) for x in visible if x.isdigit()] if visible else None
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2, indices=candidates)
        selected = [visible.index(str(x)) for x in selected] if visible else selected
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)
    return device


def select_device(device="", newline=False, verbose=True):
    """根据提供的参数选择适用的 PyTorch 设备。

    此函数接收指定设备的字符串或 torch.device 对象，并返回代表所选设备的 torch.device 对象。
    同时验证可用设备数量；请求的设备不可用时抛出异常。

    参数：
        device (str | torch.device, 可选): 设备字符串或 torch.device 对象。可选值包括 'cpu'、'cuda'、'0'、
            '0,1,2,3', 'mps', 'npu:0', 'npu:0,1', 'xpu:0', 'xpu:0,1', or '-1' for auto-select. Defaults to auto-selecting
            the first available GPU, or CPU if no GPU is available.
        newline (bool, 可选): 为 True 时在日志字符串末尾添加换行符。
        verbose (bool, 可选): 为 True 时记录设备信息。

    返回：
        (torch.device): Selected device.

    示例：
        >>> select_device("cuda:0")
        device(type='cuda', 索引=0)

        >>> select_device("cpu")
        device(type='cpu')

    注意：
        CUDA 索引就是 torch 设备索引，会反映外部设置的 CUDA_VISIBLE_DEVICES。此函数不会修改
        CUDA_VISIBLE_DEVICES；显式的单 GPU 请求会通过 torch.cuda.set_device() 设置为默认 CUDA 设备，
        使无索引的 'cuda' 操作落到该设备上，而默认的 '' 请求（解析为当前设备）和多 GPU 请求
        （DDP rank 在 trainer._setup_ddp() 中绑定自己的设备）不会改变当前设备。
    """
    if isinstance(device, torch.device):
        if device.type not in {"cuda", "npu", "xpu"}:
            return device  # 其他 torch.device 输入直接透传；加速器输入会在下方规范化并验证
    elif str(device).startswith(("tpu", "intel", "vulkan")):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    device = parse_device(device)

    if device.startswith(("npu", "xpu")):
        device_type = device.split(":", 1)[0]
        if device_type == "npu":
            try:
                import torch_npu  # noqa
            except ImportError:
                raise ValueError(
                    f"Invalid NPU 'device={device}'. Install 'torch_npu' at https://github.com/Ascend/pytorch"
                )
        if not hasattr(torch, device_type):
            raise ValueError(f"Invalid {device_type.upper()} 'device={device}' requested. Backend is not available.")
        backend = get_torch_device_backend(device_type)
        if not backend.is_available():
            raise ValueError(f"Invalid {device_type.upper()} 'device={device}' requested. Backend is not available.")

        requested = ["0"] if device == device_type else device[4:].split(",")
        indices = [int(x) for x in requested if x.isdigit()]
        if not indices or len(indices) != len(requested) or len(indices) != len(set(indices)):
            raise ValueError(
                f"Invalid {device_type.upper()} 'device={device}' format. "
                f"Use '{device_type}', '{device_type}:0', or '{device_type}:0,1'."
            )
        n = backend.device_count()
        if any(idx >= n for idx in indices):
            raise ValueError(
                f"Invalid {device_type.upper()} 'device={device}' requested. Only {n} device(s) available."
            )

        if len(indices) == 1:
            backend.set_device(indices[0])  # 多设备 DDP 的每个 rank 在 trainer._setup_ddp() 中绑定自己的设备
        if verbose:
            space = " " * len(s)
            for i, idx in enumerate(indices):
                s += f"{'' if i == 0 else space}{device_type.upper()}:{idx} ({backend.get_device_name(idx)})\n"
            LOGGER.info(s if newline else s.rstrip())
        return torch.device(device_type, indices[0])

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if not cpu and not mps and device:  # 请求了非 CPU 设备
        valid = all(x.isdigit() and int(x) < torch.cuda.device_count() for x in device.split(","))
        if not (torch.cuda.is_available() and valid):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # 如果可用，优先使用 GPU
        devices = device.split(",") if device else [str(torch.cuda.current_device())]  # '' -> current default device
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(int(d))})\n"
        arg = f"cuda:{devices[0]}"
        if device and len(devices) == 1:  # 仅处理明确的单 GPU 请求：'' 不会移动当前设备，且
            torch.cuda.set_device(int(devices[0]))  # 多 GPU DDP 的每个 rank 在 _setup_ddp() 中绑定自己的设备
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # 如果可用，优先使用 MPS
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # 回退到 CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # 为 CPU 训练重置 OMP_NUM_THREADS
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync(device: torch.device | None = None):
    """返回与 PyTorch 同步的准确时间。"""
    if device is None or device.type not in {"cpu", "mps"}:
        accelerator = get_torch_device_backend(device or "cuda")
        if accelerator.is_available() and hasattr(accelerator, "synchronize"):
            accelerator.synchronize()
    return time.perf_counter()


def fuse_conv_and_bn(conv, bn):
    """融合 Conv2d 和 BatchNorm2d 层，以优化推理。

    参数：
        conv (nn.Conv2d): 要融合的卷积层。
        bn (nn.BatchNorm2d): 要融合的批归一化层。

    返回：
        (nn.Conv2d): 已融合且禁用梯度的卷积层。

    示例：
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    # 计算融合权重：Conv2d 权重形状为 [out_channels, in_channels // groups, kH, kW]，沿轴 0 缩放。
    bn_scale = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    conv.weight.data = conv.weight * bn_scale.view(-1, 1, 1, 1)

    # 计算融合后的偏置
    b_conv = (
        torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = bn_scale * b_conv + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)


def fuse_deconv_and_bn(deconv, bn):
    """融合 ConvTranspose2d 和 BatchNorm2d 层，以优化推理。

    参数：
        deconv (nn.ConvTranspose2d): 要融合的转置卷积层。
        bn (nn.BatchNorm2d): 要融合的批归一化层。

    返回：
        (nn.ConvTranspose2d): 已融合且禁用梯度的转置卷积层。

    示例：
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    if isinstance(bn, nn.Identity):  # ConvTranspose(bn=False) 会保留 nn.Identity，无需融合
        return deconv.requires_grad_(False)
    # 计算融合权重：ConvTranspose2d 权重形状为 [in_channels, out_channels // groups, kH, kW]，
    # 因此每个输出通道的 BN 缩放沿轴 1 应用（从轴 0 按分组映射），而不是像 Conv2d 那样沿轴 0 应用。
    bn_scale = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    w_scale = bn_scale.view(deconv.groups, -1).repeat_interleave(deconv.in_channels // deconv.groups, 0)
    deconv.weight.data = deconv.weight * w_scale[:, :, None, None]

    # 计算融合后的偏置
    b_conv = (
        torch.zeros(deconv.out_channels, device=deconv.weight.device, dtype=deconv.weight.dtype)
        if deconv.bias is None
        else deconv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = bn_scale * b_conv + b_bn

    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias

    return deconv.requires_grad_(False)


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """逐层打印并返回详细的模型信息。

    参数：
        model (nn.Module): 要分析的模型。
        detailed (bool, 可选): 是否打印详细的层信息。
        verbose (bool, 可选): 是否打印模型信息。
        imgsz (int | 列表, 可选): 输入图像尺寸。

    返回：
        (tuple): Tuple containing:
            - n_l (int): 层数.
            - n_p (int): 参数数量。
            - n_g (int): 梯度数量。
            - flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # 参数数量
    n_g = get_num_gradients(model)  # 梯度数量
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # 层数
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # 没有可学习参数的层
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz 可以是整数或列表，例如 imgsz=640 或 imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """返回 YOLO 模型中的参数总量。"""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """返回 YOLO 模型中需要梯度的参数总量。"""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """返回包含有用模型信息的模型信息字典。

    参数：
            trainer (ultralytics.engine.trainer.BaseTrainer): 包含模型和验证数据的训练器对象。

    返回：
        (dict): 包含模型参数、GFLOPs 和推理速度的字典。

    示例：
        用于日志记录的 YOLOv8n 信息
        >>> results = {
        ...     "model/parameters": 3151904,
        ...     "model/GFLOPs": 8.746,
        ...     "model/speed_ONNX(ms)": 41.244,
        ...     "model/speed_TensorRT(ms)": 3.211,
        ...     "model/speed_PyTorch(ms)": 18.755,
        ... }
    """
    if trainer.args.profile:  # 分析 ONNX 和 TensorRT 耗时
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device, imgsz=trainer.args.imgsz).run()[0]
        results.pop("model/name")
    else:  # 仅返回最近一次验证的 PyTorch 耗时
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model, trainer.args.imgsz), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def _attention_ops(m, x, y):
    """统计注意力块中查询-键和注意力-值矩阵乘法的 THOP 运算量。

    两种运算都在重塑后的张量上以函数形式运行，因此子模块 hook 无法观测到它们；否则该模块只会统计
    qkv/proj/pe 卷积的运算量。两个乘积的每个输出元素在收缩轴上产生一次乘加操作，因此每个头的运算量为
    `tokens**2 * (key_dim + head_dim)`。
    """
    b, _, h, w = x[0].shape
    area = getattr(m, "area", 1)  # area attention 在指定数量的独立组内执行注意力，仅 AAttn 使用
    tokens = h * w // area
    key_dim = getattr(m, "key_dim", m.head_dim)  # Attention 按 attn_ratio 缩小 q 和 k，AAttn 不执行此操作
    m.total_ops += b * area * m.num_heads * tokens * tokens * (key_dim + m.head_dim)


def get_flops(model, imgsz=640):
    """计算模型的 FLOPs（浮点运算次数），单位为 GFLOPs。

    使用 THOP 的步长感知图像分析来提高效率，并准确统计与尺寸无关的操作。如果 thop 不可用或分析失败，则返回 0.0。

    参数：
        model (nn.Module): 要计算 FLOPs 的模型。
        imgsz (int | 列表, 可选): 输入图像尺寸。

    返回：
        (float): 模型的 GFLOPs（十亿次浮点运算）。
    """
    try:
        import thop
    except ImportError:
        thop = None  # 未安装 'ultralytics-thop' 时支持 conda 环境

    if not thop:
        return 0.0  # 未安装时返回 0.0 GFLOPs

    try:
        from ultralytics.nn.modules.block import AAttn, Attention  # 在此处导入：block.py 会导入此模块
        from ultralytics.nn.modules.head import RTDETRDecoder

        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # 输入为 int/float 时扩展为二维
        attn = tuple(m for m in model.modules() if isinstance(m, (Attention, AAttn)))
        rtdetr = any(isinstance(m, RTDETRDecoder) for m in model.modules())
        # 注意力的计算量与图像面积呈平方关系，因此禁用 THOP 的仿射代理计算。
        stride = None if attn else max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device, dtype=p.dtype)  # BCHW 格式的输入图像
        custom_ops = {Attention: _attention_ops, AAttn: _attention_ops} if attn else None
        if rtdetr:  # RT-DETR 无法运行步长大小的代理输入
            return thop.profile(model, inputs=[im], custom_ops=custom_ops, verbose=False)[0] / 1e9 * 2
        return thop.profile(model, inputs=[im], stride=stride, custom_ops=custom_ops, verbose=False)[0] / 1e9 * 2
    except Exception:
        return 0.0


def initialize_weights(model):
    """将模型权重、偏置和模块配置初始化为默认值。"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """缩放并填充图像张量，可选择保持宽高比并填充到 gs 的整数倍。

    参数：
        img (torch.Tensor): 输入图像张量。
        ratio (float, 可选): 缩放比例。
        same_shape (bool, 可选): 是否保持相同形状。
        gs (int, 可选): 用于填充的网格尺寸。

    返回：
        (torch.Tensor): 缩放并填充后的图像张量。
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # 新尺寸
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # 调整尺寸
    if not same_shape:  # 填充或裁剪图像
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """将对象 b 的属性复制到对象 a，并支持包含或排除指定属性。

    参数：
        a (Any): 接收属性的目标对象。
        b (Any): 提供属性的源对象。
        include (tuple, 可选): 要包含的属性。为空时包含所有属性。
        exclude (tuple, 可选): 要排除的属性。
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da, db, exclude=()):
    """返回形状匹配的交集键字典，排除 exclude 键，并使用 da 值。

    参数：
        da (dict): 第一个字典。
        db (dict): 第二个字典。
        exclude (tuple, 可选): 要排除的键。

    返回：
        (dict): 包含形状匹配交集键的字典。
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """如果模型类型为 DP 或 DDP，则返回 True。

    参数：
        模型 (nn.Module): 要检查的模型。

    返回：
        (bool): 模型为 DataParallel 或 DistributedDataParallel 时返回 True。
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def unwrap_model(m: nn.Module) -> nn.Module:
    """解除编译模型和并行模型的包装，获取基础模型。

    参数：
        m (nn.Module): 可能被 torch.compile（._orig_mod）或 DataParallel/DistributedDataParallel（.module）等并行包装器封装的模型。

    返回：
        (nn.Module): 移除编译和并行包装器后的基础模型。
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """返回从 y1 到 y2 的正弦渐变 lambda 函数，参见 https://arxiv.org/pdf/1812.01187.pdf。

    参数：
        y1 (float, optional): Initial value.
        y2 (float, optional): Final value.
        steps (int, optional): Number of steps.

    返回：
        (function): Lambda function for computing the sinusoidal ramp.
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """初始化随机数生成器（RNG）种子，参见 https://pytorch.org/docs/stable/notes/randomness.html。

    参数：
        seed (int, optional): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 用于多 GPU，并安全处理异常
    # torch.backends.cudnn.benchmark = True  # AutoBatch 问题：https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        unset_deterministic()


def unset_deterministic():
    """取消为确定性训练应用的所有配置。"""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)


class ModelEMA:
    """更新后的指数移动平均（EMA）实现。

    保存模型 state_dict 中所有参数和缓冲区的移动平均值。有关 EMA 的详细信息，请参阅参考资料。

    要禁用 EMA，请将 `enabled` 属性设置为 `False`。

    属性：
        ema (nn.Module): Copy of the model in evaluation mode.
        updates (int): Number of EMA updates.
        decay (function): Decay function that determines the EMA weight.
        enabled (bool): Whether EMA is enabled.

    参考：
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """使用给定参数为“模型”初始化 EMA。

        参数：
            model (nn.Module): 用于创建 EMA 的模型。
            decay (float, 可选): EMA 最大衰减率。
            tau (int, 可选): EMA 衰减时间常数。
            updates (int, 可选): 初始更新次数。
        """
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        if hasattr(self.ema, "teacher_model"):
            # DistillationModel：移除教师模型，避免 EMA 携带一份完整的重复副本。
            self.ema.teacher_model = None
        self.updates = updates  # EMA 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # 指数衰减曲线，帮助训练早期稳定
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """更新 EMA 参数。

        参数：
            model (nn.Module): 用于更新 EMA 的模型。
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = unwrap_model(model).state_dict()  # 模型 state_dict
            ema_v, model_v = [], []
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # FP16 和 FP32 满足此条件
                    ema_v.append(v)
                    model_v.append(msd[k])
            if (
                ema_v and TORCH_2_0 and ema_v[0].device.type != "npu" and (TORCH_2_4 or ema_v[0].device.type != "mps")
            ):  # 每个操作启动一次内核
                torch._foreach_lerp_(ema_v, model_v, 1 - d)
            else:  # _foreach_lerp_ 需要 torch>=2.0，MPS 需要 torch>=2.4，且 NPU 不支持
                for v, m in zip(ema_v, model_v):
                    v.mul_(d).add_(m, alpha=1 - d)

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """将模型属性复制到 EMA，并支持包含或排除指定属性。

        参数：
            model (nn.Module): 要复制属性的模型。
            include (tuple, 可选): 要包含的属性。
            exclude (tuple, 可选): 要排除的属性。
        """
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """从 f 中移除优化器以完成训练，并可选择保存为 s。

    参数：
        f (str | Path): 要从中移除优化器的模型文件路径。
        s (str, 可选): 保存已移除优化器模型的文件路径。未提供时覆盖 'f'。
        updates (dict, 可选): 保存前叠加到检查点上的更新字典。

    返回：
        (dict): 合并后的检查点字典。

    示例：
        >>> from pathlib import Path
        >>> from ultralytics.utils.torch_utils import strip_optimizer
        >>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
        ...     strip_optimizer(f)
    """
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}

    metadata = {
        "date": datetime.now().astimezone().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # 更新模型
    if x.get("ema"):
        x["model"] = x["ema"]  # 使用 EMA 模型替换原模型

    # 解包 DistillationModel，仅保存学生模型
    from ultralytics.nn.distill_model import DistillationModel

    if isinstance(x["model"], DistillationModel):
        x["model"]._remove_feature_hooks()
        x["model"] = x["model"].student_model

    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # 从 IterableSimpleNamespace 转换为字典
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # 移除损失函数
    x["model"].half()  # 转换为 FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # 更新其他键
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # 合并参数
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # 键名
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 移除非默认键
    # x['model'].args = x['train_args']

    # 保存
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # 合并字典（右侧值优先）
    mb = os.path.getsize(s or f) / 1e6  # 文件 尺寸
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """将给定优化器的 state_dict 转换为 FP16，重点转换 state 键中的张量。

    参数：
        state_dict (dict): 优化器状态字典。

    返回：
        (dict): 转换后的优化器状态字典，其中张量为 FP16。
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k not in {"step", "exp_avg_sq"} and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


@contextmanager
def cuda_memory_usage(device=None):
    """监控并管理加速器内存使用情况。

    此函数清空当前加速器缓存，返回包含内存使用信息的字典，然后记录指定设备上的已保留内存。

    参数：
        device (torch.device, 可选): 要查询内存使用情况的加速器设备。

    Yields:
        (dict): 包含键 'memory' 的字典，该键初始为 0，随后更新为已保留内存。
    """
    info = {"memory": 0}
    if device is not None and device.type in {"cpu", "mps"}:
        yield info
        return
    accelerator = get_torch_device_backend(device or "cuda")
    if accelerator.is_available() and hasattr(accelerator, "memory_reserved"):
        accelerator.empty_cache()
        try:
            yield info
        finally:
            info["memory"] = accelerator.memory_reserved(device)
    else:
        yield info


def profile_ops(input, ops, n=10, device=None, max_num_obj=0):
    """Ultralytics 速度、内存和 FLOPs 分析器。

    参数：
        input (torch.Tensor | 列表): 要分析的输入张量。
        ops (nn.Module | 列表): 要分析的模型或操作列表。
        n (int, 可选): 用于求平均值的迭代次数。
        device (str | torch.device, 可选): 执行分析的设备。
        max_num_obj (int, 可选): 模拟使用的最大目标数量。

    返回：
        (列表): 每个操作的分析结果。

    示例：
        >>> from ultralytics.utils.torch_utils import profile_ops
        >>> input = torch.randn(16, 3, 640, 640)
        >>> m1 = lambda x: x * torch.sigmoid(x)
        >>> m2 = nn.SiLU()
        >>> profile_ops(input, [m1, m2], n=100)  # 分析 100 次迭代
    """
    try:
        import thop
    except ImportError:
        thop = None  # 未安装 'ultralytics-thop' 时支持 conda 环境

    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # 尝试释放未使用的内存
    accelerator = get_torch_device_backend(device) if device.type not in {"cpu", "mps"} else None
    if accelerator is not None:
        accelerator.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
            except Exception:
                flops = 0

            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync(device)
                        y = m(x)
                        t[1] = time_sync(device)
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync(device)
                        except Exception:  # no backward method
                            # print(e)  # 调试时使用
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1e9  # (GB)
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                    if max_num_obj:  # 为 AutoBatch 模拟每张图像网格上的预测结果
                        with cuda_memory_usage(device) as cuda_info:
                            anchors = int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist()))
                            # 检测损失的内存峰值范围：TaskAlignedAssigner.get_box_metrics 会同时保存约 6 个
                            # (bs, max_num_obj, 锚框) 的 fp32 缓冲区（overlaps、bbox_scores、gathered
                            # pd_scores、两个幂运算临时量和 align_metric）；分类路径会保存约 6 个
                            # (bs, 锚框, nc) 的 fp32 等价缓冲区（预测/目标，以及 v8DetectionLoss 中未归约 BCE
                            # 的两个临时量：纯 fp32 约 4 个，AMP 下 autocast 将两个 BCE 输入提升为 fp32 副本后约 6 个）。
                            sim = (
                                torch.randn(x.shape[0], 6 * max_num_obj, anchors, device=device, dtype=torch.float32),
                                torch.randn(x.shape[0], anchors, 6 * len(m.names), device=device, dtype=torch.float32),
                            )
                        del sim
                        mem += cuda_info["memory"] / 1e9  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # 参数
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{s_in!s:>24s}{s_out!s:>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()  # 尝试释放未使用的内存
                if accelerator is not None:
                    accelerator.empty_cache()
    return results


class EarlyStopping:
    """早停类：指定数量的周期没有改进时停止训练。

    属性：
        best_fitness (float): 观察到的最佳适应度值。
        best_epoch (int): Epoch where best fitness was observed.
        patience (int): Number of epochs to wait after fitness stops improving before stopping.
        possible_stop (bool): Flag indicating if stopping may occur next epoch.
    """

    def __init__(self, patience=50):
        """初始化早停对象。

        参数：
            patience (int, 可选): 适应度停止改善后、停止训练前等待的轮数。
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # fitness 停止提升后等待的周期数
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """检查是否应停止训练。

        参数：
        epoch (int): 当前训练轮次。
        fitness (float): 当前轮次的适应度值。

        返回：
            (bool): 训练应停止时返回 True，否则返回 False。
        """
        if fitness is None:  # 检查 fitness 是否为 None（val=False 时会出现）
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:  # 允许训练早期出现适应度为零的阶段
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # 没有改善的训练轮数
        self.possible_stop = delta >= (self.patience - 1)  # 下一轮可能停止
        stop = delta >= self.patience  # 超过耐心值时停止训练
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module:
    """使用 torch.compile 编译模型，并可选择预热计算图以降低首次迭代延迟。

        此工具尝试使用 inductor 后端编译提供的模型。如果编译不可用或失败，则原样返回原始模型。
        可选的预热会使用虚拟输入执行一次前向传播，以预热编译图并测量编译/预热时间。

    参数：
        model (torch.nn.Module): 要编译的模型。
        device (torch.device): 用于预热及 autocast 判断的推理设备。
        imgsz (int, 可选): 用于创建形状为 (1, 3, imgsz, imgsz) 虚拟张量的正方形输入尺寸。
        use_autocast (bool, 可选): 是否在 CUDA 或 MPS 设备上使用 autocast 执行预热。
        warmup (bool, 可选): 是否执行一次虚拟前向传播以预热已编译模型。
        mode (bool | str, 可选): torch.compile 模式。True 表示 "default"，False 表示不编译，也可以传入类似
            "default", "reduce-overhead", "max-autotune-no-cudagraphs".

    返回：
        (torch.nn.Module): 编译成功时返回已编译模型，否则返回未修改的原始模型。

    示例：
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> # 尝试使用 640x640 输入编译并预热模型
        >>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)

    注意：
        - 如果当前 PyTorch 构建不提供 torch.compile，函数会立即返回输入模型。
        - 编译采用延迟执行，在第一次前向传播时运行。因此会预先检查 inductor 在 CPU 上所需的主机 C++ 编译器，
          如果不可用则返回原始模型。
        - 预热过程在 torch.inference_mode 下运行，并可能对 CUDA/MPS 使用 torch.autocast，以保持计算精度一致。
        - 预热后同步 CUDA 设备，以计入异步内核执行时间。
    """
    if not hasattr(torch, "compile") or not mode:
        return model

    if mode is True:
        mode = "default"
    prefix = colorstr("compile:")
    if device.type == "cpu":
        try:  # 编译是惰性的，因此先验证 Inductor CPU 所需的主机 C++ 编译器
            from torch._inductor.cpp_builder import get_cpp_compiler

            get_cpp_compiler()
        except ImportError:
            pass  # 旧版 torch 没有 cpp_builder，交由 torch.compile 处理
        except Exception as e:
            LOGGER.warning(f"{prefix} no C++ compiler found for the inductor backend, continuing uncompiled: {e}")
            return model
    LOGGER.info(f"{prefix} starting torch.compile with '{mode}' mode...")
    t0 = time.perf_counter()
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        LOGGER.warning(f"{prefix} torch.compile failed, continuing uncompiled: {e}")
        return model
    t_compile = time.perf_counter() - t0

    t_warm = 0.0
    if warmup:
        # 使用单个虚拟张量构建图的形状状态，降低首次迭代延迟。
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if use_autocast and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1

    total = t_compile + t_warm
    if warmup:
        LOGGER.info(f"{prefix} complete in {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)")
    else:
        LOGGER.info(f"{prefix} compile complete in {t_compile:.1f}s (no warmup)")
    return model
