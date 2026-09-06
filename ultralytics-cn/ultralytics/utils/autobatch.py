# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""在 PyTorch 中估算 YOLO 最佳批量大小的函数，使其使用可用 GPU 内存的一定比例。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, get_torch_device_backend, profile_ops


def check_train_batch_size(
    model: torch.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: float = -1,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """使用 autobatch() 函数计算 YOLO 训练的最佳批量大小。

    参数：
        model (torch.nn.Module): 用于检查批量大小的 YOLO 模型。
        imgsz (int, 可选): 训练使用的图像尺寸。
        amp (bool, 可选): 为 True 时使用自动混合精度。
        batch (int | float, 可选): 要使用的 GPU 内存比例。为 -1 时使用默认比例。
        max_num_obj (int, 可选): 数据集中的最大目标数量。
        dataset_size (int, 可选): 训练图像总数。大于 0 时，批量大小不会超过此值。

    返回：
        (int): 使用 autobatch() 函数计算出的最佳批量大小。

    异常：
        RuntimeError: 没有候选批量大小能够生成可用性能评估结果时抛出。

    注意：
        如果 0.0 < batch < 1.0，则将其作为要使用的 GPU 内存比例。
        否则使用默认比例 0.6。
    """
    with autocast(enabled=amp, device=next(model.parameters()).device.type):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )


def autobatch(
    model: torch.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.60,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """自动估算 YOLO 的最佳批量大小，使其使用可用 GPU 内存的一定比例。

    参数：
        model (torch.nn.Module): 用于计算批量大小的 YOLO 模型。
        imgsz (int, 可选): 作为 YOLO 模型输入的图像尺寸。
        fraction (float, 可选): 要使用的可用 CUDA 内存比例。
        batch_size (int, 可选): 检测到错误时使用的默认批量大小。
        max_num_obj (int, 可选): 数据集中的最大目标数量。
        dataset_size (int, 可选): 训练图像总数。大于 0 时，批量大小不会超过此值。

    返回：
        (int): 最佳批量大小。

    异常：
        RuntimeError: 没有候选批量大小能够生成可用性能评估结果时抛出。
    """
    # 检查设备
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% GPU memory utilization.")
    device = next(model.parameters()).device  # 获取模型所在设备
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(f"{prefix}intended for GPU devices, using default batch-size {batch_size}")
        return batch_size
    if device.type == "cuda" and torch.backends.cudnn.benchmark:
        LOGGER.warning(f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # 检查 GPU 内存
    accelerator = get_torch_device_backend(device)
    gb = 1 << 30  # 字节转换为 GiB（1024 ** 3）
    d = f"{device.type.upper()}:{device.index}"
    properties = accelerator.get_device_properties(device)  # 设备属性
    t = properties.total_memory / gb  # 总 GiB 数
    r = accelerator.memory_reserved(device) / gb  # 已保留的 GiB 数
    a = accelerator.memory_allocated(device) / gb  # 已分配的 GiB 数
    f = t - (r + a)  # 可用 GiB 数
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # 评估各个批量大小
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    if dataset_size > 0:
        batch_sizes = [b for b in batch_sizes if b <= dataset_size]
    ch = model.yaml.get("channels", 3)
    try:
        img = [torch.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)

        # 拟合内存使用曲线
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # 有效结果
            and isinstance(y[2], (int, float))  # 数值类型
            and 0 < y[2] < t  # 位于 0 和 GPU 上限之间
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # 第一个结果或内存使用量递增
        ]
        if xy:
            fit_x, fit_y = zip(*xy)
            p = np.polyfit(fit_x, fit_y, deg=1)  # 一阶（线性）多项式拟合
            b = int((round(f * fraction) - p[1]) / p[0])  # y 轴截距（最佳批量大小）
            if None in results:  # 某些批量大小评估失败
                i = results.index(None)  # 第一个失败结果的索引
                if b >= batch_sizes[i]:  # y 轴截距超过失败点
                    b = batch_sizes[max(i - 1, 0)]  # 选择之前的安全点
            if b < 1 or b > 1024:  # b 超出安全范围
                LOGGER.warning(f"{prefix}batch={b} outside safe range, using default batch-size {batch_size}.")
                b = batch_size
            if dataset_size > 0:
                b = min(b, dataset_size)

            fraction = (np.polyval(p, b) + r + a) / t  # 预测的内存占用比例
            LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
            return b
    except Exception as e:
        LOGGER.warning(f"{prefix}error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
    finally:
        accelerator.empty_cache()

    raise RuntimeError(
        f"{prefix}no usable batch size found while profiling batch={batch_sizes}. "
        f"See the errors above, free GPU memory, reduce imgsz, or set batch explicitly."
    )
