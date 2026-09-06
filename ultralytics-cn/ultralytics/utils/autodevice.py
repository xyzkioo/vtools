# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
from typing import Any

from ultralytics.utils import LOGGER


class GPUInfo:
    """通过 pynvml 管理 NVIDIA GPU 信息，并提供完善的错误处理。

    提供查询 GPU 详细统计信息（利用率、内存、温度和功耗）以及根据可配置条件选择最空闲 GPU 的方法。
    如果 pynvml 不存在或初始化失败，函数会记录警告并禁用相关功能，避免应用程序崩溃。

    如果选择 GPU 时 NVML 不可用，还包含使用 `torch.cuda` 进行基本设备计数的回退逻辑。
    NVML 的初始化和关闭由此类在内部管理。

    属性：
        pynvml (模块 | None): 成功导入并初始化后的 `pynvml` 模块，否则为 `None`。
        nvml_available (bool): 表示 `pynvml` 是否可用。`nvmlInit()` 成功时为 True，否则为 False。
        gpu_stats (列表[dict[str, Any]]): GPU 统计信息字典列表，在初始化和 `refresh_stats()` 时填充。
            每个字典对应一块 GPU，包含 'index'、'name'、'utilization'（%）、'memory_used'（MiB）、
            'memory_total'（MiB）、'memory_free'（MiB）、'temperature'（C）、'power_draw'（W）和
            'power_limit'（W 或 'N/A'）。如果 NVML 不可用或查询失败，则为空列表。

    方法：
        refresh_stats: 查询 NVML 并刷新内部的 gpu_stats 列表。
        print_status: 使用当前统计信息，以紧凑表格格式打印 GPU 状态。
        select_idle_gpu: 根据利用率和可用内存选择最空闲的 GPU。
        shutdown: 如果 NVML 已初始化，则关闭 NVML。

    示例：
        初始化 GPUInfo 并打印状态：
        >>> gpu_info = GPUInfo()
        >>> gpu_info.print_status()

        按最低内存要求选择空闲 GPU：
        >>> selected = gpu_info.select_idle_gpu(count=2, min_memory_fraction=0.2)
        >>> print(f"选中的 GPU 索引：{selected}")
    """

    def __init__(self):
        """初始化 GPUInfo，并尝试导入和初始化 pynvml。"""
        self.pynvml: Any | None = None
        self.nvml_available: bool = False
        self.gpu_stats: list[dict[str, Any]] = []

        try:
            import pynvml  # 延迟导入，因为该模块加载较慢

            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except Exception as e:
            LOGGER.warning(f"Failed to initialize pynvml, GPU stats disabled: {e}")

    def __del__(self):
        """确保对象被垃圾回收时关闭 NVML。"""
        self.shutdown()

    def shutdown(self):
        """如果 NVML 已初始化，则关闭 NVML。"""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_available = False

    def refresh_stats(self):
        """查询 NVML 并刷新内部的 gpu_stats 列表。"""
        self.gpu_stats = []
        if not self.nvml_available or not self.pynvml:
            return

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            self.gpu_stats.extend(self._get_device_stats(i) for i in range(device_count))
        except Exception as e:
            LOGGER.warning(f"Error during device query: {e}")
            self.gpu_stats = []

    def _get_device_stats(self, index: int) -> dict[str, Any]:
        """获取单个 GPU 设备的统计信息。"""
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

        def safe_get(func, *args, default=-1, divisor=1):
            try:
                val = func(*args)
                return val // divisor if divisor != 1 and isinstance(val, (int, float)) else val
            except Exception:
                return default

        temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)

        return {
            "index": index,
            "name": self.pynvml.nvmlDeviceGetName(handle),
            "utilization": util.gpu if util else -1,
            "memory_used": memory.used >> 20 if memory else -1,  # 将字节转换为 MiB
            "memory_total": memory.total >> 20 if memory else -1,
            "memory_free": memory.free >> 20 if memory else -1,
            "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
            "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),  # 将 mW 转换为 W
            "power_limit": safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),
        }

    def print_status(self):
        """使用当前统计信息，以紧凑表格格式打印 GPU 状态。"""
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("No GPU stats available.")
            return

        stats = self.gpu_stats
        name_len = max(len(gpu.get("name", "N/A")) for gpu in stats)
        hdr = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        LOGGER.info(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")

        for gpu in stats:
            u = f"{gpu['utilization']:>5}%" if gpu["utilization"] >= 0 else " N/A "
            m = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}" if gpu["memory_used"] >= 0 else " N/A / N/A "
            t = f"{gpu['temperature']}C" if gpu["temperature"] >= 0 else " N/A "
            p = f"{gpu['power_draw']:>3}/{gpu['power_limit']:<3}" if gpu["power_draw"] >= 0 else " N/A "

            LOGGER.info(f"{gpu.get('index'):<3d} {gpu.get('name', 'N/A'):<{name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")

        LOGGER.info(f"{'-' * len(hdr)}\n")

    def select_idle_gpu(
        self,
        count: int = 1,
        min_memory_fraction: float = 0,
        min_util_fraction: float = 0,
        indices: list[int] | None = None,
    ) -> list[int]:
        """根据利用率和可用内存选择最空闲的 GPU。

        参数：
            count (int): 要选择的空闲 GPU 数量。
            min_memory_fraction (float): 所需可用内存占总内存的最小比例。
            min_util_fraction (float): 所需的最小空闲利用率，范围为 0.0 到 1.0。
            indices (列表[int] | None): 将选择范围限制为这些 GPU 索引，例如外部 CUDA_VISIBLE_DEVICES
                限制下可见的 GPU。None 表示搜索所有 GPU。

        返回：
            (列表[int]): 选中 GPU 的索引，按空闲程度排序（利用率最低的优先）。

        注意：
             如果符合条件的 GPU 不足或不存在，返回数量会少于 'count'。
             如果 NVML 统计信息不可用或没有 GPU 符合条件，则返回空列表。
        """
        assert min_memory_fraction <= 1.0, f"min_memory_fraction must be <= 1.0, got {min_memory_fraction}"
        assert min_util_fraction <= 1.0, f"min_util_fraction must be <= 1.0, got {min_util_fraction}"
        criteria = (
            f"free memory >= {min_memory_fraction * 100:.1f}% and free utilization >= {min_util_fraction * 100:.1f}%"
        )
        LOGGER.info(f"Searching for {count} idle GPUs with {criteria}...")

        if count <= 0:
            return []

        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("NVML stats unavailable.")
            return []

        # 筛选并排序符合条件的 GPU
        eligible_gpus = [
            gpu
            for gpu in self.gpu_stats
            if (indices is None or gpu["index"] in indices)
            and gpu.get("memory_free", 0) / gpu.get("memory_total", 1) >= min_memory_fraction
            and (100 - gpu.get("utilization", 100)) >= min_util_fraction * 100
        ]
        # 随机打破平局，避免多个进程同时启动且所有 GPU 都同样空闲（利用率和可用内存相同）时产生竞争条件
        eligible_gpus.sort(key=lambda x: (x.get("utilization", 101), -x.get("memory_free", 0), random.random()))

        # 选择排名靠前的 'count' 个索引
        selected = [gpu["index"] for gpu in eligible_gpus[:count]]

        if selected:
            if len(selected) < count:
                LOGGER.warning(f"Requested {count} GPUs but only {len(selected)} met the idle criteria.")
            LOGGER.info(f"Selected idle CUDA devices {selected}")
        else:
            LOGGER.warning(f"No GPUs met criteria ({criteria}).")

        return selected


if __name__ == "__main__":
    required_free_mem_fraction = 0.2  # 要求 20% 的可用显存
    required_free_util_fraction = 0.2  # 要求 20% 的可用利用率
    num_gpus_to_select = 1

    gpu_info = GPUInfo()
    gpu_info.print_status()

    if selected := gpu_info.select_idle_gpu(
        count=num_gpus_to_select,
        min_memory_fraction=required_free_mem_fraction,
        min_util_fraction=required_free_util_fraction,
    ):
        print(f"\n==> Using selected GPU indices: {selected}")
        devices = [f"cuda:{idx}" for idx in selected]
        print(f"    Target devices: {devices}")
