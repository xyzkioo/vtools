# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import sys
import time
import unicodedata
from functools import lru_cache
from typing import IO, Any

from typing_extensions import Self


@lru_cache(maxsize=1)
def is_noninteractive_console() -> bool:
    """检查已知的非交互式控制台环境。"""
    return "GITHUB_ACTIONS" in os.environ


class TQDM:
    """Ultralytics 的轻量级零依赖进度条。

    适用于 Weights & Biases、控制台输出和其他日志系统等多种环境，提供简洁的 rich 风格进度条。
    其特性包括零外部依赖、简洁的单行输出、Unicode 块字符进度条、上下文管理器支持、迭代器协议支持
    以及动态描述更新。

    属性：
        iterable (Any): 要用进度条包装的可迭代对象。
        desc (str): 进度条前缀描述。
        total (int | None): 预期迭代次数。
        disable (bool): 是否禁用进度条。
        unit (str): 迭代单位名称。
        unit_scale (bool): 是否自动缩放单位。
        unit_divisor (int): 单位缩放除数。
        leave (bool): 完成后是否保留进度条。
        mininterval (float): 更新之间的最小时间间隔。
        initial (int): 初始计数值。
        n (int): 当前迭代次数。
        closed (bool): 进度条是否已关闭。
        bar_format (str | None): 自定义进度条格式字符串。
        file (IO[str]): 输出文件流。

    方法：
        update: Update progress by n steps.
        set_description: Set or update the description.
        set_postfix: Set postfix for the progress bar.
        close: Close the progress bar and clean up.
        refresh: Refresh the progress bar display.
        clear: Clear the progress bar from display.
        write: Write a message without breaking the progress bar.

    示例：
        Basic usage with iterator:
        >>> for i in TQDM(range(100)):
        ...     time.sleep(0.01)

        With custom description:
        >>> pbar = TQDM(range(100), desc="Processing")
        >>> for i in pbar:
        ...     pbar.set_description(f"Processing item {i}")

        Context manager usage:
        >>> with TQDM(total=100, unit="B", unit_scale=True) as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)

        Manual updates:
        >>> pbar = TQDM(total=100, desc="Training")
        >>> for epoch in range(100):
        ...     # Do work
        ...     pbar.update(1)
        >>> pbar.close()
    """

    # 常量
    MIN_RATE_CALC_INTERVAL = 0.01  # 计算速率的最小时间间隔
    RATE_SMOOTHING_FACTOR = 0.3  # 速率指数平滑因子
    MAX_SMOOTHED_RATE = 1000000  # 应用平滑的最大速率
    NONINTERACTIVE_MIN_INTERVAL = 60.0  # 非交互环境的最小间隔

    def __init__(
        self,
        iterable: Any = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: IO[str] | None = None,
        mininterval: float = 0.1,
        disable: bool | None = None,
        unit: str = "it",
        unit_scale: bool = True,
        unit_divisor: int = 1000,
        bar_format: str | None = None,  # 保持 API 兼容；不用于格式化
        initial: int = 0,
        **kwargs,
    ) -> None:
        """使用指定配置选项初始化 TQDM 进度条。

        参数：
            iterable (Any, 可选): 要用进度条包装的可迭代对象。
            desc (str, 可选): 进度条前缀描述。
            total (int, 可选): 预期迭代次数。
            leave (bool, 可选): 完成后是否保留进度条。
            file (IO[str], 可选): 用于显示进度的输出文件流。
            mininterval (float, 可选): 更新之间的最小时间间隔（默认 0.1 秒，GitHub Actions 中为 60 秒）。
            disable (bool, 可选): 是否禁用进度条。为 None 时自动检测。
            unit (str, 可选): 迭代单位名称（默认项目单位为 "it"）。
            unit_scale (bool, 可选): 是否自动缩放字节或数据单位。
            unit_divisor (int, 可选): 单位缩放除数（默认 1000）。
            bar_format (str, 可选): 自定义进度条格式字符串。
            initial (int, 可选): 初始计数值。
            **kwargs (Any): 兼容性所需的其他关键字参数（忽略）。
        """
        # 非详细模式下禁用进度条
        if disable is None:
            try:
                from ultralytics.utils import LOGGER, VERBOSE

                disable = not VERBOSE or LOGGER.getEffectiveLevel() > 20
            except ImportError:
                disable = False

        self.iterable = iterable
        self.desc = desc or ""
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None) or None  # 防止 total=0
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.leave = leave
        self.noninteractive = is_noninteractive_console()
        self.mininterval = max(mininterval, self.NONINTERACTIVE_MIN_INTERVAL) if self.noninteractive else mininterval
        self.initial = initial

        # 为兼容 API 保留（f-string 格式化不会使用）
        self.bar_format = bar_format

        self.file = file or sys.stdout

        # 内部状态
        self.n = self.initial
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.last_rate = 0.0
        self.closed = False
        self.is_bytes = unit_scale and unit in {"B", "bytes"}
        self.scales = (
            [(1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s")]
            if self.is_bytes
            else [(1e9, f"G{self.unit}/s"), (1e6, f"M{self.unit}/s"), (1e3, f"K{self.unit}/s")]
        )

        if not self.disable and self.total and not self.noninteractive:
            self._display()

    def _format_rate(self, rate: float) -> str:
        """格式化带单位的速率，在 it/s 和 s/it 之间切换以提高可读性。"""
        if rate <= 0:
            return ""

        inv_rate = 1 / rate if rate else None

        # 当 inv_rate > 1（即速率 < 1 it/s）时使用 s/it 格式以提高可读性
        if inv_rate and inv_rate > 1:
            return f"{inv_rate:.1f}s/B" if self.is_bytes else f"{inv_rate:.1f}s/{self.unit}"

        # 快速迭代时使用 it/s 格式
        fallback = f"{rate:.1f}B/s" if self.is_bytes else f"{rate:.1f}{self.unit}/s"
        return next((f"{rate / t:.1f}{u}" for t, u in self.scales if rate >= t), fallback)

    def _format_num(self, num: float) -> str:
        """使用可选单位缩放格式化数量。"""
        if not self.unit_scale or not self.is_bytes:
            return str(num)

        for unit in ("", "K", "M", "G", "T"):
            if abs(num) < self.unit_divisor:
                return f"{num:3.1f}{unit}B" if unit else f"{num:.0f}B"
            num /= self.unit_divisor
        return f"{num:.1f}PB"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间长度。"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
        else:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            return f"{h}:{m:02d}:{seconds % 60:02.0f}"

    def _generate_bar(self, width: int = 12) -> str:
        """生成进度条。"""
        if self.total is None:
            return "━" * width if self.closed else "─" * width

        frac = min(1.0, self.n / self.total)
        filled = int(frac * width)
        bar = "━" * filled + "─" * (width - filled)
        if filled < width and frac * width - filled > 0.5:
            bar = f"{bar[:filled]}╸{bar[filled + 1 :]}"
        return bar

    @staticmethod
    def _fit(text: str, width: int) -> str:
        """将文本截断到指定显示单元格宽度，跳过零宽度 ANSI 代码，并将中日韩字符计为 2 个单元格。"""
        cells = i = 0
        while i < len(text):
            if text[i] == "\033":  # ANSI 转义序列：宽度为零，持续到字母终止符
                while i < len(text) and not text[i].isalpha():
                    i += 1
            else:
                cells += 2 if unicodedata.east_asian_width(text[i]) in "WF" else 1
                if cells > width:
                    return f"{text[:i]}\033[0m"  # 重置颜色，避免截断后的颜色溢出
            i += 1
        return text

    def _should_update(self, dt: float, dn: int) -> bool:
        """检查是否应更新显示。"""
        if self.noninteractive:
            return False
        return (self.total is not None and self.n >= self.total) or (dt >= self.mininterval)

    def _display(self, final: bool = False) -> None:
        """显示进度条。"""
        if self.disable or (self.closed and not final):
            return

        current_time = time.time()
        dt = current_time - self.last_print_t
        dn = self.n - self.last_print_n

        if not final and not self._should_update(dt, dn):
            return

        # 计算速率（避免出现异常数值）
        if dt > self.MIN_RATE_CALC_INTERVAL:
            rate = dn / dt if dt else 0.0
            # 对正常数值进行速率平滑，数值过高时使用原始速率
            if rate < self.MAX_SMOOTHED_RATE:
                self.last_rate = self.RATE_SMOOTHING_FACTOR * rate + (1 - self.RATE_SMOOTHING_FACTOR) * self.last_rate
                rate = self.last_rate
        else:
            rate = self.last_rate

        # 完成时使用整体速率
        if self.total and self.n >= self.total:
            overall_elapsed = current_time - self.start_t
            if overall_elapsed > 0:
                rate = self.n / overall_elapsed

        # 更新计数器
        self.last_print_n = self.n
        self.last_print_t = current_time
        elapsed = current_time - self.start_t

        # 剩余时间
        remaining_str = ""
        if self.total and 0 < self.n < self.total and elapsed > 0:
            est_rate = rate or (self.n / elapsed)
            remaining_str = f"<{self._format_time((self.total - self.n) / est_rate)}"

        # 数值和百分比（向下取整，只有真正完成时才显示 100%）
        if self.total:
            percent = int(self.n / self.total * 100)
            n_str = self._format_num(self.n)
            t_str = self._format_num(self.total)
            if self.is_bytes and n_str[-2] == t_str[-2]:  # 仅在后缀相同（如 "5.4/5.4MB"）时折叠后缀
                n_str = n_str.rstrip("KMGTPB")
        else:
            percent = 0.0
            n_str, t_str = self._format_num(self.n), "?"

        elapsed_str = self._format_time(elapsed)
        rate_str = self._format_rate(rate) or (self._format_rate(self.n / elapsed) if elapsed > 0 else "")

        bar = self._generate_bar()

        # 通过 f-string 组合进度字段（有总数和无总数两种形式）
        if self.total:
            if self.is_bytes and self.n >= self.total:
                # 字节处理完成：仅显示最终大小
                fields = f"{percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
            else:
                fields = f"{percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
        else:
            fields = f"{bar} {n_str} {rate_str} {elapsed_str}"

        # 写入输出；仅在真实终端中适配宽度，使重定向日志保留完整行
        try:
            progress_str = f"{self.desc}: {fields}"
            if self.file.isatty():  # description 先生成其单元格，以保留进度字段
                try:  # 测量 self.file 自身终端，而不是 sys.__stdout__
                    width = os.get_terminal_size(self.file.fileno()).columns - 1
                except Exception:  # 没有可用 fileno 的流（io.StringIO、包装后的 stdout）
                    width = shutil.get_terminal_size().columns - 1  # COLUMNS env, else sys.__stdout__
                progress_str = self._fit(f"{self._fit(self.desc, width - len(fields) - 2)}: {fields}", width)
            # 非交互环境避免使用会产生空行的回车符
            self.file.write(progress_str if self.noninteractive else f"\r\033[K{progress_str}")
            self.file.flush()
        except Exception:
            pass

    def update(self, n: int = 1) -> None:
        """将进度更新 n 步。"""
        if not self.disable and not self.closed:
            self.n += n
            self._display()

    def set_description(self, desc: str | None) -> None:
        """设置描述。"""
        self.desc = desc or ""
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs: Any) -> None:
        """设置后缀（追加到描述后）。"""
        if kwargs:
            postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
            self.set_description(f"{base_desc} | {postfix}")

    def close(self) -> None:
        """关闭进度条。"""
        if self.closed:
            return

        self.closed = True

        if not self.disable:
            # 最终显示
            if self.total and self.n >= self.total:
                self.n = self.total
                if self.n != self.last_print_n:  # 如果已经显示 100%，则跳过
                    self._display(final=True)
            else:
                self._display(final=True)

            # 清理
            if self.leave:
                self.file.write("\n")
            else:
                self.file.write("\r\033[K")

            try:
                self.file.flush()
            except Exception:
                pass

    def __enter__(self) -> Self:
        """进入上下文管理器。"""
        return self

    def __exit__(self, *args: object) -> None:
        """退出上下文管理器并关闭进度条。"""
        self.close()

    def __iter__(self) -> Any:
        """遍历被包装的可迭代对象，并更新进度。"""
        if self.iterable is None:
            raise TypeError("'NoneType' object is not iterable")

        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __del__(self) -> None:
        """析构对象并确保完成清理。"""
        try:
            self.close()
        except Exception:
            pass

    def refresh(self) -> None:
        """刷新显示。"""
        if not self.disable:
            self._display()

    def clear(self) -> None:
        """清除进度条。"""
        if not self.disable:
            try:
                self.file.write("\r\033[K")
                self.file.flush()
            except Exception:
                pass

    @staticmethod
    def write(s: str, file: IO[str] | None = None, end: str = "\n") -> None:
        """静态写入方法，不破坏当前进度条。"""
        file = file or sys.stdout
        try:
            file.write(s + end)
            file.flush()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    print("1. Basic progress bar with known total:")
    for i in TQDM(range(3), desc="Known total"):
        time.sleep(0.05)

    print("\n2. Manual updates with known total:")
    pbar = TQDM(total=300, desc="Manual updates", unit="files")
    for i in range(300):
        time.sleep(0.03)
        pbar.update(1)
        if i % 10 == 9:
            pbar.set_description(f"Processing batch {i // 10 + 1}")
    pbar.close()

    print("\n3. Progress bar with unknown total:")
    pbar = TQDM(desc="Unknown total", unit="items")
    for i in range(25):
        time.sleep(0.08)
        pbar.update(1)
        if i % 5 == 4:
            pbar.set_postfix(processed=i + 1, status="OK")
    pbar.close()

    print("\n4. Context manager with unknown total:")
    with TQDM(desc="Processing stream", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for i in range(30):
            time.sleep(0.1)
            pbar.update(1024 * 1024 * i)  # 模拟处理 MB 级数据

    print("\n5. Iterator with unknown length:")

    def data_stream():
        """模拟长度未知的数据流。"""
        import random

        for i in range(random.randint(10, 20)):
            yield f"data_chunk_{i}"

    for chunk in TQDM(data_stream(), desc="Stream processing", unit="chunks"):
        time.sleep(0.1)

    print("\n6. File processing simulation (unknown size):")

    def process_files():
        """模拟处理数量未知的文件。"""
        return [f"file_{i}.txt" for i in range(18)]

    pbar = TQDM(desc="Scanning files", unit="files")
    files = process_files()
    for i, filename in enumerate(files):
        time.sleep(0.06)
        pbar.update(1)
        pbar.set_description(f"Processing {filename}")
    pbar.close()
