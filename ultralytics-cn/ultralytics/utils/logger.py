# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import logging
import plistlib
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from ultralytics.utils import LINUX, LOGGER, MACOS, RANK, WINDOWS


class ConsoleLogger:
    """捕获控制台输出，并批量流式写入文件、API 或自定义回调。

    捕获 stdout/stderr 输出，通过智能去重和可配置的批量策略进行流式处理。

    属性：
        destination (str | Path | None): 流式输出目标（URL、路径；为 None 时仅使用回调）。
        batch_size (int): 刷新前批量积累的行数（默认 1，立即刷新）。
        flush_interval (float): 自动刷新的间隔秒数（默认 5.0）。
        on_flush (callable | None): 可选回调函数，刷新时接收批量内容。
        active (bool): 当前是否正在捕获控制台输出。

    示例：
        文件日志（立即写入）：
        >>> logger = ConsoleLogger("training.log")
        >>> logger.start_capture()
        >>> print("This will be logged")
        >>> logger.stop_capture()

        API 批量流式传输：
        >>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
        >>> logger.start_capture()

        自定义批量回调：
        >>> def my_handler(content, line_count, chunk_id):
        ...     print(f"Received {line_count} lines")
        >>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
        >>> logger.start_capture()
    """

    def __init__(self, destination=None, batch_size=1, flush_interval=5.0, on_flush=None):
        """初始化控制台日志记录器，并配置可选的批处理策略。

        参数：
            destination (str | Path | None): API 地址（http/https）、本地文件路径，或 None。
            batch_size (int): 刷新前积累的行数（1 表示立即刷新，更大值表示批量刷新）。
            flush_interval (float): 批量模式下两次刷新的最大间隔秒数。
            on_flush (callable | None): 用于自定义处理的回调（content: str, line_count: int, chunk_id: int）。
        """
        if isinstance(destination, str) and destination.startswith("http://"):
            LOGGER.warning("ConsoleLogger destination uses plaintext HTTP; captured logs are sent unencrypted.")
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(("http://", "https://"))
        if destination is not None and not self.is_api:
            self.destination = Path(destination)

        # Batching 配置
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.on_flush = on_flush

        # 控制台捕获状态
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False
        self._log_handler = None  # 保存处理器，便于清理

        # 批处理缓冲区
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.chunk_id = 0

        # 去重状态
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_time = 0.0
        self.progress_interval = 1.0

    def start_capture(self):
        """开始捕获控制台输出，并重定向 stdout/stderr。

        注意：
            在 DDP 训练中，仅 rank 0/-1 启用，以避免重复记录日志。
        """
        if self.active or RANK not in {-1, 0}:
            return

        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

        # 接入 Ultralytics 日志记录器
        try:
            self._log_handler = self._LogHandler(self._queue_log)
            logging.getLogger("ultralytics").addHandler(self._log_handler)
        except Exception:
            pass

        # 为批量模式启动后台刷新线程
        if self.batch_size > 1:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def stop_capture(self):
        """停止捕获控制台输出，并刷新剩余缓冲区。"""
        if not self.active:
            return

        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # 移除日志处理器，避免内存泄漏
        if self._log_handler:
            try:
                logging.getLogger("ultralytics").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

        # 最终刷新
        self._flush_buffer()

    def _queue_log(self, text):
        """对控制台文本去重、添加时间戳并加入队列。"""
        if not self.active:
            return

        current_time = time.time()

        # 处理回车符并去除 ANSI 清行代码（TQDM 交互式写入 "\r\033[K<line>"）
        if "\r" in text:
            text = text.split("\r")[-1]
        text = text.replace("\x1b[K", "")

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        for line in lines:
            if not (line := line.rstrip()):
                continue  # 添加时间戳后，单独的换行符没有有效内容

            # 限制进度条重绘频率，同时始终保留已完成的进度条
            if any(pair in line for pair in ("──", "━─", "━╸", "╸─")):  # 进度条中的未填充单元格
                if current_time - self.last_progress_time < self.progress_interval:
                    continue
                self.last_progress_time = current_time

            # 常规去重
            if line == self.last_line and current_time - self.last_time < 0.1:
                continue

            self.last_line = line
            self.last_time = current_time

            # 必要时添加时间戳
            if not line.startswith("[20"):
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"

            # 加入缓冲区并检查是否需要刷新
            should_flush = False
            with self.buffer_lock:
                self.buffer.append(line)
                if len(self.buffer) >= self.batch_size:
                    should_flush = True

            # 在锁外刷新，避免死锁
            if should_flush:
                self._flush_buffer()

    def _flush_worker(self):
        """定期刷新缓冲区的后台线程。"""
        while self.active:
            time.sleep(self.flush_interval)
            if self.active:
                self._flush_buffer()

    def _flush_buffer(self):
        """将缓冲区中的行刷新到目标位置和/或回调。"""
        with self.buffer_lock:
            if not self.buffer:
                return
            lines = self.buffer.copy()
            self.buffer.clear()
            self.chunk_id += 1
            chunk_id = self.chunk_id  # 在锁内复制，避免竞态

        content = "\n".join(lines)
        line_count = len(lines)

        # 如果提供了自定义回调，则调用它
        if self.on_flush:
            try:
                self.on_flush(content, line_count, chunk_id)
            except Exception:
                pass  # 静默忽略回调错误，避免向 stderr 大量输出

        # 写入目标位置（文件或 API）
        if self.destination is not None:
            self._write_destination(content)

    def _write_destination(self, content):
        """将内容写入文件或 API 目标地址。"""
        try:
            if self.is_api:
                import requests

                payload = {"timestamp": datetime.now().astimezone().isoformat(), "message": content}
                requests.post(str(self.destination), json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(content + "\n")
        except Exception as e:
            print(f"Console logger write error: {e}", file=self.original_stderr)

    class _ConsoleCapture:
        """轻量级 stdout/stderr 捕获器。"""

        __slots__ = ("callback", "original")

        def __init__(self, original, callback):
            """初始化流包装器，将写入内容重定向到回调，同时保留原始流。"""
            self.original = original
            self.callback = callback

        def write(self, text):
            """将文本写入原始流，并转发给捕获回调。"""
            self.original.write(text)
            self.callback(text)

        def flush(self):
            """刷新包装流，使缓冲输出在捕获期间及时传递。"""
            self.original.flush()

        def isatty(self):
            """将 isatty 检查委托给原始流。"""
            return self.original.isatty()

    class _LogHandler(logging.Handler):
        """轻量级日志处理器。"""

        __slots__ = ("callback",)

        def __init__(self, callback):
            """初始化轻量级 logging.Handler，将日志记录转发给指定回调。"""
            super().__init__()
            self.callback = callback

        def emit(self, record):
            """格式化 LogRecord 消息，并转发给捕获回调以统一流式记录日志。"""
            self.callback(self.format(record) + "\n")


class _DriveInfo:
    """解析由本地磁盘提供支持的已挂载存储路径。

    此辅助类将平台相关的磁盘发现逻辑与 SystemLogger 指标采集隔离。它优先使用快速的 psutil 挂载发现，仅在多个可见挂载点需要区分时才回退到操作系统原生命令。

    示例：
        >>> logger = SystemLogger(all_drives=True)
        >>> logger.mounts
        ['/']
    """

    @staticmethod
    def mounts(psutil, all_drives=False):
        """获取需要监控的已挂载路径。"""
        partitions = [p for p in psutil.disk_partitions(all=False) if p.mountpoint]
        if not all_drives:
            return [_DriveInfo._current_mount(partitions)]

        mounts = [
            p.mountpoint for p in partitions if Path(p.mountpoint).is_dir() and "dontbrowse" not in p.opts.split(",")
        ]
        if len(mounts) <= 1:
            return _DriveInfo._sort(mounts) or [_DriveInfo._current_mount(partitions)]

        for getter in (
            _DriveInfo._macos_mounts if MACOS else None,
            _DriveInfo._linux_mounts if LINUX else None,
            _DriveInfo._windows_mounts if WINDOWS else None,
        ):
            if getter:
                try:
                    if platform_mounts := getter(partitions):
                        return _DriveInfo._sort(platform_mounts)
                except (json.JSONDecodeError, OSError, plistlib.InvalidFileException, subprocess.SubprocessError):
                    pass
        return _DriveInfo._sort(mounts)

    @staticmethod
    def _sort(mounts):
        """对已挂载路径排序，将根路径置于首位，并排除 /boot、/boot/efi 等启动或固件分区。"""
        mounts = {m for m in mounts if not (m + "/").startswith(("/boot/", "/efi/"))}
        return sorted(mounts, key=lambda mount: (mount != "/", mount))

    @staticmethod
    def _current_mount(partitions):
        """获取当前工作目录所在的已挂载文件系统。"""
        try:
            cwd = Path.cwd().resolve()
        except OSError:
            return "C:\\" if WINDOWS else "/"
        matches = []
        for partition in partitions:
            try:
                mount = Path(partition.mountpoint).resolve()
            except OSError:
                continue
            if cwd == mount or cwd.is_relative_to(mount):
                matches.append(partition.mountpoint)
        return max(matches, key=len, default=Path.cwd().anchor or "/")

    @staticmethod
    def _macos_mounts(partitions):
        """获取由物理磁盘支持且对用户可见的 macOS 挂载点。"""
        disk_info = plistlib.loads(subprocess.check_output(["diskutil", "list", "-plist", "physical"], timeout=5))
        physical_devices = set(disk_info.get("WholeDisks", []))
        for disk in disk_info.get("AllDisksAndPartitions", []):
            physical_devices.add(disk.get("DeviceIdentifier", ""))
            physical_devices.update(p.get("DeviceIdentifier", "") for p in disk.get("Partitions", []))

        mounts, volume_groups = [], set()
        for partition in partitions:
            if partition.mountpoint != "/" and "dontbrowse" in partition.opts.split(","):
                continue
            info = plistlib.loads(
                subprocess.check_output(
                    ["diskutil", "info", "-plist", partition.mountpoint],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            )
            devices = {info.get("DeviceIdentifier", "")}
            devices.update(s.get("APFSPhysicalStore", "") for s in info.get("APFSPhysicalStores", []))
            if not devices & physical_devices:
                continue
            group = info.get("APFSVolumeGroupID") or info.get("APFSContainerReference") or partition.mountpoint
            if group in volume_groups:
                continue
            volume_groups.add(group)
            mounts.append(partition.mountpoint)
        return mounts

    @staticmethod
    def _linux_mounts(_partitions):
        """获取由物理块设备支持的 Linux 挂载点。"""
        block_info = json.loads(
            subprocess.check_output(
                ["lsblk", "--json", "--output", "NAME,TYPE,MOUNTPOINT,MOUNTPOINTS"], text=True, timeout=5
            )
        )
        mounts = []

        def visit(block, physical=False):
            physical = physical or block.get("type") == "disk"
            if physical:
                values = block.get("mountpoints") or [block.get("mountpoint")]
                if isinstance(values, str):
                    values = [values]
                mounts.extend(m for m in values if isinstance(m, str) and m.startswith("/") and Path(m).is_dir())
            for child in block.get("children", []):
                visit(child, physical)

        for block in block_info.get("blockdevices", []):
            visit(block)
        return mounts

    @staticmethod
    def _windows_mounts(_partitions):
        """获取 Windows 固定本地磁盘挂载点。"""
        output = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object -ExpandProperty DeviceID",
            ],
            text=True,
            timeout=5,
        )
        return [f"{drive}\\" for drive in (line.strip() for line in output.splitlines()) if drive]


class SystemLogger:
    """记录用于训练监控的动态系统指标。

    捕获实时系统指标，包括 CPU、内存、磁盘 I/O、网络 I/O 和 NVIDIA GPU 统计信息，用于训练性能监控与分析。

    属性：
        pynvml: 成功导入时的 NVIDIA pynvml 模块实例，否则为 None。
        nvidia_initialized (bool): NVIDIA GPU 监控是否可用并已初始化。
        net_start: 用于计算累计使用量的初始网络 I/O 计数器。
        disk_start: 用于计算累计使用量的初始磁盘 I/O 计数器。

    示例：
        基本用法（单个磁盘）：
        >>> logger = SystemLogger()
        >>> metrics = logger.get_metrics()
        >>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
        >>> for disk in metrics["disk"]:
        ...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")

        监控所有磁盘：
        >>> logger = SystemLogger(all_drives=True)
        >>> metrics = logger.get_metrics()
        >>> for disk in metrics["disk"]:
        ...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")

        集成到训练循环：
        >>> system_logger = SystemLogger()
        >>> for epoch in range(epochs):
        ...     # Training code here
        ...     metrics = system_logger.get_metrics()
        ...     # Log to database/file
    """

    def __init__(self, all_drives=False):
        """初始化系统日志记录器。

        参数：
            all_drives (bool): 为 True 时监控所有已挂载磁盘；为 False 时仅监控当前磁盘。
        """
        import psutil  # scoped as slow import

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()
        self.mounts = _DriveInfo.mounts(psutil, all_drives)

        # 用于计算速率
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        """使用 pynvml 初始化 NVIDIA GPU 监控。"""
        if MACOS:
            return False

        try:
            import pynvml  # scoped as slow import

            self.pynvml = pynvml
            pynvml.nvmlInit()
            return True
        except Exception as e:
            import torch

            if torch.cuda.is_available():
                LOGGER.warning(f"SystemLogger NVML init failed: {e}")
            return False

    def get_metrics(self, rates=False):
        """获取当前系统指标，包括 CPU、内存、磁盘、网络和 GPU 使用情况。

        收集完整的系统指标，包括 CPU 使用率、内存使用率、磁盘使用量、磁盘 I/O 统计、网络 I/O 统计和 GPU 指标（如果可用）。

        示例输出（rates=False，默认）：
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
            "disk_io": {"read_mb": 156.7, "write_mb": 89.3},
            "network": {"recv_mb": 157.2, "sent_mb": 89.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
                "1": {"usage": 94.1, "memory": 82.7, "temp": 70, "power": 278},
            },
        }
        ```

        示例输出（rates=True）：
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
            "disk_io": {"read_mbs": 12.5, "write_mbs": 8.3},
            "network": {"recv_mbs": 5.2, "sent_mbs": 1.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
            },
        }
        ```

        参数：
            rates (bool): 为 True 时返回磁盘/网络 MB/s 速率，而不是累计 MB 数值。

        返回：
            (dict): 包含 cpu、ram、disk、network 和 gpus 键的指标字典。

        示例：
            >>> logger = SystemLogger()
            >>> logger.get_metrics()["cpu"]  # CPU 百分比
            >>> logger.get_metrics(rates=True)["network"]["recv_mbs"]  # MB/s 下载速率
        """
        import psutil  # scoped as slow import

        net = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        now = time.time()

        # 计算距上次调用的时间间隔
        elapsed = max(0.1, now - self._prev_time)  # 避免除零

        if rates:
            disk_io_metrics = {
                "read_mbs": round(max(0, (disk_io.read_bytes - self._prev_disk.read_bytes) / 1e6 / elapsed), 3),
                "write_mbs": round(max(0, (disk_io.write_bytes - self._prev_disk.write_bytes) / 1e6 / elapsed), 3),
            }
        else:
            disk_io_metrics = {
                "read_mb": round((disk_io.read_bytes - self.disk_start.read_bytes) / 1e6, 3),
                "write_mb": round((disk_io.write_bytes - self.disk_start.write_bytes) / 1e6, 3),
            }

        disks = []
        for mounts in (self.mounts, ["C:\\" if WINDOWS else "/"]):
            for mount in mounts:
                try:
                    usage = shutil.disk_usage(mount)
                    disks.append(
                        {
                            "mount": mount,
                            "used_gb": round(usage.used / 1e9, 3),
                            "total_gb": round(usage.total / 1e9, 3),
                        }
                    )
                except (PermissionError, OSError):
                    continue  # 跳过无法访问的磁盘
            if disks:
                break

        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "disk": disks,
            "disk_io": disk_io_metrics,
            "gpus": {},
        }

        if rates:
            metrics["network"] = {
                "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / 1e6 / elapsed), 3),
                "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / 1e6 / elapsed), 3),
            }
        else:
            metrics["network"] = {
                "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / 1e6, 3),
                "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / 1e6, 3),
            }

        # 始终更新上一次的值，以便下次准确计算速率
        self._prev_net = net
        self._prev_disk = disk_io
        self._prev_time = now

        # 添加 GPU 指标（仅 NVIDIA）
        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())

        return metrics

    def _get_nvidia_metrics(self):
        """获取 NVIDIA GPU 指标，包括利用率、显存、温度和功耗。"""
        gpus = {}
        if not self.nvidia_initialized or not self.pynvml:
            return gpus
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000

                gpus[str(i)] = {
                    "usage": round(util.gpu, 3),
                    "memory": round((memory.used / memory.total) * 100, 3),
                    "temp": temp,
                    "power": power,
                }
        except Exception:
            pass
        return gpus


if __name__ == "__main__":
    print("SystemLogger Real-time Metrics Monitor")
    print("Press Ctrl+C to stop\n")

    logger = SystemLogger(all_drives=True)

    try:
        while True:
            metrics = logger.get_metrics()

            # 清屏（适用于大多数终端）
            print("\033[H\033[J", end="", flush=True)

            # 显示系统指标
            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"Net Recv: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"Net Sent: {metrics['network']['sent_mb']:9.1f} MB")

            # 显示磁盘指标
            print("\nDisk Metrics:")
            for disk in metrics["disk"]:
                print(f"  {disk['mount']}: {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB")

            # 如果可用，显示 GPU 指标
            if metrics["gpus"]:
                print("\nGPU Metrics:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | "
                        f"Mem: {gpu_data['memory']:5.1f}% | "
                        f"Temp: {gpu_data['temp']:2}°C | "
                        f"Power: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: No NVIDIA GPUs detected")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
