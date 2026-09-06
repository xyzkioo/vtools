# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ctypes
import platform
import re
import sys
from pathlib import Path


class CPUInfo:
    """提供跨平台的 CPU 品牌和型号信息。

    查询各平台专用的信息源，获取便于阅读的 CPU 描述，并将其规范化，确保在 macOS、Linux 和 Windows
    上保持一致的显示格式。如果平台专用查询失败，则使用通用平台标识，确保始终返回稳定的字符串。

    方法：
        name: 使用平台专用信息源和可靠的回退机制返回规范化的 CPU 名称。
        _sysctl: 通过 libc 读取 macOS 的 sysctl 字符串。
        _clean: 规范化并美化常见厂商品牌字符串和频率格式。
        __str__: 在字符串上下文中返回规范化的 CPU 名称。

    示例：
        >>> name = CPUInfo.name()
        >>> text = str(CPUInfo())
    """

    @staticmethod
    def _sysctl(key: str) -> str:
        """通过 libc 读取 macOS 的 sysctl 字符串，因为启动 `sysctl` 进程会产生毫秒级开销。"""
        libc = ctypes.CDLL(None)
        libc.sysctlbyname.restype = ctypes.c_int
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        name, n = key.encode(), ctypes.c_size_t(0)
        if libc.sysctlbyname(name, None, ctypes.byref(n), None, 0) or not n.value:  # 尺寸 probe
            return ""
        buf = ctypes.create_string_buffer(n.value)
        if libc.sysctlbyname(name, buf, ctypes.byref(n), None, 0):
            return ""
        return buf.value.decode(errors="ignore").strip()

    @staticmethod
    def name() -> str:
        """从平台专用信息源返回规范化的 CPU 型号字符串。"""
        try:
            if sys.platform == "darwin":
                # 查询 macOS sysctl 获取 CPU 品牌字符串
                s = CPUInfo._sysctl("machdep.cpu.brand_string")
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
                # 解析 /proc/cpuinfo 中的第一个“model name”条目。多核主机会为每个逻辑 CPU 重复完整信息块，
                # 此处采用流式读取，因为只需要第一个条目
                p = Path("/proc/cpuinfo")
                if p.exists():
                    with p.open(errors="ignore") as f:
                        for line in f:
                            if "model name" in line:
                                return CPUInfo._clean(line.split(":", 1)[1])
            elif sys.platform.startswith("win"):
                try:
                    import winreg as wr

                    with wr.OpenKey(wr.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                        val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                        if val:
                            return CPUInfo._clean(val)
                except Exception:
                    # Windows 注册表访问失败时，继续使用通用平台回退信息
                    pass
            # 通用平台回退信息
            s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
            return CPUInfo._clean(s or "Unknown CPU")
        except Exception:
            # 即使发生意外错误，也确保始终返回字符串
            s = platform.processor() or platform.machine() or ""
            return CPUInfo._clean(s or "Unknown CPU")

    @staticmethod
    def _clean(s: str) -> str:
        """规范化并美化原始 CPU 描述字符串。"""
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
        if m := re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
            return f"{m.group(1)} {m.group(2)}"
        if m := re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
            return f"{m.group(1)} {m.group(2)}"
        return s

    def __str__(self) -> str:
        """返回规范化的 CPU 名称。"""
        return self.name()


if __name__ == "__main__":
    print(CPUInfo.name())
