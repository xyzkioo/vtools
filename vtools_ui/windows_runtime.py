"""Read the process integrity level without changing Windows security policy."""

from __future__ import annotations

import ctypes


def process_integrity_level() -> int:
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_void_p
    dword = ctypes.c_ulong
    adv.OpenProcessToken.argtypes = [handle, dword, ctypes.POINTER(handle)]
    adv.GetTokenInformation.argtypes = [handle, ctypes.c_int, handle, dword, ctypes.POINTER(dword)]
    adv.GetSidSubAuthorityCount.argtypes = [handle]
    adv.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    adv.GetSidSubAuthority.argtypes = [handle, dword]
    adv.GetSidSubAuthority.restype = ctypes.POINTER(dword)
    kernel.GetCurrentProcess.restype = handle
    kernel.CloseHandle.argtypes = [handle]
    token = handle()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = dword()
        adv.GetTokenInformation(token, 25, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(token, 25, buffer, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(buffer, ctypes.POINTER(handle))[0]
        count = adv.GetSidSubAuthorityCount(sid)[0]
        return adv.GetSidSubAuthority(sid, count - 1)[0]
    finally:
        kernel.CloseHandle(token)


def require_desktop_integrity() -> None:
    if process_integrity_level() < 8192:
        raise RuntimeError(
            "Windows 将当前程序标记为低完整性运行，禁止桌面渲染组件正常启动。"
            "请通过安装器安装；安装器会检查目标目录，不会更改 Windows 安全设置。"
        )
