from __future__ import annotations

import runpy
import os
import sys
import traceback
from pathlib import Path


def _run_bundled_script(argv: list[str]) -> int | None:
    """Run a bundled tool script when the desktop app is frozen."""

    if not getattr(sys, "frozen", False) or len(argv) < 3 or argv[1] != "--run-script":
        return None
    script = Path(argv[2]).resolve()
    if not script.is_file():
        print(f"找不到打包任务脚本：{script}", file=sys.stderr)
        return 2
    # Only pure tool sources belong on the import path; the bundle root also
    # contains extensions compiled for the desktop Python's ABI.
    sources = Path(sys._MEIPASS) / "tool-sources"
    sys.path.insert(0, str(sources))
    sys.argv = [str(script), *argv[3:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc.code), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if getattr(sys, "frozen", False) and "--run-script" in sys.argv:
        # Frozen Python does not honor PYTHONIOENCODING; TaskManager decodes
        # its pipe as UTF-8 on every platform.
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    desktop = sys.platform == "win32" and getattr(sys, "frozen", False) and "--run-script" not in sys.argv
    log_path = None
    try:
        if desktop:
            from vtools_ui.state import desktop_state_dir
            state = desktop_state_dir()
            os.environ["VTOOLS_USER_DATA"] = str(state)
            log_path = state / "logs" / "desktop.log"
            stream = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = stream
            sys.stderr = stream
            from vtools_ui.windows_runtime import require_desktop_integrity
            require_desktop_integrity()
        bundled_code = _run_bundled_script(sys.argv)
        if bundled_code is not None:
            return bundled_code
        from vtools_ui.webapp.server import run
        return run()
    except Exception:
        detail = traceback.format_exc()
        if sys.stderr is not None:
            sys.stderr.write(detail)
        if desktop and "--install-check" not in sys.argv:
            import ctypes
            location = f"详细错误已保存到：\n{log_path}" if log_path and sys.stderr is not None else detail
            ctypes.windll.user32.MessageBoxW(
                None,
                f"工作台启动失败。\n{location}\n\n"
                "请确认安装目录可读写，并保留完整的 _internal 运行库目录。",
                "vtools 启动失败", 0x10,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
