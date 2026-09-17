from __future__ import annotations

import runpy
import sys
from pathlib import Path

from vtools_ui.webapp.server import run


def _run_bundled_script(argv: list[str]) -> int | None:
    """Run a bundled tool script when the desktop app is frozen."""

    if not getattr(sys, "frozen", False) or len(argv) < 3 or argv[1] != "--run-script":
        return None
    script = Path(argv[2]).resolve()
    if not script.is_file():
        print(f"找不到打包任务脚本：{script}", file=sys.stderr)
        return 2
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


if __name__ == "__main__":
    bundled_code = _run_bundled_script(sys.argv)
    raise SystemExit(run() if bundled_code is None else bundled_code)
