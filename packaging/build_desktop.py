"""Build the vtools desktop bundle with PyInstaller."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "vtools_ui.spec"
DIST = ROOT / "vtools_ui" / "webapp" / "frontend" / "dist" / "index.html"


def _repair_conda_openssl(output: Path) -> None:
    """Keep conda Python's OpenSSL ABI when PyInstaller sees system libraries."""

    if os.name == "nt":
        return
    bundled = output / "_internal"
    for name in ("libcrypto.so.3", "libssl.so.3"):
        source = Path(sys.prefix) / "lib" / name
        target = bundled / name
        if source.is_file() and target.is_file():
            shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 vtools 跨平台桌面目录包")
    parser.add_argument("--clean", action="store_true", help="删除 PyInstaller 的临时缓存")
    args = parser.parse_args()
    if not DIST.is_file():
        raise SystemExit("缺少前端构建产物，请先在 vtools_ui/webapp/frontend 执行 npm run build")
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise SystemExit("缺少 PyInstaller，请先执行 python -m pip install pyinstaller") from exc
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if args.clean:
        command.append("--clean")
    command.append(str(SPEC))
    result = subprocess.call(command, cwd=ROOT)
    if result == 0:
        _repair_conda_openssl(ROOT / "dist" / "vtools")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
