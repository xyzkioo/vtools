"""Build the vtools desktop bundle with PyInstaller."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "vtools_ui" / "webapp" / "frontend" / "dist" / "index.html"
SOURCE_DIRS = (
    "config", "model_diagnostics", "model_visualization", "speed_test",
    "model_compression", "transform_tools", "others", "env_test", "vtools_runtime",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".cfg", ".txt", ".toml", ".typed"}
EXCLUDED_DIRS = {
    ".git", ".idea", ".vtools_ui", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "runs", "runs-profile",
    "outputs", "tests", "weights", "build", "dist",
}


def source_datas() -> list[tuple[Path, str]]:
    """Return tracked tool sources and frontend files for the one-folder bundle."""

    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", *SOURCE_DIRS],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    tracked_files = {ROOT / relative for relative in tracked if relative}
    datas: list[tuple[Path, str]] = [(ROOT / "run_tools.py", ".")]
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if (
                path not in tracked_files
                or not path.is_file()
                or any(part in EXCLUDED_DIRS for part in relative.parts[1:])
                or path.suffix.lower() not in SOURCE_SUFFIXES
            ):
                continue
            datas.append((path, relative.parent.as_posix()))
    frontend = ROOT / "vtools_ui" / "webapp" / "frontend" / "dist"
    for path in frontend.rglob("*"):
        if path.is_file():
            datas.append((path, path.relative_to(ROOT).parent.as_posix()))
    return datas


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
    try:
        import PySide6  # noqa: F401
        import webview  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 缺少桌面依赖，请使用已安装 PySide6 和 pywebview 的 Python 环境构建，"
            "例如：/home/xyzkioo/miniconda3/envs/yolo/bin/python packaging/build_desktop.py --clean"
        ) from exc
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
        "--name", "vtools", "--console", "--paths", str(ROOT),
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
    ]
    if args.clean:
        command.append("--clean")
    for source, destination in source_datas():
        command.extend(["--add-data", f"{source}{os.pathsep}{destination}"])
    for module in ("webview", "vtools_runtime", "vtools_runtime.ultralytics"):
        command.extend(["--hidden-import", module])
    for module in (
        "pytest", "_pytest", "expecttest", "tests", "torch", "torchvision",
        "ultralytics", "cv2", "PIL", "matplotlib", "onnxruntime",
        "tensorrt", "cupy",
    ):
        command.extend(["--exclude-module", module])
    command.append(str(ROOT / "vtools_ui" / "__main__.py"))
    result = subprocess.call(command, cwd=ROOT)
    if result == 0:
        _repair_conda_openssl(ROOT / "dist" / "vtools")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
