"""Build the vtools Windows installer (.exe) with Inno Setup.

前置条件：先在仓库根目录用打包环境执行 `python packaging/build_desktop.py --clean`，
生成整个 `dist/vtools/` 目录包；本脚本只负责把该目录包封装成安装程序。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = Path(__file__).resolve().parent
ISS = PACKAGING / "vtools.iss"
BUNDLE = ROOT / "dist" / "vtools"
BUNDLE_EXE = BUNDLE / "vtools.exe"
BUILD = ROOT / "build"
OUTPUT = ROOT / "dist"

sys.path.insert(0, str(PACKAGING))
from vtools_icon import ICON, ensure_icon  # noqa: E402  图标与 vtools.svg 同一几何


def read_version() -> str:
    """Return the workbench version, which the installer must match."""

    text = (ROOT / "vtools_ui" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("无法从 vtools_ui/__init__.py 读取 __version__")
    return match.group(1)


def find_iscc(explicit: str | None) -> Path:
    """Locate ISCC.exe from Inno Setup 6, which has no admin-free PATH entry."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    from_env = os.environ.get("ISCC")
    if from_env:
        candidates.append(Path(from_env))
    on_path = shutil.which("ISCC") or shutil.which("iscc")
    if on_path:
        candidates.append(Path(on_path))
    for base in (
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ):
        if base:
            candidates.append(Path(base) / "Programs" / "Inno Setup 6" / "ISCC.exe")
            candidates.append(Path(base) / "Inno Setup 6" / "ISCC.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "未找到 Inno Setup 6 的 ISCC.exe。请安装 Inno Setup 6"
        "（winget install --id JRSoftware.InnoSetup --scope user），或用 --iscc 指定路径"
    )


def staged_script(version: str, icon: Path, chinese_isl: Path | None) -> Path:
    """Copy the .iss with a UTF-8 BOM, which Inno Setup needs for Chinese text."""

    BUILD.mkdir(parents=True, exist_ok=True)
    staged = BUILD / f"vtools-{version}-setup.iss"
    defines = [
        f'#define AppVersion "{version}"',
        f'#define SourceDir "{BUNDLE}"',
        f'#define OutputDir "{OUTPUT}"',
        f'#define IconFile "{icon}"',
    ]
    if chinese_isl:
        defines.append(f'#define ChineseIsl "{chinese_isl}"')
    text = "\n".join(defines) + "\n" + ISS.read_text(encoding="utf-8")
    staged.write_text(text, encoding="utf-8-sig")
    return staged


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 vtools Windows .exe 安装包")
    parser.add_argument("--iscc", help="ISCC.exe 路径，默认自动查找 Inno Setup 6")
    parser.add_argument(
        "--chinese-language",
        help="Inno Setup 中文语言文件（.isl）路径；不提供时安装向导使用英文界面",
    )
    args = parser.parse_args()
    if not BUNDLE_EXE.is_file():
        raise SystemExit(
            f"缺少目录包入口 {BUNDLE_EXE}；请先执行 packaging/build_desktop.py --clean"
        )
    if not (BUNDLE / "_internal").is_dir():
        raise SystemExit(f"{BUNDLE} 看起来不是完整的 PyInstaller 目录包（缺少 _internal/）")
    if not (BUNDLE / "vtools-worker.exe").is_file():
        raise SystemExit("缺少 vtools-worker.exe，请重新构建完整桌面目录包")
    sources = BUNDLE / "_internal" / "tool-sources"
    if not (sources / "run_tools.py").is_file():
        raise SystemExit("缺少隔离的 tool-sources 目录，请重新构建桌面包")
    if any(path.suffix.lower() in {".pyd", ".dll", ".so", ".dylib"} for path in sources.rglob("*")):
        raise SystemExit("tool-sources 中混入二进制运行库，停止构建以避免外部 Python ABI 冲突")
    version = read_version()
    icon = ensure_icon(ICON)
    chinese_isl = Path(args.chinese_language).resolve() if args.chinese_language else None
    if chinese_isl and not chinese_isl.is_file():
        raise SystemExit(f"找不到中文语言文件：{chinese_isl}")
    iscc = find_iscc(args.iscc)
    staged = staged_script(version, icon, chinese_isl)
    command = [str(iscc)]
    if chinese_isl:
        command.append(f"/DChineseIsl={chinese_isl}")
    command.extend(
        [
            f"/DAppVersion={version}",
            f"/DSourceDir={BUNDLE}",
            f"/DOutputDir={OUTPUT}",
            f"/DIconFile={icon}",
            str(staged),
        ]
    )
    print("执行：" + subprocess.list2cmdline(command), flush=True)
    result = subprocess.call(command, cwd=ROOT)
    if result != 0:
        return result
    installer = OUTPUT / f"vtools-{version}-windows-x64-setup.exe"
    if not installer.is_file():
        raise SystemExit(f"编译结束但没有找到安装包：{installer}")
    size = installer.stat().st_size
    print(f"安装包：{installer}")
    print(f"版本：{version}")
    print(f"大小：{size} 字节（{size / 1024 / 1024:.1f} MiB）")
    print(f"SHA-256：{sha256_of(installer)}")
    if not chinese_isl:
        print("提示：未提供中文语言文件，安装向导界面为英文")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
