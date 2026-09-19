"""Wrap the Linux PyInstaller directory bundle in an installable Debian package."""

from __future__ import annotations

import argparse
import platform
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "vtools"
DESKTOP_FILE = ROOT / "packaging" / "vtools.desktop"
ICON_FILE = ROOT / "packaging" / "vtools.svg"


def main() -> int:
    argparse.ArgumentParser(description="将 Linux 桌面目录包封装为 .deb 安装包").parse_args()
    if platform.system() != "Linux":
        raise SystemExit(".deb 必须在 Linux 上构建")
    if not (BUNDLE / "vtools").is_file() or not (BUNDLE / "_internal").is_dir():
        raise SystemExit("缺少 dist/vtools 目录包；请先运行 packaging/build_desktop.py")
    if shutil.which("dpkg-deb") is None:
        raise SystemExit("缺少 dpkg-deb；请安装 dpkg-dev")

    version = runpy.run_path(str(ROOT / "vtools_ui" / "__init__.py"))["__version__"]
    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    destination = ROOT / "dist" / f"vtools_{version}_{architecture}.deb"

    with tempfile.TemporaryDirectory(prefix="vtools-deb-") as temporary:
        stage = Path(temporary)
        app_dir = stage / "opt" / "vtools"
        shutil.copytree(BUNDLE, app_dir, symlinks=True)

        applications = stage / "usr" / "share" / "applications"
        applications.mkdir(parents=True)
        shutil.copy2(DESKTOP_FILE, applications / DESKTOP_FILE.name)

        icons = stage / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"
        icons.mkdir(parents=True)
        shutil.copy2(ICON_FILE, icons / ICON_FILE.name)

        metadata = stage / "DEBIAN"
        metadata.mkdir()
        (metadata / "control").write_text(
            f"Package: vtools\n"
            f"Version: {version}\n"
            f"Architecture: {architecture}\n"
            "Maintainer: vtools project\n"
            "Section: science\n"
            "Priority: optional\n"
            "Description: Visual model workbench\n"
            " Desktop workbench for model diagnostics, visualization, benchmarks and conversion.\n",
            encoding="utf-8",
        )
        # Never build directly over a published package. If compression is
        # interrupted, or a replacement happens to be shorter, writing to the
        # final pathname can leave a truncated archive or stale trailing bytes
        # that dpkg-deb tolerates but APT rejects. Build beside the destination
        # and atomically publish only after dpkg-deb exits successfully.
        with tempfile.TemporaryDirectory(prefix=".vtools-deb-build-", dir=destination.parent) as output_dir:
            temporary_package = Path(output_dir) / destination.name
            subprocess.run(
                ["dpkg-deb", "--build", "--root-owner-group", str(stage), str(temporary_package)],
                check=True,
            )
            temporary_package.replace(destination)

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
