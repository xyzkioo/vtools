"""生成 vtools 应用图标（.ico）。

图标与 packaging/vtools.svg 使用同一套几何：128x128 画布、28 圆角矩形、白色 V 形，
因此 Windows 可执行文件和安装包不会出现两套视觉。
"""

from __future__ import annotations

import sys
from pathlib import Path


ICON = Path(__file__).resolve().with_name("vtools.ico")
BOX = 128
RADIUS = 28
BACKGROUND = "#2855d6"
FOREGROUND = "#ffffff"
POLYGON = (
    (27, 35), (44, 35), (64, 88), (84, 35), (101, 35), (73, 103), (55, 103),
)
SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 1024


def ensure_icon(path: Path = ICON) -> Path:
    """Create the .ico when it is missing and return its path."""

    if path.is_file():
        return path
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - build environment check
        raise SystemExit("生成图标需要 Pillow，请安装 packaging/requirements-bundled-tools.txt") from exc
    scale = SUPERSAMPLE / BOX
    canvas = Image.new("RGBA", (SUPERSAMPLE, SUPERSAMPLE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (0, 0, SUPERSAMPLE - 1, SUPERSAMPLE - 1), radius=RADIUS * scale, fill=BACKGROUND,
    )
    draw.polygon([(x * scale, y * scale) for x, y in POLYGON], fill=FOREGROUND)
    canvas.save(path, format="ICO", sizes=[(size, size) for size in SIZES])
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ICON
    print(ensure_icon(target))