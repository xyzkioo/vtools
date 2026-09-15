#!/usr/bin/env python3
"""图片批量改分辨率工具。

支持单张图片或目录；默认保持宽高比并将图片缩放到指定框内。
依赖：Pillow（python -m pip install Pillow）
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


PYCHARM_CONFIG: dict[str, Any] = {
    "input": r"",       # 图片文件或目录
    "output_dir": r"",                # 留空时输出到 input 同级的 resized 目录
    "width": 1024,
    "height": 1024,
    "mode": "fit",                   # fit=保持比例填充框；stretch=强制拉伸；crop=保持比例裁剪
    "recursive": False,
    "overwrite": False,
    "output_format": "same",         # same、jpg、png、webp
    "jpeg_quality": 95,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> list[object]:
    import re
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片格式：{input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted((p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)


def resize_image(image: Any, size: tuple[int, int], mode: str) -> Any:
    from PIL import Image, ImageOps
    if mode == "stretch":
        return image.resize(size, Image.Resampling.LANCZOS)
    if mode == "crop":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def save_image(image: Any, output: Path, output_format: str, jpeg_quality: int) -> None:
    from PIL import Image
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.upper() if output_format != "same" else (image.format or output.suffix.lstrip(".")).upper()
    if fmt in {"JPG", "JPEG"}:
        fmt = "JPEG"
        if image.mode in {"RGBA", "LA", "P"}:
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
            else:
                rgba = image.convert("RGBA")
                background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        image.save(output, format=fmt, quality=jpeg_quality, optimize=True)
    else:
        image.save(output, format=fmt)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量调整图片分辨率")
    parser.add_argument("--input", "-i", help="图片文件或目录")
    parser.add_argument("--output-dir", "-o", help="输出目录")
    parser.add_argument("--width", type=int, help="目标宽度")
    parser.add_argument("--height", type=int, help="目标高度")
    parser.add_argument("--mode", choices=["fit", "stretch", "crop"], help="fit 保持比例，stretch 拉伸，crop 裁剪")
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    parser.add_argument("--output-format", choices=["same", "jpg", "png", "webp"], help="输出格式")
    parser.add_argument("--jpeg-quality", type=int, help="JPEG 质量，0-100")
    return parser


def main() -> int:
    config = dict(PYCHARM_CONFIG)
    args = make_parser().parse_args()
    for key, value in vars(args).items():
        if value is not None and value is not False:
            config[key] = value
    try:
        from PIL import Image, ImageOps
        input_path = Path(config["input"]).expanduser().resolve()
        width, height = int(config["width"]), int(config["height"])
        if width < 1 or height < 1:
            raise ValueError("width 和 height 必须大于 0")
        if not 0 <= int(config["jpeg_quality"]) <= 100:
            raise ValueError("jpeg_quality 必须在 0-100 之间")
        images = collect_images(input_path, bool(config["recursive"]))
        if not images:
            print(f"没有在 {input_path} 找到图片。")
            return 0
        output_dir = Path(config["output_dir"]).expanduser().resolve() if config["output_dir"] else (
            input_path.parent / "resized" if input_path.is_file() else input_path / "resized"
        )
        if output_dir == input_path or output_dir in input_path.parents:
            raise ValueError("输出目录不能是输入目录本身或其父目录")
        if input_path.is_dir():
            images = [source for source in images if output_dir not in source.parents]
        output_format = str(config["output_format"]).lower()
        plan = []
        targets = set()
        for source in images:
            relative = Path(source.name) if input_path.is_file() else source.relative_to(input_path)
            suffix = "." + output_format if output_format != "same" else source.suffix
            target = output_dir / relative.with_suffix(suffix)
            if target in targets:
                raise ValueError(f"多张输入图片会写入同一输出文件：{target}；请保留原格式或先调整文件名")
            targets.add(target)
            if target.exists() and not config["overwrite"]:
                raise FileExistsError(f"输出文件已存在（可加 --overwrite）：{target}")
            plan.append((source, target))
        saved = 0
        for source, target in plan:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                resized = resize_image(image, (width, height), str(config["mode"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    save_image(resized, temporary, output_format, int(config["jpeg_quality"]))
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            saved += 1
            print(f"完成：{source} -> {target}（{resized.width}x{resized.height}）")
        print(f"共处理 {saved} 张图片，输出目录：{output_dir}")
        return 0
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
