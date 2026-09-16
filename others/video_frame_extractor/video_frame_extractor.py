#!/usr/bin/env python3
"""视频解帧工具。

支持：
1. 保存视频中的每一帧。
2. 每隔 N 帧保存一帧（例如 N=10 时保存第 0、10、20... 帧）。
3. 单个视频或目录批量处理，支持常见视频扩展名。

依赖：opencv-python
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".ts"}


@dataclass
class ExtractionResult:
    video: Path
    saved: int
    output_dir: Path
    fps: float
    total_frames: int


def natural_key(path: Path) -> list[object]:
    """按文件名中的数字自然排序。"""
    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式：{input_path.suffix or '无扩展名'}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"输入路径不存在或不是文件/目录：{input_path}")
    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    videos = [path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=natural_key)


def validate_config(every_n: int, start_frame: int, end_frame: int | None, extension: str) -> str:
    if every_n < 1:
        raise ValueError("--every-n 必须是大于等于 1 的整数")
    if start_frame < 0:
        raise ValueError("--start-frame 不能小于 0")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("--end-frame 不能小于 --start-frame")
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("--extension 只支持 jpg、jpeg、png")
    return normalized


def build_output_dir(video: Path, input_path: Path, output_dir: Path | None, multiple: bool) -> Path:
    stem = f"{video.stem}_{video.suffix.lstrip('.')}" if video.suffix else video.stem
    if output_dir is None:
        # 单个视频默认输出到 video_frames；批量时按视频名分目录，避免文件互相覆盖。
        root = video.parent / f"{stem}_frames"
    elif multiple:
        try:
            relative_parent = video.relative_to(input_path).parent if input_path.is_dir() else Path()
        except ValueError:
            relative_parent = Path()
        root = output_dir / relative_parent / stem
    else:
        root = output_dir
    return root


def extract_video(
    video: Path,
    output_dir: Path,
    every_n: int,
    start_frame: int,
    end_frame: int | None,
    extension: str,
    jpeg_quality: int,
    png_compression: int,
    overwrite: bool,
) -> ExtractionResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("未安装 OpenCV，请运行：python -m pip install opencv-python") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频，可能格式不受支持或文件损坏：{video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not overwrite and total_frames > 0:
        last_frame = total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
        collisions = [
            output_dir / f"frame_{index:08d}{extension}"
            for index in range(start_frame, last_frame + 1, every_n)
            if (output_dir / f"frame_{index:08d}{extension}").exists()
        ]
        if collisions:
            capture.release()
            raise FileExistsError(f"输出文件已存在（可加 --overwrite 覆盖）：{collisions[0]}")
    saved = 0
    frame_index = 0
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= start_frame and (frame_index - start_frame) % every_n == 0:
                if end_frame is not None and frame_index > end_frame:
                    break
                output_path = output_dir / f"frame_{frame_index:08d}{extension}"
                if output_path.exists() and not overwrite:
                    raise FileExistsError(f"输出文件已存在（可加 --overwrite 覆盖）：{output_path}")
                params = []
                if extension in {".jpg", ".jpeg"}:
                    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                else:
                    params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
                with tempfile.NamedTemporaryFile(dir=output_dir, suffix=extension, delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    if not cv2.imwrite(str(temporary), frame, params):
                        raise IOError(f"保存帧失败：{output_path}")
                    temporary.replace(output_path)
                finally:
                    temporary.unlink(missing_ok=True)
                saved += 1
            if end_frame is not None and frame_index >= end_frame:
                break
            frame_index += 1
    finally:
        capture.release()
    if saved == 0 and total_frames > start_frame:
        raise IOError(f"视频未能解码出目标帧：{video}")
    return ExtractionResult(video, saved, output_dir, fps, total_frames)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将视频保存为图片帧，支持每帧或每隔 N 帧保存一帧")
    parser.add_argument("--input", "-i", required=True, help="视频文件或视频目录")
    parser.add_argument("--output-dir", "-o", help="输出目录；批量处理时会按视频名建立子目录")
    parser.add_argument("--every-n", type=int, default=8, help="每隔 N 帧保存一帧，默认 8")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧编号，默认 0")
    parser.add_argument("--end-frame", type=int, help="结束帧编号（包含），默认直到视频结束")
    parser.add_argument("--recursive", action="store_true", help="递归搜索输入目录中的视频")
    parser.add_argument("--extension", default=".jpg", help="输出格式：jpg、jpeg 或 png，默认 jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 质量 0-100，默认 95")
    parser.add_argument("--png-compression", type=int, default=3, help="PNG 压缩级别 0-9，默认 3")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出帧")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    config = vars(args)

    try:
        input_path = Path(config["input"]).expanduser().resolve()
        extension = validate_config(config["every_n"], config["start_frame"], config["end_frame"], config["extension"])
        if not 0 <= config["jpeg_quality"] <= 100:
            raise ValueError("jpeg_quality 必须在 0-100 之间")
        if not 0 <= config["png_compression"] <= 9:
            raise ValueError("png_compression 必须在 0-9 之间")
        videos = collect_videos(input_path, config["recursive"])
        if not videos:
            raise ValueError(f"没有找到支持的视频文件：{input_path}")
        output_dir = Path(config["output_dir"]).expanduser().resolve() if config["output_dir"] else None
        multiple = len(videos) > 1
        for video in videos:
            result = extract_video(
                video,
                build_output_dir(video, input_path, output_dir, multiple),
                config["every_n"], config["start_frame"], config["end_frame"], extension,
                config["jpeg_quality"], config["png_compression"], config["overwrite"],
            )
            print(f"完成：{result.video} -> {result.output_dir}，保存 {result.saved} 帧（视频帧数：{result.total_frames}，FPS：{result.fps:.3f}）")
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
