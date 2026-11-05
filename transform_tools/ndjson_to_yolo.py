
#ultraliytics的ndjson下载到本地成为yolo格式的方法


import argparse
import asyncio
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_converter(input_path: str):
    """Load Ultralytics from the same checkout discovery path as other tools."""

    from vtools_runtime.ultralytics import add_repo_to_path, import_error_message

    source = Path(input_path).expanduser().resolve()
    repo = add_repo_to_path({"project": {"root": str(ROOT)}}, ROOT, (source,))
    try:
        from ultralytics.data.converter import convert_ndjson_to_yolo
    except ModuleNotFoundError as exc:
        if exc.name == "ultralytics":
            raise RuntimeError(import_error_message(repo)) from exc
        raise
    return convert_ndjson_to_yolo


async def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="将 Ultralytics NDJSON 数据集转换为 YOLO 格式")
    parser.add_argument("--input", default="l55(2).ndjson", help="NDJSON 文件")
    parser.add_argument("--output", default="l55_yolo", help="YOLO 数据集输出目录")
    parser.add_argument("--workers", type=int, default=4, help="下载并发数，1–128，默认 4")
    parser.add_argument("--auto-retry", action=argparse.BooleanOptionalAction, default=True, help="下载失败后自动重试")
    parser.add_argument("--retries", type=int, default=4, help="首次失败后的重试次数，0–20，默认 4")
    parser.add_argument("--timeout", type=int, default=120, help="单次下载超时秒数，1–3600，默认 120")
    args = parser.parse_args(argv)
    for name, low, high in (("workers", 1, 128), ("retries", 0, 20), ("timeout", 1, 3600)):
        if not low <= getattr(args, name) <= high:
            parser.error(f"--{name} 必须在 {low}–{high} 之间")
    convert_ndjson_to_yolo = _load_converter(args.input)

    if "download_workers" not in inspect.signature(convert_ndjson_to_yolo).parameters:
        raise RuntimeError("当前 Ultralytics 不支持下载配置，请使用已更新的 ultralytics-cn Fork。")
    print(f"下载配置：并发 {args.workers}，失败后重试 {args.retries if args.auto_retry else 0} 次，超时 {args.timeout} 秒", flush=True)
    yaml_path = await convert_ndjson_to_yolo(
        args.input,
        args.output,
        download_workers=args.workers,
        download_retries=args.retries if args.auto_retry else 0,
        download_timeout=args.timeout,
        retry_delay=2,
    )

    print("转换完成:")
    print(yaml_path)


if __name__ == "__main__":
    asyncio.run(main())
