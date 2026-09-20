
#ultraliytics的ndjson下载到本地成为yolo格式的方法


import argparse
import asyncio
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    args = parser.parse_args(argv)
    convert_ndjson_to_yolo = _load_converter(args.input)

    yaml_path = await convert_ndjson_to_yolo(
        args.input,
        args.output
    )

    print("转换完成:")
    print(yaml_path)


if __name__ == "__main__":
    asyncio.run(main())
