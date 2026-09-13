
#ultraliytics的ndjson下载到本地成为yolo格式的方法


import argparse
import asyncio


async def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="将 Ultralytics NDJSON 数据集转换为 YOLO 格式")
    parser.add_argument("--input", default="l55(2).ndjson", help="NDJSON 文件")
    parser.add_argument("--output", default="l55_yolo", help="YOLO 数据集输出目录")
    args = parser.parse_args(argv)
    from ultralytics.data.converter import convert_ndjson_to_yolo

    yaml_path = await convert_ndjson_to_yolo(
        args.input,
        args.output
    )

    print("转换完成:")
    print(yaml_path)


if __name__ == "__main__":
    asyncio.run(main())
