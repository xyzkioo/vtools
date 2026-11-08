
#ultraliytics的ndjson下载到本地成为yolo格式的方法


import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ultralytics 的查找规则放在共享运行时模块里，与其他工具保持一致。
from vtools_runtime.ultralytics import add_repo_to_path, import_error_message  # noqa: E402


def _load_converter(input_path: str):
    """Load Ultralytics from the same checkout discovery path as other tools."""

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
    # 旧版界面会传入下载配置，该功能已移除：参数仅为兼容旧调用而保留，取值不再使用。
    parser.add_argument("--workers", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--auto-retry", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-auto-retry", dest="auto_retry", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--retries", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    convert_ndjson_to_yolo = _load_converter(args.input)

    try:
        yaml_path = await convert_ndjson_to_yolo(args.input, args.output)
    except RuntimeError as exc:
        if "Downloaded" not in str(exc):
            raise
        # 提示走 stdout：异常信息在部分终端下会按本地编码写出而显示为乱码。
        print(
            "\n部分图片下载失败。直接重新启动本工具即可续传："
            "已下载的图片会自动跳过，只需补齐缺失的那几张。",
            flush=True,
        )
        raise

    print("转换完成:")
    print(yaml_path)


if __name__ == "__main__":
    asyncio.run(main())
