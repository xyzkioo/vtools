
#ultraliytics的ndjson下载到本地成为yolo格式的方法


import asyncio
from ultralytics.data.converter import convert_ndjson_to_yolo


async def main():

    yaml_path = await convert_ndjson_to_yolo(
        "l55(2).ndjson",
        "l55_yolo"
    )

    print("转换完成:")
    print(yaml_path)


if __name__ == "__main__":
    asyncio.run(main())
