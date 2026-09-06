# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements


def onnx2deepx(
    onnx_file: str | Path,
    imgsz: tuple[int, int],
    dataset,
    metadata: dict | None = None,
    optimize: bool = False,
    prefix: str = "",
) -> Path:
    """使用 DEEPX DX-Compiler 将 ONNX 模型转换为 DEEPX 格式。

    参数：
        onnx_file (str | Path): 输入 ONNX 模型路径。
        imgsz (tuple[int, int]): 导出图像尺寸，格式为 ``(高度, 宽度)``。
        dataset (DataLoader): 用于构建 DEEPX 配置的校准数据加载器。
        metadata (dict | None, 可选): 要保存为 YAML 的元数据，默认为 None。
        optimize (bool, 可选): 为 True 时启用更高级的编译器优化，可降低推理延迟但会增加编译时间，默认为 False。
        prefix (str, 可选): 日志前缀，默认为 ""。

    返回：
        (Path): 导出的 DEEPX 模型目录路径。
    """
    try:
        import dx_com
    except ImportError:
        check_requirements("dx_com", cmds="-f https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html")
        import dx_com

    LOGGER.info(f"\n{prefix} starting export with DEEPX...")

    onnx_file = Path(onnx_file)
    export_path = onnx_file.parent / f"{onnx_file.stem}_deepx_model"
    export_path.mkdir(exist_ok=True)
    config_path = export_path / "config.json"

    config = {
        "inputs": {"images": [1, 3, imgsz[0], imgsz[1]]},
        "calibration_num": 100,  # 校准期间使用的步骤数量
        "calibration_method": "ema",  # 量化期间使用的校准方法
        "default_loader": {
            # JSON 需要字符串；ClassificationDataset 将图像目录存储在“root”中，而不是“img_path”。
            "dataset_path": str(getattr(dataset.dataset, "img_path", None) or dataset.dataset.root),
            "file_extensions": [val for x in ["jpeg", "jpg", "png"] for val in (x.lower(), x.upper())],
            "preprocessings": [
                {"resize": {"mode": "pad", "size": imgsz[0], "pad_location": "edge", "pad_value": [114, 114, 114]}},
                {"div": {"x": 255.0}},
                {"convertColor": {"form": "BGR2RGB"}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }

    with open(config_path, "w") as file:
        json.dump(config, file)

    dx_com.compile(model=str(onnx_file), output_dir=str(export_path), config=str(config_path), opt_level=int(optimize))

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)

    return export_path
