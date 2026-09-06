# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import IS_COLAB, LOGGER, YAML


def _check_rknn_return(ret, name: str):
    """RKNN API 调用失败时抛出 RuntimeError。"""
    if ret not in {0, None}:
        raise RuntimeError(f"RKNN {name} failed with return code {ret}.")


def onnx2rknn(
    onnx_file: str,
    output_dir: Path | str,
    name: str = "rk3588",
    quantize: int | str | None = None,
    dataset: Path | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
    batch: int = 1,
) -> str:
    """将 ONNX 模型导出为 Rockchip NPU 使用的 RKNN 格式，并支持可选 INT8 量化。

    参数：
        onnx_file (str): 源 ONNX 文件路径（已导出，opset <=19）。
        output_dir (Path | str): 保存导出 RKNN 模型的目录。
        name (str): 目标平台名称（例如 ``"rk3588"``）。
        quantize (int | str | None): 精度方案，8 表示 INT8，16 或 None 表示浮点构建。
        dataset (Path | str | None): 生成的 RKNN Toolkit 校准图像列表文件路径，``quantize=8`` 时必需。
            用户应将 YOLO 数据集 YAML 传给 ``export(数据=...)``；``export_rknn()`` 会将其转换为内部图像路径列表。
        metadata (dict | None): 保存为 ``metadata.yaml`` 的元数据。
        prefix (str): 日志消息前缀。
        batch (int): 加载批次为 1 的 ONNX 模型后，由 RKNN Toolkit 应用的推理批次大小。

    返回：
        (str): 导出的 ``_rknn_model`` 目录路径。
    """
    use_int8 = quantize == 8
    if name in {"rv1103", "rv1106", "rv1103b", "rv1106b"} and not use_int8:
        raise ValueError(
            f"Rockchip target '{name}' requires quantize=8. Use a target that supports floating-point builds "
            f"(e.g. rk2118, rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b) or export with quantize=8."
        )
    if use_int8:
        if not dataset:
            raise ValueError("RKNN INT8 export requires a generated calibration image-list file.")
        dataset = Path(dataset)
        if not dataset.is_file():
            raise ValueError(f"Generated RKNN INT8 calibration image-list file not found: {dataset}")

    from ultralytics.utils.checks import check_requirements

    LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")
    # setuptools<82 会保留 pkg_resources，rknn-toolkit2 依赖该模块，而 setuptools 82 已将其移除。
    check_requirements(["rknn-toolkit2>=2.3.2", "setuptools<82"])

    if IS_COLAB:
        # 防止“exit”关闭 Notebook：https://github.com/airockchip/rknn-toolkit2/issues/259
        import builtins

        builtins.exit = lambda: None

    from rknn.api import RKNN

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)
    config = {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "target_platform": name}
    _check_rknn_return(rknn.config(**config), "config")
    _check_rknn_return(rknn.load_onnx(model=onnx_file), "load_onnx")
    build_kwargs = {"do_quantization": use_int8}
    if use_int8:
        build_kwargs["dataset"] = str(dataset)
    _check_rknn_return(rknn.build(**build_kwargs, rknn_batch_size=batch), "build")
    _check_rknn_return(rknn.export_rknn(str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn")), "export_rknn")
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
