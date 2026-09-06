# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.utils import LOGGER


def onnx_calibration_reader(dataset, transform_fn, input_name: str = "images", batch: int = 0):
    """根据 Ultralytics 校准数据加载器创建 ONNX Runtime 校准数据读取器。

    ``batch`` 是图的静态批次维度（动态批次模型为 0）：小于导出批次的数据集会产生静态图无法接受的较小批次，
    因此会复制样本，直到批次大小恰好达到 ``batch``。
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _CalibrationReader(CalibrationDataReader):
        def __init__(self):
            """初始化校准数据集迭代。"""
            self.iterator = iter(dataset)

        def get_next(self):
            """返回下一个校准样本；数据耗尽时返回 None。"""
            if (b := next(self.iterator, None)) is None:
                return None
            im = transform_fn(b)
            if batch and im.shape[0] != batch:  # 复制样本，直到达到静态批次维度
                im = np.tile(im, (-(-batch // im.shape[0]), 1, 1, 1))[:batch]
            return {input_name: im}

        def rewind(self):
            """重置迭代器，以便再次执行校准。"""
            self.iterator = iter(dataset)

    return _CalibrationReader()


def onnx_int8_quantize(
    onnx_file,
    output_file,
    dataset,
    transform_fn,
    input_name: str = "images",
    batch: int = 0,
    prefix: str = "",
) -> str:
    """使用 ONNX Runtime 静态量化将 ONNX 模型量化为 INT8。"""
    import onnx
    from onnxruntime.quantization import quantize_static

    # 仅量化带权算子，使检测头解码保持浮点：一个覆盖边界框像素（约 0-640）和类别概率（0-1）的 INT8 标度
    # 会将所有分数舍入为 0。按节点而非算子类型排除，仍可校准所有张量，避免出现
    # 避免 ONNX Runtime 在未校准的注意力 Softmax 上崩溃。
    graph = onnx.load(onnx_file).graph
    exclude = [n.name for n in graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    del graph

    LOGGER.info(f"{prefix} quantizing INT8 with ONNX Runtime...")
    quantize_static(
        onnx_file,
        output_file,
        onnx_calibration_reader(dataset, transform_fn, input_name, batch),
        nodes_to_exclude=exclude,
    )
    return str(output_file)
