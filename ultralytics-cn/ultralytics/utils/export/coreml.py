# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules import Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class IOSDetectModel(nn.Module):
    """包装 Ultralytics YOLO 模型，以便导出为 Apple iOS CoreML。"""

    def __init__(self, model: nn.Module, im: torch.Tensor, mlprogram: bool = True):
        """使用 YOLO 模型和示例图像初始化 IOSDetectModel 类。

        参数：
            model (nn.Module): 要封装的 YOLO 模型。
            im (torch.Tensor): 用于跟踪的示例输入张量，形状为 (B, C, H, W)。
            mlprogram (bool): 是否导出为 MLProgram 格式。
        """
        super().__init__()
        _, _, h, w = im.shape  # 批次、通道、高度、宽度
        self.model = model
        self.nc = len(model.names)  # 类别数量
        self.mlprogram = mlprogram
        if w == h:
            self.normalize = 1.0 / w  # 标量
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],  # 广播使用（速度较慢，但模型更小）
                device=next(model.parameters()).device,
            )

    def forward(self, x: torch.Tensor):
        """使用依赖输入尺寸的因子归一化目标检测模型的预测结果。"""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # 将类别长度填充到 80 的倍数
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)
        return cls, xywh * self.normalize


def pipeline_coreml(
    model: Any,
    output_shape: tuple[int, ...],
    metadata: dict,
    mlmodel: bool = False,
    iou: float = 0.45,
    conf: float = 0.25,
    agnostic_nms: bool = False,
    weights_dir: Path | str | None = None,
    prefix: str = "",
):
    """为 YOLO 检测模型创建带 NMS 的 CoreML 流水线。

    参数：
        模型: CoreML 模型.
        output_shape (tuple[int, ...]): 导出器返回的输出形状。
        metadata (dict): 模型元数据。
        mlmodel (bool): 模型是否为 MLModel（而不是 MLProgram）。
        iou (float): NMS 使用的 IoU 阈值。
        conf (float): NMS 使用的置信度阈值。
        agnostic_nms (bool): 是否使用类别无关 NMS。
        weights_dir (Path | str | None): MLProgram 模型的权重目录。
        prefix (str): 日志消息前缀。

    返回：
        CoreML pipeline 模型.
    """
    import coremltools as ct

    LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")

    spec = model.get_spec()
    outs = list(iter(spec.description.output))
    if mlmodel:  # mlmodel doesn't infer shapes automatically
        outs[0].type.multiArrayType.shape[:] = output_shape[2], output_shape[1] - 4
        outs[1].type.multiArrayType.shape[:] = output_shape[2], 4

    names = metadata["names"]
    nx = spec.description.input[0].type.imageType.width
    ny = spec.description.input[0].type.imageType.height
    nc = outs[0].type.multiArrayType.shape[-1]
    if len(names) != nc:  # 临时修复 MLProgram NMS 缺陷 https://github.com/ultralytics/ultralytics/issues/22309
        names = {**names, **{i: str(i) for i in range(len(names), nc)}}

    model = ct.models.MLModel(spec, weights_dir=weights_dir, skip_model_load=True)

    # 创建 NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = spec.specificationVersion
    for i in range(len(outs)):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    output_names = ["confidence", "coordinates"]
    for i, name in enumerate(output_names):
        nms_spec.description.output[i].name = name

    for i, out in enumerate(outs):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = out.type.multiArrayType.shape[-1]
        ma_type.shapeRange.sizeRanges[1].upperBound = out.type.multiArrayType.shape[-1]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
    nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
    nms.confidenceOutputFeatureName = output_names[0]
    nms.coordinatesOutputFeatureName = output_names[1]
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = iou
    nms.confidenceThreshold = conf
    nms.pickTop.perClass = not agnostic_nms
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec, skip_model_load=True)

    # 将模型组合为流水线
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=output_names,
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # 修正数据类型
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # 更新元数据
    pipeline.spec.specificationVersion = spec.specificationVersion
    pipeline.spec.description.metadata.CopyFrom(spec.description.metadata)
    pipeline.spec.description.metadata.userDefined.update(
        {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
    )

    # 保存 模型
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir, skip_model_load=True)
    model.input_description["image"] = "Input image"
    model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
    LOGGER.info(f"{prefix} pipeline success")
    return model


def _coreml_gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """沿 dim 1 选择 x 中索引为 (batch, k) 的行，并通过 MIL 保持索引为 int32。"""
    return x[torch.arange(x.shape[0])[..., None], index]


def torch2coreml(
    model: nn.Module,
    inputs: list,
    im: torch.Tensor,
    classifier_names: list[str] | None,
    output_file: Path | str | None = None,
    mlmodel: bool = False,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> Any:
    """将 PyTorch 模型导出为 CoreML ``.mlpackage`` 或 ``.mlmodel`` 格式。

    参数：
        model (nn.Module): 要导出的 PyTorch 模型。
        inputs (列表): 模型的 CoreML 输入描述。
        im (torch.Tensor): 用于跟踪的示例输入张量。
        classifier_names (列表[str] | None): 分类器配置使用的类别名称；非分类器时为 None。
        output_file (Path | str | None): 输出文件路径；为 None 时跳过保存。
        mlmodel (bool): 是否导出为 ``.mlmodel``（神经网络），而不是 ``.mlpackage``（ML 程序）。
        quantize (int | str | None): 量化方案，例如 16 表示 FP16，8/``"w8a16"`` 表示 INT8 权重。
        metadata (dict | None): 要嵌入 CoreML 模型的元数据。
        prefix (str): 日志消息前缀。

    返回：
        (ct.models.MLModel): 转换后的 CoreML 模型。
    """
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    for m in model.modules():  # MIL 类型会先将 int64 gather 索引转换为 fp32，随后拒绝这些索引
        if isinstance(m, Detect):
            m._gather = types.MethodType(_coreml_gather, m)
    ts = torch.jit.trace(model.eval(), im, strict=False)  # TorchScript 模型
    fp16 = quantize == 16
    weight_int8 = quantize in {8, "w8a16"}

    # 根据 Apple 文档，最好省略 minimum_deployment 目标，让系统根据模型转换和输出类型在内部设置。
    # 将 minimum_deployment_target 设置为 >= iOS16 时，必须设置 compute_precision=ct.precision.FLOAT32。
    # iOS16 增强了对 FP16 的支持，但 CoreML 的 NMS 规范都不接受 FP16 作为输入。
    convert_kwargs = {
        "inputs": inputs,
        "classifier_config": ct.ClassifierConfig(classifier_names) if classifier_names else None,
        "convert_to": "neuralnetwork" if mlmodel else "mlprogram",
        "skip_model_load": True,
    }
    if not mlmodel:
        # ML Program 转换默认使用 FP16。除非请求 FP16/INT8，否则固定使用 FP32。
        from ultralytics.nn.modules.head import RTDETRDecoder

        if not (fp16 or weight_int8):
            convert_kwargs["compute_precision"] = ct.precision.FLOAT32
        elif any(isinstance(m, RTDETRDecoder) for m in model.modules()):
            # RT-DETR 解码器类别 logits 和可变形采样索引在 fp16 下会漂移；将这些算子类型固定为 fp32。
            fp32_ops = {"linear", "gather", "gather_nd", "gather_along_axis"}
            convert_kwargs["compute_precision"] = ct.transform.FP16ComputePrecision(
                op_selector=lambda op: op.op_type not in fp32_ops
            )
    ct_model = ct.convert(ts, **convert_kwargs)
    bits, mode = (8, "kmeans") if weight_int8 else (16, "linear") if fp16 else (32, None)
    if bits < 32:
        if "kmeans" in mode:
            check_requirements("scikit-learn")  # k-means 量化需要 scikit-learn 软件包
        if mlmodel:
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        elif bits == 8:  # mlprogram 已经量化为 FP16
            import coremltools.optimize.coreml as cto

            op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
            config = cto.OptimizationConfig(global_config=op_config)
            ct_model = cto.palettize_weights(ct_model, config=config)

    m = dict(metadata or {})  # 复制以避免修改原始元数据
    ct_model.short_description = m.pop("description", "")
    ct_model.author = m.pop("author", "")
    ct_model.license = m.pop("license", "")
    ct_model.version = m.pop("version", "")
    ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})

    if output_file is not None:
        try:
            ct_model.save(str(output_file))  # 保存 *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            output_file = Path(output_file).with_suffix(".mlmodel")
            ct_model.save(str(output_file))
    return ct_model
