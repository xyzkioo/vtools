# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import types
import zipfile
from pathlib import Path

import torch

from ultralytics.nn.modules import Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.export.engine import _NormalizeCoords


def _litert_grouped_topk(x: torch.Tensor, k: int, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """沿 dim 1 选择 x 的前 k 项并返回 int32 索引；GPU 委托支持 int32，但不支持 int64。"""
    values, index = Detect._grouped_topk(x, k, groups)
    return values, index.int()


def _litert_gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """沿 dim 1 选择 x 中索引为 (batch, k) 的行，不使用 GPU 委托未实现的 gather_nd。"""
    b, n = x.shape[:2]
    offset = torch.arange(b, device=x.device, dtype=index.dtype)[..., None] * n
    return x.flatten(0, 1).index_select(0, (index + offset).flatten()).view(b, index.shape[1], *x.shape[2:])


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path,
    quantize: int | str | None,
    calibration_dataset: torch.utils.data.DataLoader | None,
    metadata: dict | None,
    prefix: str,
) -> Path:
    """使用 litert_torch 将 PyTorch 模型导出为 LiteRT 格式，并支持可选 INT8 量化。

    通过 ``quantize`` 支持三种 INT8 方案：``8`` 执行静态 INT8（int8 权重 + int8 激活值），
    ``'w8a16'`` 执行带 int16 激活值的静态 INT8；两者都需要 ``calibration_dataset``。
    ``'w8a32'`` 执行动态或仅权重 INT8（int8 权重 + FP32 激活值），无需校准。
    ``None``/``32`` 导出 FP32。FP16 不作为单独模型导出：LiteRT 在运行时通过 GPU 委托（默认 FP16）
    或 ARM 上的 XNNPACK ``FORCE_FP16`` 标志，以 FP16 运行 FP32 模型。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型。
        im (torch.Tensor): 用于跟踪的示例输入张量。
        file (Path | str): 用于确定输出目录的源模型文件路径。
        quantize (int | str | None): 量化方案：``8``（静态 INT8）、``'w8a16'``（静态 int8 权重 + int16 激活值）、
            ``'w8a32'``（动态 INT8）或 ``None``/``32``（FP32）。
        calibration_dataset (DataLoader | None): 静态量化使用的校准数据加载器，由 ``get_int8_calibration_dataloader`` 返回。
            ``quantize`` 为 ``8`` 或 ``'w8a16'`` 时必需。
        metadata (dict | None): 作为 ``metadata.json`` 条目嵌入 ``.tflite`` 的可选元数据。
        prefix (str): 日志消息前缀。

    返回：
        (Path): 导出的 ``.tflite`` 文件路径，其中嵌入了 ``metadata.json`` 元数据条目。
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements(("litert-torch>=0.9.0", "ai-edge-litert>=2.1.4"))
    import litert_torch

    static_int8 = quantize == 8
    static_int16 = quantize == "w8a16"
    dynamic_int8 = quantize == "w8a32"
    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    quant_tag = "_int8" if static_int8 else "_w8a16" if static_int16 else "_w8a32" if dynamic_int8 else ""

    # 根据输入尺寸对坐标通道进行归一化，使 INT8 量化能够保留分数（由 LiteRTBackend 反归一化）。
    # 端到端模型以 FP32 输出 NMS 后的像素坐标（不会合并缩放），因此保持原样。
    meta = metadata or {}
    task = meta.get("task")
    if task in {"detect", "segment", "pose", "obb"} and not meta.get("end2end", False):
        model = _NormalizeCoords(
            model, int(im.shape[2]), int(im.shape[3]), task, len(meta.get("names", {})), meta.get("kpt_shape")
        )

    for m in model.modules():  # int32 索引和无需 gather_nd 的 gather 使检测头保持在 GPU 委托中
        if isinstance(m, Detect):
            m._grouped_topk = _litert_grouped_topk
            m._gather = types.MethodType(_litert_gather, m)

    # 将 index_select 降级为 tfl.gather：默认降级会生成 GPU 委托不支持的 GATHER_ND
    litert_torch.fx_infra.decomp.add_pre_lower_decomp(
        torch.ops.aten.index_select.default, lambda x, dim, index: torch.ops.tfl.gather(x, index.int(), dim)
    )
    edge_model = litert_torch.convert(model, (im,))
    tflite_file = file.with_name(f"{file.stem}{quant_tag}.tflite")
    edge_model.export(tflite_file)

    if static_int8 or static_int16 or dynamic_int8:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import qtyping, quantizer, recipe

        qt = quantizer.Quantizer(str(tflite_file))
        if static_int8 or static_int16:  # 静态方案使用代表性图像进行校准
            act = "int8" if static_int8 else "int16"
            LOGGER.info(f"{prefix} applying static quantization (int8 weights + {act} activations)...")
            calib_samples = []
            for batch in calibration_dataset:
                imgs = batch["img"].cpu().float() / 255.0
                # litert-torch 跟踪固定批次；将较小批次平铺到 im 的批次维度（重复项
                # 校准统计数据保持一致）
                if imgs.shape[0] < im.shape[0]:
                    imgs = imgs.repeat(-(-im.shape[0] // imgs.shape[0]), 1, 1, 1)[: im.shape[0]]
                calib_samples.append({"args_0": imgs.numpy()})
            qt.load_quantization_recipe(recipe.static_wi8_ai8() if static_int8 else recipe.static_wi8_ai16())
            # 保持图的 FP32 输入/输出（权重和激活值在内部仍为 int8/int16），与历史行为一致。
            # 遵循 onnx2tf 的“fp32 输入/输出”约定，这是下游使用方（LiteRT GPU 委托和设备端运行时）所期望的，
            # 并避免强制每个使用方在边界处执行量化/反量化。必须在 load_quantization_recipe 后运行。
            for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
                qt.update_quantization_recipe(
                    regex=".*", operation_name=op, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE
                )
            result = qt.calibrate({"serving_default": calib_samples})
            qt.quantize(calibration_result=result).export_model(str(tflite_file), overwrite=True)
        else:  # 动态或仅权重 INT8：int8 权重、FP32 激活值，无需校准
            LOGGER.info(f"{prefix} applying dynamic INT8 quantization (int8 weights + FP32 activations)...")
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
            qt.quantize().export_model(str(tflite_file), overwrite=True)

    # 将元数据作为 JSON 条目附加到 .tflite（可容忍 zip 尾部数据的 flatbuffer），使模型成为单个
    # 生成独立文件，LiteRTBackend 会在加载时读取该文件。
    with zipfile.ZipFile(tflite_file, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata or {}))
    return tflite_file
