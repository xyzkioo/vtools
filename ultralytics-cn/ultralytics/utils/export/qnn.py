# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER, QNN_HTP_TARGETS, WINDOWS
from ultralytics.utils.checks import check_requirements


def qnn_library_paths() -> tuple[str | None, str]:
    """为已安装的 onnxruntime-qnn 构建解析 QNN 执行提供程序和 HTP 后端库路径。

    onnxruntime-qnn 有两种构建形式：插件构建提供 `onnxruntime_qnn` 辅助模块，单体构建则直接提供
    `QNNExecutionProvider`，并将 QNN 后端库放在 `onnxruntime/capi` 中。

    返回：
        (tuple[str | None, str]): `(ep_library_path, htp_backend_path)`。当 QNN 已内置于 ONNX Runtime、
            不需要调用 `register_execution_provider_library` 时，`ep_library_path` 为 `None`。
    """
    try:
        import onnxruntime_qnn as qnn_ep

        return qnn_ep.get_library_path(), qnn_ep.get_qnn_htp_path()
    except ImportError:
        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        if "QNNExecutionProvider" in onnxruntime.get_available_providers():
            ep_lib = None
        else:
            ep_lib = capi / ("onnxruntime_providers_qnn.dll" if WINDOWS else "libonnxruntime_providers_qnn.so")
        htp_lib = "QnnHtp.dll" if WINDOWS else "libQnnHtp.so"
        return str(ep_lib) if ep_lib else None, str(capi / htp_lib)


def onnx2qnn(
    onnx_file: str | Path,
    output_file: Path | str,
    dataset,
    transform_fn,
    name: str = "73",
    metadata: dict | None = None,
    batch: int = 0,
    prefix: str = "",
) -> str:
    """使用 ONNX Runtime QNN 执行提供程序将 ONNX 模型转换为 Qualcomm QNN 上下文二进制文件。

    转换过程完全在主机上执行，不需要 Qualcomm 账户，也不会上传到云端。模型使用 ONNX Runtime 的 QNN QDQ 流程
    量化为 16 位激活值和 8 位权重（Hexagon NPU 推荐的精度与性能平衡方案），随后由包含 Qualcomm AI Runtime
   （QAIRT）库的 `onnxruntime-qnn` 执行提供程序，将量化图编译为嵌入 `<stem>_qnn.onnx` 的 QNN 上下文二进制文件。
    不执行推理。

    参数：
        onnx_file (str | Path): 源 ONNX 文件路径（已导出）。
        output_file (Path | str): 保存导出 QNN ONNX 上下文二进制模型的路径。
        dataset (DataLoader): 用于 INT8 量化的校准数据加载器（来自 `Exporter.get_int8_calibration_dataloader`）。
        transform_fn (Callable): 预处理变换（`Exporter._transform_fn`），将校准数据项转换为归一化的 `float32` NCHW 数组。
        name (str): Hexagon Tensor Processor（HTP）架构版本，例如 `"73"`、`"75"`、`"79"`，或 `"iq-8275"` 等受支持的 SoC 名称。
            在没有 Snapdragon NPU 的主机上导出时，该参数用于完成目标芯片的图处理。
        metadata (dict | None): 确保存在上下文模型 `metadata_props` 中的 Ultralytics 模型元数据。
        batch (int): 用于填充过小校准批次的 ONNX 图静态批次维度；动态批次模型使用 0。
        prefix (str): 日志消息前缀。

    返回：
        (str): 导出的 `*_qnn.onnx` 文件路径。

    注意：
        `onnxruntime-qnn` wheels may expose QNN either as a plugin library or as a built-in ONNX Runtime provider.
    """
    check_requirements("onnxruntime-qnn")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from ultralytics.utils.export.onnx import onnx_calibration_reader

    ep_library, htp_backend = qnn_library_paths()

    onnx_file = Path(onnx_file)
    ctx_file = Path(output_file)
    ctx_file.parent.mkdir(parents=True, exist_ok=True)
    pre_file = ctx_file.with_name(f"{onnx_file.stem}_qnn_preprocessed.onnx")
    qdq_file = ctx_file.with_name(f"{onnx_file.stem}_qnn_qdq.onnx")

    LOGGER.info(f"\n{prefix} starting W8A16 quantization and export with ONNX Runtime QNN (HTP target {name})...")
    import onnx

    dims = [d.dim_value for d in onnx.load(str(onnx_file)).graph.input[0].type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[3] in {1, 3} and dims[1] not in {1, 3}:  # 通道-last graph (QNNModel export)
        nchw_transform = transform_fn

        def transform_fn(data_item):
            """将校准数据从 NCHW 转换为 NHWC。"""
            return nchw_transform(data_item).transpose(0, 2, 3, 1)

    try:
        quant_pre_process(str(onnx_file), str(pre_file))
        # 对 HTP 后端而言，16 位激活值加 8 位权重是 ORT 推荐的精度与性能平衡方案。
        qdq_config = get_qnn_qdq_config(
            str(pre_file),
            onnx_calibration_reader(dataset, transform_fn, batch=batch),
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
        )
        quantize(str(pre_file), str(qdq_file), qdq_config)

        # 注册 QNN EP，然后在会话初始化期间将量化图编译为上下文二进制文件（不执行推理）。
        # provider 目标会在没有 NPU 的主机上离线完成图的最终处理，并禁用共享内存分配器（因为没有设备）。
        # 对于 ONNX Runtime htp_arch 解析器未公开的目标，通过
        # 改用其 QNN SoC 模型。
        ep_name = "QNNExecutionProvider"
        ep_options = {
            "backend_path": htp_backend,
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_shared_memory_allocator": "0",
        }
        option, value = QNN_HTP_TARGETS[name]
        ep_options[option] = value
        options = ort.SessionOptions()
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_file_path", str(ctx_file))
        options.add_session_config_entry("ep.context_embed_mode", "1")
        if ep_library:
            ort.register_execution_provider_library(ep_name, ep_library)
        try:
            if ep_library:
                devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
                if not devices:
                    raise RuntimeError("QNN EP registered but no QNN devices were found by ONNX Runtime.")
                options.add_provider_for_devices(devices, ep_options)
                ort.InferenceSession(str(qdq_file), sess_options=options)
            else:
                ort.InferenceSession(
                    str(qdq_file), sess_options=options, providers=[ep_name], provider_options=[ep_options]
                )
        finally:
            if ep_library:
                ort.unregister_execution_provider_library(ep_name)
    finally:
        for f in (pre_file, qdq_file):  # 删除量化中间文件；上下文二进制文件是自包含的
            f.unlink(missing_ok=True)

    if not ctx_file.exists():
        raise RuntimeError(f"QNN context binary was not generated at {ctx_file}. See {prefix} logs for details.")

    if metadata:  # 确保上下文模型中包含 Ultralytics 元数据（通常由 ONNX Runtime 保留）
        import onnx

        ctx_model = onnx.load(str(ctx_file))
        existing = {p.key for p in ctx_model.metadata_props}
        if missing := {k: v for k, v in metadata.items() if str(k) not in existing}:
            for k, v in missing.items():
                entry = ctx_model.metadata_props.add()
                entry.key, entry.value = str(k), str(v)
            onnx.save(ctx_model, str(ctx_file))
    return str(ctx_file)
