# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import torch

from ultralytics.utils import IS_JETSON, LOGGER, TORCH_VERSION, ThreadingLocked, is_dgx, is_jetson
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version
from ultralytics.utils.torch_utils import TORCH_2_4


class _NormalizeCoords(torch.nn.Module):
    """包装模型，为逐张量量化提供相对于输入的边界框和姿态坐标。"""

    def __init__(self, model: torch.nn.Module, h: int, w: int, task: str, nc: int, kpt_shape: tuple | None):
        """使用包装后的模型和预测元数据初始化。"""
        super().__init__()
        self.model = model
        self.h = h
        self.w = w
        self.task = task
        self.nc = nc
        self.kpt_shape = kpt_shape

    def forward(self, x: torch.Tensor):
        """运行包装后的模型，并按输入尺寸归一化其坐标通道。"""
        y = self.model(x)
        det = y[0] if isinstance(y, (tuple, list)) else y
        box_wh = torch.tensor([self.w, self.h, self.w, self.h], dtype=det.dtype, device=det.device).view(1, 4, 1)
        parts = [det[:, :4] / box_wh]
        if self.task == "pose" and self.kpt_shape:
            parts.append(det[:, 4 : 4 + self.nc])
            b, _, a = det.shape
            kpts = det[:, 4 + self.nc :].view(b, self.kpt_shape[0], self.kpt_shape[1], a)
            kpt_wh = torch.tensor([self.w, self.h], dtype=det.dtype, device=det.device).view(1, 1, 2, 1)
            kpts = torch.cat([kpts[:, :, :2] / kpt_wh, kpts[:, :, 2:]], dim=2)
            parts.append(kpts.reshape(b, -1, a))
        else:
            parts.append(det[:, 4 : 4 + self.nc])
            if det.shape[1] > 4 + self.nc:
                parts.append(det[:, 4 + self.nc :])
        det = torch.cat(parts, dim=1)
        return (det, *y[1:]) if isinstance(y, (tuple, list)) else det


def best_onnx_opset(onnx: types.ModuleType) -> int:
    """返回当前 torch 版本支持的最大 ONNX opset；不支持时使用 ONNX 默认值。"""
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    # ONNX Runtime CUDA 没有 Resize-19 或 ReduceMax-20 内核，因此 opset>=19 会在 CPU 上运行这些节点，
    # 并来回复制张量。其静态 INT8 量化也拒绝 opset>=21。
    return min(opset, 18, onnx.defs.onnx_opset_version())


@ThreadingLocked()
def torch2onnx(
    model: torch.nn.Module,
    im: torch.Tensor | tuple[torch.Tensor, ...],
    output_file: Path | str,
    opset: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic: dict | None = None,
) -> str:
    """将 PyTorch 模型导出为 ONNX 格式。

    参数：
        model (torch.nn.Module): 要导出的 PyTorch 模型。
        im (torch.Tensor | tuple[torch.Tensor, ...]): 用于跟踪的示例输入张量。
        output_file (Path | str): 保存导出 ONNX 文件的路径。
        opset (int): 导出时使用的 ONNX opset 版本。
        input_names (列表[str] | None): 输入张量名称列表。默认为 ``["images"]``。
        output_names (列表[str] | None): 输出张量名称列表。默认为 ``["output0"]``。
        dynamic (dict | None): 指定输入和输出动态轴的字典。

    返回：
        (str): 导出的 ONNX 文件路径。

    注意：
        设置 `do_constant_folding=True` 可能会导致 torch>=1.12 的 DNN 推理出现问题。
    """
    if input_names is None:
        input_names = ["images"]
    if output_names is None:
        output_names = ["output0"]
    kwargs = {"dynamo": False} if TORCH_2_4 else {}
    torch.onnx.export(
        model,
        im,
        output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # 警告：torch>=1.12 的 DNN 推理可能需要设为 False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        **kwargs,
    )
    return str(output_file)


def modelopt_quantize_onnx(
    onnx_file: str,
    quantize: int | str | None = None,
    dataset=None,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic: bool = False,
    prefix: str = "",
) -> str:
    """使用 NVIDIA ModelOpt 将低精度固化到 ONNX 模型中，供 TensorRT 11 强类型构建使用。

    TensorRT 11 is strongly-typed only: it removed the FP16/INT8 builder flags and the ``IInt8Calibrator`` interface, so
    reduced precision must be expressed in the ONNX graph itself before building. FP16 is applied via ModelOpt AutoCast
    mixed-precision conversion and INT8 via explicit Q/DQ quantization with calibration.

    参数：
        onnx_file (str): 要转换的 FP32 ONNX 文件路径。
        quantize (int | str | None): 精度方案，8 表示 INT8 Q/DQ 节点，16 表示 FP16 精度。
        dataset (ultralytics.数据.build.InfiniteDataLoader | None): 提供 INT8 校准图像的数据加载器。
            当 ``quantize=8`` 时必需。
        shape (tuple[int, int, int, int]): 用于动态校准的输入形状 (batch, 通道, 高度, 宽度)。
        dynamic (bool): ONNX 模型是否使用动态输入形状。
        prefix (str): 日志消息前缀。

    返回：
        (str): 精度转换后的 ONNX 文件路径。
    """
    if quantize == 8 and dataset is None:
        raise ValueError("INT8 ModelOpt quantization requires a calibration dataset.")

    # 要求 modelopt >= 0.44：旧版本导入 onnx.mapping，而 onnx >= 1.18 已删除该模块，会导致崩溃
    check_requirements("nvidia-modelopt[onnx]>=0.44")
    import onnx

    input_name = onnx.load(onnx_file, load_external_data=False).graph.input[0].name
    if quantize == 8:
        from modelopt.onnx.quantization import quantize as modelopt_quantize

        out_file = str(Path(onnx_file).with_suffix(".int8.onnx"))
        # 收集最多约 500 张校准图像（TensorRT 建议值）；ModelOpt 会一次性将它们保存在内存中，
        # 因此限制数量以控制内存，避免将整个（可能包含数千张图像的）数据集实例化。
        images, n = [], 0
        for batch in dataset:
            images.append(batch["img"])
            n += images[-1].shape[0]
            if n >= 512:
                break
        calib = torch.cat(images).to(torch.float32) / 255.0
        LOGGER.info(f"{prefix} quantizing ONNX to INT8 with ModelOpt using {calib.shape[0]} calibration images...")
        kwargs = {"calibration_shapes": f"{input_name}:{'x'.join(str(d) for d in shape)}"} if dynamic else {}
        modelopt_quantize(
            onnx_file,
            quantize_mode="int8",
            calibration_data={input_name: calib.cpu().numpy()},
            calibration_method="max",
            # 在 CPU 上校准。ModelOpt 的 CUDA EP 会因其固定的 onnxruntime-gpu cuDNN 与已安装 torch 的 cuDNN ABI
            # 不一致而触发无法捕获的段错误，TensorRT EP 也会在 RTX 显卡上中止（NvTensorRTRTX）；
            # 标度与 EP 无关，因此 INT8 引擎等效，仅这一次性步骤会变慢。
            calibration_eps=["cpu"],
            output_path=out_file,
            **kwargs,
        )
        return out_file

    from modelopt.onnx import autocast

    out_file = str(Path(onnx_file).with_suffix(".fp16.onnx"))
    LOGGER.info(f"{prefix} converting ONNX to FP16 mixed precision with ModelOpt AutoCast...")
    onnx.save(
        autocast.convert_to_mixed_precision(
            onnx_file,
            low_precision_type="fp16",
            keep_io_types=True,
            calibration_data={input_name: torch.randn(*shape).cpu().numpy()},
        ),
        out_file,
    )
    return out_file


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    quantize: int | str | None = None,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """将 YOLO 模型导出为 TensorRT engine 格式。

    参数：
        onnx_file (str): 待转换的 ONNX 文件路径。
        output_file (Path | str | None): 保存生成 TensorRT 引擎文件的路径。
        workspace (int | None): TensorRT 工作空间大小，单位为 GB。
        quantize (int | str | None): 精度方案，16 表示 FP16，8 表示 INT8。
        dynamic (bool, 可选): 是否启用动态输入形状。
        shape (tuple[int, int, int, int], 可选): 输入形状 (batch, 通道, 高度, 宽度)。
        dla (int | None): 要使用的 DLA 核心（仅限 Jetson 设备）。
        dataset (ultralytics.数据.build.InfiniteDataLoader, 可选): 用于 INT8 校准的数据集。
        metadata (dict | None): 要写入引擎文件的元数据。
        verbose (bool, 可选): 是否启用详细日志。
        prefix (str, 可选): 日志消息前缀。

    返回：
        (str): 导出的引擎文件路径。

    异常：
        ValueError: 在非 Jetson 设备上启用 DLA 或未设置所需精度时抛出。
        RuntimeError: 无法解析 ONNX 文件时抛出。

    注意：
        该函数兼容不同 TensorRT 版本的工作区大小和引擎构建方式。在 TensorRT 7-10 中，INT8 校准使用
        ``IInt8Calibrator`` 遍历 ``dataset`` 并写入校准缓存，FP16/INT8 则通过构建器标志启用。
        TensorRT 11 移除了这些接口并改用强类型网络，因此在构建前使用 NVIDIA ModelOpt 将低精度信息写入
        ONNX 图（FP16 AutoCast、INT8 显式 Q/DQ），具体由 `modelopt_quantize_onnx` 完成。TensorRT 7-10 路径
        会让 Sigmoid 层保持较高精度，以保留置信度分数的校准精度（参见 #24668）。如果提供元数据，
        则会将其序列化并写入引擎文件。
    """
    # 在 CUDA 13 ARM 设备上强制重新安装 10.15.x 版本的 TensorRT，以支持 RT-DETR 导出
    # https://github.com/ultralytics/ultralytics/issues/22873
    if is_jetson(jetpack=7) or is_dgx():
        check_tensorrt("10.15")

    try:
        import tensorrt as trt
    except ImportError:
        check_tensorrt()
        import tensorrt as trt
    check_version(trt.__version__, ">=7.0.0", hard=True)
    check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    output_file = output_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # 引擎构建器
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    trt_major = int(trt.__version__.split(".", 1)[0])
    is_trt10 = trt_major >= 10
    # TensorRT >= 11 仅支持强类型：精度构建器标志和 IInt8Calibrator 已被删除
    is_trt11 = trt_major >= 11
    if workspace_bytes > 0:
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:  # TensorRT 7 fallback
            config.max_workspace_size = workspace_bytes
    # TensorRT 10 删除了 EXPLICIT_BATCH 标志（显式批次是唯一/默认模式）；为 TRT 7/8 保留该标志
    flag = 0 if is_trt10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    # TensorRT 10 从 Builder 中删除了 platform_has_fast_fp16/int8；缺失时默认设为 True
    use_fp16 = getattr(builder, "platform_has_fast_fp16", True) and quantize == 16
    use_int8 = getattr(builder, "platform_has_fast_int8", True) and quantize == 8
    if use_int8 and dataset is None:
        raise ValueError("INT8 TensorRT export requires a calibration dataset.")

    # 如果启用，则可选地切换到 DLA
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA 仅适用于 NVIDIA Jetson 设备")
        if check_version(trt.__version__, ">=11.0.0,<11.1.0"):
            # TensorRT 11.0 不支持 DLA，计划在后续版本中恢复
            # https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html
            raise ValueError("DLA is not supported in TensorRT 11.0; export with TensorRT 10.x to use DLA.")
        LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
        if not use_fp16 and not use_int8:
            raise ValueError(
                "DLA requires either quantize=16 (FP16) or quantize=8 (INT8). Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # TensorRT 11 使用强类型并移除了 FP16/INT8 构建器标志和 INT8 校准器，因此必须在解析前使用 NVIDIA
    # ModelOpt 将降低精度固化到 ONNX 图中（FP16 AutoCast、INT8 Q/DQ）。
    if is_trt11 and (use_fp16 or use_int8):
        onnx_file = modelopt_quantize_onnx(onnx_file, quantize, dataset, shape, dynamic, prefix)

    # 读取 ONNX 文件
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"无法加载 ONNX 文件：{onnx_file}")

    # 网络输入
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # 最小输入形状
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # 最大输入形状
        for inp in inputs:
            inp_min = tuple(d if d != -1 else lo for d, lo in zip(inp.shape, min_shape))
            inp_max = tuple(d if d != -1 else hi for d, hi in zip(inp.shape, max_shape))
            profile.set_shape(inp.name, min=inp_min, opt=shape, max=inp_max)
        config.add_optimization_profile(profile)
        if use_int8 and not is_trt10:  # deprecated in TensorRT 10, causes internal errors
            config.set_calibration_profile(profile)

    LOGGER.info(
        f"{prefix} building {'INT8' if use_int8 else 'FP' + ('16' if use_fp16 else '32')} engine as {output_file}"
    )
    if use_int8 and not is_trt11:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            """用于 TensorRT engine 优化的自定义 INT8 校准器。

            此校准器提供 TensorRT 使用数据集执行 INT8 量化校准所需的接口，并负责批次生成、缓存和校准算法选择。

            属性：
            dataset: 用于校准的数据集。
            data_iter: 校准数据集的迭代器。
            algo (trt.CalibrationAlgoType): 校准算法类型。
            batch (int): 校准批次大小。
            cache (Path): 保存校准缓存的路径。

            方法：
                get_algorithm：获取要使用的校准算法。
                get_batch_size：获取校准使用的批次大小。
                get_batch：获取用于校准的下一批数据。
                read_calibration_cache：使用已有缓存，避免重复校准。
                write_calibration_cache：将校准缓存写入磁盘。
            """

            def __init__(
                self,
                dataset,  # ultralytics.数据.build.InfiniteDataLoader
                cache: str = "",
            ) -> None:
                """使用数据集和缓存路径初始化 INT8 校准器。"""
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # DLA quantization needs ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                """获取要使用的校准算法。"""
                return self.algo

            def get_batch_size(self) -> int:
                """获取校准使用的批次大小。"""
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                """获取校准使用的下一批数据，返回设备内存指针列表。"""
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
            # 返回 None，表示 TensorRT 已没有剩余的校准数据。
                    return None

            def read_calibration_cache(self) -> bytes | None:
                """使用现有缓存，避免再次校准；否则隐式返回 None。"""
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache: bytes) -> None:
                """将校准缓存写入磁盘。"""
                _ = self.cache.write_bytes(cache)

        # 使用构建器加载数据集（用于分批），并执行校准。
        config.int8_calibrator = EngineCalibrator(
            dataset=dataset,
            cache=str(Path(onnx_file).with_suffix(".cache")),
        )

        # TRT 11 的隐式量化无法像 ModelOpt 那样排除特定算子类型，因此通过逐层精度约束，
        # 将检测头 Sigmoid（以其 ONNX 节点命名的 ACTIVATION 层）保持为 FP32，以保留置信度分数的校准效果，
        # 这与 OpenVINO 的 IgnoredScope 类似。
        # 详见 https://github.com/ultralytics/ultralytics/issues/24668。仅将约束应用于检测头：每个 SiLU 激活也
        # 是一个 Sigmoid，在所有层上施加约束会降低骨干网络和颈部的 INT8 速度。
        names = [network.get_layer(i).name for i in range(network.num_layers)]
        indices = [int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))]
        head = f"/model.{max(indices)}/" if indices else "/"
        count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (
                layer.type == trt.LayerType.ACTIVATION
                and "sigmoid" in layer.name.lower()
                and layer.name.startswith(head)
            ):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                count += 1
        if count:
            flag = (
                trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS
                if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS")
                else trt.BuilderFlag.STRICT_TYPES
            )
            config.set_flag(flag)  # OBEY_PRECISION_CONSTRAINTS replaced STRICT_TYPES in TensorRT 8.2
            LOGGER.info(f"{prefix} keeping {count} head Sigmoid layers in FP32 for INT8 accuracy")

    elif use_fp16 and not is_trt11:
        config.set_flag(trt.BuilderFlag.FP16)

    # 写入文件
    if hasattr(builder, "build_serialized_network"):
        engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        engine = None if engine is None else engine.serialize()
    if engine is None:
        raise RuntimeError("TensorRT engine build failed, check logs for errors")
    with open(output_file, "wb") as t:
        if metadata is not None:
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        t.write(engine)
    return str(output_file)
