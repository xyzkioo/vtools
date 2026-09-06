# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
对 YOLO 模型格式进行速度和精度基准测试。

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolo26n.yaml', 'yolov8s.yaml']).run()
    benchmark(模型='yolo26n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
"""

from __future__ import annotations

import glob
import platform
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.cuda

from ultralytics import RTDETR, YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.nn.modules import Segment26
from ultralytics.utils import (
    ARM64,
    ASSETS,
    IS_DOCKER,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    TQDM,
    WEIGHTS_DIR,
    is_github_action_running,
)
from ultralytics.utils.checks import IS_PYTHON_MINIMUM_3_13, check_imgsz, check_yolo, is_rockchip
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo26n.pt",
    data=None,
    imgsz=160,
    quantize=None,
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="",
    **kwargs,
):
    """在不同格式下测试 YOLO 模型的速度和精度。

    参数：
        model (str | Path): 模型文件或目录路径。
        data (str | None): 用于评估的数据集；未传入时从 TASK2DATA 获取。
        imgsz (int): 测试使用的图像尺寸。
        quantize (int | str | None): 导出和推理精度：16（FP16）、8（INT8）或 None/32（FP32）。
        device (str): 运行测试的设备，可选 'cpu' 或 'cuda'。
        verbose (bool | float): 为 True 或浮点数时，使用给定指标断言测试通过。
        eps (float): 防止除零的 epsilon 值。
        format (str): 测试使用的导出格式；未提供时测试所有格式。
        **kwargs (Any): 传给导出器的其他关键字参数。

    返回：
        (polars.DataFrame): 包含每种格式测试结果的 Polars DataFrame，包括文件大小、指标和推理时间。

    示例：
        使用默认设置测试 YOLO 模型：
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo26n.pt", imgsz=640)
    """
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import polars as pl  # scope for faster 'import ultralytics'

    pl.Config.set_tbl_cols(-1)  # 显示所有列
    pl.Config.set_tbl_rows(-1)  # 显示所有行
    pl.Config.set_tbl_width_chars(-1)  # 不限制宽度
    pl.Config.set_tbl_hide_column_data_types(True)  # 隐藏数据类型
    pl.Config.set_tbl_hide_dataframe_shape(True)  # 隐藏形状信息
    pl.Config.set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")

    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    data = data or TASK2DATA[model.task]  # 任务对应的数据集，例如 task=detect 时为 coco8.yaml
    key = TASK2METRIC[model.task]  # 任务对应的指标，例如 task=detect 时为 metrics/mAP50-95(B)

    y = []
    t0 = time.time()

    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format_arg in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."
    for name, export_format, suffix, cpu, gpu, valid_args, _ in zip(*export_formats().values()):
        emoji, filename = "❌", None  # export defaults
        try:
            if format_arg and format_arg != export_format:
                continue
            if IS_PYTHON_MINIMUM_3_13 and not format_arg and export_format in {"saved_model", "pb", "edgetpu"}:
                continue

            # Checks
            if export_format == "pb":
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
            elif export_format == "edgetpu":
                assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"
                assert shutil.which("edgetpu_compiler"), "Edge TPU benchmark requires edgetpu_compiler"
            elif export_format == "coreml":
                assert MACOS or (LINUX and not ARM64), "CoreML export only supported on macOS and non-aarch64 Linux"
                # macOS Python 3.13 中 coremltools 在 OpenVINO 后会因 OpenMP 运行时冲突而死锁；
                # 在非 aarch64 Linux Python 3.13 上仍执行 CoreML 基准测试。
                assert not (MACOS and IS_PYTHON_MINIMUM_3_13), (
                    "CoreML not benchmarked on macOS Python>=3.13 (coremltools/OpenVINO OpenMP deadlock)"
                )
            if export_format in {"saved_model", "pb", "edgetpu"}:
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
            if export_format == "paddle":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"
                assert (LINUX and not IS_JETSON) or MACOS, "Windows and Jetson Paddle exports not supported yet"
                # PaddlePaddle 导出在 Python 3.13 中单独运行正常，但其原生 protobuf 与 TensorFlow 在共享基准进程中
                # 先加载的 protobuf>=6.31.1 冲突，导致段错误。
                assert not IS_PYTHON_MINIMUM_3_13, (
                    "PaddlePaddle not benchmarked on Python>=3.13 (protobuf ABI conflict with TensorFlow)"
                )
            if export_format == "mnn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN exports not supported yet"
                # MNN 导出在 Python 3.13 中单独运行正常，但其 ONNX 解析 protobuf 与 TensorFlow 在共享基准进程中
                # 先加载的 protobuf>=6.31.1 冲突，导致运行中止。
                assert not IS_PYTHON_MINIMUM_3_13, (
                    "MNN not benchmarked on Python>=3.13 (protobuf ABI conflict with TensorFlow)"
                )
            if export_format == "ncnn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"
            if export_format == "imx":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX exports not supported"
                assert model.task in {"detect", "classify", "pose", "segment"}, (
                    "IMX export is only supported for detection, classification, pose estimation and segmentation tasks"
                )
                assert "C2f" in model.__str__(), "IMX only supported for YOLOv8n and YOLO11n"
            if export_format == "rknn":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 RKNN exports not supported yet"
                assert LINUX, "RKNN only supported on Linux"
                assert not is_rockchip(), "RKNN Inference only supported on Rockchip devices"
            if export_format == "executorch":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 ExecuTorch exports not supported yet"
            if export_format == "axelera":
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Axelera exports not supported"
                assert LINUX and not (ARM64 and IS_DOCKER), (
                    "export is only supported on Linux and is not supported on ARM64 Docker."
                )
                assert not (model.task == "segment" and any(isinstance(m, Segment26) for m in model.model.modules())), (
                    "Axelera export does not currently support YOLO26 segmentation models"
                )
            if export_format == "litert":
                assert MACOS or (LINUX and not ARM64), "LiteRT benchmark only supported on Linux x86 and macOS"
                # 在 macOS CI 中，当 litert 在共享进程的其他 TF 格式之后运行时，benchmark() 会在
                # ai-edge-litert/TensorFlow abseil 互斥锁（RAW: Lock blocking）上死锁；本地仍执行基准测试。
                assert not (MACOS and is_github_action_running()), (
                    "LiteRT not benchmarked on macOS CI (ai-edge-litert/TF abseil mutex deadlock)"
                )
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if export_format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = deepcopy(model)  # PyTorch 格式
            else:
                export_data = data if "data" in valid_args else None
                filename = deepcopy(model).export(
                    imgsz=imgsz,
                    format=export_format,
                    quantize=quantize,
                    data=export_data,
                    device=device,
                    verbose=False,
                    **kwargs,
                )
                exported_model = RTDETR(filename) if isinstance(model, RTDETR) else YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # 预测
            assert model.task != "pose" or export_format != "pb", "GraphDef Pose inference is not supported"
            assert export_format != "edgetpu", "inference not supported"
            assert export_format != "coreml" or platform.system() == "Darwin", "inference requires macOS>=10.13"
            assert export_format != "axelera", "inference only supported on Axelera hardware"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, quantize=quantize, verbose=False)

            # Validate
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                quantize=quantize,
                verbose=False,
                conf=0.001,  # 所有预设基准 mAP 值均基于 conf=0.001
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # frames per second
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.error(f"Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print 结果
    check_yolo(device=device)  # print system info
    df = pl.DataFrame(y, schema=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"], orient="row")
    df = df.with_row_index(" ", offset=1)  # add 索引 info
    df_display = df.with_columns(pl.all().cast(pl.String).fill_null("-"))

    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df_display}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].to_numpy()  # 用于与下限比较的值
        floor = verbose  # 通过所需的最小指标，例如 YOLOv5n 的 mAP 为 0.29
        assert all(x > floor for x in metrics if not np.isnan(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df_display


class ProfileModels:
    """用于在 ONNX 和 TensorRT 上分析不同模型性能的 ProfileModels 类。

    此类分析不同模型的性能，并返回模型速度和 FLOPs 等结果。

    属性：
        paths (列表[str]): 待分析的模型路径。
        num_timed_runs (int): 性能分析的计时运行次数。
        num_warmup_runs (int): 分析前的预热运行次数。
        min_time (float): 性能分析的最短时间（秒）。
        imgsz (int): 模型使用的图像尺寸。
        quantize (int | str | None): TensorRT 分析的导出精度，例如 16（FP16）或 8（INT8）。
        trt (bool): 是否使用 TensorRT 进行分析。
        device (torch.device): 性能分析使用的设备。

    方法：
        run: 分析不同格式 YOLO 模型的速度和精度。
        get_files: 获取所有相关模型文件。
        get_onnx_model_info: 从 ONNX 模型提取元数据。
        iterative_sigma_clipping: 应用 sigma clipping 移除异常值。
        profile_tensorrt_model: 分析 TensorRT 模型。
        profile_onnx_model: 分析 ONNX 模型。
        generate_table_row: 生成包含模型指标的表格行。
        generate_results_dict: 生成分析结果字典。
        print_table: 打印格式化的结果表格。

    示例：
        分析模型并打印结果
        >>> from ultralytics.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolo26n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.run()
    """

    def __init__(
        self,
        paths: list[str],
        num_timed_runs: int = 100,
        num_warmup_runs: int = 10,
        min_time: float = 60,
        imgsz: int = 640,
        quantize: int | str | None = 16,
        trt: bool = True,
        device: torch.device | str | None = None,
    ):
        """初始化用于模型性能分析的 ProfileModels 类。

        参数：
            paths (列表[str]): 待分析的模型路径列表。
            num_timed_runs (int): 性能分析的计时运行次数。
            num_warmup_runs (int): 实际分析开始前的预热运行次数。
            min_time (float): 模型性能分析的最短时间（秒）。
            imgsz (int): 分析期间使用的图像尺寸。
            quantize (int | str | None): TensorRT 分析的导出精度，例如 16（默认 FP16）或 8（INT8）。
            trt (bool): 是否使用 TensorRT 进行分析。
            device (torch.device | str | None): 分析使用的设备；为 None 时自动确定。

        注意：
            quantize 仅适用于 TensorRT 性能分析导出；ONNX 分析保持 FP32（CPU 上 FP16 更慢）。
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.quantize = quantize
        self.trt = trt  # 运行 TensorRT 性能分析
        self.device = device if isinstance(device, torch.device) else select_device(device)

    def run(self):
        """在多种格式（包括 ONNX 和 TensorRT）下分析 YOLO 模型的速度和精度。

        返回：
            (列表[dict]): 包含每个模型性能分析结果的字典列表。

        示例：
            分析模型并打印结果
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolo26n.yaml", "yolo11s.yaml"])
            >>> results = profiler.run()
        """
        files = self.get_files()

        if not files:
            LOGGER.warning("No matching *.pt or *.onnx files found.")
            return []

        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse(verbose=False)
                model_info = model.info(imgsz=self.imgsz)
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine",
                        quantize=self.quantize,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(
                    format="onnx",
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output

    def get_files(self):
        """返回用户指定的所有相关模型文件路径列表。

        返回：
            (列表[Path]): 模型文件的 Path 对象列表。
        """
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # add non-existing
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        LOGGER.info(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """从 ONNX 模型文件提取元数据，包括层数、参数量、梯度数量和 FLOPs。"""
        return 0.0, 0.0, 0.0, 0.0  # 返回 (num_layers, num_params, num_gradients, num_flops)

    @staticmethod
    def iterative_sigma_clipping(data: np.ndarray, sigma: float = 2, max_iters: int = 3):
        """对数据迭代应用 sigma 裁剪，以移除异常值。

        参数：
            数据 (np.ndarray): 输入数据 数组.
            sigma (float): 裁剪使用的标准差倍数。
            max_iters (int): 裁剪过程的最大迭代次数。

        返回：
            (np.ndarray): 移除异常值后的裁剪数据数组。
        """
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            # 包含位于边界和方差为零的样本。
            clipped_data = data[(data >= mean - sigma * std) & (data <= mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """使用 TensorRT 分析 YOLO 模型性能，测量平均运行时间和标准差。

        参数：
            engine_file (str): TensorRT engine 文件路径。
            eps (float): 防止除零的小 epsilon 值。

        返回：
            (tuple[float, float]): 以毫秒为单位的推理时间均值和标准差。
        """
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # 模型和输入
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # Classify 使用 uint8

        # 预热运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.perf_counter()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.perf_counter() - start_time

        # 计算运行次数，取 min_time 对应次数和 num_timed_runs 中的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        # 计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])  # 转换 to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    @staticmethod
    def check_dynamic(tensor_shape):
        """检查 ONNX 模型中的张量形状是否为动态形状。"""
        return not all(isinstance(dim, int) and dim >= 0 for dim in tensor_shape)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """分析 ONNX 模型，测量多次运行的平均推理时间和标准差。

        参数：
            onnx_file (str): ONNX 模型文件路径。
            eps (float): 防止除零的小 epsilon 值。

        返回：
            (tuple[float, float]): 推理时间的均值和标准差，单位为毫秒。
        """
        import onnxruntime as ort

        from ultralytics.nn.backends import ONNXBackend

        # 用于一致基准测试的会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  # 限制线程数量

        # 使用 CPU 设备初始化 ONNXBackend，以确保基准测试一致
        backend = ONNXBackend(onnx_file, device=torch.device("cpu"), fp16=False, session_options=sess_options)

        # 为多输入模型准备输入数据字典
        input_data_dict = {}
        for input_tensor in backend.session.get_inputs():
            input_type = input_tensor.type
            if self.check_dynamic(input_tensor.shape):
                if len(input_tensor.shape) != 4 and self.check_dynamic(input_tensor.shape[1:]):
                    raise ValueError(f"Unsupported dynamic shape {input_tensor.shape} of {input_tensor.name}")
                input_shape = (
                    (1, 3, self.imgsz, self.imgsz) if len(input_tensor.shape) == 4 else (1, *input_tensor.shape[1:])
                )
            else:
                input_shape = input_tensor.shape

            # 将 ONNX 数据类型映射为 numpy 数据类型
            if "float16" in input_type:
                input_dtype = np.float16
            elif "float" in input_type:
                input_dtype = np.float32
            elif "double" in input_type:
                input_dtype = np.float64
            elif "int64" in input_type:
                input_dtype = np.int64
            elif "int32" in input_type:
                input_dtype = np.int32
            else:
                raise ValueError(f"Unsupported ONNX datatype {input_type}")

            input_data_dict[input_tensor.name] = np.random.rand(*input_shape).astype(input_dtype)

        # 预热运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.perf_counter()
            for _ in range(self.num_warmup_runs):
                backend.forward(input_data_dict)
            elapsed = time.perf_counter() - start_time

        # 运行次数取 min_time 与 num_timed_runs 中较大的值。
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # 计时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.perf_counter()
            backend.forward(input_data_dict)
            run_times.append((time.perf_counter() - start_time) * 1000)  # 转换 to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(
        self,
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成包含模型性能指标的表格行字符串。

        参数：
            model_name (str): 模型名称。
            t_onnx (tuple): ONNX 模型推理时间统计（均值、标准差）。
            t_engine (tuple): TensorRT engine 推理时间统计（均值、标准差）。
            model_info (tuple): 模型信息（层数、参数量、梯度数量、FLOPs）。

        返回：
            (str): 包含模型指标的格式化表格行字符串。
        """
        _layers, params, _gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±"
            f"{t_engine[1]:.1f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成性能分析结果字典。

        参数：
            model_name (str): 模型名称。
            t_onnx (tuple): ONNX 模型推理时间统计（均值、标准差）。
            t_engine (tuple): TensorRT engine 推理时间统计（均值、标准差）。
            model_info (tuple): 模型信息（层数、参数量、梯度数量、FLOPs）。

        返回：
            (dict): 包含性能分析结果的字典。
        """
        _layers, params, _gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows: list[str]):
        """打印格式化的模型性能分析结果表格。

        参数：
            table_rows (列表[str]): 格式化表格行字符串列表。
        """
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        headers = [
            "Model",
            "size<br><sup>(pixels)",
            "mAP<sup>val<br>50-95",
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",
            "params<br><sup>(M)",
            "FLOPs<br><sup>(B)",
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

        LOGGER.info(f"\n\n{header}")
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
