# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.modules import Detect, Pose, Pose26
from ultralytics.utils import LINUX, LOGGER, MACOS
from ultralytics.utils.checks import (
    IS_PYTHON_MINIMUM_3_13,
    check_apt_requirements,
    check_requirements,
    check_version,
    is_sudo_available,
)
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.tal import make_anchors


def tf_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """用于 TensorFlow 导出兼容性的包装器（TensorFlow 专用处理现在位于 head 模块）。"""
    for m in model.modules():
        if not isinstance(m, Detect):
            continue
        import types

        m._get_decode_boxes = types.MethodType(_tf_decode_boxes, m)
        if isinstance(m, Pose):
            m.kpts_decode = types.MethodType(partial(_tf_kpts_decode, is_pose26=type(m) is Pose26), m)
    return model


def _tf_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """为 TensorFlow 导出解码边界框。"""
    shape = x["feats"][0].shape  # BCHW
    boxes = x["boxes"]
    if self.format != "imx" and (self.dynamic or self.shape != shape):
        self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
        self.shape = shape
    grid_h, grid_w = shape[2:4]
    grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=boxes.device).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    dbox = self.decode_bboxes(self.dfl(boxes) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
    return dbox


def _tf_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    """为 TensorFlow 导出解码关键点。"""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    # 预先计算归一化因子，以提高数值稳定性
    y = kpts.view(bs, *self.kpt_shape, -1)
    grid_h, grid_w = self.shape[2:4]
    grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    a = ((y[:, :, :2] + self.anchors) if is_pose26 else (y[:, :, :2] * 2.0 + (self.anchors - 0.5))) * norm
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def onnx2saved_model(
    onnx_file: str,
    output_dir: Path | str,
    quantize: int | str | None = None,
    images: np.ndarray | None = None,
    disable_group_convolution: bool = False,
    cuda: bool = False,
    prefix: str = "",
):
    """使用 onnx2tf 将 ONNX 模型转换为 TensorFlow SavedModel 格式。

    参数：
        onnx_file (str): ONNX 文件路径。
        output_dir (Path | str): 保存 SavedModel 的输出目录路径。
        quantize (int | str | None): 精度方案，8 表示 INT8。
        images (np.ndarray | None, 可选): 用于 INT8 量化的校准图像，格式为 BHWC。
        disable_group_convolution (bool, 可选): 是否禁用分组卷积优化。默认为 False。
        cuda (bool, 可选): 是否在 CUDA 设备上导出；为 True 时选择 GPU 版 onnxruntime，并让 TensorFlow 保持 GPU 可见，
            否则 CPU 导出不会占用 GPU 内存。默认为 False。
        prefix (str, 可选): 日志前缀。默认为 ""。

    返回：
        (keras.Model): 转换后的 Keras 模型。

    注意：
        - 如果尚未安装，则自动安装 tensorflow、onnx2tf 及所有必需依赖。
        - 启用 INT8 量化时下载校准数据。
        - 转换完成后删除临时文件并重命名量化模型。
    """
    try:
        import tensorflow as tf
    except ImportError:
        check_requirements("tensorflow>2.19.0" if IS_PYTHON_MINIMUM_3_13 else "tensorflow>=2.0.0,<=2.19.0")
        import tensorflow as tf
    if not cuda:
        with contextlib.suppress(Exception):  # 仅当用户之前的代码已初始化 TF GPU 时失败
            tf.config.set_visible_devices([], "GPU")  # hide GPUs so non-CUDA exports never allocate GPU memory
    check_requirements(
        f"onnx2tf{'>=2.3.0,<2.3.16' if IS_PYTHON_MINIMUM_3_13 else '>=1.26.3,<1.29.0'}",  # pin to avoid h5py build issues on aarch64
        cmds="--no-deps",
    )
    check_requirements(
        (
            f"tf_keras{'>2.19.0' if IS_PYTHON_MINIMUM_3_13 else '<=2.19.0'}",  # required by 'onnx2tf' package
            "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
            "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
            "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),  # required by 'onnx2tf' package
            "onnx>=1.12.0,<2.0.0",
            f"onnx2tf{'>=2.3.0,<2.3.16' if IS_PYTHON_MINIMUM_3_13 else '>=1.26.3,<1.29.0'}",
            "onnxslim>=0.1.82",
            # 这些候选项可互换，避免已安装的变体（例如 onnxruntime-gpu）被重复安装
            ("onnxruntime-gpu" if cuda else "onnxruntime", "onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn"),
            "protobuf>=6.31.1,<7.0.0"
            if IS_PYTHON_MINIMUM_3_13
            else "protobuf>=5",  # TF>2.19 (Python 3.13) needs protobuf>=6.31.1; cap <7 to match TF gencode and avoid PaddlePaddle segfault
        )
    )

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    check_version(
        tf.__version__,
        ">=2.0.0",
        name="tensorflow",
        verbose=True,
        msg="https://github.com/ultralytics/ultralytics/issues/5161",
    )

    output_dir = Path(output_dir)
    use_int8 = quantize == 8
    # 预先下载校准文件，以修复 https://github.com/PINTO0309/onnx2tf/issues/545
    onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not onnx2tf_file.exists():
        attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)
    np_data = None
    if use_int8:
        tmp_file = output_dir / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration 图像 文件
        if images is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(tmp_file), images)  # BHWC
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    # 修补 onnx.helper，以兼容 ONNX>=1.17 的 onnx_graphsurgeon
    # ONNX 1.17 删除了 float32_to_bfloat16 函数，但 onnx_graphsurgeon 仍在使用它
    import onnx.helper

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        import struct

        def float32_to_bfloat16(fval):
            """将 float32 转换为 bfloat16（截断尾数的低 16 位）。"""
            ival = struct.unpack("=I", struct.pack("=f", fval))[0]
            return ival >> 16

        onnx.helper.float32_to_bfloat16 = float32_to_bfloat16

    import importlib
    import inspect
    import pathlib

    import onnx2tf.ops.TopK as _t

    _path = pathlib.Path(inspect.getfile(_t))
    _text = _path.read_text()
    _patched = _text.replace(
        "k_tensor = int(k_tensor)",
        "k_tensor = int(k_tensor.squeeze()) if hasattr(k_tensor, 'squeeze') else int(k_tensor)",
    )
    if _patched != _text:  # write only when unpatched; site-packages may be read-only (pre-patched containers)
        try:
            _path.write_text(_patched)
            importlib.reload(_t)
        except OSError as e:  # 安装目录只读：保持未修补状态继续运行，只有包含 TopK 的模型会受影响
            LOGGER.warning(f"{prefix} unable to apply onnx2tf TopK patch: {e}")
    import onnx2tf  # 在 ONNX 导出后按需导入，以减少导入期间的冲突

    LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
    keras_model = onnx2tf.convert(
        input_onnx_file_path=onnx_file,
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="error",  # 注意 INT8-FP16 激活值问题：https://github.com/ultralytics/ultralytics/issues/15873
        output_integer_quantized_tflite=use_int8,
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=not use_int8,  # 修复 GPU 委托上检测对象数量减少的问题
        output_signaturedefs=True,  # 修复 Attention 模块组卷积错误
        disable_group_convolution=disable_group_convolution,  # 修复组卷积错误
    )

    # 删除或重命名 TFLite 模型
    if use_int8:
        tmp_file.unlink(missing_ok=True)
        for file in output_dir.rglob("*_dynamic_range_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
        for file in output_dir.rglob("*_integer_quant_with_int16_act.tflite"):
            file.unlink()  # 删除多余的 FP16 激活值 TFLite 文件
    return keras_model


def keras2pb(keras_model, output_file: Path | str, prefix: str = "") -> str:
    """将 Keras 模型转换为 TensorFlow GraphDef（.pb）格式。

    参数：
        keras_model (keras.Model): 要转换为冻结图格式的 Keras 模型。
        output_file (Path | str): 输出文件路径（后缀会改为 .pb）。
        prefix (str, 可选): 日志前缀，默认为 ""。

    返回：
        (str): 导出的 ``.pb`` 文件路径。

    注意：
        将变量转换为常量，创建用于推理优化的冻结图。
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    m = tf.function(lambda x: keras_model(x))  # full 模型
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    output_file = Path(output_file)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph, logdir=str(output_file.parent), name=output_file.name, as_text=False
    )
    return str(output_file)


def tflite2edgetpu(tflite_file: str | Path, output_dir: str | Path, prefix: str = "") -> str:
    """使用 Edge TPU 编译器将 TensorFlow Lite 模型转换为 Edge TPU 格式。

    参数：
        tflite_file (str | Path): 输入 TensorFlow Lite（.tflite）模型文件的路径。
        output_dir (str | Path): 保存编译后 Edge TPU 模型的输出目录路径。
        prefix (str, 可选): 日志前缀，默认为 ""。

    返回：
        (str): 导出的 Edge TPU 模型文件路径。

    注意：
        如果找不到 Edge TPU 编译器则自动安装。此函数编译 TFLite 模型
        以便在 Google Edge TPU 硬件加速器上获得最佳性能。
    """
    import shlex
    import subprocess

    # 如果未找到，则安装 Edge TPU 编译器
    check_cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert LINUX, f"导出仅支持 Linux 系统。参见 {help_url}"
    if (
        subprocess.run(
            check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, check=False
        ).returncode
        != 0
    ):
        LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
        sudo = "sudo " if is_sudo_available() else ""
        for c in (
            f"{sudo}mkdir -p /etc/apt/keyrings",
            f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | {sudo}gpg --no-tty --dearmor -o /etc/apt/keyrings/google.gpg",
            f'echo "deb [signed-by=/etc/apt/keyrings/google.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | {sudo}tee /etc/apt/sources.list.d/coral-edgetpu.list',
        ):
            subprocess.run(c, shell=True, check=True)
        check_apt_requirements(["edgetpu-compiler"])

    ver = subprocess.run(check_cmd, shell=True, capture_output=True, check=True).stdout.decode().rsplit(maxsplit=1)[-1]
    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")

    cmd = [
        "edgetpu_compiler",
        "--out_dir",
        str(output_dir),
        "--show_operations",
        "--search_delegate",
        "--delegate_search_step",
        "30",
        "--timeout_sec",
        "180",
        str(tflite_file),
    ]  # argv 列表可避免 output_dir/tflite_file 路径中的 shell 元字符问题
    LOGGER.info(f"{prefix} running '{shlex.join(cmd)}'")
    subprocess.run(cmd, check=True)
    return str(Path(output_dir) / f"{Path(tflite_file).stem}_edgetpu.tflite")


def gd_outputs(gd):
    """返回 TensorFlow GraphDef 模型的输出节点名称。"""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))
