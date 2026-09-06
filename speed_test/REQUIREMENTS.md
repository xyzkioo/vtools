# 环境要求

本文说明本工具包运行 PyTorch 测速、TensorRT 测速和一致性检查所需的环境。
版本分为“当前已验证组合”和“功能依赖”；不要求所有用户完全复制同一台机器，
但同一次对比必须尽量固定软件、硬件和输入条件。

## 1. 当前已验证组合

以下组合对应本项目目前实际运行过的环境：

| 组件 | 已验证版本/设备 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，Linux x86_64 |
| Python | 3.10.20 |
| Conda 环境 | `yolo` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB |
| NVIDIA 驱动 | 580.173.02 |
| CUDA | PyTorch/TensorRT CUDA 13.0 运行时 |
| PyTorch | 2.12.0+cu130 |
| TensorRT Python API | 11.2.1.2 |
| ONNX | 1.21.0 |
| NumPy | 2.2.6 |
| OpenCV | 4.13.0 |
| Ultralytics | 8.4.128（当前 Ultralytics adapter 使用的本地源码） |

当前默认测试条件为 `832×832`、`batch=1`、CUDA、FP32/FP16。更大的输入、batch
或模型可能需要更多显存；8 GB 显存不是所有模型的最低要求，只是当前模型配置
已经验证过。

## 2. 按功能划分的依赖

### 2.1 只运行 PyTorch 测速

必需：

- Python 3.10（建议使用与权重/adapter 相同的 Python 大版本）；
- PyTorch；
- NumPy；
- PyYAML；
- 当前模型 adapter 需要的额外依赖。

PyTorch 测速可以使用 CPU，但 FP16 通常会被跳过，且 CPU 与 GPU 的延迟不能放在
同一张性能表中比较。使用 CUDA 时，需要 NVIDIA GPU 和与 PyTorch CUDA 运行时
兼容的驱动。

### 2.2 使用真实图片 pipeline

如果 `benchmark.image` 有值并且启用了 `adapter_pipeline`，需要：

- `opencv-python`，除非自定义 adapter 自己实现图片读取；
- 对应 adapter 的预处理依赖；
- 可读取的图片文件（常见格式为 JPG、PNG、BMP、WEBP 等）。

`model_call` 只接收已经构造好的 Tensor，不自动代表图片读取和预处理；具体边界
以输出 CSV 的 `scope_note` 为准。

### 2.3 使用 Ultralytics 模型

需要安装与权重训练时兼容的 `ultralytics`，或者在配置中把
`project.ultralytics_repo` 指向本地源码目录。当前项目的 YOLO26 权重使用本地
Ultralytics 源码时，应确认实际导入路径和版本，而不是只看 pip 中是否存在同名包。

可用下面的命令确认导入来源：

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
```

如果权重中包含自定义层，运行环境还必须能导入这些层；否则 checkpoint 检查可能
看到文件存在，但 adapter 实际加载会失败。

### 2.4 使用 TensorRT 测速或构建 engine

必需：

- NVIDIA GPU；
- 与 CUDA 运行时兼容的 NVIDIA 驱动；
- TensorRT Python bindings（当前已验证 11.2.1.2）；
- 能被当前 TensorRT 解析的 ONNX 文件；
- ONNX 自动导出时建议安装 `onnx` 和 `onnxscript`。

当前配置使用：

```yaml
tensorrt:
  builder: python
```

这表示通过 TensorRT Python API 构建 engine，**不要求安装 `trtexec`**。只有选择
`builder: trtexec`，或选择 `builder: auto` 且希望优先使用命令行构建时，才需要
另外安装 `trtexec` 并在配置中写入文件名或绝对路径。

TensorRT 11 使用强类型网络；FP16/BF16 的实际类型主要由导出 ONNX 的 dtype
决定，不要照搬旧版 TensorRT 的 `--fp16`/`--bf16` 参数。

### 2.5 使用一致性检查

一致性检查需要：

- 可加载的 PyTorch adapter；
- CUDA GPU；
- 对应的 ONNX 和 TensorRT engine；
- TensorRT Python bindings；
- `onnx`（报告和输出图读取需要）；
- 图片模式下的图片读取依赖。

一致性检查不重复测速，也不自动证明 mAP、Precision 或 Recall 不变。TensorRT
阶段失败时，`run_all.py` 会跳过本次一致性检查，避免误用磁盘中遗留的旧 engine。

## 3. 推荐安装方式

建议新建独立 Conda 环境，避免系统 Python、PyCharm 解释器和其他项目互相污染：

```bash
conda create -n vision-benchmark python=3.10 -y
conda activate vision-benchmark
```

基础依赖：

```bash
python -m pip install numpy pyyaml opencv-python
```

然后根据功能安装：

```bash
# PyTorch：请按目标 GPU、驱动和 CUDA 运行时选择对应官方构建
python -m pip install torch

# Ultralytics 模型（使用本地源码时可改为 pip install -e /path/to/ultralytics）
python -m pip install ultralytics

# TensorRT Python API
python -m pip install tensorrt

# 自动导出 ONNX 时使用
python -m pip install onnx onnxscript
```

不要为了“凑版本”覆盖一个已经能正常工作的 PyTorch/TensorRT 环境。尤其是
PyTorch、CUDA、TensorRT 和 NVIDIA 驱动应作为一组检查；安装完成后要重新确认
`torch.cuda.is_available()` 和 `import tensorrt`。

## 4. Conda 动态库与 CXXABI 错误

如果出现：

```text
version `CXXABI_1.3.15' not found
```

通常是 ONNX 的二进制扩展加载了系统 `libstdc++.so.6`，而不是当前 Conda 环境
中的版本。可在当前环境安装运行库：

```bash
conda activate yolo
conda install -n yolo -c conda-forge libstdcxx-ng libgcc-ng
```

终端运行时可以显式优先使用 Conda 动态库：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

本工具包的入口还会在 Linux/PyCharm 中自动检查 `sys.prefix/lib`，必要时重新
启动当前 Python，并把该目录放到 `LD_LIBRARY_PATH` 前面。自动处理只能解决动态
库搜索顺序问题；如果 Conda 环境里没有 `libstdc++.so.6`，仍需先安装上面的包。

## 5. 环境自检

### PyTorch/基础依赖

```bash
python - <<'PY'
import numpy as np
import torch
import yaml

print("Python/Torch:", torch.__version__)
print("NumPy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### TensorRT/ONNX

```bash
python - <<'PY'
import onnx
import tensorrt as trt

print("ONNX:", onnx.__version__)
print("TensorRT:", trt.__version__)
print("TensorRT Python API: OK")
PY
```

### 检查是否真的导入了预期的 Ultralytics

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
```

输出的路径应与 `benchmark_config.yaml` 中的 `project.root`、
`project.ultralytics_repo` 预期一致。若同时存在 pip 包和本地源码，优先解决导入
路径，再开始测速，否则模型加载和结果可能不是同一份代码。

## 6. 版本与可复现性注意事项

- PyTorch 测速和 TensorRT 测速必须记录 GPU、驱动、CUDA、PyTorch、TensorRT、
  ONNX、Python 版本。
- TensorRT engine 与 GPU 架构、驱动、CUDA、TensorRT 版本、输入尺寸、batch、
  精度和模型图相关；这些条件变化后建议重新构建 engine。
- 修改模型源码、权重、adapter、输入尺寸或精度后，不要仅因为 engine 文件名
  相同就继续复用旧 engine；应设置 `tensorrt.rebuild_engine: true`。
- ONNX 导出算子集应与当前导出器和 TensorRT 支持情况匹配。当前 PyTorch 2.12
  环境建议使用 opset 18；请求过低的 opset 可能触发自动转换警告或转换失败。
- 不同 batch、输入尺寸和精度的 FPS 不应直接横向比较；速度 CSV 中的
  `scope`/`scope_note` 也必须一致。
- 自定义 adapter 的第三方依赖不可能由工具包统一列全，应在 adapter 文件或项目
  README 中补充其模型专属依赖。
