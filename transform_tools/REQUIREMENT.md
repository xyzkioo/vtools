# 各种转化脚本

模型有时会有不兼容的情况，这里有一些用于转换的代码

## 文件说明

| 文件                | 用途                                                |
|---------------------|-----------------------------------------------------|
| `ndjson_to_yolo.py` | ndjson下到本地的方法                                |
| `py2kmodel.py`      | pytorch模型转换成kmodel的代码（会产生一个onnx文件） |
| `PY2KM_validate.py` | pytorch模型转换成kmodel的验证代码 （需要onnx文件）  |

## 环境要求

本目录包含三类脚本：NDJSON 数据集转换、PyTorch/Ultralytics 模型转换为 K230
`.kmodel`，以及 ONNX 与 `.kmodel` 的一致性验证。三类脚本的依赖并不完全相同，
可以按实际用途安装。

### 已验证的基础环境

| 组件 | 建议/已验证版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 LTS，Linux x86_64（当前脚本默认按 Linux 路径编写） |
| Python | 3.10.x；`nncase` 官方 K230 文档目前支持 Python 3.7–3.10 |
| Conda | 推荐使用独立环境，例如 `yolo` |
| Ultralytics | 8.4.x；当前项目权重使用过 8.4.128 本地源码 |
| NumPy | 2.2.6 |
| ONNX | 1.21.0 |
| OpenCV/Pillow | 图片处理按脚本实际需要安装；转换脚本使用 Pillow |
| nncase / nncase-kpu | 两者必须使用**完全相同的版本**，并与 K230 SDK/固件匹配 |
| ONNX Runtime | 用于 `PY2KM_validate.py` 的 ONNX 推理 |
| .NET | Linux 建议安装系统 `dotnet-sdk-7.0`，不要只在 Conda 环境中安装 |

当前脚本中的 `TARGET_SIZE=320`、`input_shape=[1,3,320,320]`、K230 目标和
`uint8` 输入是配套设置。修改输入尺寸、量化类型或预处理方式时，必须同步修改
转换脚本、校验脚本和板端推理代码。

### 按脚本安装依赖

#### 1. `ndjson_to_yolo.py`

只需要 Ultralytics 及其依赖；脚本会读取 NDJSON 中的图片并写入本地 YOLO 数据集，
因此还需要网络访问和目标目录的写权限：

```bash
conda create -n k230-tools python=3.10 -y
conda activate k230-tools
python -m pip install ultralytics
```

#### 2. `py2kmodel.py`

该脚本会执行 `PyTorch/Ultralytics → ONNX → ONNX 简化 → nncase INT8 → K230 .kmodel`，需要：

```bash
python -m pip install ultralytics pillow numpy onnx onnxsim
python -m pip install nncase nncase-kpu
```

`nncase-kpu` 必须与 `nncase` 版本一致。例如先查看版本，再安装相同版本：

```bash
python -m pip show nncase
python -m pip install nncase==<相同版本> nncase-kpu==<相同版本>
```

#### 3. `PY2KM_validate.py`

该脚本在主机上用 ONNX Runtime 和 nncase Simulator 对比 ONNX 与 `.kmodel`：

```bash
python -m pip install pillow numpy onnxruntime
python -m pip install nncase nncase-kpu
```

### .NET 与 `DOTNET_ROOT`

nncase Simulator/编译器在 Linux 上可能需要 .NET 运行时。建议先安装系统 SDK：

```bash
sudo apt update
sudo apt install dotnet-sdk-7.0
export DOTNET_ROOT=/usr/share/dotnet
dotnet --version
```

两个 Python 脚本会默认尝试使用：

```text
~/miniconda3/lib/dotnet
```

如果你的 .NET 实际安装在其他位置，请在 PyCharm 的 Run Configuration 或终端中
设置正确的 `DOTNET_ROOT`，例如 `/usr/share/dotnet`。不要把不存在的目录写入
`DOTNET_ROOT`，否则可能出现 `Failed to initialize hostfxr`。

### 环境自检

```bash
python - <<'PY'
import numpy as np
import onnx
import onnxruntime as ort
import nncase
import ultralytics

print("NumPy:", np.__version__)
print("ONNX:", onnx.__version__)
print("ONNX Runtime:", ort.__version__)
print("Ultralytics:", ultralytics.__version__)
print("nncase: OK")
PY
dotnet --version
```

如果 `nncase-kpu` 已安装但版本不一致，或出现
`nncase.simulator.k230.sc: not found`，先检查两个包版本和 `PATH`。转换主机是
x86_64/amd64；生成的 `.kmodel` 能否在 K230 板端运行，还要满足板端固件、K230
SDK 和 nncase runtime 的版本对应关系。

### 运行前需要修改的路径

`py2kmodel.py` 至少检查：

```python
PT_PATH = "best.pt"
ONNX_PATH = "best.onnx"
KMODEL_PATH = "best.kmodel"
CALIB_DIR = "/path/to/calibration/images"
TARGET_SIZE = 320
CALIB_SAMPLES = 200
```

`PY2KM_validate.py` 至少检查：

```python
ONNX_PATH = "best.onnx"
KMODEL_PATH = "best.kmodel"
IMAGE_PATH = "/path/to/test.jpg"
TARGET_SIZE = 320
NORMALIZE = True
```

校准图片的 `letterbox`、填充色 `(128,128,128)`、RGB/CHW 顺序和 `/255` 归一化
必须与实际板端预处理保持一致，否则即使模型文件生成成功，校验结果也没有参考
价值。
