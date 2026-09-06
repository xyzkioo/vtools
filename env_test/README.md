# Ultralytics 本地源码安装与环境检验向导

这个文件夹用于确认 `vtools/ultralytics-cn/` 这份精简版 Ultralytics 是否安装成功，以及当前 Python 是否真的加载了本地源码。

仓库中有两种容易混淆的东西：

- **项目目录名**：当前是 `ultralytics-cn`，可以改成别的名字；
- **Python 包名和导入名**：当前仍是 `ultralytics`，代码使用 `from ultralytics import YOLO`。

建议先保留 Python 包名和导入名，只改变外层目录名。这样已有训练脚本、`.pt` 权重和 Ultralytics 的内部模块路径可以继续使用。

## 一、确认目录结构

从 GitHub 下载或克隆后，目录应类似这样：

```text
vtools/
├── install_check/
│   ├── check_install.py
│   └── README.md
└── ultralytics-cn/
    ├── pyproject.toml
    └── ultralytics/
        ├── __init__.py
        ├── nn/
        ├── models/
        └── cfg/
```

如果你把 `ultralytics-cn` 改名为 `ultralytics-custom`，只需要把下面命令中的目录名替换掉。`--source` 必须指向包含 `pyproject.toml` 和包目录的项目根目录。

## 二、已有 Conda 环境安装（推荐）

如果当前的 `yolo` 环境已经能够运行 PyTorch，可以直接接入本地源码：

```bash
cd /path/to/vtools
conda activate yolo
python -m pip install -e ./ultralytics-cn
```

`-e` 是可编辑安装。源码修改后，重新启动 Python 或 PyCharm 运行即可生效。

如果这个环境的依赖已经完整，不想让 pip 改动已有的 PyTorch、NumPy 或 CUDA 组合，可以使用：

```bash
python -m pip install -e ./ultralytics-cn --no-deps
```

`--no-deps` 只安装本地源码的入口，不会补齐缺失依赖；空环境不要使用它。

## 三、新电脑创建环境

先创建独立环境：

```bash
conda create -n yolo-custom python=3.10 -y
conda activate yolo-custom
python -m pip install --upgrade pip setuptools wheel
```

然后根据目标电脑的 NVIDIA 驱动、CUDA 和显卡，从 PyTorch 官方安装页面选择匹配的 PyTorch 与 torchvision。不要直接把另一台电脑的 CUDA 版本照搬过来。

确认 PyTorch 后安装本地源码：

```bash
cd /path/to/vtools
python -m pip install -e ./ultralytics-cn
```

本仓库 `pyproject.toml` 声明的基础范围包括 Python `>=3.8`、PyTorch `>=1.8.0`、torchvision `>=0.9.0`、NumPy、OpenCV、Pillow、PyYAML、Matplotlib、Requests、psutil、polars、nvidia-ml-py 和 ultralytics-thop。实际使用时，PyTorch 与 torchvision 必须互相匹配。

当前已经验证过的组合是：Ubuntu 22.04.5、Python 3.10.20、PyTorch 2.12.0+cu130、RTX 5060 Laptop 8 GB、默认输入 832×832。这个组合用于参考，不要求所有电脑完全相同。

## 四、运行环境检验

在 `vtools` 根目录运行：

```bash
python install_check/check_install.py --source ./ultralytics-cn
```

脚本会检查：

1. 当前使用的 Python 解释器；
2. `ultralytics` 的真实导入文件和版本；
3. 是否导入了 `ultralytics-cn/ultralytics/`，避免误用 pip 版；
4. PyTorch、torchvision、NumPy、PyYAML、OpenCV、Pillow；
5. pip 依赖冲突；
6. `yolo26.yaml` 是否能构建模型并完成一次 Tensor 前向传播；
7. CUDA 是否可用。没有 GPU 时会改用 CPU，不会因此判定基础安装失败。

看到下面这样的结果，表示源码来源正确：

```text
[通过] 源码来源: Python 实际加载的是指定的本地源码。
       期望：/path/to/vtools/ultralytics-cn/ultralytics/__init__.py
       实际：/path/to/vtools/ultralytics-cn/ultralytics/__init__.py
```

如果实际路径包含 `site-packages/ultralytics/`，说明当前仍然使用 pip 版。重新执行：

```bash
conda activate yolo
cd /path/to/vtools
python -m pip install -e ./ultralytics-cn
python install_check/check_install.py --source ./ultralytics-cn
```

如果你是在 PyCharm 中运行，检查 PyCharm 使用的解释器是否与终端中的 `sys.executable` 相同。修改解释器后重新启动运行配置。

## 五、检查自己的权重

安装检查默认不需要训练权重。要额外确认自定义 `.pt` 能否由这份源码加载并预测：

```bash
python install_check/check_install.py \
    --source ./ultralytics-cn \
    --weights /absolute/path/to/best.pt
```

脚本会用一张黑色合成图片执行一次预测，不会下载数据集，也不会修改权重。你的权重如果依赖自定义模块，必须确保这些模块已经包含在当前源码中，并且实际导入路径检查通过。

如果只想排查基础导入而暂时不进行任何模型构建/预测，可以写：

```bash
python install_check/check_install.py \
    --source ./ultralytics-cn \
    --weights /absolute/path/to/best.pt \
    --skip-model
```

`--skip-model` 会同时跳过 YAML 构建和权重预测，因此通常只在排查基础导入时使用。

## 六、常用参数

| 参数 | 作用 |
| --- | --- |
| `--source PATH` | 指定本地源码项目根目录；推荐始终填写 |
| `--weights PATH` | 加载并预测指定 `.pt` |
| `--yaml PATH` | 指定要构建的模型 YAML |
| `--device auto/cpu/cuda:0` | 选择运行设备；默认自动选择 |
| `--input-size 64` | 合成测试输入尺寸，脚本会调整为 32 的倍数 |
| `--check-export` | 检查 ONNX、TensorRT Python API |
| `--skip-model` | 跳过模型构建和推理 |
| `--no-local-check` | 允许使用 pip 版，不比较源码路径 |
| `--expected-version none` | 跳过版本号比较 |
| `--strict` | 将版本不一致、pip check 冲突和可选依赖缺失判为失败 |

TensorRT 不是普通 PyTorch 训练的必需项。只有需要导出或 TensorRT 测速时才运行：

```bash
python install_check/check_install.py \
    --source ./ultralytics-cn \
    --check-export
```

## 七、如果要改包名

### 只改外层目录名

这是推荐做法。例如：

```text
ultralytics-cn/  →  ultralytics-custom/
```

安装命令改为：

```bash
python -m pip install -e ./ultralytics-custom
python install_check/check_install.py --source ./ultralytics-custom
```

代码仍然使用：

```python
from ultralytics import YOLO
```

### 只改安装显示名称

可以把 `pyproject.toml` 中 `[project]` 的 `name` 改成例如 `ultralytics-vtools`，但 Python 导入名仍是 `ultralytics`。此时建议检验时使用：

```bash
python install_check/check_install.py \
    --source ./ultralytics-cn \
    --distribution ultralytics-vtools
```

如果没有同步修改代码中的自引用依赖，不要随意替换 `pyproject.toml` 中依赖列表里的 `ultralytics`。

### 改 Python 导入名

例如把 `ultralytics/` 改成 `ultralytics_custom/`，需要同步修改源码内部所有 `from ultralytics...`、动态导入、命令行入口和已有权重可能保存的模块路径。旧 `.pt` 权重可能因此无法直接加载，除非做兼容映射。

只有完成这些修改后，才使用：

```bash
python install_check/check_install.py \
    --source ./ultralytics-custom \
    --module ultralytics_custom \
    --distribution ultralytics-vtools
```

## 八、常见错误

### 1. `实际加载的不是指定本地源码`

当前解释器导入了另一份 `ultralytics`。确认 `python -m pip` 和运行脚本的 `python` 是同一个解释器，再重新执行本地可编辑安装。不要只看 `pip list`，以检验脚本打印的实际 `__file__` 为准。

### 2. `No module named torch` 或 torchvision 导入失败

先安装与显卡/驱动匹配的 PyTorch 和 torchvision，再安装本地源码。若已有能训练的环境，优先保留原有 PyTorch 组合，不要为了安装源码重复覆盖它。

### 3. `CXXABI_1.3.15 not found`

在 Conda 环境中补齐 C++ 运行库：

```bash
conda activate yolo
conda install -c conda-forge libstdcxx-ng libgcc-ng -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

然后重新运行检验脚本。

### 4. YAML 构建失败或自定义层找不到

先确认源码来源检查通过，再确认自定义模块已在 `ultralytics/nn/modules/` 中实现，并在 `nn/modules/__init__.py` 和 `nn/tasks.py` 中正确导出/导入。最后用 `--yaml` 指向实际 YAML。

### 5. 终端通过、PyCharm 失败

这是两个运行配置使用了不同 Python 的典型表现。分别在终端和 PyCharm 打印 `sys.executable`，并将 PyCharm 解释器设置为已经通过检验的 Conda 环境。

## 九、通过标准

以下项目全部显示“通过”时，表示本地安装和基本模型运行链路已经确认：

- Python 版本符合要求；
- `ultralytics` 能导入，并存在 `YOLO`；
- 实际导入路径指向指定的本地源码；
- PyTorch、torchvision 等核心依赖可导入；
- `yolo26.yaml` 完成一次构建和前向传播；
- 如果指定了权重，权重加载和合成图片预测通过。

CUDA、TensorRT、ONNX 属于按功能启用的检查；CPU 机器没有 CUDA 时，脚本会明确标记为警告并保留基础结论。
