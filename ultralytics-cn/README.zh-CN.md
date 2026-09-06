# Ultralytics 中文注释精简版

基于上传的 **Ultralytics 8.4.128** 仓库整理，用于阅读网络架构、修改模型，以及训练、验证、推理和按需导出。

本版删除了官方自动化维护、文档站点、Docker 和测试套件，保留完整的 `ultralytics/` 运行时源码。原有 Python 源码、模型 YAML 和中文注释保持不变。

## 1. 留下了什么

### 项目根目录

| 文件或目录 | 作用 |
| --- | --- |
| `ultralytics/` | 模型和运行代码，包括网络层、训练、验证、推理、数据处理等 |
| `pyproject.toml` | 安装入口、基本依赖、可选依赖、打包规则和命令行入口；支持 `pip install -e .` |
| `README.zh-CN.md` | 当前说明文档 |
| `LICENSE` | 原项目 AGPL-3.0 许可证 |
| `.gitignore` | Git 忽略规则，减少缓存、运行结果等文件被误提交 |

交付压缩包不包含 Git 历史。将文件放回已有 Git 仓库时，应保留该仓库自身的 `.git/`。

### ultralytics 包内部

下表路径均相对于 `ultralytics/`。

| 路径 | 保留内容 | 修改模型时的用途 |
| --- | --- | --- |
| `nn/` | 模型构建、基础模块、检测头和推理后端 | 修改网络层、连接方式和模型解析逻辑 |
| `cfg/` | 模型 YAML、默认参数、数据集与跟踪配置、命令行入口 | 修改网络结构和默认运行参数 |
| `models/` | YOLO、RT-DETR、SAM、FastSAM、NAS 等模型接口及任务实现 | 修改检测、分割等任务的训练、验证和预测逻辑 |
| `engine/` | 通用模型接口、训练器、验证器、预测器、结果对象、导出器和调参器 | 修改训练流程、推理流程与结果处理 |
| `data/` | 数据集读取、增强、加载器和数据检查 | 修改数据处理与增强方式 |
| `utils/` | 损失函数、标签分配、指标、绘图、设备工具和回调 | 修改损失、匹配策略和评价逻辑 |
| `optim/` | Muon、MuSGD 优化器 | 为训练器提供优化器实现 |
| `assets/` | `bus.jpg`、`zidane.jpg` 示例图片 | 简单推理检查，不是训练数据集 |
| `trackers/` | ByteTrack、BoT-SORT 等跟踪代码 | 使用 `model.track()` 时需要 |
| `solutions/` | 计数、区域分析等应用封装 | 使用对应应用接口或命令时需要 |
| `__init__.py` | 包版本和公共导入入口 | 支持 `from ultralytics import YOLO` 等用法 |
| `py.typed` | Python 类型信息标记 | 为编辑器和类型检查工具提供支持 |

本版保留了各模型系列和运行接口。`trackers/`、`solutions/` 等不是普通检测训练的必需功能，但仍有入口引用，当前没有进一步裁剪这些接口。

仅留下 `nn/` 和模型 YAML 无法维持现有 `YOLO` API：模型接口依赖 `models/`、`engine/`，训练与推理还依赖 `data/`、`utils/`、`cfg/` 等。训练器会直接导入 `optim` 中的 MuSGD。

## 2. 修改网络主要看哪些文件

下表路径相对于项目根目录。

| 想修改的内容 | 对应位置 |
| --- | --- |
| YOLO26 检测网络结构 | `ultralytics/cfg/models/26/yolo26.yaml` |
| YOLO26 实例分割网络结构 | `ultralytics/cfg/models/26/yolo26-seg.yaml` |
| 已有 P2 / P6 结构配置 | `ultralytics/cfg/models/26/yolo26-p2.yaml`、`yolo26-p6.yaml` |
| 其他系列的网络结构 | `ultralytics/cfg/models/` 下对应目录 |
| 卷积及相关模块 | `ultralytics/nn/modules/conv.py` |
| 特征提取、融合等模块 | `ultralytics/nn/modules/block.py` |
| 检测头、分割头等 | `ultralytics/nn/modules/head.py` |
| Transformer 相关模块 | `ultralytics/nn/modules/transformer.py` |
| 激活函数 | `ultralytics/nn/modules/activation.py` |
| 模块导出入口 | `ultralytics/nn/modules/__init__.py` |
| YAML 解析和模型构建 | `ultralytics/nn/tasks.py`，重点看 `parse_model()` |
| 损失函数 | `ultralytics/utils/loss.py` |
| 标签分配与匹配 | `ultralytics/utils/tal.py` |
| 数据增强 | `ultralytics/data/augment.py` |
| YOLO 检测训练、验证、预测 | `ultralytics/models/yolo/detect/` |
| YOLO 实例分割训练、验证、预测 | `ultralytics/models/yolo/segment/` |
| 通用训练循环与优化器选择 | `ultralytics/engine/trainer.py` |

新增自定义模块通常需要依次处理：

1. 在 `nn/modules/` 中实现模块。
2. 在 `nn/modules/__init__.py` 中导出，并在 `nn/tasks.py` 中导入。
3. 根据模块接口调整 `parse_model()` 的通道计算、参数传递和重复次数处理。
4. 在模型 YAML 中引用模块，检查输入层编号、通道数和特征图尺寸。
5. 先检查模型构建与前向传播，再开始训练。

自定义 YAML 应明确使用的模型尺度；复制结构文件改名时，注意 `scales` 与 `scale` 的关系，避免意外使用其他尺度。

`ultralytics/cfg/models/26/change2onnx.py` 是上传仓库中已有的 ONNX 导出辅助脚本，已原样保留。它不属于网络结构定义，不用导出时不需要运行它。

## 3. 环境要求

以下为本仓库 `pyproject.toml` 声明的部分基础要求，完整版本限制和平台条件以该文件为准。

| 项目 | 声明要求或用途 |
| --- | --- |
| Python | `>=3.8` |
| PyTorch | `>=1.8.0`；Windows 有额外版本排除条件 |
| torchvision | `>=0.9.0`，需要与 PyTorch 兼容 |
| NumPy | `>=1.23.0`；macOS 有额外版本限制 |
| OpenCV | `>=4.7.0`，排除 `4.13.0.90` |
| 其他基础依赖 | filelock、matplotlib、Pillow、PyYAML、requests、psutil、polars、nvidia-ml-py、ultralytics-thop |
| ultralytics-platform | Python `>=3.11` 时由基础依赖声明引入 |
| 构建依赖 | setuptools `>=70.0.0,<=84.0.0`、wheel |

这些是包声明的约束，不表示所有版本组合和硬件都经过验证。已有可正常训练的 `yolo` 环境可以继续使用；GPU 运行需要驱动、PyTorch CUDA 构建和 torchvision 相互兼容。

ONNX、TensorRT、OpenVINO 等不是普通 PyTorch 训练的基础要求。相应导出代码和可选依赖分组仍保留，需要使用时再配置。

## 4. 安装并确认使用本地源码

在包含 `pyproject.toml` 的项目根目录执行：

```bash
conda activate yolo
python -m pip install -e .
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
```

打印的路径应指向当前项目中的 `ultralytics/__init__.py`。`-e` 表示可编辑安装，之后修改源码会在重新启动的 Python 进程中生效。

如果现有环境的依赖已经齐全，只需要将导入切换到这份源码，可以使用：

```bash
python -m pip install -e . --no-deps
```

`--no-deps` 不会补齐缺少的依赖，因此不适合直接用于空环境。PyCharm 应选择相同的 Conda 解释器；修改源码后重新运行脚本。

## 5. 构建、训练、验证和推理

### 仅检查网络能否构建

```python
from ultralytics import YOLO

model = YOLO("yolo26s.yaml")
model.info()
```

该示例从架构构建模型，不加载预训练权重。YOLO26 的不同尺度使用共享结构配置，`yolo26s.yaml` 会由框架解析到对应结构和 s 尺度。

### 训练检测模型

将下面路径换成自己的文件路径。参数用于展示调用方法，不是已经验证的最优训练配置。

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26s.yaml")
    model.load("/absolute/path/yolo26s.pt")
    model.train(
        data="/absolute/path/data.yaml",
        imgsz=832,
        epochs=150,
        batch=2,
        device=0,
        project="runs",
        name="train_custom",
    )
```

结构改动后，预训练参数的加载量取决于参数名称与张量形状是否匹配，应检查加载日志。若从头训练，可省略 `model.load()`。

### 验证和预测

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/absolute/path/best.pt")
    model.val(data="/absolute/path/data.yaml", imgsz=832, device=0)
    model.predict(
        source="/absolute/path/test.jpg",
        imgsz=832,
        conf=0.25,
        device=0,
        save=True,
    )
```

使用 CPU 时将 `device=0` 改为 `device="cpu"`。实例分割模型可从 `yolo26s-seg.yaml` 构建，并使用对应分割权重和分割标注数据。

压缩包提供源码和示例图片，不包含你的训练集或训练好的模型权重。

## 6. 已移除的内容

| 内容 | 原用途 |
| --- | --- |
| `.github/` | 12 个 GitHub Actions 工作流、Issue/PR 模板和维护配置 |
| `docs/`、`mkdocs.yml` | 文档站点及构建配置 |
| `docker/`、`.dockerignore` | Docker 镜像构建 |
| `tests/` | 官方自动化测试套件 |
| `AGENTS.md`、`CLAUDE.md`、`CONTRIBUTING.md` | 原项目开发协作说明 |
| `CITATION.cff` | 引用元数据 |
| `__pycache__/`、`.pyc` | Python 生成的缓存 |
| 原压缩包中的 `.git/` | Git 历史与本地仓库元数据，不随精简包交付 |

`pyproject.toml` 同步清理了开发 extras、测试文件打包路径和相关工具配置，并将 README 引用修正为 `README.zh-CN.md`。运行依赖和导出等可选功能分组保留。

原 `Conda Builds` 工作流定时检查 `conda-forge/ultralytics-feedstock` 的 PR。提供的失败日志显示：该外部仓库的 PR #645（Ultralytics v8.4.140）出现 `linux_64_` 检查失败，工作流执行 `assert False` 后报红。这与本地模型训练和中文注释修改无关。精简版已移除该工作流。

将精简版同步到已有 GitHub 仓库时，要实际删除旧 `.github/` 等已移除内容，再提交变更。仅覆盖解压不会删除旧文件，也不会自动停止远程工作流。

## 7. 验证范围与许可证

精简时已确认保留的 364 个运行时文件与上传版本逐字节一致，通过 Python 语法检查、TOML 配置检查和 wheel 安装包构建。处理环境未安装 PyTorch，未执行模型前向传播、训练或 GPU 测试。

许可证见根目录 `LICENSE`，源码中的原有授权声明保留。
