# 视觉妙妙小工具（自用）

视觉深度学习中往往会遇到各种需要测试的场景，**vtools** 将模型源码、环境检验、检测质量诊断、性能测速和转换脚本放在一起，方便量化模型表现、对比改进效果和准备部署。

项目以 Python 脚本和 YAML 配置为主要入口，可在终端或 PyCharm 中运行。测速与诊断支持通过 adapter（模型适配器）接入自定义模型；内置 Ultralytics 接口可直接使用兼容的 YOLO 权重。

## 项目组成

| 目录 / 文件 | 主要内容 | 使用入口 / 说明 |
| --- | --- | --- |
| [ultralytics-cn/](ultralytics-cn/) | Ultralytics 8.4.128 中文注释精简版；保留模型构建、训练、验证、推理和按需导出所需的运行时源码 | [源码说明](ultralytics-cn/README.zh-CN.md)、[安装配置](ultralytics-cn/pyproject.toml) |
| [env_test/](env_test/) | 检查 Python、依赖、实际导入的源码路径、CUDA、模型构建与前向传播 | [check_install.py](env_test/check_install.py)、[安装与检验向导](env_test/README.md) |
| [model_diagnostics/](model_diagnostics/) | 目标检测错误分析：漏检、错分类、多余框、重复框、定位偏差、框重叠和差图；常规 mAP 等交给 Ultralytics 验证 | [run_model_diagnostics.py](model_diagnostics/run_model_diagnostics.py)、[完整说明](model_diagnostics/docs/README.md) |
| [model_visualization/](model_visualization/) | 通用特征图、聚合激活图、可选 Grad-CAM/LayerCAM，以及检测头 raw candidate→top-k→final 阶段追踪 | [run_visualization.py](model_visualization/run_visualization.py)、[PyCharm 配置](model_visualization/config/pycharm_run.yaml) |
| [speed_test/](speed_test/) | PyTorch / TensorRT 测速、checkpoint 可恢复性检查、转换前后输出一致性检查；功能由 `benchmark_config.yaml` 的 `modules` 控制 | [run_all.py](speed_test/run_all.py)、[使用说明](speed_test/README.md) |
| [transform_tools/](transform_tools/) | NDJSON 数据集转 YOLO、Ultralytics 权重转 ONNX / K230 `.kmodel`、ONNX 与 `.kmodel` 输出对比 | [脚本与环境说明](transform_tools/REQUIREMENT.md) |
| [模型勘误方法.md](模型勘误方法.md) | 模型问题排查的思路与参考方法 | 阅读文档 |
| [LICENSE](LICENSE) | 仓库根目录许可证 | 引入的 Ultralytics 源码另保留其许可证 |

主要子目录的分工：

| 位置 | 作用 |
| --- | --- |
| `ultralytics-cn/ultralytics/nn/` | 网络模块、检测头、模型解析与构建 |
| `ultralytics-cn/ultralytics/cfg/models/` | 各模型系列的网络结构 YAML |
| `ultralytics-cn/ultralytics/models/`、`engine/`、`data/`、`utils/` | 模型任务接口、训练/验证/预测流程、数据增强、损失与指标 |
| `model_diagnostics/config/` | 诊断配置；日常主要修改 `pycharm_run.yaml` |
| `model_diagnostics/diagnostics/`、`adapters/` | 评估引擎、模型运行器、自定义适配器模板 |
| `model_diagnostics/docs/`、`data/` | 使用教程、指标解释和输入数据格式说明 |
| `model_visualization/core/`、`adapters/` | 输入变换、激活捕获、特征/CAM 渲染、Ultralytics raw 输出和检测头阶段追踪 |
| `model_visualization/config/`、`runs/` | PyCharm/CLI 配置和离线可视化报告；`runs/` 由运行时自动创建 |
| `speed_test/core/`、`backends/`、`checks/` | 配置与运行目录管理、测速后端、一致性检查 |
| `speed_test/adapters/`、`tests/` | 自定义模型适配器模板与测试 |

**范围说明：** 当前 `model_diagnostics` 评估的是目标检测框，不评估实例分割掩码。其他模型接入需要提供统一预测文件或实现相应 adapter。网络结构阅读可从 `ultralytics-cn` 入手；通用模型可视化独立放在 `model_visualization/`，首版内置 Ultralytics/YOLO26 适配器。

## 环境要求

### 基础环境

日常使用建议采用 **Python 3.10 + 独立 Conda 环境**。仓库主要在 Ubuntu 22.04 / Linux x86_64 下使用；其他系统需要调整路径，并确认对应依赖能够安装。

本地 Ultralytics 的 `pyproject.toml` 声明 Python `>=3.8`、PyTorch `>=1.8.0`、torchvision `>=0.9.0` 等基础约束，**这些下限不等于整套工具和 YOLO26 在所有版本组合上均已验证**。完整限制以 [pyproject.toml](ultralytics-cn/pyproject.toml) 为准，PyTorch 与 torchvision 应配套安装。

以下为仓库现有环境文档记录的已验证组合，供复现参考，并非每台电脑都必须完全一致：

| 组件 | 记录的版本 / 设备 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，Linux x86_64 |
| Python | 3.10.20 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB |
| NVIDIA 驱动 | 580.173.02 |
| PyTorch | 2.12.0+cu130 |
| CUDA 运行时 | 13.0 |
| Ultralytics | 8.4.128，本仓库本地源码 |
| NumPy / OpenCV | 2.2.6 / 4.13.0 |
| ONNX | 1.21.0 |
| TensorRT Python API | 11.2.1.2 |

8 GB 是上述设备的显存容量，不是所有模型的最低要求。实际显存需求与模型、输入分辨率、batch 和精度有关。

### 按功能安装依赖

| 功能 | 所需依赖 |
| --- | --- |
| 已有预测文件的检测诊断（模式 A） | 评估核心使用 Python 标准库；YAML 配置需要 PyYAML，读取 YOLO 图片尺寸需要 Pillow；无需为此加载 PyTorch 模型 |
| 本地 YOLO 模型开发、训练、推理、诊断（模式 B） | 配套的 PyTorch / torchvision，以及本地 `ultralytics-cn` 声明的依赖 |
| 自定义模型诊断（模式 C） | 诊断基础依赖，加上 adapter 所用框架和模型依赖 |
| PyTorch 测速 | PyTorch、NumPy、PyYAML 和模型 adapter 依赖；内置图片流程还需要 OpenCV |
| 特征图 / 阶段可视化 | PyTorch、NumPy、PyYAML、OpenCV、Pillow；使用本地 Ultralytics 权重时还需 `ultralytics-cn` 的依赖 |
| TensorRT 测速与一致性检查 | NVIDIA GPU、兼容驱动、TensorRT Python API；导出/检查另需 ONNX，导出链路可能需要 onnxscript |
| NDJSON 转 YOLO | Ultralytics 及其依赖；下载清单中的图片需要网络 |
| K230 转换与校验 | PyTorch / Ultralytics、NumPy、Pillow、ONNX、onnxsim、ONNX Runtime，以及与目标 SDK 配套的 nncase、nncase-kpu 和所需 .NET 运行时 |

普通 PyTorch 训练和测速不要求安装 TensorRT 或 nncase。CPU 可以用于基础检验和适用模型的运行，PyTorch 测速在 CPU 上会跳过 FP16。

详细环境说明见 [测速环境](speed_test/REQUIREMENTS.md)、[诊断依赖](model_diagnostics/requirements.txt) 和 [转换环境](transform_tools/REQUIREMENT.md)。K230 建议单独配置环境，`nncase` 与 `nncase-kpu` 版本须一致，并与板端 SDK/runtime 配套。

## 安装与首次检验

以下终端命令以 **vtools 仓库根目录** 为工作目录；文中的 `/path/to/...` 都需要替换为自己的真实路径。

### 1. 获取仓库

```bash
git clone https://github.com/xyzkioo/vtools.git
cd vtools
```

### 2. 准备 Python 和 PyTorch

新电脑可以创建环境：

```bash
conda create -n vtools python=3.10 -y
conda activate vtools
python -m pip install --upgrade pip setuptools wheel
```

已有能正常训练的环境时，直接激活原环境即可，例如 `conda activate yolo`。

需要运行模型时，先根据本机系统、显卡和驱动，在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择配套的 PyTorch / torchvision 安装命令。只使用 CPU 时选择 CPU 构建；GPU 机器应选择支持该显卡的 CUDA 构建，不必照搬上表的 CUDA 版本。

### 3. 安装本地源码与诊断依赖

```bash
python -m pip install -e ./ultralytics-cn
python -m pip install -r ./model_diagnostics/requirements.txt
```

`-e` 表示可编辑安装：之后修改 `ultralytics-cn/ultralytics/` 源码，重新启动 Python 进程即可使用修改后的代码。外层文件夹叫 `ultralytics-cn`，Python 导入名仍然是 `ultralytics`。

已有环境的依赖确实完整、只想切换到本地源码时，可以将第一条命令换成：

```bash
python -m pip install -e ./ultralytics-cn --no-deps
```

`--no-deps` 不会补齐缺失依赖，空环境不要使用。若只使用诊断模式 A，安装 `model_diagnostics/requirements.txt` 即可，读取 NDJSON 等额外转换流程除外。

### 4. 确认安装与源码来源

```bash
python env_test/check_install.py --source ./ultralytics-cn
```

脚本会报告解释器、核心依赖、Ultralytics 实际导入位置、pip 依赖冲突、CUDA 状态，并从 YAML 构建模型执行合成输入前向传播。默认不需要自己的数据集或训练权重；无可用 CUDA 时可使用 CPU。

导入路径应指向：

```text
/path/to/vtools/ultralytics-cn/ultralytics/__init__.py
```

如果实际导入另一份 `site-packages/ultralytics/`，本地源码来源检查会判为失败。需要检查自己的权重时运行：

```bash
python env_test/check_install.py --source ./ultralytics-cn --weights /path/to/best.pt
```

通过基础检验表示安装、导入和基本前向链路可用；自己的权重、数据集和部署后端仍需分别验证。`--check-export` 可额外检查 ONNX / TensorRT 导入，`--strict` 可将部分警告提升为失败。

## 使用各工具

### 1. 修改模型架构

最常用的源码位置如下，路径均相对于仓库根目录：

| 要修改的内容 | 文件 |
| --- | --- |
| YOLO26 检测 / 实例分割结构 | `ultralytics-cn/ultralytics/cfg/models/26/yolo26.yaml`、`yolo26-seg.yaml` |
| 卷积、特征提取与融合模块 | `ultralytics-cn/ultralytics/nn/modules/conv.py`、`block.py` |
| 检测头 / 分割头 | `ultralytics-cn/ultralytics/nn/modules/head.py` |
| 模块导出与模型解析 | `ultralytics-cn/ultralytics/nn/modules/__init__.py`、`ultralytics-cn/ultralytics/nn/tasks.py` 中的 `parse_model()` |
| 损失函数 / 标签分配 | `ultralytics-cn/ultralytics/utils/loss.py`、`tal.py` |
| 数据增强 | `ultralytics-cn/ultralytics/data/augment.py` |

新增模块后，需要同步检查模块导出、`tasks.py` 导入、YAML 引用和通道/参数解析。训练、验证和预测示例见 [源码 README](ultralytics-cn/README.zh-CN.md)。

### 2. 目标检测诊断

修改 [model_diagnostics/config/pycharm_run.yaml](model_diagnostics/config/pycharm_run.yaml)，选择一种模式：

| 模式 | 输入 | 需要修改 |
| --- | --- | --- |
| A | 已有预测 JSON/JSONL 或 COCO 预测结果 | `mode: A`、`mode_a.predictions` |
| B | 兼容的 Ultralytics `.pt` | `mode: B`、`mode_b.weights` |
| C | 自定义模型与 adapter | `mode: C`、`mode_c.weights`、`mode_c.adapter` |

三种模式都要填写数据集配置，例如 `dataset.data` 指向检测数据集 `data.yaml`，`dataset.split: val` 选择验证集。

以仓库根目录作为路径基准，模式 B 可将现有配置中的对应项改为以下内容；**合并到原有配置节，保留其余参数，不要重复添加同名 YAML 节**：

```yaml
mode: B

project:
  root: ..
  ultralytics_repo: ultralytics-cn

run:
  root: model_diagnostics/runs
  name: auto

dataset:
  data: /path/to/dataset/data.yaml
  split: val

mode_b:
  name: ultralytics_pt
  weights: /path/to/best.pt
  task: detect

modules:
  diagnostics.missed: true
  diagnostics.classification: true
  diagnostics.overlap: true
  diagnostics.threshold_sweep: false
  output.bad_cases: true
  output.images: false
  output.html: false
```

然后运行：

```bash
python model_diagnostics/run_model_diagnostics.py
```

该入口默认读取 `model_diagnostics/config/pycharm_run.yaml`，也可以用 `--config` 指定另一份 YAML。按上面的路径设置，结果保存在 `model_diagnostics/runs/runN/ultralytics_pt/`。

主要查看：

| 输出文件 | 用途 |
| --- | --- |
| `report.md`、`summary.json` | 总览、整体指标和诊断摘要 |
| `per_gt.csv`、`per_prediction.csv` | 每个真实目标的检出/漏检情况、每个预测框的匹配/错误类型 |
| `per_image.csv` | 每张图片的检测和计数表现 |
| `group_metrics.csv` | 按目标尺寸、密度、类别等分组对比 |
| `threshold_sweep.csv`、`fp_budget.csv` | 仅在开启 `diagnostics.threshold_sweep` 时生成，用于观察置信度阈值与召回的取舍 |
| `predictions_adapter.json` | 模式 B/C 本次推理生成的预测，便于复查或后续用模式 A 分析 |

`benchmark.conf: 0.001` 用于推理时保留低分候选，`diagnostics.score_threshold: 0.25` 是正式 P/R/F1 的工作阈值，两者用途不同。候选召回还由 `diagnostics.candidate_threshold` 控制；已被推理阶段删掉的框，后续降低诊断阈值也无法找回。AP/mAP 与常规验证使用 Ultralytics 原生 `val`，不会由诊断入口重复计算。

前后处理阶段对比需要另外提供真实的 raw 预测，才能生成 `stage_comparison.csv`；普通模式 B 的最终检测框不自动等于“NMS 前候选”。指标口径见 [METRICS.md](model_diagnostics/docs/METRICS.md)，数据格式与 adapter 接口见 [完整教程](model_diagnostics/docs/README.md)。

### 3. 特征图、CAM 与检测阶段追踪

这个入口用于回答“问题从哪一层开始出现”以及“候选框在哪个阶段被删掉”。它和 Ultralytics 已有的训练曲线、验证指标、预测保存功能分开，输出一份可离线打开的 `index.html`。

先安装可视化的轻量依赖，并确保当前环境使用仓库内的 Ultralytics 源码：

```bash
python -m pip install -r ./model_visualization/requirements.txt
python -m pip install -e ./ultralytics-cn
```

复制 [model_visualization/config/pycharm_run.yaml](model_visualization/config/pycharm_run.yaml) 为自己的配置，至少修改 `model.weights` 和 `input.source`，然后运行：

```bash
python model_visualization/run_visualization.py --config model_visualization/config/pycharm_run.yaml
```

可视化功能由配置中的 `modules` 独立控制：

```yaml
modules:
  visualization.features: true
  visualization.cam: false
  visualization.stage_trace: false
```

默认会捕获检测头进入的 P 层特征，保存以下内容：

| 输出 | 用途 |
| --- | --- |
| `layers.csv` | 当前权重实际加载后的模块名、类型和参数量；先用 `mode: list_layers` 找到要观察的层 |
| `features/`、`activations/` | 指定通道图、聚合激活图和原图叠加图；`layers.modules` 可改成任意模块名 |
| `cams/` | 开启 `modules.visualization.cam` 后生成 Grad-CAM/LayerCAM；解释前向固定使用 FP32 |
| `stage_trace/` | 开启 `modules.visualization.stage_trace` 后保存 raw candidate、head top-k、final、事件 JSONL、P 层/网格索引和 overlay |
| `canonical/` | raw/final 的 canonical JSON，可直接交给 `model_diagnostics` 做阶段对照 |

阶段追踪只使用检测头自己的索引：每个候选会记录 `raw_index`、`source_level`（如 P3/P4/P5）、`source_index`、stride 和网格坐标；YOLO26 的 `one2one` 分支按 top-k 与置信度形成 final，`one2many` 分支沿用 Ultralytics NMS 并保留 `return_idxs=True` 的精确索引。不会用近似 IoU 推断候选来源。首版目标是检测任务，其他模型需要实现同等 adapter 接口。

TensorRT engine 仍由已有的 `rebuild_engine` 开关控制：成功构建后旁边会写一个 `.engine.build.json`，记录权重、ONNX、输入尺寸、batch、精度、融合、opset、workspace、TensorRT/CUDA/GPU 等来源。复用已有 engine 时只提示元数据差异，不会悄悄替换或自动引入新的哈希缓存策略；需要重建时设置 `rebuild_engine: true`。

### 4. 性能测速与一致性检查

修改 [speed_test/benchmark_config.yaml](speed_test/benchmark_config.yaml)，至少配置项目根目录、源码目录、模型权重和图片。以仓库根目录为基准，可合并以下片段到原配置：

```yaml
project:
  root: ..
  ultralytics_repo: ultralytics-cn

run:
  enabled: true
  root: speed_test/runs
  name: auto

models:
  - name: my_detector
    weights: /path/to/best.pt
    adapter: ultralytics
    task: detect
    enabled: true

benchmark:
  image: /path/to/test.jpg
  device: auto
  input_size: [832, 832]
  batch_size: 1

modules:
  checkpoint.inspect: true
  checkpoint.load_check: true
  model.parameters: false
  model.flops: false
  speed.pytorch_call: true
  speed.pytorch_pipeline: false
  speed.tensorrt_call: true
  memory.pytorch_peak: false
  export.onnx: false
  build.tensorrt: true
  consistency.tensor: true
  consistency.detection: false
```

根据用途选择入口：

| 功能 | 在仓库根目录运行 |
| --- | --- |
| 检查 checkpoint 结构与模型可加载性 | `python speed_test/inspect_checkpoint.py` |
| 只测 PyTorch | `python speed_test/benchmark_pytorch.py` |
| 导出 / 构建并测试 TensorRT | `python speed_test/benchmark_tensorrt.py` |
| 检查 PyTorch 与已有 TensorRT engine 的输出一致性 | `python speed_test/check_consistency.py` |
| 按 YAML 配置执行各阶段并汇总 | `python speed_test/run_all.py` |

首次使用可先运行 `benchmark_pytorch.py`。当前 YAML 默认启用 TensorRT 和一致性检查；未配置 TensorRT 时，若使用 `run_all.py`，应将 `run_all.run_tensorrt`、`run_all.run_consistency` 改为 `false`。

需要 TensorRT 时，按 [NVIDIA 官方安装说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) 安装与环境匹配的版本，并按需要安装导出依赖：

```bash
python -m pip install onnx onnxscript
```

`tensorrt.builder: python` 使用 TensorRT Python API 构建 engine，不需要 `trtexec`。更改模型、权重、输入形状或导出配置后，应检查缓存并设置 `tensorrt.rebuild_engine: true` 重新生成。

测速明细主要是 `vision_speed_v2.csv` 和 `vision_tensorrt_v2.csv`；一键运行另生成 `vision_benchmark_summary.csv`，一致性报告为 `consistency.csv` / `consistency.json`。实际位置以终端打印路径为准。

比较速度时应保持硬件、输入尺寸、batch、精度和计时范围一致。`model_call` 不含预处理/后处理；`adapter_pipeline` 的范围由 adapter 决定，请查看 CSV 的 `scope_note`。输出一致性检查通过也不代替验证集上的检测精度评估。详见 [CSV 指标说明](speed_test/CSV_METRICS_GUIDE.md)。

测速 test 内的参数量、峰值显存、PyTorch 纯调用、PyTorch pipeline、ONNX 导出、TensorRT 构建、engine 调用和一致性张量/框匹配现在都可在 `benchmark_config.yaml` 的 `modules` 中单独开关。终端可用同样的模块 ID 临时覆盖配置，例如：

```bash
python speed_test/benchmark_pytorch.py --only speed.pytorch_call
python speed_test/benchmark_tensorrt.py --only export.onnx
python speed_test/run_all.py --disable speed.pytorch_pipeline,memory.pytorch_peak
```

目标检测诊断也按同一粒度拆分：漏检、错分类、背景多余框、重复框、定位偏差、框重叠、差图清单、图片和 HTML 均可独立开关。诊断入口支持 PyCharm 配置和终端的 `--only/--enable/--disable`；mAP 等常规指标继续使用 Ultralytics 原生验证，避免重复测量。完整模块列表和输出规则见 [模块化 Spec](docs/specs/modular-tests-and-diagnostics.md)。

也可以使用仓库根目录的统一入口选择子工具：

```bash
python run_tools.py --tool diagnostics --config config/tools.yaml
python run_tools.py --tool pytorch --only speed.pytorch_call
python run_tools.py --tool visualization --only visualization.features
```

`run_all.py --only ...` 会按模块所属阶段执行，避免同一个 PyTorch 测试被 TensorRT 阶段再次调用。

### 5. 数据集与 K230 模型转换

这些脚本当前主要使用文件顶部的参数配置，相对路径按**运行时工作目录**解释：

| 脚本 | 操作与运行前配置 |
| --- | --- |
| `transform_tools/ndjson_to_yolo.py` | 将 NDJSON 下载/转换为本地 YOLO 数据集；修改 `convert_ndjson_to_yolo(...)` 中的输入文件与输出目录 |
| `transform_tools/py2kmodel.py` | Ultralytics `.pt` → ONNX → 简化 → nncase 量化生成 K230 `.kmodel`；检查 `PT_PATH`、`ONNX_PATH`、`KMODEL_PATH`、`CALIB_DIR` 和输入尺寸 |
| `transform_tools/PY2KM_validate.py` | 用 ONNX Runtime 和 nncase Simulator 比较输出；设置 ONNX、kmodel、测试图片和预处理参数，也可传命令行参数 |

修改好配置后，在仓库根目录分别运行对应脚本。校验脚本示例：

```bash
python transform_tools/PY2KM_validate.py --onnx /path/to/best.onnx --kmodel /path/to/best.kmodel --image /path/to/test.jpg --size 320
```

当前转换与校验脚本默认使用 `320×320`、RGB、CHW、灰色 `(128, 128, 128)` 填充；ONNX 端默认归一化，kmodel 接收 `uint8` 输入。更改预处理时要同步转换、校验与板端代码。转换脚本发现 ONNX 已存在会跳过导出，更换权重后要处理旧 ONNX，防止继续编译旧模型。

## 路径、runs 保存位置与 PyCharm

### 路径解析规则

| 设置 / 入口 | 相对路径的基准 |
| --- | --- |
| 测速 YAML 的 `project.root` | 相对于配置文件所在目录，即 `speed_test/`；当前值由 YAML 明确指定 |
| 诊断 YAML 的 `project.root` | 相对于 `run_model_diagnostics.py` 所在目录，即 `model_diagnostics/`；当前值由 YAML 明确指定 |
| 两套配置中的权重、数据集、adapter、`run.root` 等路径 | 各自解析后的 `project.root` |
| 环境检验的显式 `--source`、`--weights` 等参数 | 终端 / PyCharm 的工作目录 |
| `transform_tools` 脚本中的相对文件路径 | 终端 / PyCharm 的工作目录 |

上面的示例用 `project.root: ..` 指向 `vtools/`。仓库附带 YAML 中的
`GoodModel/`、`datasets/`、`test.jpg` 等是个人项目示例，克隆仓库后需要提供自己的权重、
图片和数据集并修改路径；如果保留 `../..`，这些路径就会相对于上两级目录解析。

将 `project.root` 改成仓库根目录后，自定义 adapter 也应写全相对于根目录的路径，例如 `speed_test/adapters/my_detector.py` 或 `model_diagnostics/adapters/my_detector.py`。

### 修改结果保存位置

修改相应 YAML 的 `run.root`，即可选择报告的父目录。例如诊断结果：

```yaml
run:
  root: /path/to/results/diagnostics
  name: auto
```

测速配置还需保留 `run.enabled: true`，并可将 `run.root` 设为另一个目录。每次独立调用自动创建 `run1`、`run2`……；测速的 `run_all.py` 各阶段共用本次运行目录，诊断结果再按模式名称分子目录。

**测速报告路径还有一层配置：** 当前代码会保留相对 `output` 中未被运行根目录匹配掉的子目录，旧值 `runs-profile/vision_speed_v2.csv` 可能形成 `runN/runs-profile/...`。若希望报告直接放在 `runN/`，将下列项改为纯文件名：

| 配置项 | 建议值 |
| --- | --- |
| `pytorch.output` | `vision_speed_v2.csv` |
| `tensorrt.output` | `vision_tensorrt_v2.csv` |
| `consistency.output` | `consistency.csv` |
| `consistency.json_output` | `consistency.json` |
| `run_all.summary_output` | `vision_benchmark_summary.csv` |

明确指定在运行根目录之外的绝对 `output` 路径不会强制归档到 `runN/`。ONNX / engine 是独立缓存，分别由 `tensorrt.onnx_dir`、`tensorrt.engine_dir` 控制；只修改 `run.root` 不会移动它们。

Ultralytics 训练/验证/预测自己的输出目录另由调用参数 `project`、`name` 等控制，与上述测速和诊断的 `run.root` 分开。

### PyCharm 直接运行

1. 用 PyCharm 打开 `vtools`，选择已经安装依赖的 Conda 解释器。
2. 修改对应 YAML 或转换脚本顶部参数，推荐先使用绝对数据路径。
3. 对转换脚本和带相对参数的环境检查，将 Working directory 设置为 `vtools` 根目录。
4. 直接运行需要的入口脚本，按终端打印路径查看结果。

终端能运行而 PyCharm 报错时，先比较两边的解释器和导入来源：

```python
import sys
import ultralytics

print(sys.executable)
print(ultralytics.__version__)
print(ultralytics.__file__)
```

## 文档索引与许可证

- [本地源码安装与环境检验](env_test/README.md)
- [Ultralytics 精简版与网络修改说明](ultralytics-cn/README.zh-CN.md)
- [目标检测诊断完整教程](model_diagnostics/docs/README.md)
- [诊断指标解释](model_diagnostics/docs/METRICS.md)
- [测速工具使用教程](speed_test/README.md)
- [测速环境要求](speed_test/REQUIREMENTS.md)
- [测速与一致性 CSV 指标说明](speed_test/CSV_METRICS_GUIDE.md)
- [数据和模型转换说明](transform_tools/REQUIREMENT.md)
- [模型勘误方法](模型勘误方法.md)

根目录许可证见 [LICENSE](LICENSE)；`ultralytics-cn/` 保留 [Ultralytics AGPL-3.0 许可证](ultralytics-cn/LICENSE) 及源码中的原有授权声明。
