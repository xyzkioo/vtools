# vtools 版本记录

## 当前版本

- 更新批次：`2026.09.08`
- 功能标识：`visualization-stage-trace`
- 版本性质：通用视觉模型辅助工具的可视化与检测阶段追踪更新
- 适用仓库：`xyzkioo/vtools`

这不是 Python 包的运行时版本号，而是仓库更新清单。以后每次增加功能时，继续在本文件追加一节即可。

## 本次更新目的

补充模型内部检查工具，用于回答以下问题：

1. 检测头输入的 P2/P3/P4/P5 特征是否有响应；
2. 候选框在筛选前、检测头 Top-K、最终检测结果三个阶段如何变化；
3. 候选框来自哪一个特征层，以及在哪一步被过滤；
4. 特征图、聚合图和 CAM 是否集中在目标区域；
5. TensorRT engine 当前由哪些参数和环境构建而来。

## 文件变更清单

### 新增：模型可视化与阶段追踪

新增目录：`model_visualization/`

| 文件 | 作用 |
| --- | --- |
| `model_visualization/run_visualization.py` | 命令行和 PyCharm 运行入口 |
| `model_visualization/adapters/ultralytics.py` | Ultralytics/YOLO 模型适配、特征捕获和检测阶段追踪 |
| `model_visualization/core/capture.py` | 激活值捕获、特征图、GradCAM/LayerCAM |
| `model_visualization/core/common.py` | 配置读取、图片预处理、绘图、统计和 HTML 索引 |
| `model_visualization/config/pycharm_run.yaml` | PyCharm/YAML 配置模板 |
| `model_visualization/requirements.txt` | 可视化功能的额外依赖 |
| `model_visualization/README.md` | 使用说明和输出文件说明 |

主要输出包括：

- `layers.csv`：模型层清单；
- `features/`、`activations/`：特征图和聚合图；
- `cam/`：GradCAM/LayerCAM 结果；
- `raw_candidates.json`：筛选前候选框；
- `head_topk.json`：检测头 Top-K 候选；
- `final_detections.json`：最终检测框；
- `candidates.jsonl`、`stage_summary.csv`、`stage_events.jsonl`：阶段追踪明细；
- `canonical/`：可直接给 `model_diagnostics` 使用的标准预测文件；
- `index.html`：结果索引页面。

### 修改：检测诊断适配器

文件：`model_diagnostics/diagnostics/adapter_runtime.py`

- 保留并传递候选框的来源层、原始索引等 provenance 字段；
- 增加运行状态信息，便于确认实际加载的模型和设备。

### 新增和修改：TensorRT 构建记录

文件：

- `speed_test/core/build_metadata.py`（新增）；
- `speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py`；
- `speed_test/backends/tensorrt_engine_builder.py`；
- `speed_test/tests/test_build_metadata.py`（新增）。

作用：

- 在 engine 旁边记录模型、ONNX、TensorRT/CUDA、GPU 和构建参数；
- 构建成功后才发布新 engine；
- 构建失败时保留旧 engine；
- 不改变现有 `rebuild_engine` 缓存策略，也不强制引入哈希。

### 文档

- 根目录 `README.md`：补充项目结构、环境和可视化工具说明；
- `speed_test/README.md`：补充 engine 构建记录说明。

## 当前不需要修改的内容

- `ultralytics-cn/` 网络源码不需要重新复制；
- 训练脚本、训练权重和数据集不需要放入 vtools；
- `__pycache__/` 和 `*.pyc` 不属于更新内容；
- 不需要重新执行 `git am`；
- 不需要把模型权重复制到 vtools 内。

## 安装/更新方式

本次更新采用普通文件覆盖方式：

```bash
cd ~/PycharmProjects/vtools
unzip -o ~/下载/vtools-visualization-complete.zip
```

如果已经完成复制，只需要确认本文件放在：

```text
~/PycharmProjects/vtools/VERSION.md
```

## 运行前配置

编辑：`model_visualization/config/pycharm_run.yaml`

至少修改：

```yaml
model:
  weights: /absolute/path/to/best.pt

input:
  source: /absolute/path/to/image-or-directory

stage_trace:
  enabled: true
```

模型权重和数据可以继续放在 `lichiflower` 项目目录中；它们只是输入数据，不代表 vtools 属于该项目。

## 验证

```bash
cd ~/PycharmProjects/vtools
python -m compileall -q model_visualization speed_test/core
```

返回 `0` 表示语法检查通过。正式运行：

```bash
python model_visualization/run_visualization.py \
  --config model_visualization/config/pycharm_run.yaml
```

结果默认位于：

```text
model_visualization/runs/runN/index.html
```

## 后续版本记录格式

以后新增功能时，在本文件顶部追加：

```markdown
## 更新批次：YYYY.MM.DD

- 功能：
- 新增文件：
- 修改文件：
- 配置变化：
- 验证方式：
- 已知限制：
```
