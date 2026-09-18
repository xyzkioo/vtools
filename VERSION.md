# vtools 版本记录

## 当前版本

- 更新批次：`2026.09.18`
- 功能标识：`dataset-quality-ui`
- 版本性质：独立数据集质量检查入口与工作台 UI 接入
- 适用仓库：`xyzkioo/vtools`

### v0.3.3

- 新增独立的数据集质量检查工具，扫描坏图、标签问题、尺寸分布、重复图片和数据划分泄漏。
- 工作台新增“数据集质量检查”入口，可选择数据集 YAML、报告目录和抽样图数量。
- 结果统一写入独立运行目录，包含 `summary.json`、`issues.csv`、`report.md` 和抽样图。

## 更新批次：`2026.09.15`

- 持续验收：使用已安装 PyTorch、PySide6、Ultralytics 的 `yolo` 环境运行压缩与真实 Qt 控件测试；系统 Python 缺少这些包不再作为整个机器的环境结论。
- 修复：统一入口拒绝非法 `tools.yaml`，不再静默改跑诊断；K230 转换延迟加载可选依赖、校验数值参数、稳定选择校准样本并原子发布 Kmodel。
- 修复：图片/视频编码失败保留旧输出，批量格式转换预检目标冲突；COCO 与注册表写入、改名回滚和 UI 历史记录增强事务保护。
- 验证：当前通用回归测试 38 项、模型压缩测试 21 项、诊断测试 6 项通过；测速测试 44 项通过（其中 1 项按环境跳过）。全仓语法编译和差异格式检查通过；完整 GPU TensorRT 和 K230 板端流程仍需对应硬件。
- 修复：可视化图片写入采用原子替换，离线 HTML 转义外部名称；输入尺寸、检测阈值、特征通道模式和聚合方式会被严格校验。视频解帧拒绝把不支持的单文件当作视频。

- 第三轮修复：图片 ID 始终保留扩展名，自定义布局使用 input.dataset_root/--dataset-root；旧预测需重新生成。配置编辑器切页同步状态，手动修改配置路径刷新模块。新增隔离回归验证，真实 Qt 窗口仍待验收。

- 修复：一键测速接受一致性 `passed` 状态；新分支同轮检测压缩后可继续导出；checkpoint 检查 CSV 保留失败详情。
- 修复：FP16 ONNX 归一化后保持声明 dtype；旧 models 配置也必须服从 `predict.generate`；单图与目录可视化采用同一数据集相对 ID。
- 修复：UI YAML 高级页的后续修改不再被图形旧值覆盖；模块按钮在完整 YAML 模块集合上应用增量，保留未显示模块。
- 历史验证：使用系统 `/usr/bin/python3` 时 56 项通过、2 项因该解释器未安装 torch 失败；后续已改用含 PyTorch/PySide6 的 `yolo` 环境补跑相关测试，结果见本节最新条目。

## 更新批次：`2026.09.14`

- 修复：结构化检测剪枝保存时旧 EMA 覆盖、检测/分类分支权重血缘不一致、蒸馏教师/学生来源登记错误、注册表并发写入丢失。
- 修复：诊断重叠重复框误判、稳定图片 ID、相对路径解析、差图图片和阈值过滤、模块 allow-list 与 CLI 覆盖；可视化 CAM 类别/多目标选择、BF16 输入和缺失 source 覆盖。
- 修复：UI 两个配置页字段丢失、未知下拉值、差图依赖、TensorRT 首次构建、文件改名递归/确认、进程树停止；批处理改名回滚、COCO 精确匹配、视频/图片输出冲突和 Kmodel 严格校验。
- 修复：统一入口和测速入口正确传播失败退出码；默认检测压缩配置关闭分类动态 INT8，并同步 README/Spec。
- 验证：Python compileall、git diff --check；无额外依赖的 model_compression/model_diagnostics 11 项和 speed_test 43 项通过。2 项需要本环境未安装的 PyTorch，PySide6/TensorRT/K230 端到端路径未在当前环境执行；另用临时注册表、模块选择、图片 ID 和 Kmodel 数值样例做了回归验证。

## 更新批次：`2026.09.10`

- 功能：测速与目标检测诊断的 test 内部模块开关；差图清单、图片/HTML 输出和检测框重叠分析；可视化的特征图/CAM/阶段追踪开关；诊断默认不重复计算 Ultralytics 的 mAP/AP 与阈值扫描。
- 新增文件：`docs/specs/modular-tests-and-diagnostics.md`、`model_diagnostics/diagnostics/modules.py`、`model_diagnostics/diagnostics/overlap.py`、`model_diagnostics/diagnostics/bad_cases.py`、`speed_test/core/module_selection.py`、`run_tools.py`、`config/tools.yaml`。
- 修改文件：诊断引擎和桌面 UI/终端入口、PyTorch/TensorRT 测速入口、一致性检查、相关配置与 README。
- 配置变化：新增 `modules`；支持 `--only`、`--enable`、`--disable`、`--list-modules`；诊断 YAML 移除 AP 阈值旧参数，speed_test 移除 `test_model_call/test_pipeline/test_pytorch` 独立开关，统一由 `modules` 控制。重叠分析支持 IoU、交集占较小框比例、规则和类别范围配置。
- 验证方式：`python -m compileall -q model_diagnostics speed_test`；在 `speed_test/` 中运行 `python -m unittest discover -s tests -v`（43 项，1 项按环境跳过），在仓库根目录运行 `python -m unittest discover -s model_diagnostics/tests -v`（5 项）；各入口的 `--list-modules`；使用 canonical GT/预测进行重叠和差图离线冒烟。
- 已知限制：`validation.ultralytics` 仅作为预留开关，具体验证仍由 Ultralytics 原生入口执行；图片绘制需要 Pillow；TensorRT 模块仍需要 CUDA/TensorRT 环境。

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
| `model_visualization/run_visualization.py` | 命令行后端入口 |
| `model_visualization/adapters/ultralytics.py` | Ultralytics/YOLO 模型适配、特征捕获和检测阶段追踪 |
| `model_visualization/core/capture.py` | 激活值捕获、特征图、GradCAM/LayerCAM |
| `model_visualization/core/common.py` | 配置读取、图片预处理、绘图、统计和 HTML 索引 |
| `model_visualization/config/config.yaml` | YAML 配置模板 |
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
cd /path/to/vtools
unzip -o ~/下载/vtools-visualization-complete.zip
```

如果已经完成复制，只需要确认本文件放在：

```text
/path/to/vtools/VERSION.md
```

## 运行前配置

编辑：`model_visualization/config/config.yaml`

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
cd /path/to/vtools
python -m compileall -q model_visualization speed_test/core
```

返回 `0` 表示语法检查通过。正式运行：

```bash
python model_visualization/run_visualization.py \
  --config model_visualization/config/config.yaml
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
