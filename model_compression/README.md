# 模型压缩与知识蒸馏

## 入口

从仓库根目录运行主入口，默认读取 `config/config.yaml`：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --list-modules
python run_tools.py --tool compression --config model_compression/config/config.yaml
```

每次运行都是独立实验：先做基线，再按模块顺序执行量化、剪枝或蒸馏。运行目录保存完整配置、汇总结果和全部模型产物；跨运行继续处理时，把上一次运行的产物路径填入 `model.weights`。它和 `speed_test/` 的职责分开：这里负责产生优化后的模型与评估结果，目标硬件测速仍使用通用测速工具完成。

YOLO 检测模型的知识蒸馏使用 Ultralytics 原生 `model.train(..., distill_model=teacher.pt)`；本工具的 `distillation.classification` 只处理分类 logits。检测压缩的基线和产物精度通过 Ultralytics `YOLO.val()` 评估，结构化剪枝前后的固定输入计时仅用于同一次实验内比较，正式部署测速仍使用目标硬件上的测速工具。

## 快速开始

先安装 YAML 依赖和与本机匹配的 PyTorch。然后复制并修改 [config/config.yaml](config/config.yaml) 中的 `model.weights`，在仓库根目录运行：

```bash
python model_compression/run_model_compression.py
```

显式指定 `--device cuda:0` 等 CUDA 设备时，若设备不可用会报错；`auto` 才会在无 CUDA 时选择 CPU。`--run-dir` 需要指向尚不存在的目录，避免覆盖旧实验结果。

默认会在 `model_compression/runs/runN/` 生成 `effective_config.json` 和扁平化的 `summary.json`。模型文件统一位于 `artifacts/structured_pruning`、`artifacts/unstructured_pruning`、`artifacts/distillation`、`artifacts/quantization` 或 `artifacts/multi_processing`；不再创建 `store/registry.json`，也不写入分散的模块 JSON。分类流程没有配置数据集时，基线仍会记录参数量并将评测标记为跳过；YOLO 检测基线需要填写 `dataset.data` 才能进行 mAP 评估。执行内容只由 `modules` 开关决定。

典型运行目录如下：

```text
runN/
├── effective_config.json
├── summary.json
├── baseline_validation/              # 按需生成的评估图表
└── artifacts/
    ├── structured_pruning/
    ├── unstructured_pruning/
    ├── distillation/
    ├── quantization/
    └── multi_processing/
```

`summary.json` 只在 `model` 段记录一次公共模型信息，阶段结果使用 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization` 和 `multi_processing` 六个类别；每类按需包含 `parameters`、`metrics` 和相对于 runN 的 `artifact` 路径。

默认配置的任务是 YOLO 检测，因此动态 INT8 分类量化默认关闭；只有将 `model.task` 改为 `classify` 并准备分类数据后，才应显式启用 `compression.quantize.dynamic_int8`。检测压缩使用 `compression.prune.unstructured` 或 `compression.prune.structured`。

需要执行某个模块时，可以直接使用统一模块 ID：

```bash
python model_compression/run_model_compression.py --only compression.quantize.dynamic_int8
python model_compression/run_model_compression.py --only compression.prune.unstructured
python model_compression/run_model_compression.py --only compression.prune.structured
python model_compression/run_model_compression.py --only distillation.classification
python model_compression/run_model_compression.py --list-modules
```

也可以从仓库统一入口调用：

```bash
python run_tools.py --tool compression --config config/tools.yaml
```

## 配置要点

`model.adapter: torch` 支持完整的 `torch.nn.Module` checkpoint，或在配置了 `model.factory: package.module:function` 时加载 `state_dict`。自定义网络和数据集可以复制 [adapters/torch_adapter_template.py](adapters/torch_adapter_template.py)，实现 `build_model`、`load_weights` 和可选的 `build_dataloader`。

分类流程的 `dataset.train` 与 `dataset.val` 默认按 ImageFolder 目录读取；adapter 返回的 DataLoader 也可以产生 `(images, labels)` 或 `{"image": ..., "label": ...}`。YOLO 检测流程使用 `model.task: detect` 和 `dataset.data` 指向 `data.yaml`，保留每张图的多框标签，支持基线评估、非结构化剪枝和结构化通道缩放。结构化模块默认根据 `compression.structured.target_scale` 从模型 YAML 的 `scales` 生成更小的 dense 网络；也可以将 `compression.structured.method` 设为 `torch_pruning`，使用 `torch-pruning` 的依赖图物理删除主干/颈部阶段通道，并同步调整依赖的卷积、BN、Concat 与残差分支。依赖感知模式需要额外安装 `torch-pruning>=1.4.1`，检测头输出层默认保留。推荐填写对应规模的 `initial_weights`，并使用与原模型接近的微调轮数，避免把未经充分训练的模型当成可用结果。默认精度门禁要求 mAP50-95 下降不超过 0.05，未通过时结果标记为失败且不能作为后续阶段输入；需要保留实验产物时可将 `compression.structured.quality_gate` 设为 `false`。报告记录参数量、文件大小、mAP 和固定输入测速；把 `finetune_epochs` 设为 0 可只做结构检查。完整接口见 [YOLO 检测压缩 Spec](../docs/specs/yolo-detection-compression.md)。分类蒸馏使用：
```text
Loss = (1 - alpha) * CE + alpha * T^2 * KL
```

教师模型始终处于 `eval()` 且冻结参数，验证集只用于选择最佳 checkpoint。蒸馏结果记录教师和学生初始化权重路径。

配置键使用完整路径：结构化质量门禁字段为 `compression.structured.quality_gate`。

`summary.json` 按 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization` 和 `multi_processing` 分类保存参数、指标和相对产物路径。公共模型信息只在 `model` 中出现，完整配置只在 `effective_config.json` 中保留。多模块串行运行会额外保存最终模型到 `artifacts/multi_processing`；单模块结果仍保留在对应类别目录。

跨运行不再自动选择父模型或维护分支指针。需要继续处理时，将上一运行 `summary.json` 中的相对 `artifact` 路径与该 runN 目录拼成实际路径，再作为下一运行的 `model.weights`。

非结构化剪枝只改变权重稀疏度，不自动声称文件变小或推理变快；需要在目标硬件上用 `speed_test` 实测。
