# 模型压缩与知识蒸馏工具 Spec

状态：已落地首版（PyTorch 分类与 YOLO 检测流程）
范围：`model_compression/`

检测模型扩展见 [YOLO 目标检测压缩 Spec](yolo-detection-compression.md)，包含 UI/CLI、模块调用、Ultralytics 接口和分阶段验收。

## 目标

在 vtools 的既有目录和运行约定下，把“模型导入 → 基线 → 压缩/蒸馏 → 对比 → 导出”做成可重复的本地实验工具。每次运行都是独立目录，summary 记录本次输入、阶段指标和产物路径；跨运行继续处理由用户显式提供上一运行的模型路径。

首版支持完整的 `torch.nn.Module` checkpoint、带 `factory` 的 state_dict、动态 INT8、非结构化 L1 剪枝和分类 logits 蒸馏。评测协议与测速职责仍由 `speed_test/` 负责，压缩工具只记录可执行的分类评测和模型参数统计。

## 模块和入口

| 模块 ID | 默认 | 作用 |
|---|---:|---|
| `branch.create` | 关 | 兼容旧配置，运行目录自包含时返回跳过 |
| `model.parameters` | 开 | 统计总参数、可训练参数和零值参数 |
| `baseline.evaluate` | 开 | 记录基线参数并在有验证集时评测分类准确率 |
| `compression.quantize.dynamic_int8` | 关 | 复制模型并执行 CPU 动态 INT8 |
| `compression.prune.unstructured` | 关 | 全局 L1 非结构化剪枝并移除重参数化 |
| `distillation.classification` | 关 | 冻结教师模型，按 CE + 温度缩放 KL 训练学生 |
| `comparison.report` | 关 | 输出本次运行各阶段的可比较指标 |
| `artifact.export` | 关 | 导出本次运行最后一个模型，重新加载并做样本推理验证 |

统一入口：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --only compression.quantize.dynamic_int8
python run_tools.py --tool compression --config config/tools.yaml
```

`--only`、`--enable`、`--disable` 与其他 vtools 工具具有相同优先级和冲突规则；`--list-modules` 不加载 PyTorch。

检测默认配置不会启用动态 INT8；该模块只适用于分类模型。模型产物使用唯一文件名，重复运行不会覆盖已有 checkpoint。

## 版本和产物

每次运行在 `model_compression/runs/runN/` 创建独立目录，包含 `effective_config.json`、扁平化的 `summary.json`、评估输出和 `artifacts/`。产物按 `structured_pruning`、`unstructured_pruning`、`distillation`、`quantization`、`multi_processing` 分类保存；不再创建 `store/registry.json` 或分散的模块 JSON。任务失败时保留错误记录，依赖失败阶段的后续模块标记为跳过。

`summary.json` 顶层只保留运行状态、公共模型信息、执行模块和错误；阶段结果按 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization`、`multi_processing` 分类，每类包含 `status`、`parameters`、`metrics` 和 `artifact` 路径。运行目录内的产物使用相对路径，外部输入保留绝对路径。完整配置继续只保存在 `effective_config.json`，不在 summary 重复写入内部版本号、分支、SHA256 或环境元数据。

非结构化稀疏度只代表权重中零值比例，不自动推断文件压缩或推理加速；动态量化默认转为 CPU 模型，必须用 `speed_test` 在目标设备上重新测量延迟、内存和吞吐。

## 数据和 adapter

未提供 adapter 时，`dataset.train`/`dataset.val` 按 ImageFolder 目录读取，batch 由 `distillation.batch_size` 控制。自定义 adapter 实现 `build_model`、`load_weights`，可选实现 `build_dataloader` 和 `save_model`；模板在 `model_compression/adapters/torch_adapter_template.py`。

蒸馏损失为：

```text
Loss = (1 - alpha) * CE(student, labels)
     + alpha * T^2 * KL(softmax(teacher / T) || softmax(student / T))
```

约束 `T > 0`、`0 ≤ alpha ≤ 1`。教师始终 `eval()` 且冻结，验证集只用于选择最佳 checkpoint。

## 后续扩展

分类结构化剪枝、静态校准量化、检测/分割任务 adapter、分布式训练、硬件自动 benchmark、Web UI 和团队权限不属于分类首版；YOLO 检测的结构化宽度缩放另见检测 Spec。新增实现应继续遵守独立运行目录、共享配置和产物可追溯约定。
