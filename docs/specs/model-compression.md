# 模型压缩、实验分支与知识蒸馏工具 Spec

状态：已落地首版（PyTorch 分类 MVP）  
范围：`model_compression/`

## 目标

在 vtools 的既有目录和运行约定下，把“模型导入 → 基线 → 分支 → 压缩/蒸馏 → 对比 → 导出”做成可重复的本地实验工具。每个模型版本不可覆盖，注册表记录父版本、教师版本、学生初始化来源、配置、产物校验和运行 ID。

首版支持完整的 `torch.nn.Module` checkpoint、带 `factory` 的 state_dict、动态 INT8、非结构化 L1 剪枝和分类 logits 蒸馏。评测协议与测速职责仍由 `speed_test/` 负责，压缩工具只记录可执行的分类评测和模型参数统计。

## 模块和入口

| 模块 ID | 默认 | 作用 |
|---|---:|---|
| `branch.create` | 关 | 创建或确认实验分支 |
| `model.parameters` | 开 | 统计总参数、可训练参数和零值参数 |
| `baseline.evaluate` | 开 | 记录基线参数并在有验证集时评测分类准确率 |
| `compression.quantize.dynamic_int8` | 关 | 复制模型并执行 CPU 动态 INT8 |
| `compression.prune.unstructured` | 关 | 全局 L1 非结构化剪枝并移除重参数化 |
| `distillation.classification` | 关 | 冻结教师模型，按 CE + 温度缩放 KL 训练学生 |
| `comparison.report` | 关 | 输出当前分支血缘和可比较指标 |
| `artifact.export` | 关 | 导出当前分支 head，重新加载并做样本推理验证 |

统一入口：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --only compression.quantize.dynamic_int8
python run_tools.py --tool compression --config config/tools.yaml
```

`--only`、`--enable`、`--disable` 与其他 vtools 工具具有相同优先级和冲突规则；`--list-modules` 不加载 PyTorch。

## 版本和产物

`storage.registry` 默认指向 `model_compression/store/registry.json`。每次运行在 `model_compression/runs/runN/` 创建独立目录，包含有效配置、模块报告、模型文件、蒸馏 checkpoint、导出 manifest 和 `summary.json`。任务失败时保留错误记录，分支指针不推进。

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

结构化剪枝、静态校准量化、检测/分割任务 adapter、分布式训练、硬件自动 benchmark、Web UI 和团队权限不属于首版；新增实现应继续遵守独立模块、共享配置、不可覆盖版本和产物可追溯约定。

