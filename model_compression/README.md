# 模型压缩、实验分支与知识蒸馏

这个工具把一个模型的优化过程保存成可追溯的版本链：先做基线，再从任意版本建立分支，执行量化、剪枝或蒸馏，最后对比并导出模型。它和 `speed_test/` 的职责分开：这里负责产生优化后的模型与实验血缘，测速仍使用通用测速工具完成。

## 快速开始

先安装 YAML 依赖和与本机匹配的 PyTorch。然后复制并修改 [config/pycharm_run.yaml](config/pycharm_run.yaml) 中的 `model.weights`，在仓库根目录运行：

```bash
python model_compression/run_model_compression.py
```

PyCharm 也可以直接运行同目录的 `run_all.py`，两者读取同一份配置。

默认会在 `model_compression/runs/runN/` 生成 `baseline.json`、`model_parameters.json`、`summary.json`，并在 `model_compression/store/registry.json` 保存分支和模型版本。没有配置数据集时，基线仍会记录参数量，但评测会明确标记为跳过。也可以把配置中的 `operation` 改成 `quantize`、`prune`、`distill` 等，只执行对应模块；`all` 按 `modules` 开关执行。

需要执行某个模块时，可以直接使用统一模块 ID：

```bash
python model_compression/run_model_compression.py --only branch.create
python model_compression/run_model_compression.py --only compression.quantize.dynamic_int8
python model_compression/run_model_compression.py --only compression.prune.unstructured
python model_compression/run_model_compression.py --only distillation.classification
python model_compression/run_model_compression.py --list-modules
```

也可以从仓库统一入口调用：

```bash
python run_tools.py --tool compression --config config/tools.yaml
```

## 配置要点

`model.adapter: torch` 支持完整的 `torch.nn.Module` checkpoint，或在配置了 `model.factory: package.module:function` 时加载 `state_dict`。自定义网络和数据集可以复制 [adapters/torch_adapter_template.py](adapters/torch_adapter_template.py)，实现 `build_model`、`load_weights` 和可选的 `build_dataloader`。

`dataset.train` 与 `dataset.val` 默认按 ImageFolder 目录读取；adapter 返回的 DataLoader 也可以产生 `(images, labels)` 或 `{"image": ..., "label": ...}`。分类蒸馏使用：

```text
Loss = (1 - alpha) * CE + alpha * T^2 * KL
```

教师模型始终处于 `eval()` 且冻结参数，验证集只用于选择最佳 checkpoint。蒸馏结果会同时记录教师版本和学生初始化版本。

## 版本与分支

注册表是人类可读的 JSON。每次变换都会生成不可覆盖的新版本，并推进当前分支指针；失败任务不会推进指针。典型血缘如下：

```text
base v1
├── dynamic_int8 v2
├── unstructured_l1 v3
└── distillation_classification v4
```

非结构化剪枝只改变权重稀疏度，不自动声称文件变小或推理变快；需要在目标硬件上用 `speed_test` 实测。
