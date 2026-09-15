# 模型压缩、实验分支与知识蒸馏

## 入口

从仓库根目录运行主入口，默认读取 `config/pycharm_run.yaml`：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --list-modules
python run_tools.py --tool compression --config model_compression/config/pycharm_run.yaml
```

PyCharm 可直接运行同目录的 `run_all.py`。

这个工具把一个模型的优化过程保存成可追溯的版本链：先做基线，再从任意版本建立分支，执行量化、剪枝或蒸馏，最后对比并导出模型。它和 `speed_test/` 的职责分开：这里负责产生优化后的模型与实验血缘，测速仍使用通用测速工具完成。

## 快速开始

先安装 YAML 依赖和与本机匹配的 PyTorch。然后复制并修改 [config/pycharm_run.yaml](config/pycharm_run.yaml) 中的 `model.weights`，在仓库根目录运行：

```bash
python model_compression/run_model_compression.py
```

PyCharm 也可以直接运行同目录的 `run_all.py`，两者读取同一份配置。

默认会在 `model_compression/runs/runN/` 生成 `baseline.json`、`model_parameters.json`、`summary.json`，并在 `model_compression/store/registry.json` 保存分支和模型版本。分类流程没有配置数据集时，基线仍会记录参数量并将评测标记为跳过；YOLO 检测基线需要填写 `dataset.data` 才能进行 mAP 评估。也可以把配置中的 `operation` 改成 `quantize`、`prune`、`distill` 等，只执行对应模块；`all` 按 `modules` 开关执行。

默认配置的任务是 YOLO 检测，因此动态 INT8 分类量化默认关闭；只有将 `model.task` 改为 `classify` 并准备分类数据后，才应显式启用 `compression.quantize.dynamic_int8`。检测压缩使用 `compression.prune.unstructured` 或 `compression.prune.structured`。

需要执行某个模块时，可以直接使用统一模块 ID：

```bash
python model_compression/run_model_compression.py --only branch.create
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

分类流程的 `dataset.train` 与 `dataset.val` 默认按 ImageFolder 目录读取；adapter 返回的 DataLoader 也可以产生 `(images, labels)` 或 `{"image": ..., "label": ...}`。YOLO 检测流程使用 `model.task: detect` 和 `dataset.data` 指向 `data.yaml`，保留每张图的多框标签，支持基线评估、非结构化剪枝和结构化通道缩放。结构化模块默认根据 `compression.structured.target_scale` 从模型 YAML 的 `scales` 生成更小的 dense 网络；也可以将 `compression.structured.method` 设为 `torch_pruning`，使用 `torch-pruning` 的依赖图物理删除主干/颈部阶段通道，并同步调整依赖的卷积、BN、Concat 与残差分支。依赖感知模式需要额外安装 `torch-pruning>=1.4.1`，检测头输出层默认保留。推荐填写对应规模的 `initial_weights`，并使用与原模型接近的微调轮数，避免把未经充分训练的模型当成可用结果。默认精度门禁要求 mAP50-95 下降不超过 0.05，未通过时不会把模型推进到注册表；需要保留实验产物时可将 `compression.structured.quality_gate` 设为 `false`。报告记录参数量、文件大小、mAP 和固定输入测速；把 `finetune_epochs` 设为 0 可只做结构检查。完整接口见 [YOLO 检测压缩 Spec](../docs/specs/yolo-detection-compression.md)。分类蒸馏使用：
```text
Loss = (1 - alpha) * CE + alpha * T^2 * KL
```

教师模型始终处于 `eval()` 且冻结参数，验证集只用于选择最佳 checkpoint。蒸馏结果会同时记录教师版本和学生初始化版本。

## 版本与分支


配置键使用完整路径：结构化质量门禁字段为 `compression.structured.quality_gate`。

注册表是人类可读的 JSON。每次变换都会生成不可覆盖的新版本，并推进当前分支指针；失败任务不会推进指针。典型血缘如下：

```text
base v1
├── dynamic_int8 v2
├── unstructured_l1 v3
└── distillation_classification v4
```

非结构化剪枝只改变权重稀疏度，不自动声称文件变小或推理变快；需要在目标硬件上用 `speed_test` 实测。
