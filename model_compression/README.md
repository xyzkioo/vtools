# 模型压缩与知识蒸馏

## 入口

从仓库根目录运行主入口，默认读取 `config/config.yaml`：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --list-modules
python run_tools.py --tool compression --config model_compression/config/config.yaml
```

每次运行都是独立实验：通常先做基线，再按模块顺序执行量化、剪枝或蒸馏；纯检测蒸馏仅执行学生自身的基线，不依赖公共模型。运行目录保存完整配置、汇总结果和全部模型产物；跨运行继续处理时，把上一次运行的产物路径填入 `model.weights`。它和 `speed_test/` 的职责分开：这里负责产生优化后的模型与评估结果，目标硬件测速仍使用通用测速工具完成。

工作台按任务选择一种处理模式：基线评估、结构化剪枝、非结构化剪枝、检测蒸馏、分类蒸馏或动态 INT8 量化，仅展示和提交该模式使用的输入。检测任务不提供分类蒸馏/动态 INT8，分类任务不提供结构化剪枝/检测蒸馏。处理模式转换为现有 CLI 的 `--only` 模块列表，覆盖 YAML 中其他模块开关；图形配置编辑器和 CLI/YAML 仍保留显式模块组合，命令行 `--only ''` 表示全部关闭。

YOLO 检测模型的 `distillation.detection` 通过支持蒸馏的 Ultralytics Fork 调用 `model.train(..., distill_model=teacher.pt, dis=6.0)`，负责输入校验、训练数据隔离、检查权重缓存复用、训练后学生模型重新加载和前后评估；特征损失与训练由 Fork 实现，不在 vtools 中重复实现。`distillation.classification` 只处理分类 logits。检测压缩的基线和产物精度通过 Ultralytics `YOLO.val()` 评估，结构化剪枝前后的固定输入计时仅用于同一次实验内比较，正式部署测速仍使用目标硬件上的测速工具。

## 快速开始

先安装 YAML 依赖和与本机匹配的 PyTorch。然后复制并修改 [config/config.yaml](config/config.yaml) 中的 `model.weights`，在仓库根目录运行：

```bash
python model_compression/run_model_compression.py
```

显式指定 `--device cuda:0` 等 CUDA 设备时，若设备不可用会报错；`auto` 才会在无 CUDA 时选择 CPU。`--run-dir` 需要指向尚不存在的目录，避免覆盖旧实验结果。

默认会在 `model_compression/runs/runN/` 生成 `effective_config.json` 和扁平化的 `summary.json`。模型文件统一位于 `artifacts/structured_pruning`、`artifacts/unstructured_pruning`、`artifacts/distillation`、`artifacts/quantization` 或 `artifacts/multi_processing`。分类流程没有配置数据集时，基线仍会记录参数量并将评测标记为跳过；YOLO 检测基线需要填写 `dataset.data` 才能进行 mAP 评估。执行内容只由 `modules` 开关决定。

`effective_config.json` 的 `modules` 记录 YAML 与 `--only`、`--enable`、`--disable` 合并后的实际开关。结构化检测压缩启动前检查 CUDA 显存；总量低于 8 GiB 或当前空闲低于 6 GiB 时提示风险。日志标注剪枝前后评估、测速、依赖图追踪、微调和保存阶段，附阶段边界的 CUDA 空闲、已分配、已保留及本进程阶段峰值；OOM 错误会标明阶段。这不是精确容量预测，实际峰值还受输入尺寸、batch size、其他进程及模型结构影响。

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

显式传入 `--only ''` 表示关闭全部模块，不恢复 YAML 默认选择；不能与 `--enable/--disable` 混用。

也可以从仓库统一入口调用：

```bash
python run_tools.py --tool compression --config config/tools.yaml
```

## 配置要点

`model.adapter: torch` 支持完整的 `torch.nn.Module` checkpoint，或在配置了 `model.factory: package.module:function` 时加载 `state_dict`。自定义网络和数据集可以复制 [adapters/torch_adapter_template.py](adapters/torch_adapter_template.py)，实现 `build_model`、`load_weights` 和可选的 `build_dataloader`。

分类流程的 `dataset.train` 与 `dataset.val` 默认按 ImageFolder 目录读取；adapter 返回的 DataLoader 也可以产生 `(images, labels)` 或 `{"image": ..., "label": ...}`。YOLO 检测流程使用 `model.task: detect` 和 `dataset.data` 指向 `data.yaml`，保留每张图的多框标签，支持基线评估、非结构化剪枝和结构化通道缩放。结构化模块默认根据 `compression.structured.target_scale` 从模型 YAML 的 `scales` 生成更小的 dense 网络；也可以将 `compression.structured.method` 设为 `torch_pruning`，使用 `torch-pruning` 的依赖图物理删除主干/颈部阶段通道，并同步调整依赖的卷积、BN、Concat 与残差分支。依赖感知模式需要额外安装 `torch-pruning>=1.4.1`，检测头输出层默认保留。推荐填写对应规模的 `initial_weights`，并使用与原模型接近的微调轮数，避免把未经充分训练的模型当成可用结果。微调和检测蒸馏期间 Ultralytics 的 AMP 检查权重共用用户缓存 `~/.cache/ultralytics/weights/`（可由 `XDG_CACHE_HOME` 更改缓存根目录），不依赖安装目录的写权限，也不随运行目录重复下载。默认精度门禁要求 mAP50-95 下降不超过 0.05，未通过时结果标记为失败且不能作为后续阶段输入；需要保留实验产物时可将 `compression.structured.quality_gate` 设为 `false`。报告记录参数量、文件大小、mAP 和固定输入测速；把 `finetune_epochs` 设为 0 可只做结构检查。配置字段与依赖见 [config/config.yaml](config/config.yaml) 和 [REQUIREMENTS.md](REQUIREMENTS.md)。分类蒸馏使用：
```text
Loss = (1 - alpha) * CE + alpha * T^2 * KL
```

教师模型始终处于 `eval()` 且冻结参数，验证集只用于选择最佳 checkpoint。蒸馏结果记录教师和学生初始化权重路径。

### 检测知识蒸馏

需要外部 Ultralytics Fork 提供 `distill_model`、`dis` 和 `ultralytics.nn.distill_model.DistillationModel`。普通 Ultralytics 安装缺少该接口时会明确失败。设置 `model.task: detect`，并填写 `distillation.teacher_weights`、`distillation.student_weights` 和 `dataset.data`。教师的类别数量、名称及顺序必须与数据集 `names` 一致；教师和学生须具有兼容的检测头特征层索引及空间尺寸。数据集须采用 `images/train`、`images/val` 与对应 `labels/` 的标准布局，train/val 应为目录。

学生可直接使用通用 80 类预训练检测权重。工具按学生 YAML 重建目标类别检测头，迁移形状兼容的权重；分类输出按类别名称匹配迁移，新类别保留初始化。同类别数量但顺序不同也会重映射。无法按 YAML 忠实重建的已剪枝主干明确拒绝。发生适配时另存 `artifacts/distillation/student-initialized.pt`，基线、训练和产物父来源均使用该初始化；原始输入路径保存在报告中，输入权重不覆盖。

```bash
python run_tools.py --tool compression --task detect \
  --config model_compression/config/config.yaml \
  --teacher-weights /path/to/teacher.pt --student-weights /path/to/student.pt \
  --data /path/to/data.yaml --only distillation.detection
```

工作台选择“模型压缩 → detect → 检测知识蒸馏”，填写教师、学生权重和数据集，无需公共模型权重。CLI/YAML 中仅检测蒸馏及其报告/导出配套模块开启时，多余的公共参数统计和公共基线标为 `skipped`，学生自身的参数统计和适配后基线仍执行；包含剪枝/量化的组合继续执行公共基线。训练轮数、批大小、学习率、`imgsz`、特征损失权重 `dis`、优化器、早停、AMP 和 workers 位于 `distillation` 配置节。`temperature/alpha/student_factory` 仅用于分类蒸馏。默认的一轮训练用于流程检查，不能作为充分训练或精度提升的保证。

本次训练使用独立的数据集镜像，跳过原有 `.cache/.npy` 缓存；AMP 检查权重下载使用上述用户级公共缓存。蒸馏产物为 `artifacts/distillation/student-best.pt`，同时保留训练目录、最佳/最后 checkpoint 和训练报告。交付的学生 checkpoint 会重新加载，确认已移除蒸馏包装器且网络结构与初始学生一致，再进行精度评估。参数量和精度差值写入 `summary.json.distillation`；这些指标没有预设提升门禁。

配置键使用完整路径：结构化质量门禁字段为 `compression.structured.quality_gate`。

`summary.json` 按 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization` 和 `multi_processing` 分类保存参数、指标和相对产物路径。公共模型信息只在 `model` 中出现，完整配置只在 `effective_config.json` 中保留。多模块串行运行会额外保存最终模型到 `artifacts/multi_processing`；单模块结果仍保留在对应类别目录。

跨运行不再自动选择父模型或维护分支指针。需要继续处理时，将上一运行 `summary.json` 中的相对 `artifact` 路径与该 runN 目录拼成实际路径，再作为下一运行的 `model.weights`。

非结构化剪枝只改变权重稀疏度，不自动声称文件变小或推理变快；需要在目标硬件上用 `speed_test` 实测。
