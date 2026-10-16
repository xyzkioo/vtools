# 模型压缩

[返回模块导航](../README.md)

生成量化、剪枝或蒸馏后的模型，记录基线和处理后的评估结果。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python model_compression/run_model_compression.py --help
python model_compression/run_model_compression.py --config model_compression/config/config.yaml
```

## 输入与功能

配置见 [config.yaml](../../model_compression/config/config.yaml)。按任务填写 `model.task`、权重及数据集；分类常用 ImageFolder 的 train/val，检测使用 `dataset.data`。

| 模块 ID | 用途 |
| --- | --- |
| `model.parameters` / `baseline.evaluate` | 参数统计和基线评估 |
| `compression.quantize.dynamic_int8` | 分类动态 INT8 量化 |
| `compression.prune.unstructured` | 权重稀疏剪枝 |
| `compression.prune.structured` | 规模缩放或依赖感知通道剪枝 |
| `distillation.classification` | 分类 logits 蒸馏 |
| `distillation.detection` | 检测特征蒸馏，需支持蒸馏的外部 Ultralytics Fork |
| `comparison.report` / `artifact.export` | 对比报告和模型导出 |

结构化方法由 `compression.structured.method` 选择 `scale` 或 `torch_pruning`；后者额外需要 torch-pruning。相对数据和权重路径按 `project.root` 解析。

检测蒸馏需要教师/学生权重、标准 train/val 图片和标签目录，以及兼容的检测头特征层。教师类别数量、名称和顺序须与数据集一致；通用预训练学生自动适配目标类别，按名称迁移分类输出，无法忠实重建的已剪枝主干拒绝适配。适配时保存 `artifacts/distillation/student-initialized.pt`，学生基线和训练使用同一初始化，输入权重不覆盖。

工作台按 detect/classify 选择单一处理模式，仅显示和提交对应输入；检测蒸馏无需公共模型权重。模式映射为 `--only` 列表，CLI/YAML 的显式模块组合仍可使用。仅检测蒸馏及报告/导出配套模块开启时，公共参数统计和公共基线标为跳过，学生自身评估照常执行。统一入口支持 `--teacher-weights`、`--student-weights`。训练参数在 `distillation` 节配置；训练后重新加载独立学生模型并比较蒸馏前后精度，结果写入 `artifacts/distillation` 和汇总报告。

## 输出

默认输出 `model_compression/runs/runN/`，含 `effective_config.json`、`summary.json` 和 `artifacts/` 下各处理阶段的模型。跨运行继续处理时，将上一阶段产物明确填入下一次的 `model.weights`。

`effective_config.json.modules` 是 YAML 与 CLI 模块开关合并后的实际选择。结构化剪枝会在日志中标出评估、测速、依赖图追踪和微调阶段及 CUDA 显存信息；低于 8 GiB 总显存或 6 GiB 空闲显存时给出风险提示，不自动跳过任务。

## 使用边界

非结构化稀疏不保证文件变小或推理更快；正式性能要在目标硬件上测速。检测蒸馏使用 Ultralytics 原生训练接口，分类蒸馏模块不能替代它。微调需要训练时间和资源，结构化质量门禁失败应查看精度结果。

详细说明：[原模块文档](../../model_compression/README.md)。
