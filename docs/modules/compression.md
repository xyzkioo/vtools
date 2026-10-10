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
| `comparison.report` / `artifact.export` | 对比报告和模型导出 |

结构化方法由 `compression.structured.method` 选择 `scale` 或 `torch_pruning`；后者额外需要 torch-pruning。相对数据和权重路径按 `project.root` 解析。

## 输出

默认输出 `model_compression/runs/runN/`，含 `effective_config.json`、`summary.json` 和 `artifacts/` 下各处理阶段的模型。跨运行继续处理时，将上一阶段产物明确填入下一次的 `model.weights`。

`effective_config.json.modules` 是 YAML 与 CLI 模块开关合并后的实际选择。结构化剪枝会在日志中标出评估、测速、依赖图追踪和微调阶段及 CUDA 显存信息；低于 8 GiB 总显存或 6 GiB 空闲显存时给出风险提示，不自动跳过任务。

## 使用边界

非结构化稀疏不保证文件变小或推理更快；正式性能要在目标硬件上测速。检测蒸馏使用 Ultralytics 原生训练接口，分类蒸馏模块不能替代它。微调需要训练时间和资源，结构化质量门禁失败应查看精度结果。

详细说明：[原模块文档](../../model_compression/README.md)。
