# 检测诊断

[返回模块导航](../README.md)

基于真实标注和预测结果定位错误原因，保留逐图、逐目标和差图证据。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python model_diagnostics/run_model_diagnostics.py --help
python model_diagnostics/run_model_diagnostics.py --config model_diagnostics/config/config.yaml
```

## 输入与功能

默认配置见 [config.yaml](../../model_diagnostics/config/config.yaml)。三种互斥模式为 `predictions_file`（已有预测）、`ultralytics_model`（加载权重生成预测）、`custom_adapter`（自定义模型），共用 `dataset.data` 和 `dataset.split`。

| 模块 ID | 用途 |
| --- | --- |
| `diagnostics.missed` | 漏检分析 |
| `diagnostics.classification` | 错分类分析 |
| `diagnostics.background` | 背景误检分析 |
| `diagnostics.duplicate` | 重复框分析 |
| `diagnostics.localization` | 定位偏差分析 |
| `diagnostics.overlap` | GT 与预测框重叠分析 |
| `diagnostics.groups` | 分组诊断统计 |
| `diagnostics.threshold_sweep` | 工作点与 FP/图预算分析 |
| `diagnostics.bootstrap` | 重采样统计分析 |
| `diagnostics.stage_comparison` | raw 与 final 预测阶段对比 |
| `output.bad_cases` / `output.images` / `output.html` | 差图清单、渲染与 HTML 报告 |

预测使用原图像素坐标 `[x1, y1, x2, y2]`、`score`、`class_id`。详细格式和指标见 [诊断文档](../../model_diagnostics/docs/README.md)。

## 输出

默认输出 `model_diagnostics/runs/runN/<mode_name>/`。`run_info.json` 保存配置和运行信息，`summary.json` 保存汇总，`raw_data/` 保存详细 CSV 与预测，差图按错误类别保存。

## 使用边界

常规 mAP/AP 和标准验证使用 Ultralytics 原生 val。`validation.ultralytics` 是未实现的预留开关，启用会报错。已有预测路径不加载模型；模型路径需要相应运行库，NDJSON GT 转换仍需要 Ultralytics。

详细说明：[原模块文档](../../model_diagnostics/README.md)。
