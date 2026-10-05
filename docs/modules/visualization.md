# 模型可视化

[返回模块导航](../README.md)

观察指定层特征、指定目标 CAM，以及检测头候选到最终框的筛选过程。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python model_visualization/run_visualization.py --help
python model_visualization/run_visualization.py --config model_visualization/config/config.yaml
python model_visualization/run_visualization.py --mode list_layers
```

## 输入与功能

配置见 [config.yaml](../../model_visualization/config/config.yaml)，主要填写 `model.weights`、`input.source` 和层选择。

| 模块 ID | 用途 |
| --- | --- |
| `visualization.features` | 通道特征图、聚合图和叠加图 |
| `visualization.cam` | 指定 raw candidate 或 final detection 的 Grad-CAM / LayerCAM |
| `visualization.stage_trace` | raw、head top-k 与 final 检测阶段追踪 |

`list_layers` 输出实际层名，便于配置 `layers.modules`。自定义图片布局需填写 `input.dataset_root`，与诊断数据集根目录一致；YAML 相对路径按 `project.root` 解析。

## 输出

默认输出 `model_visualization/runs/runN/`，含 `index.html`、`run_metadata.json` 和按需生成的特征、CAM、阶段 JSON/CSV。`canonical/` 的逐图 raw/final JSON 可作为诊断预测输入。层检查输出 `layers.csv`。

## 使用边界

需要可加载的 PyTorch 模型，不能仅靠 ONNX 图获得相同结果。CAM 需要梯度且固定 FP32，会增加耗时；热图不证明因果关系。普通预测可视化优先使用 Ultralytics 原生接口。结果阅读见 [解释指南](../../model_visualization/RESULT_INTERPRETATION.zh-CN.md)。

详细说明：[原模块文档](../../model_visualization/README.md)。
