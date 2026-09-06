# 数据目录

把验证集数据和预测结果放在这里，或在 `config/pycharm_run.yaml` 中填写相对/绝对路径。推荐直接使用 Ultralytics 的 `data.yaml`，配置为 `dataset.data` + `dataset.split`；程序会根据 `path`、`val` 和标准 `images/`、`labels/` 目录自动读取数据。

推荐的 Ultralytics 数据集结构：

```text
dataset/
├── data.yaml
├── images/val/
└── labels/val/
```

`data.yaml` 最少包含：

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: flower
```

配置只需写：

```yaml
dataset:
  data: data.yaml
  split: val
```

Ultralytics Platform 的 `.ndjson` 清单也可以直接填到 `dataset.data`；程序会调用 Ultralytics 官方兼容转换，再读取生成的 `data.yaml`。

也兼容 canonical JSON/JSONL。其最小运行需要：

- `gt.json`：每张图的 `image_id`、图像尺寸和 `ground_truth`；
- `predictions.json`：每张图的 `image_id` 和 `predictions`；如果由模式 B/C 生成，可不预先创建，运行器会写入当前 `runs/runN/<mode_name>/predictions_adapter.json`。

当前目录不放示例数据，避免把测试数据误当成真实验证结果。
