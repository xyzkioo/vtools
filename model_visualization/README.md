# 模型可视化与检测阶段追踪

这个目录只负责“看模型内部发生了什么”，不重复 Ultralytics 已有的训练曲线、验证指标和普通预测保存。
首版适配本仓库的 Ultralytics/YOLO26 检测模型，其他 PyTorch 模型可按 `adapters/ultralytics.py` 的边界扩展。

## 快速运行

在仓库根目录执行：

```bash
python -m pip install -r model_visualization/requirements.txt
python -m pip install -e ./ultralytics-cn
python model_visualization/run_visualization.py --config model_visualization/config/pycharm_run.yaml
```

配置中至少替换：

```yaml
model:
  weights: /absolute/path/to/best.pt
input:
  source: /absolute/path/to/image-or-directory
```

每次运行创建一个新的 `model_visualization/runs/runN/`，打开其中的 `index.html` 查看图片和链接。

## 三类输出

1. **层与特征图**：`mode: list_layers` 只输出 `layers.csv`，先用它确认实际模块名；`layers.modules` 填模块名后，`features/` 保存通道图，`activations/` 保存聚合图和叠加图。
2. **CAM**：设置 `cam.enabled: true`，`cam.method` 可选 `gradcam` 或 `layercam`。CAM 前向固定用 FP32；它需要梯度，运行会明显慢于普通特征图。
3. **阶段追踪**：设置 `stage_trace.enabled: true`，保存 `raw_candidates.json`、`head_topk.json`、`candidates.jsonl`、`stage_summary.csv`、`stage_events.jsonl`、`final_detections.json` 和可选 `raw_arrays.npz`。每个 raw 候选都带有 `raw_index`、`source_level`、stride、网格坐标、`in_head_topk`、`in_final` 和 `filter_reason`。

YOLO26 默认选择 `one2one` 分支。选择 `one2many` 时，final 阶段调用 Ultralytics 的 NMS 并使用其 `return_idxs=True` 索引；不会通过 IoU 反推候选来源。`canonical/` 下的 raw/final JSON 可以直接作为 `model_diagnostics` 的预测输入。

## 配置边界

- `input.size` 当前使用固定尺寸 letterbox；记录了原图尺寸、缩放比例和 padding，叠加图会还原到原图坐标。
- `features.channels` 支持 `first`、`variance`、`explicit`；默认每层保存前 16 个通道和一个 `mean_abs` 聚合图。
- `cam.target.kind` 支持 `raw_candidate`、`final_detection`；`index` 是检测头的精确 anchor 索引，不是绘图排序后的序号。
- TensorRT 缓存仍只由 `rebuild_engine` 控制。engine 旁边的 `.build.json` 只是记录构建来源和参数，不参与自动重建，也不引入全量哈希策略。
