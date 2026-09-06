# 通用目标检测诊断工具

这套代码只关心目标检测质量与错误来源，不计算参数量、FLOPs、显存和延迟。核心评估不强制导入 Ultralytics，也不假设模型有 NMS、P2/P3、anchor 或特定输出头；只有使用 `adapter: ultralytics` 或 `.ndjson` 官方转换时才需要它。YOLO、DETR、Faster R-CNN 和自定义 PyTorch/ONNX 模型都可以通过统一预测格式或 vtools 风格 adapter 接入。

## 最短使用路径

1. 用 PyCharm 打开 `model_diagnostics` 文件夹。
2. 安装 `pip install -r requirements.txt`。
3. 打开 `config/pycharm_run.yaml`，填写 `dataset.data`，再选择一个 A/B/C 模式。
4. 直接运行 `run_model_diagnostics.py`。
5. 在 `runs/runN/<mode_name>/` 查看 `report.md`、`summary.json` 和 CSV 明细。

每次运行会创建新的 `runN`，避免把上一次结果混进本次结果。入口配置只保留一个 `mode` 选择，不再使用容易混淆的 `models[].enabled` 列表。

## A/B/C 三种模式

三种模式共用数据集配置：

```yaml
dataset:
  data: data/dataset.yaml
  split: val
```

### 模式 A：已有预测文件

不加载模型，只评估已经生成的预测结果：

```yaml
mode: A

mode_a:
  name: existing_predictions
  predictions: data/predictions.json
  pred_format: auto
```

预测文件支持 canonical JSON/JSONL 或 COCO detection results。预测应尽量保留低分框，诊断程序会自行做阈值扫描。

### 模式 B：Ultralytics `.pt`

直接使用现成的 Ultralytics 权重逐图生成预测：

```yaml
mode: B

mode_b:
  name: ultralytics_pt
  weights: /path/to/best.pt
  task: detect
```

如果权重依赖本地修改版 Ultralytics，在 `project.ultralytics_repo` 填写训练时使用的源码目录；如果环境中已经安装兼容版本，可以不填。

### 模式 C：自定义 adapter

适合普通 PyTorch、DETR、Faster R-CNN 或自定义网络。复制 `adapters/vision_adapter_template.py`，例如保存为 `adapters/my_detector.py`：

```yaml
mode: C

mode_c:
  name: custom_adapter
  weights: weights/best.pt
  adapter: adapters/my_detector.py
  task: detect
```

模式 B/C 会自动逐图推理，并把中间预测保存为当前运行目录中的 `predictions_adapter.json`；模式 A 不会加载模型。

## vtools 对齐的 adapter 契约

接口名称与 [vtools `vision_adapter_template.py`](https://github.com/xyzkioo/vtools/blob/main/speed_test/adapters/vision_adapter_template.py) 一致。函数参数可以比下面少，运行器会按函数签名传参：

```python
def build_model(weights, device):
    ...

def make_inputs(batch_size, image_size, device, dtype, source=None, for_export=False):
    ...  # 返回 InputBundle

def prepare_model(model, device, precision, fuse=False):
    ...  # 可选

def predict(model, source, config):
    ...  # 可选；返回 raw output 或检测列表

def postprocess_detections(outputs, source, image_info, config):
    ...  # raw output 必须实现

def count_outputs(outputs):
    ...  # 可选，诊断不依赖它
```

`core/vision_benchmark_common.py` 提供与 vtools 同名的 `InputBundle`、`BenchmarkConfig`、`dtype_for_precision`、`resolve_device` 和 `normalize_precision`，所以适配器中常见的导入无需修改。诊断运行器会给 `BenchmarkConfig` 传入 `device`、`input_size`、`batch_size`、`precision`、`conf`、`iou`、`max_det` 以及 vtools 常见的 `warmup`、`repeats`、`fuse` 字段。

注意事项：

- `postprocess_detections` 必须返回 `[{"bbox": [x1, y1, x2, y2], "score": 0.8, "class_id": 0}, ...]`。
- `predict` 的 `source` 按 vtools 约定是图片路径字符串；`image_info` 额外提供 `image_id`、`width`、`height`、`file_name` 和 GT 记录中的扩展字段。
- bbox 必须是**原图像素坐标**，不是 resize/letterbox 后的坐标。
- `score` 应是可比较的 `[0, 1]` 分数；如果模型输出 logits，请在 adapter 中完成 sigmoid/softmax 或明确校准。
- 不要在 adapter 中按 `benchmark.conf` 删除低分框，否则无法确认“候选存在但被阈值过滤”的问题。
- `make_inputs` 仅用于没有 `predict` 的简单模型；真实图片预处理和坐标还原更适合放入 `predict` 与 `postprocess_detections`。

## 数据格式

### Ultralytics `data.yaml`（推荐）

诊断入口兼容 Ultralytics 检测数据集 YAML。常见写法如下：

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test       # 可选
names:
  0: flower
```

其中 `train`、`val`、`test` 可以是图片目录、图片列表文件（`.txt`）、单张图片或路径列表。标准 YOLO 标签目录与图片目录同级：

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

配置中只需要写：

```yaml
dataset:
  data: /path/to/dataset/data.yaml
  split: val
```

程序会根据 `path` 与 `val` 自动找到图片，再将对应的 `labels/.../*.txt` 中的 `class xc yc w h` 归一化坐标转换成原图像素框。`names` 会保存到报告的类别信息中。当前诊断目标是 `detect`；如果 YAML 明确写了 `task: segment`，程序会直接提示当前版本暂不支持实例分割；没有明确任务但标签列数不是 5 时，也会警告并跳过，避免把 polygon 误当成检测框。

如果输入本身是 Ultralytics Platform 的 `.ndjson` 数据集清单，工具会调用 Ultralytics 官方 `convert_ndjson_to_yolo_if_needed` 兼容转换，再读取生成的 `data.yaml`；这种模式需要当前环境安装 `ultralytics`。

推荐的 canonical GT：

```json
{
  "class_names": {"0": "flower"},
  "records": [
    {
      "image_id": "img001",
      "file_name": "img001.jpg",
      "width": 832,
      "height": 832,
      "ground_truth": [
        {"bbox": [100, 120, 180, 220], "class_id": 0}
      ]
    }
  ]
}
```

推荐的 canonical 预测：

```json
{
  "records": [
    {
      "image_id": "img001",
      "predictions": [
        {"bbox": [102, 118, 179, 221], "score": 0.83, "class_id": 0},
        {"bbox": [500, 500, 530, 530], "score": 0.12, "class_id": 0}
      ]
    }
  ]
}
```

COCO GT（`images` + `annotations` + `categories`）、COCO prediction results（`image_id` + `bbox: [x,y,w,h]`）和旧式 YOLO 标签目录也可直接使用。旧式写法才需要同时填写 `dataset.gt` 与 `dataset.images`；新项目优先使用上面的 `data.yaml`。

## 诊断指标与确认方法

| 指标 | 作用 | 下一步确认 |
|---|---|---|
| AP50、AP75、mAP50–95 | 整体检测质量与严格定位质量 | 对照不同 IoU，区分检出率和框回归 |
| Precision、Recall、F1 | `diagnostics.score_threshold` 工作点表现 | 对照阈值扫描，判断是否只是工作点选择 |
| Recall@IoU、候选覆盖率 | 宽松候选是否已经产生 | 低 IoU 高、严格 IoU 低时优先查定位 |
| 低置信度可找回 GT | 目标是否被低分过滤 | 降低阈值并观察 FP/图，确认校准或类别区分度 |
| 错误类型 | 漏检、背景、重复、定位、类别错误 | 查 `per_gt.csv` 与 `per_prediction.csv` |
| 尺寸/密度/类别分组 | 小目标、拥挤场景或特定类别短板 | 固定数据分组做分辨率、裁剪和标签复核 |
| raw vs final | 后处理/筛选是否额外丢失候选 | 传 `dataset.raw_predictions`，查看 `stage_comparison.csv` |

核心确认顺序是：

1. `summary.json`：先看正式 Recall、Recall@IoU、候选覆盖率和低置信度可找回比例。
2. `per_gt.csv`：区分 `miss_no_candidate`、`miss_low_confidence`、`miss_localization` 和类别错误。
3. `group_metrics.csv`：只有组内样本足够时才比较小目标、密度和类别差异。
4. `threshold_sweep.csv` / `fp_budget.csv`：确认分数阈值和 FP/图约束下的取舍。
5. `stage_comparison.csv`：raw 明显好于 final 时，才支持筛选、NMS、top-k 或 query 截断造成额外损失；具体是哪一步仍需单独导出和标注。

本工具报告的是可观测错误，不会仅凭相关性把问题归因到某个网络模块。输入信息不足、特征表示不足、标签问题和训练问题分别需要分辨率/裁剪对照、特征探针、标注复核和小数据集过拟合等实验确认。

## 命令行运行

已有预测文件可以绕过 PyCharm 入口：

```bash
cd model_diagnostics
python diagnostics/engine.py \
  --gt data/dataset.yaml \
  --gt-format ultralytics \
  --split val \
  --pred data/predictions.json \
  --config config/config.example.yaml \
  --output runs/manual
```

直接运行入口会自动加载 vtools 风格 YAML；命令行引擎仍接受扁平配置，方便脚本集成。

## 项目目录

```text
model_diagnostics/
├── installer/                     # 安装向导和环境检查
│   ├── install.py
│   └── check_environment.py
├── run_model_diagnostics.py       # PyCharm 入口
├── config/                        # 带注释的 YAML/JSON 配置
├── core/                          # vtools 兼容类型
├── diagnostics/                   # 通用评估引擎和 adapter 运行器
├── adapters/                      # 可复制的模型适配器
├── data/                          # GT、图片和预测文件
├── docs/                          # 指标与确认方法
└── runs/                          # 自动生成的结果
```

核心评估只使用 Python 标准库；读取 YAML 需要 PyYAML，YOLO 标签需要 Pillow，具体模型依赖由 adapter 自己声明。
