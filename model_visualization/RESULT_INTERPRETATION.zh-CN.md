# 模型可视化结果解读指南

本文件用于解释 `model_visualization/runs/runN/` 中的输出。建议不要一开始逐个打开所有图片，而是按下面的顺序判断：

```text
index.html
  -> stage_summary.csv
  -> final_detections.json / raw_candidates.json
  -> features/activations/
  -> cams/
  -> canonical/
```

## 一、先确认本次运行是否有效

打开运行目录中的 `index.html`，先看每张输入图片是否都有结果链接。

再检查：

- `runtime_state.json`：实际使用的模型、设备、精度和源码路径；
- `config_used.yaml` 或 `config_used.json`：本次运行真正使用的配置；
- `images/<短ID>/image_meta.json`：原图尺寸、输入尺寸、缩放比例和 padding，同时保留完整 `image_id` 与原始路径。

如果模型路径、图片路径、设备或输入尺寸不对，后面的热力图没有分析价值。

## 二、阶段追踪：最重要的三个文件

阶段追踪需要配置：

```yaml
modules:
  visualization.stage_trace: true
```

### 1. `stage_summary.csv`

这是最先看的统计表。每张图片通常有三行：

| stage | 含义 |
| --- | --- |
| `raw_candidate` | 检测头产生的全部候选框 |
| `head_topk` | 检测头筛选后保留的 Top-K 候选 |
| `final` | 经过置信度、NMS、`max_det` 等处理后的最终框 |

建议计算三个比例：

```text
Top-K 保留率 = head_topk 数量 / raw_candidate 数量
最终保留率 = final 数量 / head_topk 数量
总保留率 = final 数量 / raw_candidate 数量
```

这些比例不是准确率或召回率，只用于定位候选框在哪一个阶段大量减少。

### 2. `raw_candidates.json`

保存筛选前的候选框。它可以回答：模型在最开始有没有“看到”目标。

常用字段：

| 字段 | 含义 |
| --- | --- |
| `raw_index` | 检测头原始候选索引，阶段之间追踪候选时使用 |
| `candidate_id` | 当前阶段的可读候选编号 |
| `bbox_input` | 输入尺寸 letterbox 坐标下的框 |
| `bbox` | 还原到原图坐标后的框 |
| `score` | 候选置信度 |
| `class_id` | 类别编号 |
| `source_level` | 来源特征层，例如 `P3`、`P4`、`P5` |
| `source_level_index` | 特征层在检测头输入列表中的编号 |
| `source_index` | 该层内部的候选位置 |
| `stride` | 该层步长；通常 P3/P4/P5 对应 8/16/32 |
| `grid_x`、`grid_y` | 候选在该层网格中的位置 |
| `in_head_topk` | 是否进入检测头 Top-K |
| `in_final` | 是否成为最终检测框 |
| `filter_reason` | 没有进入下一阶段的原因 |

### 3. `final_detections.json`

保存最终输出框。它应当与普通预测结果基本一致，可以用来检查：

- 最终框数量是否合理；
- 位置是否落在花簇上；
- 置信度是否普遍过低；
- 最终框主要来自 P3、P4 还是 P5。

## 三、`filter_reason` 怎么看

`raw_candidates.json` 中的 `filter_reason` 是定位问题的核心字段：

| 值 | 解释 | 主要含义 |
| --- | --- | --- |
| `kept_final` | 保留为最终框 | 候选完整通过筛选 |
| `below_conf` | 低于置信度阈值 | 候选分数没有超过 `detection.conf` |
| `removed_by_nms` | 被 NMS 去除 | 与更高分框重叠过大 |
| `removed_after_head_topk` | Top-K 后被去除 | 进入过 Top-K，但后续没有成为最终框 |
| `removed_by_topk` | 被 Top-K 去除 | 候选在检测头阶段就被截掉 |

### 常见判断

- `raw_candidate` 很少：模型在检测头前端就没有产生足够响应，优先检查特征图、输入尺寸和模型结构；
- raw 很多，但 `below_conf` 很多：模型有响应，但置信度不足，可能是小目标、训练不足或类别区分困难；
- `removed_by_topk` 很多：候选竞争激烈，Top-K 可能过小；
- `removed_by_nms` 很多：同一目标产生了大量相互重叠的框，检查 NMS 的 IoU 阈值和重复预测；
- final 很少但 raw 很多：不要直接判断模型没有看到目标，要先看这些候选的位置和来源层；
- final 框数量正常但位置偏移：检查 letterbox 的输入尺寸、padding 和原图坐标还原。

## 四、P3/P4/P5 来源怎么解读

`source_level` 表示候选来自哪个检测头输入层，不表示该目标只由这一层独立识别。

一般规律：

| 来源 | 常见特点 | 重点观察 |
| --- | --- | --- |
| `P3` | 分辨率较高，适合较小目标 | 小花簇是否有响应、候选是否足够 |
| `P4` | 中等尺度 | 中等大小目标和密集区域 |
| `P5` | 分辨率较低、语义较强 | 大目标或整体结构 |

判断时不要只看最终框数量。应同时比较：

1. 各层 raw 候选数量；
2. 各层进入 Top-K 的数量；
3. 各层最终保留数量；
4. 各层候选框是否实际覆盖目标。

如果 P3 的特征图明显有响应，但 P3 候选大多被 `removed_by_topk` 或 `below_conf`，问题可能在候选排序、阈值或 Top-K，而不一定是 backbone 没有提取到特征。

## 五、特征图和聚合图怎么读

### `features/`

这里通常是每个捕获层的多个通道图。亮的区域表示该通道激活较强，不等于“那里一定是目标”。

重点看：

- 目标位置是否比背景更亮；
- 目标变小后响应是否消失；
- 响应是否集中在花簇边缘、枝条或背景纹理；
- 不同 P 层的响应范围是否发生变化。

### `activations/`

这是多个通道的聚合图，默认通常为 `mean_abs`。它更适合快速判断某一层整体有没有响应，但不能代替单通道分析。

如果聚合图全图都亮，可能是背景纹理或低层边缘响应；如果整张图都暗，可能是层选择不合适、输入路径错误或该层没有有效激活。

### `layers.csv`

如果不知道该观察哪些模块，先运行：

```yaml
mode: list_layers
```

它只生成层清单。先查看模块名称，再把需要的模块写入：

```yaml
layers:
  modules:
    - model.16
    - model.19
```

## 六、CAM 热力图怎么读

启用：

```yaml
modules:
  visualization.cam: true
cam:
  method: layercam
```

CAM 表示“为了当前候选或最终框，哪些区域对该输出更有贡献”。它不是目标分割图，也不是置信度图。

### 正常现象

- 热点大致覆盖花簇，而不是完全落在背景；
- 小目标的热点可能比目标框小或略有偏移；
- 不同层的 CAM 范围不同，浅层更细，深层更粗。

### 异常现象

- 热点集中在枝条、天空、叶片纹理：模型可能学到了背景捷径；
- 热点远离最终框：检查 target 类型、候选索引和坐标还原；
- 热点几乎全图均匀：层选择不合适，或目标梯度信号太弱；
- CAM 没有输出：确认使用 FP32，并检查目标候选确实存在。

CAM 前向固定使用 FP32，并且需要梯度，所以速度会明显慢于普通特征图。

## 七、`canonical/` 与检测诊断的关系

开启：

```yaml
stage_trace:
  export_canonical: true
```

会按图片生成：

- `canonical/raw_<短ID>.json`：该图片的筛选前预测；
- `canonical/final_<短ID>.json`：该图片的最终预测。

`<短ID>` 形如 `img_a1b2c3d4e5f6`。JSON 的 `image_id`、`file_name`、`metadata.source_image_path` 保留完整图片标识和原始路径，`run_metadata.json` 与 `index.html` 也保存同一映射。不会再生成跨图片的 `raw_predictions.json` 或 `final_predictions.json`。

这些文件可以直接作为 `model_diagnostics` 的预测输入，用于继续分析 Precision、Recall、误检和漏检。

推荐流程是：

```text
可视化阶段追踪
  -> 找到候选在哪一层、哪一步消失
  -> canonical/final_<短ID>.json
  -> 检测诊断统计整体指标
```

阶段追踪不能替代验证集指标，单张图上的热力图也不能证明模型整体性能提升。

## 八、建议的实际分析顺序

### 第一次运行：只看结构和候选数量

```yaml
modules:
  visualization.features: true
  visualization.cam: false
  visualization.stage_trace: true
```

先看 `index.html`、`stage_summary.csv`、`raw_candidates.json` 和 `final_detections.json`。

### 第二次运行：对比 P3/P4/P5

选取同一张小目标图片，比较各层特征图和候选来源。不要同时改变输入尺寸、阈值和模型权重。

### 第三次运行：只对少量目标做 CAM

```yaml
modules:
  visualization.cam: true
cam:
  max_targets: 1
```

CAM 主要用于解释“为什么这个框被保留”或“为什么模型关注了背景”，不建议一开始对整个数据集全部生成。

## 九、常见误区

- raw 候选数量多，不代表 Recall 高；候选可能全是背景；
- final 框数量少，不一定是特征提取失败，可能是置信度、Top-K 或 NMS 筛掉了；
- 热力图亮，不等于检测框一定正确；
- P3/P4/P5 来源是候选归属，不是严格的因果证明；
- `display_conf` 只影响图片显示，不会删除 raw 记录；
- `bbox_input` 和 `bbox` 不在同一坐标系，不能直接混用；
- 不同运行只有在模型、输入尺寸、阈值和分支一致时才适合比较。

## 十、最小结论模板

每次分析完一张图，可以按下面格式记录：

```text
图片：
模型/分支：
输入尺寸：
raw / Top-K / final：
主要来源层：
主要过滤原因：
P3 特征是否覆盖目标：
P4 特征是否覆盖目标：
P5 特征是否覆盖目标：
CAM 是否集中在目标：
初步结论：
下一步只改变的变量：
```
