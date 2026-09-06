# CSV 结果指标说明

本文只解释工具生成的 CSV 文件，不解释配置、checkpoint 或命令行参数。CSV 中的
时间单位均为 **ms（毫秒）**；除非特别说明，速度统计来自同一次运行目录中的
有效重复测量。

## 1. 先看哪个 CSV

| CSV 文件 | 用途 | 主要回答的问题 |
| --- | --- | --- |
| `vision_speed_v2.csv` | PyTorch 明细 | PyTorch 纯模型调用和 adapter 完整流程各多快？ |
| `vision_tensorrt_v2.csv` | TensorRT 明细 | ONNX/engine 构建是否成功，engine 调用多快？ |
| `vision_benchmark_summary.csv` | `run_all.py` 汇总 | 本次运行各阶段成功/失败及测量结果是什么？ |
| `consistency.csv` | PyTorch/TensorRT 一致性明细 | 同一输入上两边输出是否一致？ |
| `consistency.json` | 一致性完整嵌套报告 | 需要查看全部输入、输出和匹配细节时使用；CSV 是它的扁平化版本 |

每次运行的 CSV 应优先和同一个 `runN` 目录中的其他文件一起解读。不要把不同
运行、不同 GPU、不同输入尺寸或不同精度的行直接混在一起比较。

## 2. PyTorch 测速 CSV：`vision_speed_v2.csv`

### 2.1 模型和测试条件字段

| 列 | 含义 | 如何解读 |
| --- | --- | --- |
| `model` | 配置中的模型名称 | 用于区分同一 CSV 中的多个模型。 |
| `weights` | 实际使用的权重文件路径 | 确认比较的是同一个权重；路径不同即使模型名相同也不能直接比较。 |
| `backend` | 后端名称 | 当前 PyTorch CSV 通常为 `pytorch`。 |
| `adapter` | 模型适配器名称 | 决定如何加载模型、构造输入和调用 pipeline，例如 `ultralytics`。 |
| `precision` | 测试精度 | `fp32`、`fp16` 或 `bf16`。精度不同应分开报告。 |
| `height` | 输入张量高度 | 单位为像素；与 `width` 一起表示模型调用输入尺寸。 |
| `width` | 输入张量宽度 | 单位为像素。 |
| `batch` | 每次模型调用的样本数 | `batch=1` 时是一张图一次；大于 1 时，`fps_per_sample` 是每秒样本数，不是每秒 batch 数。 |
| `params` | 已加载 PyTorch 模型的参数量 | 是参数个数，不是权重文件 MB，也不是 FLOPs；模型融合、包装或 adapter 不同可能使数值不同。 |
| `scope` | 计时范围 | 重点看 `model_call` 和 `adapter_pipeline` 的区别，见下表。 |
| `scope_note` | 该行计时边界的文字说明 | 用于防止把纯模型调用误当成端到端时间。 |

`scope` 的三种常见值：

| `scope` | 计时内容 | 适合回答 |
| --- | --- | --- |
| `model_call` | 已在设备上的输入直接调用模型；不含图像读取、预处理和后处理 | 网络前向本身的计算速度 |
| `adapter_pipeline` | `adapter.predict` 的完整边界；是否含预处理/后处理取决于 adapter | 当前适配器在实际调用入口上的速度 |
| `engine_call` | 固定输入直接调用 TensorRT engine；通常不含图像读取、预处理和后处理 | TensorRT engine 本身的推理速度 |

因此，`model_call` 与 `adapter_pipeline` 不是重复数据：前者更接近网络算子
耗时，后者更接近适配器对外提供的调用耗时。当前 Ultralytics 检测模型的
`adapter_pipeline` 通常包含其 `predict` 过程；自定义 adapter 的具体范围应以
`scope_note` 为准。

### 2.2 延迟和速度字段

| 列 | 含义 | 如何解读 |
| --- | --- | --- |
| `mean_ms` | 所有有效重复的平均延迟 | 受偶发抖动影响较大，适合观察总体平均，不建议单独作为主结论。 |
| `median_ms` | 延迟中位数（P50） | 一半测量不超过该值；通常作为主要“典型延迟”指标。 |
| `p90_ms` | P90 延迟 | 90% 测量不超过该值，剩余 10% 更慢；用于观察尾延迟。 |
| `min_ms` | 最小延迟 | 理想情况下的一次最快测量，不代表稳定运行速度。 |
| `max_ms` | 最大延迟 | 本次重复中的最慢测量，可用于发现抖动或首次异常。 |
| `std_ms` | 延迟标准差 | 越小通常表示越稳定；不同场景应结合 `median_ms` 一起看。 |
| `fps_per_sample` | 按中位延迟换算的吞吐 | 公式为 `1000 × batch / median_ms`；`batch=1` 时就是 `1000 / median_ms`。它不是数据集准确率，也不自动包含图片读盘时间。 |
| `peak_delta_mb` | 纯模型调用期间的 CUDA 峰值显存增量 | 近似为该次调用新增的峰值显存，单位 MB；不是进程总显存，也不是显卡总显存。当前 pipeline 行通常留空。 |
| `output_count` | adapter 定义的输出数量 | 对 Ultralytics 检测通常是阈值后的检测框数量；普通原始模型输出可能为空。它不是类别数、参数量或 mAP。 |

延迟统计已经包含 GPU 同步，因此不能用“异步 CUDA 调用返回时间”自行推断出同样
的数值。比较两个模型时，应固定 `height`、`width`、`batch`、`precision`、
`warmup`、`repeats` 和硬件。

### 2.3 一个 PyTorch 行的示例

例如下面这一行：

```text
backend=pytorch, precision=fp16, scope=model_call,
median_ms=6.68, p90_ms=8.88, fps_per_sample=149.67,
peak_delta_mb=44.45, batch=1
```

含义是：在指定输入尺寸和 GPU 上，已经准备好的单个 FP16 输入直接走模型前向，
典型延迟约 **6.68 ms**，P90 尾延迟约 **8.88 ms**，按中位数折算约 **149.7
样本/秒**，这次调用的峰值显存增量约 **44.45 MB**。它不表示完整图片处理流程
也不表示模型精度。

## 3. TensorRT 测速 CSV：`vision_tensorrt_v2.csv`

TensorRT CSV 使用的速度字段与 PyTorch 基本相同，另外记录 ONNX、engine 和阶段
状态。列顺序以当前版本为准：

| 列 | 含义 | 如何解读 |
| --- | --- | --- |
| `model`、`weights`、`adapter` | 模型身份和加载方式 | 与 PyTorch 行核对，确认两边来自同一权重/adapter。 |
| `backend` | `pytorch` 或 `tensorrt` | TensorRT CSV 可能同时写入用于参考的 PyTorch 行和 TensorRT 行。比较 engine 时筛选 `backend=tensorrt`。 |
| `precision` | engine/参考模型精度 | 例如 `fp16`；不要把 FP16 TensorRT 和 FP32 PyTorch 当作同一条件。 |
| `height`、`width`、`batch` | engine 输入形状 | engine 通常绑定这些形状；形状不一致不能直接比较。 |
| `params` | PyTorch 参考模型参数量 | TensorRT `engine_call` 行通常为空，因为 engine 不是 `nn.Module`。 |
| `scope`、`scope_note` | 计时边界 | TensorRT 正常速度行通常为 `engine_call`。 |
| `mean_ms`、`median_ms`、`p90_ms`、`min_ms`、`max_ms`、`std_ms` | engine 调用延迟统计 | 含义与 PyTorch 相同。 |
| `fps_per_sample` | 按中位 engine 延迟换算的吞吐 | 公式同上；只是 engine 内部吞吐，不自动等于端到端应用 FPS。 |
| `peak_delta_mb` | 峰值显存增量 | 当前 TensorRT engine 测速通常不填该值；空白不等于零。 |
| `output_count` | adapter 定义的输出数量 | engine 原始输出通常不提供检测框计数，因此可能为空。 |
| `onnx` | 本次使用或生成的 ONNX 文件路径 | 用于追溯 engine 的输入图。 |
| `engine` | 使用或生成的 TensorRT engine 路径 | 同一 engine 与 GPU、TensorRT/CUDA 版本相关，换部署机通常需要重新构建。 |
| `status` | 该行/阶段状态 | 常见为 `succeeded` 或 `failed`。 |
| `error` | 失败原因 | 成功时通常为空；失败时应先看该列，不要把空的延迟字段当作 0。 |

### TensorRT 行的比较方式

正常情况下，优先比较：

1. PyTorch `scope=model_call` 与 TensorRT `scope=engine_call`：看网络/engine
   内部速度差异。
2. PyTorch `scope=adapter_pipeline`：看适配器完整调用的额外开销。

`engine_call` 不包含完整的图片读取、预处理、后处理时，不能直接宣称它是用户
最终看到的端到端 FPS。若要比较端到端速度，必须让两边使用相同的输入读取、
预处理、后处理和计时边界。

## 4. 汇总 CSV：`vision_benchmark_summary.csv`

该文件是 `run_all.py` 汇总的“本次运行内存结果”，不是把旧 CSV 追加在一起。
它会同时包含两类行：

| `record_type` | 含义 |
| --- | --- |
| `measurement` | 具体模型/精度/计时范围的测量行；字段与后端明细 CSV 相近。 |
| `stage_summary` | 阶段级状态行，例如 `pytorch`、`tensorrt`、`consistency` 是否成功、跳过或失败。 |

汇总 CSV 的通用字段：

| 列 | 含义 |
| --- | --- |
| `record_type` | 该行是测量还是阶段摘要。先按此列筛选再分析。 |
| `backend` | `pytorch`、`tensorrt` 或 `consistency` 等阶段名称。 |
| `status` | 测量行或阶段的状态。 |
| `error` | 错误或跳过原因；空值表示没有记录错误。 |
| 其余模型/速度字段 | 对 `measurement` 行有效；对 `stage_summary` 行常为空，这是正常的。 |

常见状态：

| 状态 | 含义 |
| --- | --- |
| `succeeded` | 该测速阶段完成并返回测量结果。 |
| `failed` | 阶段发生错误或存在失败测量；结合 `error` 查看原因。 |
| `skipped` | 按流程跳过，例如本次 TensorRT 失败后不使用旧 engine 继续做一致性检查。 |
| `warning` | 有诊断性差异，但按当前验收模式仍保留为警告。 |
| `inconclusive` | 输入没有足够证据得出通过/失败，例如双方都没有检测框。 |
| `error` | 一致性或加载阶段出现未处理的运行错误。 |

`run_all.py` 为了把各阶段结果写入汇总，进程可能仍以退出码 0 结束；是否成功
应以 `vision_benchmark_summary.csv` 的 `status` 和 `error` 为准，而不是只看
退出码。

## 5. 一致性 CSV：`consistency.csv`

该文件不是速度表，而是把 PyTorch 与 TensorRT 在同一输入上的比较结果扁平化为
多行。通过 `record_type` 区分三种记录：

| `record_type` | 一行代表什么 |
| --- | --- |
| `case` | 一个模型 + 一张图片（或一个 adapter 示例输入）的总体结论。 |
| `tensor` | 一个 ONNX 输出张量的逐元素数值比较。 |
| `detections` | 一个 batch 内一张图片的检测框匹配结果。 |

### 5.1 `case` 行

| 列 | 含义 |
| --- | --- |
| `model` | 模型名称。 |
| `source` | 本次比较的图片路径；adapter 示例输入时可能为 `adapter_example_input`。 |
| `case_status` / `status` | 该输入的总体状态；两列通常表示同一结论，`status` 方便通用 CSV 工具筛选。 |
| `precision` | TensorRT 与参考 PyTorch 的精度设置。 |
| `raw_tensor_status` | 所有输出逐元素比较的原始状态，仅作为诊断。 |
| `detection_status` | 若启用检测框匹配，表示检测框层面的状态。 |
| `acceptance_mode` | 配置的验收模式：`auto`、`all` 或 `detection_only`。 |
| `acceptance_mode_effective` | 本案例实际采用的模式；`auto` 可能变成 `detection_only` 或 `all`。 |
| `acceptance_note` | 模式转换说明，例如 `auto→detection_only`。 |
| `error` | 该输入无法比较时的错误。 |

验收模式的意义：

| 模式 | 总体结论依据 |
| --- | --- |
| `all` | 要求所有输出逐元素通过；适合输出顺序和数值都必须完全对齐的模型。 |
| `detection_only` | 有检测框输出时，以类别、IoU 和置信度匹配为主要验收；原始张量差异仍保留在 CSV 中。形状不匹配、NaN/Inf 等硬错误仍会失败。 |
| `auto` | 检测模型自动采用 `detection_only`；没有可识别检测框输出时回退到 `all`。 |

### 5.2 `tensor` 行：逐元素指标

| 列 | 含义 | 如何解读 |
| --- | --- | --- |
| `name` | ONNX 输出张量名称 | 例如 `cat_27`、`silu_52`；这些通常是中间/原始导出输出，不一定是最终检测框。 |
| `status` | 该张量是否在容差内 | `passed` 表示全部元素通过；`failed` 表示至少一个元素超出容差。 |
| `reference_shape` / `actual_shape` | PyTorch / TensorRT 输出形状 | 形状不同属于结构性问题，不能用提高容差掩盖。 |
| `reference_dtype` / `actual_dtype` | 两侧数据类型 | dtype 差异可能来自精度设置；整数/布尔输出按严格相等处理。 |
| `elements` | 参考张量元素个数 | 用于理解 `close_fraction` 的分母。 |
| `atol` | 绝对误差容差 | 允许的固定误差部分。 |
| `rtol` | 相对误差容差 | 允许随参考值大小增加的误差部分。 |
| `close_fraction` | 通过容差的元素比例 | 例如 `0.95` 表示约 95% 元素通过；仍可能有少量关键元素失败。 |
| `mismatched_elements` | 超出容差的元素个数 | 与 `elements` 一起看，不要只看百分比。 |
| `max_abs_error` | 最大绝对误差 | `max(abs(actual-reference))`；容易被一个异常元素主导。 |
| `mean_abs_error` | 平均绝对误差 | 所有元素绝对误差的平均值。 |
| `rmse` | 均方根误差 | 对较大的误差更敏感。 |
| `max_rel_error` | 最大相对误差 | 参考值接近 0 时可能很大，应与绝对误差一起看。 |
| `worst_index` | 最大绝对误差所在的索引 | 用于定位具体 batch/channel/位置。 |
| `reference_at_worst` / `actual_at_worst` | 该位置两侧的数值 | 结合 `max_abs_error` 判断偏差规模。 |
| `reason` | 通过/失败原因 | 例如 `elementwise_tolerance`、`shape_mismatch`、`nan_or_inf`。 |

浮点逐元素判断使用：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

所以 `failed` 不一定意味着最终任务结果错误；检测模型可能因为候选框排序、
补零框、拼接顺序或 TensorRT 数值差异造成原始张量不逐元素相同。`cat_*`、
`silu_*` 这类名称通常更适合作为导出/数值诊断，而不是直接作为检测准确率。

### 5.3 `detections` 行：检测框匹配指标

该部分只在输出符合 `[B, N, 6]` 的 `xyxy + score + class` 格式且检测匹配被启用
时出现。它比较的是 PyTorch 参考输出，不是标注数据集上的 Recall、Precision
或 mAP。

| 列 | 含义 | 如何解读 |
| --- | --- | --- |
| `batch_index` | batch 内图片索引 | `0` 表示第一张图片。 |
| `pytorch_count` | PyTorch 侧置信度达到阈值的框数 | 阈值由 `conf` 记录。 |
| `tensorrt_count` | TensorRT 侧达到同一阈值的框数 | 与 PyTorch 数量不一致时通常会有未匹配框。 |
| `matched_count` | 成功匹配的框数 | 同时满足类别相同、IoU 达标、置信度差不超过 `score_atol`。 |
| `unmatched_pytorch` | PyTorch 未匹配框数 | `pytorch_count - matched_count`。 |
| `unmatched_tensorrt` | TensorRT 未匹配框数 | `tensorrt_count - matched_count`。 |
| `min_matched_iou` | 已匹配框中的最小 IoU | 越接近 1 表示坐标越接近；阈值由 `iou_threshold` 指定。 |
| `max_matched_score_diff` | 已匹配框中的最大置信度差 | 应不大于 `score_atol`。 |
| `pairs` | 匹配的索引对 | `[PyTorch 框索引, TensorRT 框索引]`；允许两侧框的排列顺序不同。 |
| `conf` | 比较前的置信度过滤阈值 | 低于该值的框不参与匹配。 |
| `iou_threshold` | 框匹配的最小 IoU | 不是训练/评估时的 NMS IoU，也不是 mAP 的 IoU 扫描。 |
| `score_atol` | 匹配框允许的置信度绝对差 | 越小越严格。 |
| `reason` | 匹配方式或不通过原因 | 常见为 `class_iou_score_matching`；双方都为空时为 `empty_detections_are_not_evidence`。 |

状态含义：

| 状态 | 含义 |
| --- | --- |
| `passed` | 过滤后的两侧框数量相同，且每个框都找到满足类别/IoU/分数条件的匹配。 |
| `failed` | 数量、类别、IoU 或置信度条件至少一项不满足。 |
| `inconclusive` | 两边都没有达到 `conf` 的框；这不能证明模型一致，所以不算通过。 |

## 6. 如何解读一次完整结果

下面是一个典型的 FP16、`batch=1` 结果的读法（数值仅作格式示例）：

| 项目 | `median_ms` | `p90_ms` | `fps_per_sample` | 结论 |
| --- | ---: | ---: | ---: | --- |
| PyTorch `model_call` | 6.17 | 7.47 | 162.0 | 网络纯前向速度 |
| PyTorch `adapter_pipeline` | 6.72 | 7.46 | 148.7 | 适配器调用边界速度 |
| TensorRT `engine_call` | 2.78 | 3.76 | 359.8 | engine 内部速度 |

这组数据可以说 TensorRT engine 的内部调用中位延迟低于 PyTorch 纯模型调用；
不能仅凭这三行断言完整应用端到端速度提升同样多，也不能由速度 CSV 推断 mAP。
一致性 CSV 还要单独查看 `case_status`、`detection_status` 和 `inconclusive`
情况。

## 7. 旧版 CSV 字段对照

如果目录中还有旧版 `detection_speed_832.csv` 或
`tensorrt_fp16_comparison.csv`，它们的字段含义如下。旧版和新版计时边界可能
不同，建议不要直接拼表比较。

### `detection_speed_832.csv`

| 旧字段 | 对应含义 |
| --- | --- |
| `imgsz` | 正方形输入尺寸；新版拆为 `height`、`width`。 |
| `forward_median_ms` / `forward_p90_ms` | 纯前向延迟的中位数/P90；大致对应新版 `model_call` 的延迟字段。 |
| `forward_fps` | 按纯前向延迟换算的吞吐。 |
| `e2e_mean_ms`、`e2e_median_ms`、`e2e_p90_ms`、`e2e_min_ms`、`e2e_max_ms` | 旧版端到端计时统计；具体是否包含读图要以旧脚本实现为准。 |
| `e2e_fps` | 按旧版端到端中位延迟换算的吞吐。 |
| `reported_preprocess_ms` / `reported_inference_ms` / `reported_postprocess_ms` | adapter/框架报告的分阶段时间，不一定和外层 `e2e_*` 完全相加。 |
| `peak_forward_delta_mb` | 旧版纯前向峰值显存增量。 |
| `detections` | 旧版输出的检测框数量。 |

### `tensorrt_fp16_comparison.csv`

| 旧字段 | 对应含义 |
| --- | --- |
| `model_file_mb` | 旧版记录的模型/engine 文件大小，单位 MB；不是显存，也不是参数量。 |
| `fps` | 旧版按 `median_ms` 换算的吞吐；新版名称为 `fps_per_sample`。 |
| `preprocess_mean_ms` / `inference_mean_ms` / `postprocess_mean_ms` | 旧版平均预处理、推理、后处理时间。 |
| `speedup_vs_pytorch` | 旧版相对 PyTorch 基线的速度倍数；通常约为 `PyTorch median / 当前 median`，必须确认两行条件完全一致后才可使用。 |
| 其余 `mean_ms`、`median_ms`、`p90_ms`、`min_ms`、`max_ms`、`detections` | 含义与本文前面的同名字段相同。 |

## 8. 论文/报告建议

建议至少报告以下信息：

- GPU、PyTorch/TensorRT/CUDA 版本；
- 输入 `height × width`、`batch`、精度；
- `warmup` 和 `repeats`；
- PyTorch `model_call` 的 `median_ms`、`p90_ms`、`fps_per_sample`；
- PyTorch `adapter_pipeline` 的同名指标（若需要反映应用调用开销）；
- TensorRT `engine_call` 的同名指标，并注明它不是完整端到端 pipeline；
- 一致性结果中的样本数、`passed`/`warning`/`inconclusive` 数量及验收模式。

速度 CSV 说明“快不快”，一致性 CSV 说明“同一输入是否接近”；二者都不能替代
数据集上的 Precision、Recall、F1 或 mAP 评估。
