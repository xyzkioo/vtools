# 功能模块化与检测诊断 Spec

状态：已落地第一版；后续可继续扩展独立的 Ultralytics 验证包装和更多可视化模块。
日期：2026-09-10
范围：问题清单第 2、3、4、5 项。

## 1. 目标与边界

每个 test 内部的每项功能都必须成为可单独调用、可单独开关的模块。保留一键执行，同时支持桌面 UI 和终端运行，二者调用同一套实现、读取同一套配置。

- 第 2 项：复用 Ultralytics 已有常规评估能力，不默认重新计算官方 mAP/AP 或阈值曲线；诊断所需的固定工作点匹配计数可以共享一次，用于定位错误来源。
- 第 3 项：自动定位漏检、类别错误、多余框、重复框和定位偏差，筛选差图并提供可视化。
- 第 4 项：分析目标框重叠与漏检的关系，单独识别疑似重复预测。
- 第 5 项：测速、一致性、诊断、可视化内部功能分别模块化，关闭功能即不执行其业务计算。

本期不扩展类别级 CAM，不制作应用 UI，也不编写 AI 调用指南。已有 CAM、特征图等功能只接入开关和模块接口。首期新诊断以轴对齐目标检测框为范围，其他任务明确报告不支持，不猜测输出格式。

## 2. 已确认的现状

- `speed_test` 已有独立入口和部分开关，但参数量、显存等仍与测速实现耦合。
- `model_diagnostics/run_model_diagnostics.py` 通过统一配置解析和模块计划调用诊断引擎；已有预测模式读取一次数据集，B/C 生成预测时会为生成和评估分别读取数据集。
- 诊断引擎已有基于标注框 IoU 的重叠分组，阈值固定为 0.10/0.30，缺少专门的差图展示。
- 多个子项目使用 `core` 等同名顶层导入，组合运行需要消除模块名冲突。
- 保留用户已有模型路径、数据路径和本地未提交修改，不以重构覆盖它们。

## 3. 模块清单

每项使用唯一 ID；YAML 键和终端参数使用相同 ID。

| 分组 | 模块 ID | 职责 | 默认 |
|---|---|---|---|
| 权重检查 | `checkpoint.inspect` | 检查 checkpoint 结构 | 关闭，按需启用 |
| 权重检查 | `checkpoint.load_check` | 验证模型可加载 | 关闭，实际推理仍正常加载 |
| 模型信息 | `model.parameters` | 参数量 | 关闭 |
| 模型信息 | `model.flops` | 复用已有统计能力；不支持时说明 | 关闭，不另写统计器 |
| 测速 | `speed.pytorch_call` | PyTorch 纯模型调用耗时 | 测速入口开启 |
| 测速 | `speed.pytorch_pipeline` | PyTorch 完整 adapter 流程耗时 | 关闭 |
| 测速 | `speed.tensorrt_call` | TensorRT engine 调用耗时 | TensorRT 入口开启 |
| 显存 | `memory.pytorch_peak` | 单独测量 PyTorch 峰值显存增量 | 关闭 |
| 导出 | `export.onnx` | 导出 ONNX | 关闭 |
| 构建 | `build.tensorrt` | 构建 engine | 关闭 |
| 一致性 | `consistency.tensor` | 张量误差比较 | 一致性入口开启 |
| 一致性 | `consistency.detection` | 检测框一对一匹配 | 关闭，按模型能力启用 |
| 推理 | `predict.generate` | 生成可复用预测文件 | 由诊断输入模式决定 |
| 常规验证 | `validation.ultralytics` | 调用仓库已有验证能力 | 关闭 |
| 诊断 | `diagnostics.missed` | 漏检分析 | 诊断入口开启 |
| 诊断 | `diagnostics.classification` | 类别错误分析 | 诊断入口开启 |
| 诊断 | `diagnostics.background` | 无对应目标的多余框 | 诊断入口开启 |
| 诊断 | `diagnostics.duplicate` | 重复预测分析 | 诊断入口开启 |
| 诊断 | `diagnostics.localization` | 定位偏差分析 | 诊断入口开启 |
| 诊断 | `diagnostics.overlap` | 重叠目标及其漏检分析 | 诊断入口开启 |
| 输出 | `output.bad_cases` | 差图排序与清单 | 诊断入口开启 |
| 输出 | `output.images` | 绘制结果图片 | 按配置，示例默认关闭 |
| 输出 | `output.html` | 生成可浏览索引 | 按配置，示例默认关闭 |
| 可视化 | `visualization.features` | 现有特征图功能 | 沿用现有配置 |
| 可视化 | `visualization.cam` | 现有 CAM 功能 | 沿用现有配置 |
| 可视化 | `visualization.stage_trace` | 检测头阶段追踪 | 沿用现有配置 |

默认指各入口随附的示例配置，不允许入口代码在后台覆盖用户显式配置。存在 `modules` 节时，未列出的模块视为关闭；speed_test 不再读取旧的 `test_*` 测试选择字段。旧版配置字段直接报错，用户应按当前配置示例修改。

TensorRT 完整图片 pipeline 本期不新增；不得将 PyTorch pipeline 标成 TensorRT pipeline。显存测量使用独立采样阶段，不将它的额外开销混入耗时结果；CPU 情况记录不支持，不填写 0。

## 4. 运行接口：桌面 UI 与终端

### 4.1 保留入口

- `speed_test/run_all.py`
- `speed_test/benchmark_pytorch.py`
- `speed_test/benchmark_tensorrt.py`
- `speed_test/check_consistency.py`
- `speed_test/inspect_checkpoint.py`
- `model_diagnostics/run_model_diagnostics.py`
- `model_visualization/run_visualization.py`

新增仓库根入口 `run_tools.py`，用于跨工具组合执行。无需为每个指标新建脚本；独立模块通过 `--only` 单独运行，也可由 Python API 调用。

### 4.2 桌面 UI

1. 启动 `python -m vtools_ui`，选择检测诊断、可视化、测速或压缩。
2. 在页面中加载对应 YAML，填写模型、数据路径及 `modules` 开关。
3. 点击运行，UI 会调用同一套命令行后端并显示日志和结果目录。

入口默认配置为 `speed_test/benchmark_config.yaml`、`model_diagnostics/config/config.yaml`、`model_visualization/config/config.yaml` 和 `model_compression/config/config.yaml`；根入口使用 `config/tools.yaml`。

### 4.3 终端

以下为实现后的目标用法，在仓库根目录运行：

```bash
# 与桌面 UI 使用同一套后端
python model_diagnostics/run_model_diagnostics.py

# 使用另一份配置
python speed_test/run_all.py --config config/my_speed.yaml

# 只做重叠分析，读取已有预测；不加载模型
python model_diagnostics/run_model_diagnostics.py --only diagnostics.overlap --predictions runs/predictions.json

# 只测速，临时关闭显存功能
python speed_test/benchmark_pytorch.py --only speed.pytorch_call

# 在 YAML 选择的功能基础上调整
python model_diagnostics/run_model_diagnostics.py --enable diagnostics.overlap --disable output.images,output.html

# 跨工具组合执行
python run_tools.py --config config/tools.yaml

# 查看可用模块，不加载模型、不执行推理
python run_tools.py --tool diagnostics --list-modules
```

从任意目录均可使用入口的绝对路径；同时支持 `python -m speed_test.run_all` 等包方式（仓库位于 Python 搜索路径时）。

统一参数：`--config`、`--only`、`--enable`、`--disable`、`--run-dir`、`--list-modules`；诊断包装入口另支持 `--predictions`。

- `--only` 替换 YAML 的模块选择，仅执行列出的业务模块及必要数据准备；不会附带可视化或常规验证。
- `--enable/--disable` 在配置选择上增减模块；可重复传入或逗号分隔。同一 ID 同时启用和禁用时报错。
- `--only` 不与 `--enable/--disable` 混用，避免覆盖歧义。
- 未知 ID、错误配置、缺少依赖在执行前报错；显示可用模块和具体修复方式。
- `--predictions` 显式选择已有预测模式，覆盖 B/C 推理选择，打印实际生效模式。

优先级：命令行 > YAML > 配置默认值。实际生效配置写入结果目录。

路径：默认配置相对入口文件定位；显式 `--config` 相对终端工作目录解析；当前配置的 `project.root` 相对 YAML 所在目录解析，模型/数据/输出相对该 root；命令行显式文件路径相对工作目录。旧配置不会自动迁移，需按当前配置示例手动确认路径。

## 5. 配置示例

下例为新诊断配置的关键字段；现有模型 A/B/C 配置继续支持。

```yaml
schema_version: 2
project:
  root: ../..

mode: A
mode_a:
  predictions: data/predictions.json
dataset:
  data: data/data.yaml
  split: val

modules:
  predict.generate: false
  validation.ultralytics: false
  diagnostics.missed: true
  diagnostics.classification: true
  diagnostics.background: true
  diagnostics.duplicate: true
  diagnostics.localization: true
  diagnostics.overlap: true
  output.bad_cases: true
  output.images: true
  output.html: true

diagnostics:
  score_threshold: 0.25
  candidate_threshold: 0.001
  match_iou: 0.5
  localization_iou_floor: 0.1
  overlap:
    iou_threshold: 0.3
    smaller_area_threshold: 0.7
    criterion: either
    class_scope: all
  bad_cases:
    top_k: 50

run:
  root: runs
  name: auto
```

B/C 模式需要推理时显式开启 `predict.generate`。无预测且推理关闭时报配置错误，不自行打开推理。阈值为起始默认值，可按数据调整；推理保留分数不得高于诊断需要的候选阈值，否则标记证据不足。

## 6. 模块与共享依赖

建议新增 `vtools_runtime/`，只管理配置、模块注册、运行计划、共享资源和输出记录。具体算法仍放在对应子项目。

```text
vtools_runtime/
  config.py
  registry.py
  context.py
  runner.py
speed_test/modules/
  model_info.py
  latency.py
  memory.py
  export.py
  consistency.py
model_diagnostics/modules/
  predictions.py
  matching.py
  missed.py
  classification.py
  background.py
  duplicate.py
  localization.py
  overlap.py
  bad_cases.py
  rendering.py
  reporting.py
model_visualization/modules/
  features.py
  cam.py
```

当前首版将重叠分析、差图排序和渲染实现集中在
`model_diagnostics/diagnostics/modules.py`，并提供 `overlap.py`、`bad_cases.py`
稳定导入路径；后续算法增多时可按此目录继续拆分。

模块接口：`run(context, options) -> ModuleResult`；结果包括模块 ID、状态、统计、输出文件、耗时及原因。模块不得自行解析命令行、改 `sys.argv` 或调用其他入口的 `main()`。

共享资源包括标注、图片索引、模型、输入、预测、匹配关系。必要的数据读取和匹配属于底层准备，允许按需执行；关闭的业务模块不得因依赖而被隐式启用。关闭错分类输出仍可为漏检分析计算基础匹配，但不得生成错分类报告。

- 已有预测模式中，同一配置的标注只读取一次；B/C 生成预测时会先为 adapter 读取一次标注索引，再由诊断引擎读取一次以完成评估。普通预测只执行一次；测速的 warmup/repeats 属于测量要求，不算重复业务推理。
- 模型按权重、adapter、设备、精度、融合状态等完整配置区分；存在原地修改风险时隔离实例，不强行共享。
- 一致性检查需要其专用同输入、同输出边界，不能用经过后处理的诊断预测替代。
- 已有预测可跨次显式复用；不默默复用上次结果。记录权重来源、图片 ID、类别表、坐标格式、推理阈值、输入尺寸、后处理方式等元数据。
- 元数据缺失可读取，但报告证据限制；坐标或类别无法确定、图片 ID 冲突时报错。
- 依赖按需导入：仅分析 JSON 时不要求安装 PyTorch、CUDA、TensorRT；未开图片输出时不要求绘图库。
- 所有子项目使用包限定导入；薄入口负责兼容直接运行，避免同名 `core` 污染。

## 7. 常规验证去重

新默认诊断流程停止计算官方 AP/mAP、阈值扫描、FP 预算和 bootstrap 等常规汇总。固定工作点的 P/R/F1 仍由一次匹配计算产生，作为差图和错误类型的计数基础，不替代 Ultralytics 的验证结果。旧算法如暂时保留只作为迁移兼容，不得被新默认流程调用。

`validation.ultralytics` 是可选包装模块，调用当前本地仓库的现成验证能力，保存其原始结果和来源，不另写一套 mAP。具体导出适配在实施时依据本地验证代码确认。

若验证和诊断同时开启，优先从验证阶段捕获或导入符合诊断要求的逐图预测。必须保留无预测图片，并校验图片映射、类别 ID、坐标、置信度截断和 max_det。缺少低分候选时不能宣称已分析低置信度漏检。

若已有验证结果只有汇总指标，则无法直接用于差图分析。输入不足时要求提供逐图预测或显式启用预测生成；不为了差图而隐式重新跑完整验证。自定义模型仍可通过 adapter 做诊断，无需强制接入 Ultralytics 验证。

参数量/FLOPs 也属于可选功能，默认关闭；有现成能力就调用，无需每次测速都执行。

## 8. 错误诊断与差图规则

在原图像素坐标下统一处理 `image_id/class_id/score/xyxy`。固定工作阈值做确定性一对一匹配，优先同类别 IoU 达标的匹配；并列时采用固定排序以确保重复运行结果稳定。忽略标注、crowd 标注和无效框沿用并明确现有数据处理规则。

预测主分类互斥：正确匹配 → 对已匹配同类 GT 的重复预测 → 对尚未匹配 GT 的类别错误 → 定位偏差 → 背景多余框。精确阈值和匹配优先级纳入测试。

GT 未被正确匹配计为未正确检出，同时附原因（低分、类别错误、定位偏差、无候选等）。同一个类别错误可能关联一个 GT 和一个预测：报告保留关联 ID；差图总错误数按事件去重，不能把两张明细表直接相加。

首版差图按错误事件数量降序、图片 ID 稳定排序；比例分母为有效 GT 数，无 GT 图片比例记为空。清单保留所有有错误的图片，可视化数量受 `bad_cases.top_k` 限制，后续可继续增加按类型筛选。

输出包括图片路径、各类问题数量、关联目标/预测 ID、排序依据。图片展示标注、预测、问题高亮三个面板，提供颜色图例和置信度，HTML 链接相对输出目录，整个目录可拷贝查看。

## 9. 重叠分析规则

框对计算 `IoU=交集/并集` 与 `IoS=交集/较小框面积`。支持任一阈值满足或同时满足，支持同类/全部类别，排除自身和零面积框；框对用稳定 ID 去重。

- GT 重叠：记录相交框对、两个指标、目标是否正确检出；按目标去重统计重叠目标数、漏检数和召回率，单列框对数。
- 预测重叠：报告疑似重复对；有 GT 时通过一对一匹配和目标归属判断是否属于重复预测。两个真实相邻目标的预测不能只因重叠被判成重复。
- 只有最终预测时，不将漏检断定为 NMS 导致；只有存在可比较的前后处理预测证据时才提供相关分析。
- 报告写“检测框重叠”，不将其等同于真实遮挡。
- 无 GT 可执行预测重叠筛查；漏检、错分类等需要 GT 的模块在执行前报输入不足，不给出虚假结论。

## 10. 输出与状态

每次运行创建新 `runN`。统一根入口组织示例：

```text
runs/runN/
  config_used.yaml
  run_manifest.json
  predictions/          # 仅本次生成预测时写入
  speed/                # 仅开启相应模块时生成
  consistency/
  diagnostics/
    missed.csv
    classification.csv
    overlap_pairs.csv
    overlap_summary.json
    bad_cases.csv
  images/
  index.html
```

关闭的模块不生成业务文件；运行清单仍记录其禁用状态。开启但无问题的分析输出明确的零计数，明细 CSV 保留表头。HTML 可以只展示清单；关闭图片绘制时不触发绘制。开启 HTML 或图片输出却没有任何可消费分析模块时，配置校验报错。

状态：`passed/failed/error/inconclusive/skipped`。发现差图是正常分析结果，默认不等同于任务失败；一致性不达标属于 failed。退出码：0 表示所选功能成功完成；1 表示检查失败、运行错误或证据不足；2 表示参数/配置错误。显式请求而不支持的功能不能仅记 skipped 后返回成功。

诊断入口的 `--run-dir` 必须指向尚不存在的目录。测速和压缩入口允许一键流程的多个阶段复用同一运行目录；各自的运行管理器负责把输出定位到本次目录，模型版本产物使用唯一文件名避免覆盖。

## 11. 实施阶段

1. 建立模块接口、配置解析、执行计划和直接运行/包运行兼容；保留现有入口。
2. 拆分测速、模型信息、显存、导出、构建、一致性和现有可视化功能；落实关闭即不计算。
3. 拆分诊断数据读取与预测接口，保持已有预测模式只由引擎读取一次 GT，关闭默认常规评估，接入可选现成验证能力。
4. 实现各类诊断独立模块、重叠分析、差图排序和可视化输出。
5. 增加配置说明、桌面 UI/终端示例，完成下面的验收。

## 12. 验收标准

- 同一配置在桌面 UI、终端脚本和 `python -m` 运行时，选择模块和业务结果一致；从其他工作目录启动仍可定位默认配置。
- 每项开关独立有效；用调用计数或故意抛错的替身验证禁用功能没有被执行，不能只检查输出文件缺失。
- 一键开启多个诊断模块时，标注只读一次、预测只生成一次；关闭常规验证时绝不调用 Ultralytics 验证或旧 AP 计算器。
- 只读已有预测并分析时，不导入模型后端；TensorRT 模块关闭时不需要 TensorRT 环境。
- 覆盖正确检测、漏检、错分类、定位偏差、重复框、相邻真实目标、包含框、无 GT 图片、无预测图片、忽略标注及无效框。
- 检验差图事件去重、稳定排序、Top K、零分母、重叠框对去重与目标统计去重。
- 一致性两项开关分别验证；只开框匹配时不执行完整张量误差统计，仅保留解码和合法性检查所需准备。
- 验证当前 YAML 能直接运行；保留用户路径；未知开关和冲突配置有可操作的错误提示。
- 有 GPU 时做真实模型测速与一致性冒烟；无 GPU 时记录未验证项，不将替身测试视为真实 GPU 验证。
- 检查导出图片和 HTML 的可读性、相对链接及中文显示；已有结果不会混入新运行。

完成定义：上述模块均能独立选择执行，2/3/4/5 的行为落实，文档中的桌面 UI 和终端用法通过对应验证；仅拆文件但仍全量执行不算完成。
