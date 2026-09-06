# 通用 PyTorch 视觉模型测速



by:小尛ovo


最常用的流程：

1. 修改 `benchmark_config.yaml` 中的项目根目录、模型权重和输入图片。
2. 可先运行 `inspect_checkpoint.py` 检查 `.pt` 是否能恢复模型架构。
3. 在 PyCharm 直接运行 `run_all.py`。
4. 查看终端打印的 `runs-profile/runN` 目录；每次运行都会使用新的编号，
   其中包含本次的 PyTorch/TensorRT/一致性 CSV 或 JSON 结果。

新增：直接运行 `check_consistency.py` 检查已有 ONNX/engine 与 PyTorch 的输出。
`run_all.py` 也可以通过 `consistency.enabled` 和 `run_all.run_consistency` 执行。
读取同一份输入，不重复测速；报告为 `consistency.csv` 和 `consistency.json`。
使用方法、容差、YOLO26 框匹配及自定义 adapter 接入见完整教程第 8 节。

注意：须先成功构建 TensorRT engine；一致性通过不代替 mAP 验证。

默认 `run.enabled: true`，结果会依次保存到 `runs-profile/run1`、`run2`……；
ONNX 和 TensorRT engine 仍使用 `tensorrt.onnx_dir`、`tensorrt.engine_dir` 的
公共缓存。需要补跑到已有目录时，可在 PyCharm 参数中加入
`--run-dir runs-profile/run2`；设置 `run.enabled: false` 可恢复固定输出路径。

---

## 目录结构

```text
.
├── benchmark_config.yaml          # 唯一需要日常修改的配置
├── run_all.py                     # 一键入口
├── benchmark_pytorch.py           # PyCharm：PyTorch
├── benchmark_tensorrt.py          # PyCharm：TensorRT
├── check_consistency.py           # PyCharm：一致性检查
├── inspect_checkpoint.py          # PyCharm：checkpoint 检查
├── core/                          # 配置、计时、checkpoint、run 管理
├── backends/                      # PyTorch/TensorRT 测速实现
├── checks/                        # 一致性检查和误差指标
├── adapters/                      # 自定义模型 adapter 模板
├── legacy/                        # 原始上传脚本备份
└── tests/                         # 离线回归测试
```

根目录只保留配置文件、说明文档和可直接运行的入口；实现代码统一放在上述功能子目录，
避免同一份代码出现多份副本。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `benchmark_config.yaml` | 唯一建议日常修改的配置文件 |
| `core/vision_benchmark_config.py` | 读取 YAML、解析项目根目录和相对路径 |
| `core/vision_run_manager.py` | 为每次入口调用创建 `run1`、`run2` 等独立目录 |
| `core/vision_benchmark_common.py` | 计时、统计、设备、精度和 adapter 核心 |
| `core/vision_runtime.py` | PyCharm/conda 动态库路径兼容处理 |
| `core/vision_checkpoint_inspector.py` | 检查 checkpoint 是否包含可恢复的模型架构 |
| `benchmark_pytorch.py` | PyCharm 直接运行的 PyTorch 入口 |
| `benchmark_tensorrt.py` | PyCharm 直接运行的 TensorRT 入口 |
| `run_all.py` | 按配置依次运行 PyTorch 和 TensorRT，并合并汇总 CSV |
| `inspect_checkpoint.py` | 单独检查 `.pt` 是否包含可恢复架构 |
| `check_consistency.py` | PyCharm 独立运行 PyTorch/TensorRT 输出一致性检查 |
| `checks/vision_consistency.py` | 共用输入、I/O 对齐和 CSV/JSON 报告 |
| `checks/vision_consistency_metrics.py` | 张量误差与端到端检测框匹配 |
| `adapters/vision_adapter_template.py` | 非内置模型的 adapter 模板 |
| `backends/tensorrt_engine_builder.py` | 使用 TensorRT Python API 或 trtexec 构建 engine |
| `backends/pytorch_vision_speed_benchmark_v2.py` | PyTorch 后端实现 |
| `backends/pytorch_vision_tensorrt_benchmark_v2.py` | TensorRT 后端实现 |
| `tests/test_consistency.py` | 不依赖 GPU 的一致性与运行目录回归测试 |
| `legacy/yolo26_tensorrt_benchmark(1).py` | 原始上传的 TensorRT 脚本备份（未改动） |
| `legacy/yolo_detection_speed_benchmark(1).py` | 原始上传的 PyTorch 脚本备份（未改动） |

## 1. 第一次使用：只改 YAML

打开 `benchmark_config.yaml`，最少修改以下内容：

```yaml
project:
  root: /home/你的用户名/PycharmProjects/你的项目
  ultralytics_repo: ultralytics

models:
  - name: yolo26s
    weights: GoodModel/lycheeflo.pt
    adapter: ultralytics
    task: detect
    enabled: true

benchmark:
  image: test.jpg
  device: auto
  input_size: [832, 832]
  batch_size: 1
  warmup: 30
  repeats: 100

run:
  enabled: true
  root: runs-profile
  name: auto

pytorch:
  enabled: true
  precisions: [fp32, fp16]
  output: runs-profile/vision_speed_v2.csv

tensorrt:
  enabled: true
  builder: python          # TensorRT Python API；不需要 trtexec
  precision: fp16
  onnx_opset: 18
  rebuild_engine: false
  output: runs-profile/vision_tensorrt_v2.csv
```

路径规则：

- `project.root` 是项目根目录，可以写绝对路径。
- `weights`、`image` 等输入路径的相对路径相对于 `project.root`。
- 启用 `run.enabled: true` 后，每次入口调用会在 `run.root` 下创建下一个
  `run1`、`run2`、`run3`……；CSV/JSON 报告和汇总默认放到该目录。
  配置中原有的 `runs-profile/...` 报告写法会自动转换成当前 run 目录下的相对路径。
- ONNX 和 TensorRT engine 仍保留在 `tensorrt.onnx_dir`、`tensorrt.engine_dir`
  公共目录，便于 `rebuild_engine: false` 复用，不会因分 run 而重复构建。
- 输出路径写成 `run.root` 之外的绝对路径时会保留原位置；这可用于固定外部日志目录。
- `enabled: false` 的模型不会加入测速。
- 一个 YAML 可以配置多个模型，每个模型可以使用不同 adapter 和 task。

### 每次运行的目录

默认配置已经启用自动分目录：

```yaml
run:
  enabled: true
  root: runs-profile
  name: auto
```

例如连续执行三次 `run_all.py` 后，目录结构类似：

```text
runs-profile/
├── run1/
│   ├── vision_speed_v2.csv
│   ├── vision_tensorrt_v2.csv
│   ├── vision_benchmark_summary.csv
│   ├── consistency.csv
│   ├── consistency.json
│   └── run_info.json
├── run2/
└── run3/

runs-profile/onnx/       # 公共 ONNX 缓存
runs-profile/engines/    # 公共 TensorRT engine 缓存
```

`run_all.py` 会把同一次调用的 PyTorch、TensorRT 和一致性检查放在同一个
`runN` 中。单独运行 `benchmark_pytorch.py`、`benchmark_tensorrt.py` 或
`check_consistency.py` 时，也会各自申请下一个 `runN`。目录通过原子创建，
即使两个终端同时启动也不会使用同一个编号。

如果需要复用已经创建的目录（例如只补跑某个阶段），在 PyCharm 的
Parameters 中填写：

```text
--run-dir runs-profile/run2
```

也可以在命令行使用同样的参数。`run2` 已存在时会在该目录继续写入，因此只
建议把它用于明确的补跑；普通重复运行请省略参数，让程序自动创建新目录。

如需恢复旧版“固定输出路径”行为，可设置：

```yaml
run:
  enabled: false
```

启动测速前，默认会自动检查每个 `.pt/.pth`：它会报告文件是完整模型对象、
包含 `model/ema` 的训练 checkpoint，还是只有 `state_dict`。同时会用当前
adapter 在 CPU 上尝试加载一次，以便尽早发现自定义模块或权重不匹配问题。
可以在 `benchmark` 中调整：

```yaml
benchmark:
  inspect_checkpoint: true
  verify_model_load: true
  fail_on_checkpoint_inspection: false
```

`fail_on_checkpoint_inspection: true` 会在检查失败时立即停止；默认 `false`
会先打印诊断信息，再让测速阶段继续执行。

## 2. 在 PyCharm 中运行

建议把工作目录设为本项目目录，然后直接右键运行：

1. `run_all.py`：一键运行两个后端，并在当前 `runN` 中生成汇总 CSV。
2. `benchmark_pytorch.py`：只运行 PyTorch。
3. `benchmark_tensorrt.py`：只运行 TensorRT。
4. `check_consistency.py`：只检查已有 engine 的输出一致性，不重新测速、不构建 engine。

运行结束后，终端会打印本次目录，例如：

```text
本次运行目录：/home/你的项目/runs-profile/run1
```

如果只设置了 `pytorch.enabled: true`，该目录中只会出现 PyTorch 报告；被跳过
的阶段不会生成空的 CSV。

`pytorch.enabled`、`tensorrt.enabled` 以及 `run_all.run_*` 用于控制一键入口；
单独运行某个入口文件时，表示你明确要求执行该后端。

不需要在 Run/Debug Configuration 中填写参数。若有另一份 YAML，也可以在
Parameters 中填写：

```text
--config /path/to/another_config.yaml
```

## 3. adapter 怎么选

`adapter` 决定“如何加载权重、如何构造输入、如何调用模型”：

- `ultralytics`：Ultralytics YOLO、RT-DETR 等模型。
- `checkpoint`：权重文件本身保存的是完整 `torch.nn.Module`。
- `adapters/vision_adapter_template.py` 或其他 `adapter.py`：自定义 PyTorch 模型。

也可以只运行检查而不测速：

```bash
python inspect_checkpoint.py --config benchmark_config.yaml
```

如果只想查看 checkpoint 外层、不加载模型（适合先排查自定义模块缺失），使用：

```bash
python inspect_checkpoint.py --config benchmark_config.yaml --raw-only
```

检查结果中：

- `full_module`：文件本身是完整 `nn.Module`。
- `full_checkpoint`：字典中包含 `model` 或 `ema` 完整模型对象。
- `state_dict_only`：只有参数，必须在 adapter 中先实例化网络。
- `adapter_load_status=succeeded`：当前源码环境和 adapter 已经可以恢复它。

例如同时测速一个 Ultralytics 模型和一个自定义分类模型：

```yaml
models:
  - name: yolo26s
    weights: GoodModel/lycheeflo.pt
    adapter: ultralytics
    task: detect
    enabled: true
  - name: resnet50
    weights: weights/resnet50.pth
    adapter: adapters/resnet50_adapter.py
    task: classify
    enabled: true
```

## 4. 自定义模型 adapter

复制 `adapters/vision_adapter_template.py` 为自己的文件，例如
`adapters/resnet50_adapter.py`，至少实现：

```python
def build_model(weights, device):
    # 实例化网络，并 load_state_dict；返回 model
    ...

def make_inputs(batch_size, image_size, device, dtype, source=None):
    # 返回 InputBundle，或直接返回 model 的输入
    ...
```

常用可选函数：

```python
def prepare_model(model, device, precision, fuse=False): ...
def run(model, inputs): ...
def predict(model, source, config): ...
def postprocess(outputs): ...
def count_outputs(outputs): ...
def export_onnx(model, output_path, inputs, opset, dynamic=False): ...
```

说明：

- `make_inputs` 负责纯模型调用和 ONNX 导出的输入；分类、分割、检测、多输入
  模型都可以在这里返回自己的输入结构。
- `predict` 是可选的真实 pipeline 边界，只有配置了 `benchmark.image` 且
  `pytorch.test_pipeline: true` 时才会执行。
- state_dict 不能直接被默认 `checkpoint` adapter 恢复，必须在
  `build_model` 中先实例化网络再 `load_state_dict`。
- TensorRT 阶段需要模型能导出 ONNX；复杂输出或多输入模型建议自行实现
  `export_onnx`。

## 5. 命令行临时覆盖

YAML 是默认配置，但调试时仍可覆盖关键值：

```bash
python benchmark_pytorch.py \
  --model debug=/tmp/model.pt \
  --adapter checkpoint \
  --device cpu --input-size 640 --precision fp32
```

可重复的 `--model` 和 PyTorch 的 `--precision` 会覆盖 YAML 中对应列表。
`--config` 可指定另一份 YAML。

## 6. 依赖和 TensorRT 注意事项

基础环境至少需要 PyTorch、NumPy 和 PyYAML：

```bash
pip install torch numpy pyyaml
```

Ultralytics 模型还需要 `ultralytics`；使用图片 pipeline 通常需要
`opencv-python`。TensorRT 对比需要 TensorRT Python bindings；默认
`builder: python` 不需要额外安装 `trtexec`。如果选择 `builder: trtexec`，
还需要自行安装并在配置中填写 `trtexec` 路径。自动导出 ONNX 时建议安装
`onnx`、`onnxscript`。

TensorRT 11 使用强类型网络，FP16/BF16 的实际类型由 ONNX 模型中的 dtype
决定，不能依赖旧版 `--fp16` 或 `--bf16` 参数。脚本会在 TensorRT 11 中按
强类型规则构建；修改模型、输入尺寸、batch、精度或 TensorRT/GPU 环境后，
建议设置 `rebuild_engine: true` 重新构建。

如果 engine 已经存在且 `tensorrt.rebuild_engine: false`，脚本会复用它；改了
模型、输入尺寸、batch、精度或 TensorRT/GPU 环境后，建议设置为 `true` 重新构建。

### TensorRT engine 构建方式

默认配置使用 TensorRT Python API 构建 engine：

```yaml
tensorrt:
  builder: python
```

这要求当前环境能执行 `import tensorrt`，不要求安装 `trtexec`。也可以临时
选择外部命令：

```bash
python benchmark_tensorrt.py --builder trtexec --trtexec /path/to/trtexec
```

`builder: auto` 会优先使用可找到的 `trtexec`，找不到时自动回退到 Python API。
TensorRT 11 使用强类型网络，FP16/BF16 的实际类型由导出的 ONNX dtype 决定；
脚本不会给 TensorRT 11 传入已经移除的旧式精度参数。

## 7. 输出指标

- `model_call`：输入已经在设备上，只测模型调用，适合比较网络本身。
- `adapter_pipeline`：测 adapter 的 `predict` 边界，是否包含预处理和后处理由
  adapter 决定。
- `engine_call`：固定输入的 TensorRT engine 调用，不包含读图、预处理和后处理。
- 推荐报告 `median_ms`、`p90_ms` 和 `fps_per_sample`；FPS 已按真实 batch 换算。

TensorRT engine 与 GPU、CUDA、TensorRT 版本相关，换部署机器后应重新构建和
测速。

## 8. 新增：PyTorch/TensorRT 结果一致性检查

### 怎么运行

先保证 `benchmark_tensorrt.py` 能成功生成当前模型的 ONNX 和 engine。
如果在 PyCharm 中出现 `CXXABI_1.3.15 not found`，入口会在导入 ONNX 前自动
把当前 Python/conda 环境的 `lib` 放到 `LD_LIBRARY_PATH` 最前面并重启当前
入口。这样通常不需要手工配置 PyCharm；如果当前环境没有 `libstdc++.so.6`，
仍需先安装 `libstdcxx-ng`/`libgcc-ng`。TensorRT 需要可用的 CUDA GPU。

在 PyCharm 直接右键运行 `check_consistency.py`，无需填写命令行参数。
也可以在 `run_all.py` 中一键执行：

```yaml
tensorrt:
  test_pytorch: false      # 避免重复测速；一致性检查不依赖这个开关
  test_pipeline: false     # 这是额外的 PyTorch pipeline，不是 TensorRT 图片流程

consistency:
  enabled: true
  input_mode: image
  source: null             # 使用 benchmark.image，也可填一张图片或图片目录
  max_images: 10
  reference_precision: same
  atol: 0.001
  rtol: 0.01
  acceptance_mode: auto      # auto / detection_only / all
  detection_mode: auto
  detection_conf: 0.25
  detection_iou: 0.95
  score_atol: 0.01
  output: runs-profile/consistency.csv
  json_output: runs-profile/consistency.json

run_all:
  run_consistency: true
```

这是需要合并到现有 YAML 的配置片段，不要在同一 YAML 中重复添加同名顶层键。
完整文件已经包含逐项中文注释；升级时保留你自己的 project/models 路径。

命令行运行：

```bash
python check_consistency.py
python check_consistency.py --config benchmark_config.yaml --source /path/to/images
```

独立检查不受 `consistency.enabled` 或 `run_all.run_*` 开关限制。
只有全部通过才退出码为 0；warning、failed、error、inconclusive 都返回 1。
`run_all.py` 会保留这些状态并继续写汇总，不把它们当成“成功”。

### 实际比较的是什么

1. 默认沿用 TensorRT 测速的权重、batch、尺寸、精度、ONNX 和 engine 路径。
   检查脚本只读取文件，不修改 `.pt`、ONNX 或 engine。换权重、修改模型源码或融合设置后，
   先用 `tensorrt.rebuild_engine: true` 重建，再检查；文件名相同不证明模型相同。
   报告包含本次权重/ONNX/engine 哈希，但没有替旧 engine 追溯构建来源。
2. 输入仅准备一次，分别克隆给两个后端，防止原地操作污染另一边。
   内置 Ultralytics **detect** 图片模式采用固定尺寸 letterbox、BGR→RGB、
   CHW、除以 255。单图只算一个案例，batch>1 时复制同图；目录按路径排序读取最多
   `max_images` 张不同图片。报告保存输入形状、dtype、哈希和真实来源。
3. PyTorch 直接调用 `adapter.forward_model(model)`，与默认 ONNX 导出边界一致。
   不调用 `.predict()`，避免其自动融合或后处理改变参考模型。
   默认 `reference_precision: same` 比较同精度转换；`fp32` 改用 PyTorch FP32
   参考，也观察低精度计算带来的差异。两者都使用已准备的同一输入数值，
   因此不是“FP32 独立预处理”的完整应用对比。
4. 按 ONNX 图中的输出名称逐项比较，不依赖 TensorRT 的枚举顺序。
   数量/名称对不上时明确报错；不丢弃额外输出、不广播不同形状。
   对浮点数要求 `abs(TRT-PyTorch) <= atol + rtol * abs(PyTorch)`；
   整数、布尔值严格相等。输出形状、非有限值、最大/平均绝对误差、RMSE、
   最大相对误差和满足容差的元素比例均写入报告。
   接近零的参考值会放大相对误差，需结合绝对误差看，不能只看 max_rel_error。
5. `acceptance_mode: auto` 在存在任务级检测输出时采用检测框匹配作为最终验收，
   但仍保留所有原始张量的诊断结果；`detection_only` 强制采用该规则，`all`
   要求所有输出逐元素通过。对于没有检测框匹配的分类、分割、姿态或自定义模型，
   `auto` 会自动回退为 `all`；显式 `detection_only` 也会安全回退并在报告中说明。
6. `detection_mode: auto` 仅对 Ultralytics 的端到端 detect（例如 YOLO26）额外
   比较第一个 `[B,N,6]` 输出，格式必须为 xyxy/conf/class、输入图片坐标。
   在 `detection_conf` 以上的框中，按类别相同、IoU≥`detection_iou`、
   置信度差≤`score_atol` 做一对一最大数量匹配，允许框的顺序改变。
   两边有任何未匹配框都失败；每张图最多匹配 500 个阈值以上框。
   这不是 NMS；非端到端 YOLO、分割、姿态等默认只做张量比较，不猜它们的后处理。

`image` 模式只对内置 Ultralytics detect 自动提供预处理；其他模型需要自己的
adapter 钩子。`input_mode: adapter` 使用 adapter 示例输入，可能是随机张量，
通过只表示这个示例没发现异常，不能代表真实图像。请用有代表性的真实图片检查。

### 如何看结论

| 状态 | 含义 |
| --- | --- |
| `passed` | 当前验收模式通过；`all` 要求所有张量通过，检测模式要求任务级输出通过 |
| `warning` | `all` 模式下原始张量不满足容差但检测框通过；检测模式通常不会把这种诊断差异作为失败 |
| `failed` | 数值超差、形状不符、NaN/Inf、类别/框/置信度不匹配等 |
| `inconclusive` | 空张量或阈值以上无检测框，证据不足，不当作验收通过 |
| `error` | 缺依赖/文件、输入输出定义不一致、无法执行等；不是实测数值不一致 |
| `skipped` | 一键入口中本次 TensorRT 阶段失败，因此未检查遗留 engine |

`consistency.csv` 按 `record_type` 区分 `case`、`tensor`、`detections`；
JSON 还包括参数、文件哈希、逐样本框匹配列表及完整状态。
`warning` 不会自动提高阈值或被当成 `passed`。张量/框两层结论分别保留，
框通过不会掩盖 NaN/Inf 等硬错误。两边都没框可能只是阈值高或图像中没目标，
不意味着模型错误，但不能据此认定检测一致性充分。

这些容差是可调整的初始诊断值，不是各类网络的通用验收标准。
一致性通过 **不等于 mAP、Recall、Precision 不变**；它不加载 GT，不做数据集精度验证。
它也不测相机采集/解码、完整 TensorRT 图片 pipeline、温度/长时稳定性。

### 自定义模型如何接入

普通单/多张量输出使用现有 adapter 即可，默认与导出时的 Tensor 扁平化顺序一致。
如果需要真实图片预处理，或自定义导出改变了输出结构，可在 adapter 中增加：

```python
def make_validation_inputs(source, config):
    # 自己按训练/部署要求做预处理，返回 InputBundle。
    # image 模式的 source 是 Path；adapter 模式为 None。
    # config.device / precision / batch_size / image_size 均可使用。
    ...

def prepare_validation_model(model, config):
    # 可选：使当前模型的 forward 与你的自定义 ONNX 导出模式一致。
    # 可原地准备并返回 None，或返回准备好的 model。
    ...

def validation_forward(model, inputs):
    # 可选：实现与自定义导出 wrapper 完全相同的 PyTorch 调用。
    return model(*inputs.args, **inputs.kwargs)

def validation_outputs(outputs, output_names):
    # 可选：返回 {每一个 ONNX 输出名称: 对应的 torch.Tensor}，不能遗漏输出。
    ...
```

不要原样复制包含 `...` 的钩子；只实现确实需要的函数。
模板文件也有相应说明。多输入如果 ONNX 顺序与 adapter Tensor 顺序不同，
在 `consistency.input_names` 按 adapter 顺序填全 ONNX 输入名称。
如使用自定义导出的路径，可在对应 `models` 条目中设置 `onnx` 和 `engine`
供检查器读取；测速脚本仍使用其原来的自动命名路径。
第三方输出 schema、带额外头信息的 engine、HOST/非 LINEAR I/O、数据相关动态
输出等不是默认 runner 的通用承诺，不支持时明确报错，不会强行判定通过。

### 汇总与测试说明

- 新配置已把 `tensorrt.test_pytorch/test_pipeline` 设为 false，避免一键运行重复测速。
  正确性检查仍会各做必要的推理，但不执行测速的 30 次预热/100 次计时循环。
- 速度汇总只使用本次返回的明细，不在运行失败后拼入磁盘上的旧 CSV。
  `record_type: measurement` 是测速记录，`stage_summary` 是阶段状态，不是一次测量。
  TensorRT 明细中有失败时，阶段也会显示失败，并跳过本轮一致性检查。
- 无 GPU 单元测试：`python -m unittest discover -s tests -v`。
  覆盖输出名称、形状/空值/非有限值、容差、检测框重排/漏框/类别变化和失败状态传播。
  这些单元测试不替代目标机器的真实 PyTorch/TensorRT 执行验证。
- `.pt` 中的完整模型反序列化可能执行 Python 代码，只使用你信任的权重文件。

实现参考：
- PyTorch 容差定义：https://docs.pytorch.org/docs/stable/generated/torch.allclose.html
- TensorRT 精度注意事项：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html
- Ultralytics 检测头输出：https://docs.ultralytics.com/reference/nn/modules/head/
