# vtools 冗余与失效代码补充审查（2026-11-01）

基准仍为 HEAD `cc39d05`、v0.4.7。上一轮重点检查功能缺陷，本轮补查重复实现、不可达分支、无消费者配置、仓库内无调用的封装及未使用导入，继续应用 `review-agent` 的证据优先规则。

扫描了 117 个 Python 文件，其中 87 个不在 `tests/` 目录内；进行 AST 语法、函数体重复和符号引用候选检查，再用全仓检索、调用链及临时样例复核。前端检查了源码中的导入和旧配置分组引用，没有执行 TypeScript 编译。AST 和文本扫描均不是完整的动态调用图，也不能证明某个公开 API 没有外部使用者。

## 确认的问题

### S01 · [P2] 清理无消费者的旧 checkpoint 配置并落实失败策略

位置：[speed_test/benchmark_config.yaml](../../speed_test/benchmark_config.yaml) 第 74–80 行；[PyTorch 后端](../../speed_test/backends/pytorch_vision_speed_benchmark_v2.py) 第 300–315 行；[测速 README](../../speed_test/README.md) 第 215–226 行；[UI 字段说明](../../vtools_ui/webapp/config_fields.py) 第 57–59 行。

`benchmark.inspect_checkpoint`、`benchmark.verify_model_load` 已被 `modules.checkpoint.inspect/load_check` 替代，当前后端不读取旧字段，但默认配置和界面仍保留它们。`benchmark.fail_on_checkpoint_inspection` 也没有执行端消费者，README 却承诺设为 true 会立即终止。检查与模型信息同时开启时，失败的 inspection_rows 还会被舍弃，返回的模型信息行不带检查失败。

**验证：**对真实 CPU Identity 模型启用 `checkpoint.inspect`、`model.parameters` 和 `fail_on_checkpoint_inspection: true`，向 `inspect_entries()` 注入一个 `architecture_status=unknown` 的失败结果。后端继续返回 1 条模型信息行，失败行数量为 0。这里验证的是故障处理控制流，检查失败通过 mock 注入。

建议：检查开关以 `modules.*` 为唯一入口，同步清理旧 YAML 和 UI 提示；明确落实或移除公开的立即终止策略，最终结果须保留实际检查状态。

### S02 · [P2] 从诊断注册表移除不会执行的跨工具模块

位置：[model_diagnostics/diagnostics/modules.py](../../model_diagnostics/diagnostics/modules.py) 第 18–29、45–47 行。

诊断注册表仍包含 15 个测速、模型信息、导出、一致性及可视化模块 ID；它们在诊断引擎没有执行路径，却可通过未知模块校验。用户只选择这些 ID 时，可以得到成功退出和空诊断结果。`validation.ultralytics` 已对未实现功能显式报错，这些残留 ID 没有同等处理。

**实测：**临时 canonical GT/预测输入配合 `modules: {speed.pytorch_call: true}` 调用实际诊断引擎，返回 `exit=0`，`run_info.modules.speed.pytorch_call=True`，`summary.aggregate={}`，未执行任何测速。

建议：诊断入口仅接受自己拥有的模块 ID，使用现有未知模块错误明确拒绝这类配置，并同步对应模块说明。

### S03 · [P3] 删除 TensorRT 后端不可达的 PyTorch 测速分支

位置：[TensorRT 后端](../../speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py) 第 420–421、560–587 行；parser 第 107–109 行。

主流程已无条件将 `speed.pytorch_call` 和 `speed.pytorch_pipeline` 设为 false，后面仍根据这两个值维护 `run_pytorch/run_pipeline` 和整段测速逻辑。在此期间没有重新开启它们的赋值，分支不可达。`--skip-pytorch`、`--skip-pipeline` 因此也不改变执行计划，`--source` 仍被读取但其唯一消费分支不可达；相关测速导入也只服务于这段旧实现。

建议：删除不可达的测速块和失效选项，清理只为旧块存在的 source 处理及导入；PyTorch 测速继续由已有独立后端和 `run_all` 调度。保留 TensorRT 实际使用的模型信息、导出与构建路径。

### S04 · [P3] 共用模块选择的基础规则

位置：[speed_test/core/module_selection.py](../../speed_test/core/module_selection.py) 第 31 行；[model_compression/core/module_selection.py](../../model_compression/core/module_selection.py) 第 69 行；[诊断模块解析](../../model_diagnostics/diagnostics/modules.py) 第 77 行；[可视化覆盖解析](../../model_visualization/run_visualization.py) 第 48 行。

`--only/--enable/--disable` 的拆分、未知 ID、互斥、显式空集合及 allow-list 规则维护了四份。AST 对比确认，测速与压缩的解析函数体在忽略说明文档、错误文本和默认表名称后完全相同；诊断和可视化再分别实现同一基础政策。修改共同规则需要多处同步，已有多个子项目回归分别维护这些约定。

建议：仅共享无框架依赖的基础选择逻辑，默认模块表、领域标签及依赖校验继续保留在各 owner。两组额外的精确函数体重复见下方清单，不必仅为短函数重复立即增加抽象。

### S05 · [P3] 让层检查只枚举和写出一次 layers.csv

位置：[model_visualization/run_visualization.py](../../model_visualization/run_visualization.py) 第 226–228、256–258 行。

`run()` 先调用 `adapter.list_layers()` 并写 `layers.csv`，随后在 `list_layers` 模式调用 `_run_list_layers()`，再次枚举并覆盖同一文件。第一次枚举和写入完全冗余。

**验证：**以假 adapter 和 mock 文件写入运行实际 `list_layers` 控制流，同一个 `layers.csv` 的 `write_csv()` 调用次数为 2。

建议：让文件写入只归一个位置负责；调用已有 helper 时不再提前写一遍。

## 清理候选与重复清单

以下不计入上面 5 项问题的严重度统计；它们用于规划后续清理，当前尚未删除。

### 仓库内未找到调用的 5 个函数

| 位置 | 符号 | 复核结果 |
| --- | --- | --- |
| [model_compression/modules/evaluation.py](../../model_compression/modules/evaluation.py) 第 10、15 行 | `evaluate`、`model_metrics` | 文件无执行端导入，只有 2 个转发封装；实际入口直接调用 model_runtime |
| [speed_test/core/vision_benchmark_common.py](../../speed_test/core/vision_benchmark_common.py) 第 752 行 | `parse_model_specs` | 当前入口统一使用 `get_model_entries`，未找到调用 |
| [speed_test/core/vision_benchmark_config.py](../../speed_test/core/vision_benchmark_config.py) 第 265 行 | `config_path_value` | 未找到调用、测试或文档引用 |
| [env_test/check_install.py](../../env_test/check_install.py) 第 258 行 | `format_versions` | 未找到调用、测试或文档引用 |

这些函数合计覆盖 26 行函数定义与函数体。前三个模块中的公开名称可能存在仓库外使用者；清理时检查公开接口约定，再决定删除或保留。没有把 adapter 模板的动态钩子按“无直接调用”判为死代码。

### 未使用导入

AST 候选经检索复核后，确认有 13 个 Python 导入名没有被当前文件使用：

- `env_test/check_install.py`：`importlib.metadata`、`Iterable`。
- `model_compression/core/model_runtime.py`：`Iterable`、`Optional`。
- `model_diagnostics/diagnostics/adapter_runtime.py`：`Sequence`。
- `model_visualization/adapters/ultralytics.py`：`sys`。
- `model_visualization/core/common.py`：`os`、`re`。
- `model_visualization/run_visualization.py`：`_cam_outputs()` 内的 `cv2`。
- `others/video_frame_extractor/video_frame_extractor.py`：`Any`。
- `speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py`：`time`、`numpy as np`。
- `vtools_ui/webapp/server.py`：`ROOT`。

前端 [App.tsx](../../vtools_ui/webapp/frontend/src/App.tsx) 的 `LoaderCircle` 也只出现在导入中。

已排除 `packaging/build_desktop.py` 中的依赖探测导入，以及 `detection_distillation.py` 用于检查原生接口是否存在的 `DistillationModel` 导入；它们的用途是导入本身。

### 两组较长函数体完全重复

去除 docstring 和源码位置后，AST 识别到以下函数体完全相同：

- `model_compression/core/config.py::_resolve` 与 `speed_test/core/vision_benchmark_config.py::_resolve_path`。
- 两个测速后端各自的 `parse_input_size`。

这只是精确函数体重复结果，不是全仓重复率。两个 helper 都较短；应结合后续修改频率与调用边界决定复用，不作为独立功能缺陷。

### 旧模式配置残留

`model_diagnostics/config/config.yaml` 第 101 行的 `mode_b.task` 没有后端消费者；UI 的 `config_fields.py` 第 138 行和 `App.tsx` 第 150 行仍保留字段说明/分组。当前真正的模式配置是 `predictions_file`、`ultralytics_model`、`custom_adapter`。建议同步移除旧字段及其展示元数据。

## 保留边界与验证

- `model_diagnostics/core/vision_benchmark_common.py` 的兼容类型、两类 adapter 模板、稳定重导出模块，以及拒绝不支持操作的 guard，都有明确接口或运行时目的，未因名字相似或缺少直接调用就判为可删除。
- 没有合并业务代码或新增兼容层。本轮新增/更新审查与交接文档；静态扫描和临时 CPU/假 adapter/故障注入验证不需要真实数据或硬件。
- 本轮未重跑上一轮已经通过的 216 项回归，业务代码未变；本轮新增的扫描、控制流复现和 Markdown 本地链接检查用于验证本补充报告。
