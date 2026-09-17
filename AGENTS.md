# AGENTS.md

本文件是 vtools 仓库的 AI coding agent 和协作者编辑指南。它适用于仓库根目录及所有没有更具体 `AGENTS.md` 的子目录。

## 项目范围

vtools 是一个以 Python 和 YAML 为核心的视觉模型工具集，包含检测诊断、模型可视化、PyTorch/TensorRT 测速、模型压缩、K230 转换、数据文件处理和 PySide6 桌面工作台。

Ultralytics Fork 是 vtools 同级的独立仓库，不属于 vtools 的源码提交范围。修改 Fork 时遵循其仓库内的约定；不要把 vtools 的工具逻辑复制进上游包内部。vtools 通过 `project.ultralytics_repo`、`VTOOLS_ULTRALYTICS_REPO` 或同级目录发现它。

## 核心原则

1. 先搜索再修改。优先复用现有入口、配置解析器、运行目录管理器和 adapter；不要创建并行实现。
2. 在行为 owner 处修复问题。诊断问题放在 `model_diagnostics`，测速问题放在 `speed_test`，可视化问题放在 `model_visualization`，不要在 UI 层掩盖后端错误。
3. 保持改动最小。删除或修改错误路径优先于新增兼容层、重复 helper、猜测性参数和无调用代码。
4. 配置、代码和文档必须同步。新增或改名入口、模块 ID、参数、输出文件时，同时更新对应 YAML、README、根目录 README 和必要的 Spec。
5. 运行失败必须可见。不要把导入错误、配置错误、输出写入失败或子任务失败静默转换成成功；返回码、summary 和报告应反映真实状态。
6. 输出文件要可复查。运行目录使用唯一目录；写 JSON、CSV、图片或模型时避免半成品覆盖已有结果，失败应清理临时文件。
7. 不修改用户输入和权重。除非用户明确要求，工具只读取输入文件；生成物写入对应的 `runs/`、`outputs/` 或用户指定目录。
8. 新增 YOLO 功能前，先检查外部 Ultralytics Fork 的公共 API 和对应文档。常规训练、验证指标与逐图 TP/FP/FN、普通 CAM、标准格式导出和整模型 benchmark 优先调用 Ultralytics；vtools 只实现错误原因、指定候选与层的追踪、自定义模型、跨后端一致性及 K230 等额外需求。确需独立实现时，在相应 README 说明原生接口不能满足的具体输出或计时边界。

## 目录与真实入口

所有命令默认从仓库根目录执行。

| 功能 | 主入口 | 配置或说明 |
| --- | --- | --- |
| 桌面工作台 | `python -m vtools_ui` | `vtools_ui/README.md` |
| 统一命令行 | `python run_tools.py --tool ...` | `config/tools.yaml` |
| 检测诊断 | `python model_diagnostics/run_model_diagnostics.py` | `model_diagnostics/config/config.yaml` |
| 模型可视化 | `python model_visualization/run_visualization.py` | `model_visualization/config/config.yaml` |
| 一键测速 | `python speed_test/run_all.py` | `speed_test/benchmark_config.yaml` |
| PyTorch/TensorRT 单项测速 | `speed_test/benchmark_pytorch.py`、`speed_test/benchmark_tensorrt.py` | 同上 |
| 一致性检查 | `python speed_test/check_consistency.py` | 同上 |
| checkpoint 检查 | `python speed_test/inspect_checkpoint.py` | 同上 |
| 模型压缩 | `python model_compression/run_model_compression.py` | `model_compression/config/config.yaml` |
| 环境检验 | `python env_test/check_install.py` | `env_test/README.md` |
| 图片/视频/文件名工具 | `others/image_resize/image_resize.py`、`others/video_frame_extractor/video_frame_extractor.py`、`others/filename_transform/image_filename_converter.py` | 各自 README |
| 数据和 K230 转换 | `transform_tools/ndjson_to_yolo.py`、`py2kmodel.py`、`PY2KM_validate.py` | `transform_tools/REQUIREMENT.md` |

使用入口前先运行 `python <入口> --help`（桌面 UI 除外）。不要假设 `run_tools.py` 的工具名；当前有效值以其 parser 为准：`diagnostics`、`pytorch`、`tensorrt`、`consistency`、`visualization`、`compression`。

## 配置与路径

- 相对路径必须说明相对于哪个 `project.root` 或配置文件目录解析。
- `--config`、`--weights`、`--source`、`--data` 等 CLI 覆盖项必须进入实际运行状态和可复现元数据。
- 模块选择使用各子项目已有的 `--only`、`--enable`、`--disable` 解析器；显式空集合表示关闭，不要偷偷恢复默认模块。
- 运行产物放在自动创建的 `runN` 目录；测试不得依赖仓库中已有的历史 run。
- 生成的 runs、缓存、临时会话和 UI 历史不属于源码提交内容。用户明确要求清理时，只删除已确认的生成路径，不删除权重、数据集或源码。

## 代码约定

- Python 代码使用类型标注、明确异常和 `pathlib.Path`；跨平台路径不要手拼字符串。
- 可选依赖延迟导入，让 `--help` 和无相关依赖的基础功能仍可用。
- adapter 负责模型差异，公共引擎负责配置、计时、报告和生命周期；不要在公共引擎中加入模型特例分支。
- 新增输出字段时保持 JSON/CSV 可序列化，并同步更新报告说明和 README。
- 不要提交真实数据集、私有权重、TensorRT engine、运行报告或本地环境路径。

## 验证要求

修改后至少运行与改动相关的检查，并在交付时报告未执行的硬件路径。

```bash
# 全仓语法和空白检查
python -m compileall -q .
git diff --check

# 通用回归（已安装 PyTorch/PySide6 时）
QT_QPA_PLATFORM=offscreen PYTHONPATH=.:speed_test:model_diagnostics \
  python -m unittest discover -s tests -q

# 子项目回归
PYTHONPATH=. python -m unittest discover -s model_compression/tests -q
PYTHONPATH=.:speed_test python -m unittest discover -s speed_test/tests -q
PYTHONPATH=.:model_diagnostics python -m unittest discover -s model_diagnostics/tests -q
```

没有 GPU、TensorRT、nncase 或真实权重时，运行无硬件单元测试、错误路径和 `--help` 检查，并明确说明端到端路径未执行。测试输出中的“跳过”必须与“成功”区分。

## 文档更新

README 中的命令必须从仓库根目录可定位到真实文件；入口、默认配置、模块名称、输出路径和依赖变化要同步更新。修改公开 Python API 或配置字段时，检查 `docs/` 下的 Spec 和示例。Markdown 本地链接应在交付前检查，避免引用已删除的 run 或临时文件。

## 交付检查

交付前检查完整 diff，而不是只看新增文件；确认没有调试输出、死代码、未使用导入、私有绝对路径、秘密、临时产物或误删用户文件。总结改动、验证命令、实际结果和仍受环境限制的部分。
