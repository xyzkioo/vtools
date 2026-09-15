# 第三轮修复后复检

修复状态：上述四项已在后续修复中处理，验证范围见 `vtools-fix-3-2026-09-15.md`；下文保留修复前记录。

日期：2026-09-15。本轮检查上轮修复及相关调用路径，未修改业务代码；以下清单不代表穷尽全仓问题。

## 确认的问题

1. **P1：图片 ID 仍依赖输入集合。** `model_visualization/run_visualization.py:253` 与 `model_diagnostics/diagnostics/engine.py:468` 分别按各自输入集合决定是否追加扩展名。实际函数复现：单独可视化 `/dataset/images/val/a.jpg` 得到 `images/val/a`，诊断数据集同时包含 `a.jpg`、`a.png` 时同一图片得到 `images/val/a#jpg`。`max_images` 截断也会触发这一问题。跨工具导入预测可能产生虚假 FN/FP。应使用不依赖当前选择集合的统一标识，并验证单图/子集预测加载到完整 GT 的行为。

2. **P2：非 images 目录布局仍不兼容。** `model_visualization/run_visualization.py:124` 通过目录名猜测数据集根，而诊断使用 data.yaml 的根。实际函数复现：`/dataset/photos/val/a.jpg` 单图可视化 ID 为 `a`，诊断为 `photos/val/a`。应明确传入数据集根或提供来源路径映射，避免将 images 命名约定当成完整协议。

3. **P2：跨图形页修改后到 YAML 页保存仍丢修改。** `vtools_ui/app.py:648` 按常用页、全部页顺序合并，合并时又用已经修改的 data 判断 YAML 是否变化。抽取实际 `_save/_get/_set`，用控件替身复现：初值 cpu，常用页改 cuda:0，全部页后改 auto，进入未编辑的 YAML 页保存，落盘为 cuda:0。没有记录编辑顺序，也没有页间同步。应统一编辑状态并验证三页之间的切换组合；此次为实际保存方法的隔离验证，非真实 Qt 窗口测试。

4. **P2：手动修改配置路径后仍使用旧模块缓存。** `vtools_ui/app.py:881` 只在初始化、选择文件和配置编辑器保存后刷新 `_configured_modules`；可编辑的 config_edit 未连接路径修改信号。`_run` 读取新路径却使用旧模块状态，用户随后调整按钮时将旧配置中的隐藏模块通过 `--only` 传给新配置，可能关闭新配置的 predict.generate 或运行错误模块。静态确认。应在配置路径变化时重新加载状态，并在运行前校验缓存对应路径。

## 验证范围

使用 `/usr/bin/python3`，设置 `PYTHONPATH=.:speed_test:model_diagnostics` 和 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，运行三个子项目测试：56 passed、2 failed。两项失败均在 import torch 时终止，当前解释器未安装 PyTorch。`git diff --check` 通过。上述前三项有实际函数隔离复现，第四项为静态调用路径确认。未执行真实训练、Qt、GPU、TensorRT 或 K230 端到端验收。

上轮修复记录中“统一单图/目录数据集图片 ID”和配置编辑已修复的表述覆盖范围过大；现有测试未覆盖上述输入子集和跨页组合，不能据测试通过认定全部问题关闭。
