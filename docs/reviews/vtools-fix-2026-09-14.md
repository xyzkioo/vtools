# vtools 修复记录

日期：2026-09-14

本记录对应 `vtools-recheck-2026-09-14.md` 中的 R01–R21。修复覆盖运行逻辑、产物保护、模块选择、可视化/诊断一致性、UI 配置保存、转换脚本和文档。

## 已修复

- R01：注册表在锁内读取最新文件并合并实例增量；分支嵌套字段按字段合并，同一运行 ID 更新而不重复；Unix 使用 `fcntl`，Windows 使用 `msvcrt`，同一字段并发推进会报冲突。
- R02/R03：压缩运行开始时只校验一次显式起点；后续模块跟随当前 head；检测导出始终从当前 head 加载；可视化与诊断使用统一相对路径 ID，输出目录增加哈希并处理重复 ID。
- R04：蒸馏目录和导出 manifest 均使用不会覆盖旧结果的路径；仅有 `student_factory` 时允许随机初始化，不登记虚假的学生初始权重。
- R05/R19：checkpoint-only 返回逐模型状态；未知架构、加载失败、warning/inconclusive 均不会被入口当成成功。
- R06–R08：factory-only 学生模型可独立构建；诊断推理模块服从最终 `--only/--enable/--disable`；未知 disable 立即报错。
- R09–R12：图形配置页按编辑值合并，YAML 高级页也会保留图形页已改字段；模块按钮从 YAML 初始化，只有用户改动时才发送 `--only`；停止任务会清理已发现子进程；批量改名在未勾选跳过确认时由 UI 先确认。
- R13–R15：CAM 校验 selection、class_id、index、max_targets；最终检测只选择真实保留且类别匹配的候选；raw 类别选择使用类别分数而非 argmax 类别过滤；空/非法 modules 配置拒绝或按 allow-list 处理。
- R16–R18：Kmodel 比较拒绝空输出、处理零范数并以 float64 计算；ONNX 输入支持 double/float16/整数声明；差图渲染优先使用 `--images-dir`。
- R20：更新压缩、模块化 Spec、NDJSON、抽帧、环境检查和文件改名文档；修正质量门禁配置路径和当前 `--run-dir` 语义。
- R21：版本记录改为列出当前环境的验证边界，不再声称未安装依赖的硬件流程已经验收。

## 验证结果

- `python -m compileall -q ...`：通过。
- `git diff --check`：通过。
- `PYTHONPATH=speed_test pytest -q speed_test/tests`：43 passed。
- `pytest -q model_compression/tests/test_module_selection.py model_compression/tests/test_structured_config.py`：6 passed；`PYTHONPATH=model_diagnostics pytest -q model_diagnostics/tests`：5 passed。
- 额外回归：注册表并发字段合并/冲突、运行 ID 去重、模块未知参数、图片 ID/哈希碰撞、诊断入口显式 argv、NDJSON CLI 帮助。

> 后续更新：这条环境结论只适用于当时使用的系统 Python。`yolo` Conda 环境已确认包含 PyTorch、PySide6、Ultralytics 和 nncase，并已补跑 PyTorch 与真实 Qt 控件测试；完整 GPU TensorRT 和 K230 板端流程仍需对应硬件。
