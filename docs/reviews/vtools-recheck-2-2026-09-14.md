# 第二次修复后复检

> 2026-09-15 更新：本文列出的 1–9 项已按后续修复处理，最新验证边界见 `vtools-fix-2-2026-09-15.md`。以下内容保留为修复前复现记录。

本轮只检查，没有修改业务代码。上轮“R01–R21 已完成修复”的结论仍不成立。以下为本轮确认的问题，不代表已经穷尽全仓所有问题。

## 已复现

1. **P1：一致性通过却返回失败。** `speed_test/run_all.py:198` 只接受 succeeded/skipped；一致性后端返回 passed。用实际 main 和返回 passed 的后端替身验证，退出码为 1。应统一成功状态映射，补通过/失败/证据不足三个用例。
2. **P1：新分支串行检测压缩仍被中断。** `model_compression/core/detection.py:193` 要求起始 head 非空才忽略旧 weights；新分支起始 head 为 None，剪枝推进后导出仍拒绝原始权重。用临时注册表构造同一运行的 baseline→pruned，调用实际 export 分派，在加载模型前即复现 ValueError。应区分“已初始化运行上下文”与“起始 head 非空”。
3. **P2：YAML 高级页的新编辑被旧图形值覆盖。** `vtools_ui/app.py:645` 按固定顺序覆盖 YAML。抽取实际保存方法、用控件替身复现：初值 cpu，图形改 cuda:0，随后 YAML 改 auto，最终保存 cuda:0。应同步三页数据或记录字段修改顺序；本轮不是实际 Qt 窗口测试。
4. **P2：FP16 归一化仍传错类型。** `transform_tools/PY2KM_validate.py:52` 只有不归一化时选择 float16，默认 normalize=True 最终传 float32。执行实际 onnx_infer 函数并使用声明 tensor(float16) 的会话替身，观察到 float32。应先确定模型 dtype，归一化后转换回该 dtype。
5. **P2：checkpoint CSV 丢失检查状态。** `speed_test/backends/pytorch_vision_speed_benchmark_v2.py:223` 固定测速列且 extrasaction=ignore。实际 save_rows 接收 failed/error/architecture_status 后，文件只留下模型名，错误字段全部丢失。返回值已修复，但落盘报告未修复。应为检查结果保留专用字段。

## 静态确认

6. **P2：UI 修改模块会丢失隐藏模块。** `vtools_ui/app.py:1069` 从少量常用按钮拼出 --only；诊断按钮中没有 predict.generate，B/C 调整按钮后会关闭预测生成并报错。测速分支仍无条件发送 --only，导致 YAML 中 checkpoint/consistency.tensor 等未展示模块消失。应在完整 YAML 模块集合上应用按钮增量。
7. **P2：旧 models 配置仍可绕过推理关闭。** `model_diagnostics/run_model_diagnostics.py:323` 仅检查 mode=B/C；legacy models 缺预测时，即使最终 predict.generate=False 仍进入 _generate_predictions。应对所有需要生成预测的路径统一校验。
8. **P2：单图输入的 ID 协议仍不一致。** `model_visualization/run_visualization.py:253` 单图使用 stem，而诊断用数据集根相对路径。例如 /dataset/images/val/a.jpg 两边为 a 与 images/val/a。重复 stem 的 #2 还依赖输入集合/顺序。应使用与输入选择无关的稳定标识，并验证预测加载匹配。
9. **P2：文档和完成记录仍与实现冲突。** 模块化 Spec 第21行称单次读取，第219行附近又承认 B/C 读取两次；第284行称所有 --run-dir 必须不存在，但测速和压缩运行管理器仍允许复用。修复记录也宣称上述未关闭项已经修复。应按入口记录实际行为，并逐项标注验证范围。

## 验证

使用 /usr/bin/python3，设置 PYTHONPATH=.:speed_test:model_diagnostics，关闭外部 pytest 插件后运行三个子项目现有测试：56 passed，2 failed；两项失败均为未安装 torch。当前默认 Conda python 没有 pytest，不能把一个解释器的依赖状况推广到所有环境。

git diff --check 通过。额外执行了上述5个针对实际函数的隔离复现；它们并未被现有单元测试覆盖。未执行真实 GPU、训练、Qt 窗口或 K230 验收。
