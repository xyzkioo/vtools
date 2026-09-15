# 第四轮复检

本轮复核上一轮修复及关联调用路径，未修改业务代码。确认以下两项问题；不代表已穷尽全仓。

## P1：单图 canonical 文件仍使用输出文件名作为图片 ID

位置：model_visualization/run_visualization.py:283、model_visualization/adapters/ultralytics.py:422、468。

主流程传给 trace_stage 的 image_stem 是 image_output_name 生成的安全文件名。适配器将其同时用于逻辑 image_id，并先写出单图 canonical 文件；主流程随后仅修改返回记录副本的外层 ID，再生成汇总文件。单图文件与阶段记录因此没有同步修正。

实际辅助函数计算：同一图片在 GT/汇总中为 images/val/a.jpg，传入 trace_stage 的值为 images_val_a.jpg_04c849f559。静态写出路径确认单图文件使用后者。将单图文件导入诊断仍会错误关联并产生虚假 FN/FP。此次未运行真实模型 trace_stage。

建议：逻辑 image_id 与产物文件名分开传递，所有记录使用逻辑 ID，只有落盘路径使用安全文件名；补单图文件与汇总文件均能匹配 GT 的集成验证。

## P2：缺省 modules 的合法旧配置被 UI 拦截

位置：vtools_ui/app.py:922、1053。

_load_module_defaults 对没有 modules 的配置将全部按钮设为 false，_run 再以没有选中模块为由提前返回。诊断后端 resolve_modules({}) 实际默认启用 diagnostics.missed 等模块，可视化和压缩也有各自的旧配置规则。

抽取实际 _load_module_defaults/_run，使用控件替身和临时 YAML 复现：无 modules 时 UI 提示“未选择模块”，后端默认 diagnostics.missed=True。这是上一轮无条件清空按钮带来的回归。

建议：区分 modules 缺省与显式空集合，按各后端最终模块规则初始化界面；未修改按钮时允许后端处理默认配置。补缺省、空映射、显式集合以及切换配置的回归测试。

## 验证

新增 tests 与三个子项目全量测试共 60 passed、2 failed。两项失败均因 /usr/bin/python3 未安装 torch。git diff --check 通过。UI 复现使用实际方法与控件替身；未执行真实 Qt 窗口、模型训练、GPU、TensorRT 或 K230 流程。
