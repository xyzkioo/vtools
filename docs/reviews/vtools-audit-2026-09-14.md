# vtools 全仓审核报告

审查日期：2026-09-14（审查始于 2026-09-13）。基准为当时的当前工作区，包含用户已有未提交修改和未跟踪源码。本报告保留原始问题编号；修复后的复查确认仍有遗漏和回归。

## 修复状态

此前“C01–C40 全部处理完成”的表述已撤回。部分问题已修复，但注册表、连续压缩、图片 ID/输出覆盖、配置保存及退出码等仍存在问题。详见 [修复后复查报告](vtools-recheck-2026-09-14.md)。GPU TensorRT、真实数据集训练和 K230 板端仍未完整验收。

## 审查范围与验证

- 全仓扫描 315 个 Python 文件、94,188 行：全部通过 Python AST 语法解析；检查本地 Ultralytics 的绝对包导入目标，未发现对应模块缺失。
- Markdown 本地链接检查未发现断链。该检查不代表所有示例命令、参数与行为都正确。
- 对自有工具的入口、配置、运行目录、模块选择、模型压缩与注册表、诊断数据/匹配/输出、可视化、测速/一致性、UI 和数据转换代码进行了重点人工审查。
- Ultralytics 源码进行了全量语法和内部导入检查，并重点核对模型加载/保存/检测头接口；没有逐项验证所有模型族、训练器、追踪器、导出后端的运行行为。
- 现有 unittest：speed_test 43 项（42 通过、1 按环境跳过）；model_diagnostics 5 项通过；model_compression 10 项通过。合计 58 项，57 通过、1 跳过。
- 本地环境检查确认实际导入本仓库 ultralytics 8.4.128，CPU 上 YOLO26 YAML 构建和 64×64 前向通过。
- 使用仓库 yolo26n.pt 做 CPU 依赖感知剪枝、前向、带旧 EMA 保存重载；使用 bus.jpg、64×64 输入完成特征图+CAM+阶段追踪冒烟，返回 0，CAM 状态为 ok。
- 通过最小复现核验了重叠误判、配置编辑数据损坏/丢失、UI 模块参数、COCO 匹配、注册表并发快照丢失、改名回滚、CAM 类别选择、错误退出码等。
- 未执行真实数据集完整 mAP/训练、GPU TensorRT engine 构建/测速、K230 板端或 Windows UI 验证。不把这些未执行路径标成“已通过”。

P1：可能导致模型/数据/验收结论错误，优先处理。P2：功能错误、配置不一致或条件性失败。以下分别标明实测或静态确认，不把推测写成已复现。

## 代码与集成问题

### C01 · P1 · 结构化剪枝保存时旧 EMA 覆盖真实剪枝模型

位置：[model_compression/core/detection.py:224](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:224)；[ultralytics-cn/ultralytics/engine/model.py:351](/home/xyzkioo/PycharmProjects/vtools/ultralytics-cn/ultralytics/engine/model.py:351)；[ultralytics-cn/ultralytics/nn/tasks.py:1902](/home/xyzkioo/PycharmProjects/vtools/ultralytics-cn/ultralytics/nn/tasks.py:1902)

影响：依赖感知剪枝且 finetune_epochs=0 时，checkpoint 中原有 ema 未清除。YOLO.save 只更新 model，重载优先 ema，实际拿到未剪枝模型。

依据：真实 yolo26n.pt 增加 EMA 后剪一个阶段：原参数 2,572,280，剪枝后 2,570,856，保存重载后 2,572,280。非结构化分支已清理 ema，但结构化分支没有。

建议：结构化保存前清理旧 EMA/优化器；重载后校验网络形状、参数量和实际权重，而不仅验证能推理。

### C02 · P1 · 检测分支选中的权重与导出、父版本、指标归属不一致

位置：[model_compression/core/detection.py:160](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:160)；[model_compression/core/detection.py:265](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:265)；[model_compression/core/detection.py:289](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:289)

影响：已有压缩 head、YAML 仍填写原始 weights 时，代码选择 origin；artifact.export 因而导出原模型，第二次剪枝从原模型开始却把已有压缩 head 记成父版本，baseline/model.parameters 还会把原模型指标写入 head。

依据：静态确认：weights 按 origin/current 选择，后续版本 parent_version_id 和 metadata 写回仍统一使用 head。

建议：分别解析“本次输入版本”“当前分支 head”“导出版本”；导出默认使用 head；指标和父关系必须使用实际输入版本。

### C03 · P1 · 分类压缩已有分支时静默忽略新的 weights

位置：[model_compression/run_model_compression.py:123](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:123)；[model_compression/run_model_compression.py:175](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:175)

影响：同一 branch.name 更换 model.weights 或 CLI --weights 后，基线、量化、剪枝继续使用原 head；用户以为处理新模型，实际处理旧模型。

依据：静态确认：_base_version 有 head 直接返回；_run_baseline 也直接采用 head 的 artifact，没有与显式 weights 比较。检测分支有冲突检查，分类分支没有。

建议：校验显式权重与当前版本的关系，冲突时报错或要求配置新分支，禁止静默替换。

### C04 · P2 · 蒸馏教师与学生血缘被硬编码成同一个父版本

位置：[model_compression/run_model_compression.py:304](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:304)；[model_compression/run_model_compression.py:334](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:334)

影响：配置独立 teacher_weights/student_weights 后，登记的 teacher_version_id 和 student_init_version_id 仍都是 parent_id，无法追溯实际训练来源。

依据：静态确认：模型从 teacher_config/student_config 加载，关系写入却固定使用 parent_id。

建议：分别登记实际教师、学生初始化 checkpoint，并记录各自路径、哈希与版本关系。

### C05 · P2 · student_factory 无法按预期从头构建学生模型

位置：[model_compression/run_model_compression.py:289](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:289)；[model_compression/core/model_runtime.py:100](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/model_runtime.py:100)

影响：只填写 student_factory 时会回退到父 checkpoint；完整 nn.Module checkpoint 直接返回，factory 根本不执行，可能把教师规模模型当作学生继续训练；不提供 weights 的内置 loader 则直接报错。

依据：静态确认：factory 仅在 state_dict 分支使用，完整 Module 分支先返回。

建议：显式 factory 应能创建模型；学生初始化权重作为独立可选步骤，不能自动用父模型对象替代新架构。

### C06 · P2 · comparison.report 读取的指标字段与写入字段不匹配

位置：[model_compression/run_model_compression.py:349](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:349)；[model_compression/core/detection.py:232](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:232)

影响：检测结构化报告写 after_benchmark_inference_ms_median，对比只读 benchmark_latency_p50_ms；蒸馏精度在 metadata.training.best_accuracy，对比只读顶层 best_accuracy，导致已有指标显示为空。

依据：静态确认：生产和读取键名/嵌套层级不一致。

建议：统一版本指标 schema，增加检测延迟、分类蒸馏精度到对比报告的集成测试。

### C07 · P2 · 压缩默认配置同时启用检测任务和分类专用 INT8

位置：[model_compression/config/pycharm_run.yaml:18](/home/xyzkioo/PycharmProjects/vtools/model_compression/config/pycharm_run.yaml:18)；[model_compression/config/pycharm_run.yaml:76](/home/xyzkioo/PycharmProjects/vtools/model_compression/config/pycharm_run.yaml:76)；[model_compression/core/detection.py:157](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:157)

影响：默认 task: detect，但 compression.quantize.dynamic_int8: true。用户补全权重和 data 后直接运行仍必然有一个模块失败；未配置 data 时基线也不会按 README 所述跳过。

依据：静态确认：检测路由明确拒绝 dynamic_int8。

建议：将默认模块与默认任务对齐，分别提供检测和分类示例。

### C08 · P2 · 压缩配置中的多条权重路径没有按 project.root 归一化

位置：[model_compression/core/config.py:120](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/config.py:120)；[model_compression/modules/structured.py:87](/home/xyzkioo/PycharmProjects/vtools/model_compression/modules/structured.py:87)

影响：distillation.teacher_weights、student_weights、compression.structured.initial_weights 保持原始相对路径，运行时按工作目录解释；与 model.weights/teacher.weights 的规则不同。

依据：最小复现：project.root=/tmp/example，加载后 teacher_weights 仍是 weights/t.pt，initial_weights 仍是 weights/n.pt。

建议：对所有输入路径统一规范化，并测试从仓库外目录运行。

### C09 · P2 · 依赖感知剪枝不接受其他入口支持的数字设备编号

位置：[model_compression/modules/dependency_pruning.py:110](/home/xyzkioo/PycharmProjects/vtools/model_compression/modules/dependency_pruning.py:110)

影响：--device 0 被 torch.device("0") 拒绝；同项目其他入口支持 0→cuda:0。

依据：实测 RuntimeError: Invalid device string: 0。

建议：复用统一 resolve_device；不要在剪枝模块重复实现设备解析。

### C10 · P1 · 注册表原子写入不能避免并发更新丢失

位置：[model_compression/core/registry.py:44](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/registry.py:44)；[model_compression/core/registry.py:62](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/registry.py:62)

影响：UI 与终端或两个终端同时操作同一 registry 时，各自持有旧快照；后写者覆盖前写者新分支、版本和运行记录。

依据：最小复现：两个 ModelRegistry 实例先读空表，依次新增 a、b，重新读取只剩 b。

建议：对完整读改写事务加文件锁或使用数据库；仅 os.replace 不足以处理并发。

### C11 · P2 · 模型版本不可覆盖的承诺没有被产物存储落实

位置：[model_compression/core/run_manager.py:18](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/run_manager.py:18)；[model_compression/core/registry.py:109](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/registry.py:109)；[model_compression/run_model_compression.py:259](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:259)

影响：--run-dir 允许复用已有目录，而量化、剪枝、蒸馏文件名固定；再次运行可覆盖旧版本引用的 artifact。导入版本也直接引用外部可变文件，后续加载不校验登记哈希。

依据：静态确认：目录 exist_ok=True，save_model 覆盖固定路径，add_version 只记录路径和当时哈希。

建议：每个版本使用唯一产物目录；复用 run 时禁止覆盖已有模型产物；读取时验证哈希。

### C12 · P1 · 多个入口把失败报告转换成退出码 0，UI 会显示成功

位置：[run_tools.py:130](/home/xyzkioo/PycharmProjects/vtools/run_tools.py:130)；[speed_test/run_all.py:194](/home/xyzkioo/PycharmProjects/vtools/speed_test/run_all.py:194)；[speed_test/benchmark_tensorrt.py:11](/home/xyzkioo/PycharmProjects/vtools/speed_test/benchmark_tensorrt.py:11)；[speed_test/core/vision_checkpoint_inspector.py:255](/home/xyzkioo/PycharmProjects/vtools/speed_test/core/vision_checkpoint_inspector.py:255)

影响：run_tools 对 dict/list 返回值一律视为 0；run_all 不返回汇总状态；TensorRT wrapper 不检查 failed rows。checkpoint 独立检查也不根据检查结果返回失败。

依据：同一份配置强制 CPU 执行 TensorRT 一致性：run_tools --tool consistency 退出 0，独立 check_consistency.py 退出 1，run_all 退出 0；报告均明确为 error。

建议：统一 RunResult 和退出码转换；UI 依据明确业务状态；覆盖 error/failed/inconclusive 和多模型部分失败。

### C13 · P2 · 仅导出、仅构建或仅 checkpoint 检查会被 run_all 误报失败

位置：[speed_test/run_all.py:63](/home/xyzkioo/PycharmProjects/vtools/speed_test/run_all.py:63)；[speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py:606](/home/xyzkioo/PycharmProjects/vtools/speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py:606)；[speed_test/backends/pytorch_vision_speed_benchmark_v2.py:312](/home/xyzkioo/PycharmProjects/vtools/speed_test/backends/pytorch_vision_speed_benchmark_v2.py:312)

影响：这些阶段成功时可以返回空 rows；_invoke 却把空列表判为“后端未返回有效结果”。构建成功但不测速时还可能因此跳过后续一致性。checkpoint-only 输出 CSV 只有表头，实际检查详情未写入。

依据：静态确认：导出成功 continue、构建不测速不 append、checkpoint-only save_rows([])/return []，与 _invoke 要求冲突。

建议：把执行状态与测量记录分离；无测量记录的成功操作也返回明确成功状态及产物/检查详情。

### C14 · P1 · 重叠分析使用错误的匹配表，误判正常目标为重复框

位置：[model_diagnostics/diagnostics/modules.py:163](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/modules.py:163)；[model_diagnostics/diagnostics/engine.py:1198](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/engine.py:1198)；[model_diagnostics/tests/test_modules.py:44](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/tests/test_modules.py:44)

影响：pred_gt_map 从 diagnostic_per_gt 读取 prediction_id/matched_gt_id，但真实 per_gt 没有这些字段；两个预测分别匹配不同真实目标也被当成疑似重复。

依据：真实 evaluate_operating 结果接入 overlap_analysis：TP=2，疑似重复框对=1。现有测试把 per_gt 和 per_prediction 人工混在同一列表，因此未发现。

建议：分别传入 per_gt/per_prediction，以 image_id+prediction_id 取对应 GT；测试使用引擎真实输出。

### C15 · P1 · 可视化 canonical 与诊断的 image_id 规则不一致

位置：[model_visualization/run_visualization.py:231](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:231)；[model_visualization/core/common.py:24](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/common.py:24)；[model_diagnostics/diagnostics/engine.py:410](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/engine.py:410)

影响：可视化对中文/符号文件名做 safe_name，并给重名图片加 _2；诊断保留原 stem，重名时使用相对路径。直接导入可视化预测可能被当成另一张图，产生虚假的 FN/FP。

依据：复现中文 stem：可视化为 item，诊断为 图片；重复 stem 的处理规则也明确不同。预测加载按 image_id 关联，不用 file_name 自动对齐。

建议：为跨工具定义稳定统一 image_id，或显式通过完整来源路径建立映射；文件名安全化仅用于输出文件。

### C16 · P2 · predict.generate 开关没有控制 B/C 模式的推理

位置：[model_diagnostics/run_model_diagnostics.py:325](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:325)；[model_diagnostics/run_model_diagnostics.py:445](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:445)

影响：B/C 无已有预测时始终 _generate_predictions，selected_modules 没有参与决定；即使 predict.generate:false 仍会加载模型和推理。

依据：静态确认；Spec 明确约定无预测且推理关闭时应报配置错误。

建议：在生成前落实模块依赖检查；模式和模块开关应只有一种明确优先规则。

### C17 · P2 · 诊断 GT 读取两次，且两次使用不同路径解析上下文

位置：[model_diagnostics/run_model_diagnostics.py:311](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:311)；[model_diagnostics/diagnostics/engine.py:1182](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/engine.py:1182)

影响：包装入口读取 GT 时传 base_dir=project_root；引擎重新读取时不传。data.yaml 的相对 path 需要 project.root 才能找到时，推理阶段成功后评估阶段可能找错目录或失败。

依据：静态确认两个调用参数不同；同时违背 Spec 的“一次读取、共享数据”。

建议：将规范化 Dataset 直接传入引擎，或只解析一次并传递相同已解析路径。

### C18 · P2 · 部分 CLI 覆盖在读取配置时就被提前校验挡住

位置：[model_diagnostics/run_model_diagnostics.py:185](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:185)；[model_visualization/core/common.py:93](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/common.py:93)；[model_visualization/run_visualization.py:340](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:340)

影响：模式 A 的 YAML predictions 为空时，即使传 --predictions 仍先报错；可视化 input.source 为空时，--source 或 --mode list_layers 也来不及覆盖。

依据：诊断最小复现：predictions_override=True 仍触发“模式 A 必须填写已有预测文件”。可视化先 load_config 后 _set_cli_overrides。

建议：先合并 YAML 与 CLI，再验证完整的有效配置。

### C19 · P2 · 差图渲染没有复用输入图片解析和正式置信度筛选

位置：[model_diagnostics/diagnostics/modules.py:256](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/modules.py:256)；[model_diagnostics/run_model_diagnostics.py:114](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:114)

影响：canonical/COCO file_name 为相对路径时，render_bad_cases 不使用 images_dir/project_root，可能直接跳过图片；绘制时又把低于正式阈值的所有预测都画出，和 bad_cases 计数不一致。

依据：静态确认：Path(source).is_file() 按工作目录判断，预测绘制循环不筛 score；Ultralytics data.yaml 的 source_path 情况不受前一个问题影响。

建议：读取阶段统一记录绝对 source_path；绘制正式预测与候选框时分层、标明阈值。

### C20 · P2 · 诊断同时 enable/disable 时漏验 disable 中的未知模块

位置：[model_diagnostics/diagnostics/modules.py:101](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/modules.py:101)

影响：requested=set(only_values or enable_values or disable_values) 仅检查首个非空列表；存在合法 enable 时，disable 中的拼写错误被静默接受。

依据：复现 enable=diagnostics.missed、disable=typo 返回含 typo:false 的配置，没有报错。

建议：对三个列表取并集，复用测速/压缩已正确实现的校验逻辑。

### C21 · P1 · UI 把空列表推断成布尔列表，保存时破坏路径/模块配置

位置：[vtools_ui/app.py:532](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:532)；[vtools_ui/app.py:617](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:617)

影响：all(...) 对空列表返回 True，python_paths: []、layers.modules: []、compression.exclude: [] 都被标成 list-bool。用户输入路径或模块名后被保存成 false。

依据：实测在 project.python_paths 输入 /tmp/custom，保存结果为 [false]。

建议：空列表按字段 schema 判定元素类型；未知类型保留 YAML 表达式或按字符串列表处理。

### C22 · P2 · UI 常用/全部/YAML 三页编辑状态不同步，切页保存丢修改

位置：[vtools_ui/app.py:622](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:622)

影响：保存只读取当前页的表单；其他页的修改没有合并，旧表单还会覆盖高级 YAML 的修改。

依据：实测常用页把 old.pt 改为 new.pt，再切全部页保存，最终文件仍为 old.pt。

建议：统一一个数据模型，切页时同步，保存时合并已修改字段。

### C23 · P2 · UI 常用配置下拉框静默替换合法自定义值

位置：[vtools_ui/app.py:405](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:405)；[vtools_ui/app.py:467](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:467)

影响：非可编辑 QComboBox 不包含 cuda:1、自定义 adapter 等值，setCurrentText 不会保留它们；保存常用页会用第一选项覆盖原配置。

依据：实测 custom.adapter、cuda:1 未做修改直接保存常用页，变为 ultralytics、auto。

建议：保留当前未知选项，或使用可编辑下拉框；未触碰字段不得重写。

### C24 · P2 · UI 模块按钮不是 YAML 的镜像，却总用 --only 覆盖 YAML

位置：[vtools_ui/app.py:817](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:817)；[vtools_ui/app.py:1009](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:1009)

影响：按钮初始化只选第一项，不从配置读取；点击开始后用 --only 关闭所有未出现在按钮中的模块。用户在图形配置开启的额外诊断、对比、导出等可能不会运行；compression 的 operation 也被 CLI 模块覆盖。

依据：静态确认：setChecked(index==0)，_run 按按钮生成 --only，配置选择后不刷新模块状态。

建议：按钮从 YAML 初始化并同步；明确区分“遵循配置”和“临时只运行所选模块”。

### C25 · P2 · UI 差图输出没有补齐 output.bad_cases 依赖

位置：[vtools_ui/app.py:889](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:889)；[vtools_ui/app.py:1009](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:1009)

影响：选择“差图输出”传 --only output.images，关闭了必需的 output.bad_cases，因此无法运行；即使 YAML 已启用依赖也会被 --only 关闭。

依据：实测默认漏检+差图按钮生成 diagnostics.missed、output.images，resolve_modules 报 output.images 依赖 output.bad_cases。

建议：选择差图时同时加入 output.bad_cases，按需加入 output.html，并明确诊断依赖。

### C26 · P2 · UI TensorRT 测速无法按 YAML 构建新 engine

位置：[vtools_ui/app.py:1014](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:1014)

影响：TensorRT 后端最终只传 --only speed.tensorrt_call；UI 没有 export.onnx/build.tensorrt 对应按钮，YAML 的构建模块被关闭。首次运行没有 engine 就失败。

依据：实测 TensorRT 页面生成参数仅 --only speed.tensorrt_call。

建议：提供导出/构建选择，或明确复用 YAML 的构建模块；不要把配置中的依赖步骤隐藏关闭。

### C27 · P2 · UI 文件改名的递归与确认选项没有正确映射

位置：[vtools_ui/app.py:1261](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:1261)；[others/filename_transform/image_filename_converter.py:44](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:44)；[others/filename_transform/image_filename_converter.py:503](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:503)

影响：UI 取消递归只是不传 --recursive，脚本仍继承 recursive=True；另外取消“跳过确认”后，子进程会等待 input()，UI 没有输入通道，任务会卡住。

依据：静态确认：未递归时没有 --no-recursive；runner 不提供 stdin 交互。

建议：明确传 --no-recursive；在 UI 做执行确认后传 --yes，或提供可用的交互输入。

### C28 · P2 · UI 停止任务可能留下忽略 SIGTERM 的工作进程

位置：[vtools_ui/runner.py:137](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/runner.py:137)

影响：先给子孙 SIGTERM 再立即 kill 主进程；主进程结束后 waitForFinished 成功，就不会对仍存活子进程补 SIGKILL；再查主进程树也无法找到已被收养的后代。

依据：静态确认；触发条件是 adapter/训练工作进程忽略或延迟处理 SIGTERM。

建议：停止前保存 PID 集合或创建进程组，等待后逐一清理仍存活的后代，不能只以主进程退出判断完成。

### C29 · P2 · 启用 stage_trace 会改变指定类别 CAM 的目标候选

位置：[model_visualization/adapters/ultralytics.py:386](/home/xyzkioo/PycharmProjects/vtools/model_visualization/adapters/ultralytics.py:386)；[model_visualization/run_visualization.py:270](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:270)

影响：stage_trace 选择全类别最高分候选，不考虑 cam.target.class_id；run 再把这个 raw index 强行传给 CAM，解释的不是指定类别最高分目标。final_detection 同样没有按 class_id 筛选最终框。

依据：合成两个候选：不传阶段索引时选类别1的候选1（0.9997）；阶段全局索引0传入后选择类别1候选0（0.0474）。

建议：目标选择由 CAM 统一执行，阶段仅提供索引映射；按类别和目标类型筛选后再确定候选。

### C30 · P2 · 可视化模块 allow-list 与其他工具约定不同

位置：[model_visualization/run_visualization.py:48](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:48)

影响：modules 只写 visualization.cam:true 时，未列出的 features 仍默认开启；YAML 未知模块 ID 也没有检查，enable/disable 同项冲突不报错。

依据：最小复现 partial modules 得到 features:true、cam:true；源码只检查 CLI 的 unknown，没有检查 YAML keys。

建议：复用统一模块解析器，存在 modules 时未列出的模块关闭，并统一冲突规则。

### C31 · P2 · 可视化 BF16 模型与输入 dtype 不匹配

位置：[model_visualization/adapters/ultralytics.py:84](/home/xyzkioo/PycharmProjects/vtools/model_visualization/adapters/ultralytics.py:84)；[model_visualization/core/common.py:196](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/common.py:196)

影响：适配器接受 bf16 并把模型转为 bfloat16，但 image_tensor 只处理 fp16，其余一律 float32；CUDA bf16、CAM 关闭时前向会出现 dtype 不匹配。

依据：静态确认模型转换和输入转换路径不一致；本次未进行 CUDA bf16 实测。

建议：集中实现 precision→dtype 映射，或显式拒绝未支持的精度。

### C32 · P1 · 文件改名 auto 模式把 YOLO 数据集根目录识别为普通图片

位置：[others/filename_transform/image_filename_converter.py:432](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:432)

影响：--dir 指向含 images/labels 的数据集根目录且 --dataset-format auto 时，只识别当前目录本身名为 images 的情况；递归执行会改图片却不改 labels，破坏图像标注对应关系。

依据：静态确认：resolve_yolo_roots 支持根目录，但 auto 判断没有检查 directory/images 与 directory/labels。UI 默认格式为 auto。

建议：auto 检测复用 resolve_yolo_roots 的识别逻辑；正式改名前校验图片和标签计划完整性。

### C33 · P1 · 文件名互换中途失败后回滚不完整

位置：[others/filename_transform/image_filename_converter.py:287](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:287)

影响：a→b、b→a 已完成后其他改名失败，回滚看到旧名字已经存在就跳过，留下部分交换成功的数据集。

依据：注入第三次提交失败：初始 a=a、b=b、c=c，结束 a=b、b=a、c=c，未回到原状态。

建议：失败回滚也必须两阶段临时命名，不能直接按“旧路径不存在”判断能否恢复。

### C34 · P2 · COCO 同名图片即使完整相对路径明确仍无法匹配

位置：[others/filename_transform/image_filename_converter.py:234](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:234)

影响：完整路径匹配结果又与 basename 匹配结果混合。a/1.jpg、b/1.jpg 会互相造成歧义，默认报未引用；若用户放行未引用，图片会改名而 JSON 不更新。

依据：最小复现两条完整 file_name，build_coco_updates 返回零条 updates，两张都被判为 unreferenced。

建议：优先精确相对路径，仅精确匹配失败时才尝试唯一 basename。

### C35 · P2 · 批量视频抽帧输出目录只取 stem，重名视频会碰撞

位置：[others/video_frame_extractor.py:79](/home/xyzkioo/PycharmProjects/vtools/others/video_frame_extractor.py:79)

影响：递归处理 a/1.mp4 和 b/1.mp4 或同目录 1.mp4/1.avi，指定统一 output 后都写 out/1。默认第二个失败，overwrite=True 则覆盖第一个视频帧。

依据：最小复现 build_output_dir 对两个不同来源返回相同目录。

建议：保留相对目录，并包含扩展名或生成唯一视频标识；处理前检测输出冲突。

### C36 · P2 · 图片缩放递归扫描会把已有输出目录重新纳入输入

位置：[others/image_resize.py:36](/home/xyzkioo/PycharmProjects/vtools/others/image_resize.py:36)；[others/image_resize.py:108](/home/xyzkioo/PycharmProjects/vtools/others/image_resize.py:108)

影响：默认输出 input/resized，recursive=True 再运行时会扫描上一次输出。允许覆盖时还会生成 resized/resized，放大数据和重复处理。

依据：静态确认先扫描 input.rglob，再选择 output_dir，没有排除输出子树。

建议：在收集图片前解析输出目录，排除输出子树，并处理同名不同后缀转同格式的目标冲突。

### C37 · P1 · Kmodel 一致性“验证”不判定失败，且截断不等长输出

位置：[transform_tools/PY2KM_validate.py:82](/home/xyzkioo/PycharmProjects/vtools/transform_tools/PY2KM_validate.py:82)

影响：只循环 min(两边输出数)，未验证 shape、有限值或误差阈值。少一个输出、全错结果、NaN 都可能打印后正常退出；不能作为自动验收。

依据：静态确认 compare 无返回状态，main 不检查误差；同元素数不同 shape 的数组还可能触发广播后计算错误 MAE。

建议：严格检查输出数/名称/shape/dtype/有限值，设显式容差并返回非零退出码，保存结构化报告。

### C38 · P2 · Kmodel 校验 --no-norm 连输入 dtype 一起变成 uint8

位置：[transform_tools/PY2KM_validate.py:47](/home/xyzkioo/PycharmProjects/vtools/transform_tools/PY2KM_validate.py:47)

影响：不归一化本应允许 FP32 的 0–255 输入，但关闭 normalize 后保留 uint8，常见 FP32 ONNX 会直接拒绝类型。

依据：静态确认 astype(float32) 和 /255 被绑在同一 if normalize 中；没有读取 ONNX 输入类型。

建议：根据模型输入类型转换 dtype，把缩放与类型转换分开。

### C39 · P2 · 根入口 tools.yaml 的 tool 字段未使用，部分参数还转发到错误工具

位置：[run_tools.py:39](/home/xyzkioo/PycharmProjects/vtools/run_tools.py:39)；[run_tools.py:92](/home/xyzkioo/PycharmProjects/vtools/run_tools.py:92)；[config/tools.yaml:3](/home/xyzkioo/PycharmProjects/vtools/config/tools.yaml:3)

影响：只修改 tools.yaml 的 tool 无法改变默认 diagnostics；入口仅按文件名 tools.yaml 判定聚合配置。测速分支还错误转发 structured-scale/epochs/initial-weights 到不接受这些参数的解析器。

依据：静态确认：args.tool 默认 diagnostics，从未读取 values.tool；structured 参数位于测速转发分支。

建议：给统一入口定义独立配置 schema，落实 CLI>配置的优先级；每个工具只允许并转发本工具参数。

### C40 · P2 · 部分可视化配置看似可调，实际没有执行逻辑

位置：[model_visualization/config/pycharm_run.yaml:49](/home/xyzkioo/PycharmProjects/vtools/model_visualization/config/pycharm_run.yaml:49)；[model_visualization/run_visualization.py:270](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:270)；[model_visualization/core/capture.py:143](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/capture.py:143)

影响：cam.max_targets 和 cam.target.selection 没有被消费，仍只解释一个候选；features.statistics:false 也没有控制统计文件写出。配置改变却不改变功能，容易误认为已经支持多目标/类别解释。

依据：静态确认配置字段搜索及实际执行流程。

建议：实现对应语义，或移除未支持字段并在解析时明确拒绝；不要用无效参数表示计划功能。

## README、Spec 和版本文档欠账

1. **VERSION 当前版本落后且自相矛盾**：[VERSION.md](/home/xyzkioo/PycharmProjects/vtools/VERSION.md:5) 顶部当前批次是 2026.09.08，下面已有 09.10 更新；未归纳后续桌面 UI、模型压缩、检测结构化及依赖感知剪枝。不是 README 完全未更新，而是版本记录没有跟上。
2. **压缩快速开始与当前默认配置不符**：根 README 和 model_compression/README.md 要求主要修改 weights，并称无数据时评估跳过；当前默认是检测且需要 data，同时启用了分类 INT8。参见 C07。README 的分叉示意也不能代替代码实际顺序推进 head 的说明。
3. **质量门禁配置路径写错**：model_compression/README.md 以及检测失败提示写 structured.quality_gate；实际应为 compression.structured.quality_gate。在 YAML 顶层加 structured 无效。检测 Spec 还写 models/structured-<scale>.yaml，实际生成模型族命名的 <family><scale>-vtools.yaml。
4. **模块化 Spec 把尚未落实的架构写成现状**：docs/specs/modular-tests-and-diagnostics.md 声称一次读取、消除 sys.argv 调用、统一 --run-dir、根入口默认消费 tools.yaml；实际仍有两次 GT 读取、sys.argv 调度，根入口及诊断 wrapper 不接受 --run-dir，tools.yaml 的 tool 未使用。schema_version:2 的路径迁移约定也没有对应实现。应区分“已实现”“规划”。
5. **文件改名的“默认只预览”与脚本默认值不一致**：文件头和 README_image_filename_converter.md 写默认预览，PYCHARM_CONFIG.apply 却为 True；yes=False 仍有确认，因此不是启动即无提示改名，但 CLI/PyCharm 的行为与文档相反。UI 显式 --dry-run 是另一套默认，需统一说明。
6. **转换和环境示例没有完全更新**：根 README 仍主要指导修改 ndjson_to_yolo 的函数调用，而脚本现已有 --input/--output；K230 有 --rebuild，但说明主要仍要求删旧 ONNX；env_test/check_install.py 文件头示例使用不存在的 install_check/check_install.py。视频抽帧 --every-n 帮助写默认 1，而 PYCHARM_CONFIG 为 8。
7. **UI 说明夸大配置与输入支持的一致性**：README 称 YAML 是真实参数来源，但模块按钮默认状态会生成 --only 覆盖配置；说明称诊断模型/预测/数据三个独立输入，实际模型与预测共用一个 resource_edit，按扩展名分流；可视化输入提示支持 .onnx，但运行逻辑要求可挂钩的 PyTorch Module。需同步实际支持边界。
8. **少量低成本文档整理**：根 README 把 TensorRT engine 元数据说明放在可视化章节；压缩“职责与测速分开”的开场应说明现已内置固定输入延迟测量；各依赖说明的命名 REQUIREMENT.md/REQUIREMENTS.md/requirements.txt 不统一。可读性问题与功能错误分开处理。

## 架构、格式与测试方面的系统性问题

- **核心类型/配置/模块解析重复**：speed_test 与 model_diagnostics 都用顶层 core 包名；speed_test/__init__.py 和入口通过修改 sys.path 维持兼容。同一进程先后加载两个 core 的结果依赖导入顺序。诊断 adapter_runtime 的包内导入也依赖外部预先调整路径。建议统一包限定导入，共享最小的 InputBundle/运行结果/模块解析协议；不建议为了统一目录复制第三份 core。
- **路径和设备协议不统一**：诊断 project.root 相对入口目录，测速/压缩/可视化相对配置目录；CLI 相对路径有的按 project.root，有的按工作目录。压缩 resolve_device 会在显式 CUDA 不可用时回退 CPU，测速/可视化通常报错。已有历史兼容需求可以保留，但要版本化并写入有效配置，避免静默差异。
- **main 返回协议不统一**：int/list/dict/None 混用是 C12/C13 的根源；模块支持状态也不统一。model.flops 和 validation.ultralytics 等属于占位/受限能力，应在模块清单中明确可用性并输出结构化 unsupported，而不是让用户从日志猜测。
- **格式与边界划分尚未建立统一约束**：没有仓库级自有代码的 formatter/linter/test 配置；存在中英文注释、单双引号、typing.List 与 list、超长字典/语句等混用。app.py 约 2,031 行、诊断 engine.py 约 1,275 行，分别集中太多职责。应先拆配置编辑/页面/进程管理、数据解析/匹配/报告等边界，再做统一格式。没有约定时，不能仅凭风格偏好把每一处写法判为 bug。
- **测试覆盖偏向纯函数，缺关键集成回归**：未见可视化、UI、转换/数据工具测试目录；压缩新结构化测试只测配置读取。重叠测试人工拼出生产代码不会生成的数据（C14），说明“测试通过”尚不能保证模块真正对得上。优先补本报告实测案例、模块-only 流程、产物重载一致性和失败退出码测试。
- **版本控制卫生**：.vtools_ui/history.json、.idea 项目文件和 ultralytics.egg-info 已被 Git 跟踪；当前新增 .gitignore 不能自动取消已跟踪文件。history 已产生工作区修改。新 detection/structured/dependency_pruning 源码及相应 Spec 仍未跟踪，若只提交跟踪文件会漏掉运行所需代码。此为当前交付状态，不把用户尚未提交的工作判为实现错误。
- **UI 日志和状态仍有小缺口**：QProcess 启动 Python 未加 -u/PYTHONUNBUFFERED，部分 print 日志会缓冲到较晚才显示；FailedToStart 没有对应 finished/history 完成处理；多字节 UTF-8 被分块解码可能显示替代字符。属于进程交互健壮性改进，低于数据/模型正确性优先级。

## 本机环境问题（与仓库代码缺陷分开）

- 基础 Python 环境没有 pytest；yolo 环境发现 pytest 8.4.2，但缺 iniconfig，python -m pytest 无法启动。因此改用仓库现有 unittest，不将 pytest 启动失败冒充业务测试失败。
- env_test 的 pip check 还发现 ncnn 缺 portalocker。基础 YOLO CPU 构建/前向仍通过；NCNN 路径没有实测。
- 本次未自动安装/升级依赖，也没有更改原环境以让结果“变绿”。

## 建议修复顺序

1. 模型与数据正确性：C01/C02/C03、C10、C14/C15、C21、C32/C33/C37。
2. 可信验收与可执行入口：C12/C13、默认检测配置、UI 差图/engine 构建、CLI 覆盖。
3. 统一版本血缘、指标 schema、配置保存和路径/模块协议，并增加集成回归。
4. 同步 README/Spec/VERSION，再统一格式与清理生成文件追踪。

这是一轮全仓范围的审核，不是所有硬件、模型族、数据集组合的无缺陷证明。上述问题为本次实际发现项；未完整执行的后端和训练路径保留为验证边界。
