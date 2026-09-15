# vtools 修复后复查

日期：2026-09-14。检查对象是当前未提交工作区，包括上次修复及原有未跟踪源码。本轮核对原始清单、检查修复差异，并在临时目录和无界面 Qt 环境复现关键问题；没有修改业务代码。

**结论：上次“C01–C40 全部修复”的结论不成立。以下列出 21 组仍存在的问题或修复引入的回归，包含 5 组 P1、16 组 P2。** 同一组可能涉及多个原始编号；该数字不是未修复原编号的个数。未实测的硬件路径不计为已通过。

## P1：模型、数据与验收结果

### R01 注册表仍会回退分支，并重复保存同一个运行记录（C10）

- 位置：[registry.py:82](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/registry.py:82)。
- 实测：两个实例读取同一注册表。A 推进 main，B 仅创建 other 分支；B 的旧 main 快照覆盖磁盘上的新 head，main 回退。文件锁并未解决旧快照覆盖。
- 实测：同一 run_id 先写 running 再写 succeeded，最终 runs 同时保留两条记录。列表按完整 JSON 去重，不能实现按运行 ID 更新。
- Windows 下 fcntl 缺失时直接不加锁，未提供等效互斥。
- 建议：锁内以最近读取快照为基准应用增量变更，对同一分支的并发推进检测冲突；runs 按 ID 合并，提供跨平台锁。

### R02 正常多模块压缩流程被新增权重冲突检查中断（C02/C03）

- 位置：[run_model_compression.py:134](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:134)、[detection.py:194](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/detection.py:194)。
- 实测：同一 run 中 head 从原模型推进到压缩模型，但配置仍是初始 weights；后续分类模块的 _base_version 拒绝继续。检测 artifact.export 在加载模型前即被同样的检查拒绝。
- 影响：分类“量化→剪枝/蒸馏”、检测“剪枝→结构化剪枝/导出”不能正常串行完成。比较模块仍可能执行，导致部分完成的报告。
- 建议：运行开始时校验显式输入归属，后续模块使用本次运行推进的版本；检测导出明确使用 head。

### R03 可视化输出名仍碰撞，并且诊断 ID 仍不兼容（C15）

- 位置：[run_visualization.py:236](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:236)、[common.py:24](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/common.py:24)、[engine.py:410](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/engine.py:410)。
- 实测：苹果、香蕉两个名字都经 safe_name 变成 item；a/b 与 a_b 都变成 a_b。上次还移除了计数去重，使特征图、CAM、阶段文件和 image_meta 相互覆盖。
- 实测：images/sub/a.jpg 在可视化中 ID 为 sub/a，诊断首次遇到该 stem 时 ID 为 a；无法匹配。同 stem 不同扩展名也会得到相同的可视化 ID。
- 汇总 canonical 仅替换外层 image_id，单图阶段输出仍用清洗后的 stem，输出之间不一致。
- 建议：统一逻辑 ID 与诊断匹配协议；输出目录采用可逆路径编码或路径哈希，不能用有损字符替换保证唯一性。

### R04 蒸馏 checkpoint 仍覆盖已登记模型（C11）

- 位置：[run_model_compression.py:357](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:357)、[distillation.py:108](/home/xyzkioo/PycharmProjects/vtools/model_compression/modules/distillation.py:108)。
- 静态确认：蒸馏仍写固定 distillation/best.pt、last.pt；复用 --run-dir 会覆盖旧版本引用的文件。新的唯一文件名辅助函数未覆盖这一路径。
- 同目录的导出 manifest 也仍使用固定 model_manifest.json，多个导出结果会引用被后次覆盖的说明。
- 建议：每次调用独立产物目录，版本登记不可变快照；清单也与对应产物一一绑定。

### R05 checkpoint-only 检查仍可把失败报告成成功（C12/C13）

- 位置：[pytorch_vision_speed_benchmark_v2.py:300](/home/xyzkioo/PycharmProjects/vtools/speed_test/backends/pytorch_vision_speed_benchmark_v2.py:300)。
- 静态确认：检查失败只在 fail_on_checkpoint_inspection=true 时抛错。仅启用检查时，默认路径写空 CSV 并返回 []；修复后的入口将 [] 视为成功，失败详情丢失。TensorRT 对应检查分支有相同结构。
- 独立检查入口也只把 missing 和 adapter_load_status=failed 判失败，raw-only 的 unknown（例如不可解析 checkpoint）不在失败判定内。
- 建议：区分“成功执行但没有测速行”与“检查失败”，返回包含实际检查状态的结果。

## P2：配置、功能与文档

### R06 student_factory 仍无法独立初始化较小学生（C05）

- 位置：[run_model_compression.py:320](/home/xyzkioo/PycharmProjects/vtools/model_compression/run_model_compression.py:320)、[model_runtime.py:99](/home/xyzkioo/PycharmProjects/vtools/model_compression/core/model_runtime.py:99)。
- 实测：只指定 student_factory，没有 student_weights，仍回退父模型权重并强制加载。父 Linear(4,3)、学生 Linear(2,3) 立即报 size mismatch；strict=false 也不能忽略张量形状不一致。
- 独立随机初始化学生不应被登记为父 checkpoint 初始化；factory-only 应允许无 weights。

### R07 predict.generate 仍不服从最终模块选择（C16）

- 位置：[run_model_diagnostics.py:321](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/run_model_diagnostics.py:321)。
- 静态确认：有非空 modules 时只读 YAML 值，忽略 --enable；不处理 --disable；--only 排除生成时仍可能因为 YAML=true 执行推理。旧 models 模式不受该检查约束。
- 建议：直接复用 resolve_modules 的最终结果，不单独实现另一套优先级。

### R08 未知 disable 模块仍被接受（C20）

- 位置：[modules.py:101](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/modules.py:101)。
- 实测：enable=[diagnostics.overlap]、disable=[typo] 未报错，结果新增 typo:false。requested 仍使用 or 而非集合并集。

### R09 三个 UI 配置页仍丢失编辑（C22）

- 位置：[app.py:639](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:639)。
- 无界面 Qt 实测：初值 cpu，常用页改 cuda:0，再在全部页改 auto，保存全部页后落盘仍是 cuda:0。因为固定先应用全部页再应用常用页。
- 切至 YAML 页保存时完全不合并图形页修改；无编辑顺序跟踪，仍会丢修改。

### R10 UI 模块按钮仍覆盖 YAML（C24）

- 位置：[app.py:824](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:824)。
- 静态确认：仍只选中 index==0，然后在启动时传 --only；未从所选配置同步模块，也会覆盖 compression.operation。该问题上次没有实际修复。

### R11 停止任务仍遗漏顽固子进程（C28）

- 位置：[runner.py:164](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/runner.py:164)。
- Mock 控制流实测：父进程 waitForFinished=true 时只调用 SIGTERM，没有任何保存的子 PID 的 SIGKILL。新增清理仍放在父进程等待超时的分支内。
- 影响：父进程被 kill 迅速退出，忽略 SIGTERM 的训练工作进程继续运行。没有对真实系统进程执行本次模拟。

### R12 UI 改名“跳过确认”选项失效（C27）

- 位置：[app.py:1282](/home/xyzkioo/PycharmProjects/vtools/vtools_ui/app.py:1282)。
- 静态确认：只要 apply=true 就添加 --yes，即使用户取消“跳过执行确认”。解决 stdin 等待的方式改变了用户选择；UI 也没有替代确认步骤。

### R13 CAM 仍可能解释错误类别/非最终目标（C29）

- 位置：[run_visualization.py:289](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:289)、[ultralytics.py:394](/home/xyzkioo/PycharmProjects/vtools/model_visualization/adapters/ultralytics.py:394)。
- 静态确认：指定 class_id 的 final_rows 为空时，回退到未按类别过滤的 target_index（第一条最终检测），从而解释该候选的另一类别分数。
- raw 选择只保留 argmax 类别等于 class_id 的行，不等价于在全部候选上寻找指定类别最高分。目标类别不是候选 argmax 时会错选。
- 显式 index 也未确认属于最终保留集合。final_detection 不能仅因 index 存在就视为有效。

### R14 空 modules 和非法模块配置仍未按约定处理（C30）

- 位置：[run_visualization.py:50](/home/xyzkioo/PycharmProjects/vtools/model_visualization/run_visualization.py:50)。
- 静态确认：modules:{} 被当作“没有配置”，退回默认 features=true。非映射 modules 也悄悄忽略，与诊断/压缩 allow-list 规则不一致。

### R15 CAM selection 仍未落实完整校验（C40）

- 位置：[capture.py:257](/home/xyzkioo/PycharmProjects/vtools/model_visualization/core/capture.py:257)。
- 静态确认：未知 selection 被当最高分处理。selection=index 且没填写 index 时，主流程先自动生成 candidate_index，绕过新增检查。max_targets<=0 被悄悄改成 1。
- features.statistics 的当前实现确实控制 CSV 写出，不能继续按旧报告描述为完全无效；多目标循环已有实现，但选择协议尚未完整。

### R16 Kmodel 比较对空输出和零输出判断错误（C37）

- 位置：[PY2KM_validate.py:92](/home/xyzkioo/PycharmProjects/vtools/transform_tools/PY2KM_validate.py:92)。
- 隔离执行当前 compare 函数实测：compare([],[]) 返回 True；两个完全一致的零向量返回 False（cosine=0、MAE=0）。数值在原 dtype 中进行点积、乘法和减法，也缺少对低精度溢出的保护。
- 建议：空输出应为失败/证据不足；零范数走绝对误差规则；先升精度计算并校验阈值。

### R17 ONNX 输入类型修复不完整（C38）

- 位置：[PY2KM_validate.py:48](/home/xyzkioo/PycharmProjects/vtools/transform_tools/PY2KM_validate.py:48)。
- 静态确认：normalize=true 强制 float32，无视模型需要 float16/uint8；tensor(double) 未匹配到 float 分支。README 新增的“按声明 dtype 选择”承诺范围超过实现。

### R18 差图路径仍未采用图片查找协议（C19）

- 位置：[engine.py:1237](/home/xyzkioo/PycharmProjects/vtools/model_diagnostics/diagnostics/engine.py:1237)。
- 静态确认：相对 file_name 使用 base_dir 或 output.parent，未使用 --images-dir/GT 文件所在目录；canonical 输入仍可能找不到图片后直接跳过，整个任务却报告成功。
- 置信度过滤已有修复，不能把该子问题继续列为未处理。

### R19 run_all 对非通过状态仍返回 0（C12）

- 位置：[run_all.py:195](/home/xyzkioo/PycharmProjects/vtools/speed_test/run_all.py:195)。
- 静态确认：只认 failed/error；一致性 warning 或 inconclusive 最终返回 0，而 check_consistency.py 只将 passed 当通过。不同入口验收结论仍不一致。

### R20 原文档欠账仍未修完

- [model_compression/README.md:15](/home/xyzkioo/PycharmProjects/vtools/model_compression/README.md:15) 仍声称无数据时基线跳过评估，但默认检测基线要求 dataset.data；快速开始未补全必要数据配置。
- 同文档第 40 行及检测失败消息仍写 structured.quality_gate，正确路径是 compression.structured.quality_gate。
- [模块化 Spec:21](/home/xyzkioo/PycharmProjects/vtools/docs/specs/modular-tests-and-diagnostics.md:21) 仍宣称一次 GT 读取，wrapper 与 engine 实际各读一次；统一 --run-dir、schema_version:2 路径迁移等也未按实现区分规划。
- [改名配置:57](/home/xyzkioo/PycharmProjects/vtools/others/filename_transform/image_filename_converter.py:57) 仍为 apply:true，与默认预览描述不一致。
- 环境检查文件头仍使用不存在的 install_check 路径；抽帧 --every-n 帮助仍写默认 1，配置为 8。转换快速开始仍有过时调用说明。

### R21 完成记录及验证表述过度

- 原审计报告和 VERSION.md 写全部完成，与上述复查结果矛盾。原审计状态已在本次更正，VERSION 中的功能完成条目应作为待重新验收记录阅读。
- 上轮展示的 EMA 重载验证只保存了未发生结构改变的模型，不能证明“实际剪枝后保存重载”完整通过。
- 本轮现有 58 项 unittest 仍然 57 通过、1 跳过，但没有覆盖本轮复现出的顺序模块调用、旧注册表快照、UI 同字段跨页编辑、输出目录碰撞等问题。

## 已确认修复与验证边界

当前代码确已补上结构化保存前 EMA 清理和参数形状比对、若干权重路径规范化、数字设备解析、检测默认关闭分类 INT8、重叠分析读取 per_prediction、空列表类型保留、CLI source 提前校验例外、BF16 输入转换、YOLO 根目录识别、COCO 精确匹配优先及部分产物唯一命名。本轮没有重新运行真实剪枝/训练、完整 mAP、GPU TensorRT 或 K230 流程，不能据此认定这些后端端到端无问题。

本轮运行了：speed_test 43 项（42 通过、1 跳过），model_compression 10 项通过，model_diagnostics 5 项通过，git diff --check 通过；额外复现注册表旧写入者回退/重复 run、顺序压缩冲突、学生工厂形状冲突、图片 ID/目录碰撞、未知 disable、Kmodel 空/零输出；Qt 无界面复现同字段编辑丢失，并模拟停止任务控制流。

下一轮应先补以上复现的回归用例，再完成业务修复，并按原始编号分别标明“已验证修复 / 部分修复 / 未修复 / 硬件未验收”。不要再次仅凭既有测试全部通过宣称所有问题已关闭。
