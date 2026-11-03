# vtools 项目审查（2026-11-01）

审查基准：当前工作树，HEAD `cc39d05`，版本 `0.4.7`。按用户要求应用本地 `review-agent` 技能，审查当前项目的可执行行为，不将发现归因于某次提交。检查覆盖统一入口、桌面任务与日志、配置、压缩、检测诊断、可视化、测速及一致性、文件工具、转换与打包入口；旧审查结论仅作线索，下面的问题均重新核对了当前源码。

确认 **9 项问题：2 项 P1、7 项 P2**。本轮未修复业务代码；新增本报告、更新文档索引与交接记录。

同日后续进行了[冗余与失效代码补充审查](vtools-redundancy-2026-11-01.md)，另列 2 项 P2、3 项 P3，以及无调用函数和未使用导入清理候选。

## 发现（按优先级）

### R01 · [P1] 用独立文件隔离可能被上游重写的输入图片

位置：[model_compression/core/detection.py](../../model_compression/core/detection.py)，第 111–116 行，尤其 `os.link(source, target)`。

验证和检测蒸馏的镜像目录通过硬链接引用原图。外部 Ultralytics Fork 的 `ultralytics/data/utils.py::check_image()` 会对末尾不是 JPEG EOI 标记的可解码 JPEG 重新编码保存；写镜像文件即写同一个 inode，因此普通验证也可能修改原始数据，违反仓库的输入只读约定。

**实测：**在临时数据集中创建 JPEG 并追加尾部字节，经 `_isolated_validation_data()` 镜像后，两条路径的 inode 相同。调用实际 Fork 的 `check_image()` 修复镜像，原图 SHA-256 发生变化：`same_inode=True`、`original_changed=True`。所有样例均位于临时目录。

建议：图片使用独立副本或具备写时复制语义的副本；不能用共享 inode 作为可被上游写入的隔离边界。

### R02 · [P1] 在直接脚本入口导入共享模块前建立仓库导入路径

位置：[speed_test/core/vision_benchmark_config.py](../../speed_test/core/vision_benchmark_config.py)，第 193 行；[model_diagnostics/run_model_diagnostics.py](../../model_diagnostics/run_model_diagnostics.py)，第 55 行；[transform_tools/ndjson_to_yolo.py](../../transform_tools/ndjson_to_yolo.py)，第 16 行。

文档中的直接脚本调用只把脚本所在子目录加入 `sys.path`。上述位置在仓库根目录可导入前就导入 `vtools_runtime`，使未设置 `PYTHONPATH` 的常规 CLI 启动失败。测速配置中的 `project.root` 尚未来得及参与路径设置。桌面子进程会注入仓库路径，现有导入式单测也带仓库路径，因而没有暴露这一问题。

**实测：**从仓库根目录、清除继承的 `PYTHONPATH` 后，以已有项目虚拟环境运行下列入口；各自 `--help` 均返回 0，实际进入运行逻辑后均报 `No module named 'vtools_runtime'`：

- `model_diagnostics/run_model_diagnostics.py`：退出码 2。
- `speed_test/benchmark_pytorch.py --config <临时有效配置>`：退出码 1。
- `transform_tools/ndjson_to_yolo.py`：退出码 1。

建议：复用已存在的直接脚本启动约定，在导入共享包前加入 vtools 根目录，并用清除 `PYTHONPATH` 的独立子进程验证公开入口。

### R03 · [P2] 为结构化微调也隔离训练和验证数据

位置：[model_compression/core/detection.py](../../model_compression/core/detection.py)，第 593–594 行；[model_compression/modules/structured.py](../../model_compression/modules/structured.py)，第 200–201 行。

结构化微调把 `detection_data(config)` 返回的原始 YAML 直接传给原生训练器，未走运行目录中的数据隔离。原生数据扫描会在原标签目录旁生成或刷新 `.cache`，JPEG 修复也会写入原图片路径。因此，即便修复 R01 的镜像实现，这条微调路径仍会写入用户数据目录。

**证据：**源码确认微调传入原始 `data`；另在临时数据集上实际构造 Fork 的 `YOLODataset`，确认产生源目录中的 `labels/train.cache`。本轮没有执行结构化微调训练。

建议：训练和验证均指向独立、可写且不会共享原图 inode 的运行目录副本。

### R04 · [P2] 按整个进程组的存活状态完成任务停止

位置：[vtools_ui/webapp/tasks.py](../../vtools_ui/webapp/tasks.py)，第 382–392 行。

停止逻辑发出进程组 SIGTERM 后只等待主进程；仅当主进程超时才补发 SIGKILL。若主进程退出而某个工作进程忽略 SIGTERM，工作进程会继续占用资源；若它继承日志管道，还可能使日志读取线程无法结束。

**实测：**临时父进程启动一个忽略 SIGTERM 的子进程。调用实际 `_terminate_process()` 后得到 `parent_exit=-15`、`child_still_alive=True`。复现结束后显式清理了该测试进程组。该情形也曾在旧审查中出现，当前实现仍存在。

建议：保留进程组身份，在宽限期后检查并终止仍存活的整个进程组，不能只以主进程的 `wait()` 结果判断停止完成。

### R05 · [P2] 让分类训练集和验证集共用类别编号

位置：[model_compression/core/model_runtime.py](../../model_compression/core/model_runtime.py)，第 234–236 行。

两个划分分别构造 `ImageFolder`，各自按现有目录重新编号，未检查或共享 `class_to_idx`。验证集缺少某个类别目录时，后续类别的标签会移动，产生错误的验证精度；这些精度还会用于分类蒸馏的最佳 checkpoint 选择。

**实测：**训练集包含 A/B，验证集只包含 B；训练映射为 `{'A': 0, 'B': 1}`，验证映射为 `{'B': 0}`。始终预测 B（输出编号 1）的模型在纯 B 验证集上被记录为 `status=succeeded, accuracy=0.0`。

建议：以训练集映射建立验证标签，或在映射不一致时明确拒绝执行；不能静默重编号。

### R06 · [P2] 显式报告无法执行的 pipeline 测速

位置：[speed_test/backends/pytorch_vision_speed_benchmark_v2.py](../../speed_test/backends/pytorch_vision_speed_benchmark_v2.py)，第 392–394 行及第 414–418 行。

只启用 `speed.pytorch_pipeline`、adapter 提供真实 `predict()`、未设置 source 时，条件分支直接略过测速。函数返回空列表，统一入口和一键入口将其视为成功，CSV 只有表头，用户请求的测速没有执行。

**实测：**用临时真实 adapter 文件（`predict()` 若被调用则抛出标记异常）、合成 checkpoint 和 CPU 配置运行 `run_tools.py --tool pytorch`。结果为 `pipeline_only_exit=0`、`csv_lines=1`、`predict_called=False`。

建议：对需要真实输入的 pipeline 提前要求 source，或返回明确的跳过状态和原因，并使调用入口据此呈现结果。

### R07 · [P2] 在有 FP/FN 且没有 TP 时输出 F1=0

位置：[model_diagnostics/diagnostics/engine.py](../../model_diagnostics/diagnostics/engine.py)，第 700–704 行。

当前先算 Precision/Recall，再用它们的调和平均计算 F1。TP=0、FP/FN 均大于零时，两者都是 0，最终走零分母分支返回 `None`。该情形下计数定义 `2TP/(2TP+FP+FN)` 的分母为正，F1 应为 0；现实现使汇总、逐图和阈值扫描报告出现空值。

**实测：**`_prf(0, 1, 1)` 返回 `{'precision': 0.0, 'recall': 0.0, 'f1': None}`。

建议：直接从 TP/FP/FN 计算 F1，并单独约定三者全零时的值。

### R08 · [P2] 用一次计数检查改名目标冲突

位置：[others/filename_transform/image_filename_converter.py](../../others/filename_transform/image_filename_converter.py)，第 234–235 行。

对每个目标调用一次 `targets.count(path)`，使改名前的冲突检测达到 O(N²)，无冲突的正常数据集也要完整重复扫描。数万张图的批量操作会长时间停在计划校验阶段。

**实测：**使用不同路径、无需改名且不触碰文件的计划，1,000 项耗时约 0.156 秒，3,000 项约 1.402 秒；输入增至 3 倍，耗时约增至 9 倍。该测量仅对应本机这两种规模，不是大型数据集实测。

建议：用 `Counter` 或集合完成 O(N) 重复检测，保留当前冲突报告。

### R09 · [P2] 按 UTF-8 BOM 的实际异常行为读取 COCO JSON

位置：[others/filename_transform/image_filename_converter.py](../../others/filename_transform/image_filename_converter.py)，第 192–195 行。

UTF-8 BOM 本身可以被 `utf-8` 正常解码；拒绝 BOM 的是随后的 `json.loads()`，抛出 `JSONDecodeError`。当前只捕获 `UnicodeDecodeError`，所以为 BOM 设置的 `utf-8-sig` 回退永远不会在这个场景执行，合法的带 BOM 标注文件被拒绝。

**实测：**将 `{"images": [{"file_name": "image.jpg"}]}` 以 `utf-8-sig` 写入临时文件，调用 `load_coco_json()` 得到 `JSONDecodeError: Unexpected UTF-8 BOM`。

建议：直接使用同时兼容有无 BOM 的 `utf-8-sig` 读取。

## 验证结果与边界

使用已有的 vtools 虚拟环境解释器执行仓库约定的四组回归；命令中的 `python` 在本轮替换为该解释器的实际路径：

| 检查 | 实际结果 |
| --- | --- |
| `QT_QPA_PLATFORM=offscreen PYTHONPATH=.:speed_test:model_diagnostics python -m unittest discover -s tests -q` | 109 项通过 |
| `PYTHONPATH=. python -m unittest discover -s model_compression/tests -q` | 42 项通过；首次受用户缓存目录的沙箱写入限制，获准重跑后通过 |
| `PYTHONPATH=.:speed_test python -m unittest discover -s speed_test/tests -q` | 52 项中 51 项通过、1 项跳过（已安装 torch，跳过面向无 torch 环境的用例） |
| `PYTHONPATH=.:model_diagnostics python -m unittest discover -s model_diagnostics/tests -q` | 13 项通过 |
| `python3 -m compileall -q .` | 通过 |
| `git diff --check` | 通过 |

合计 216 项回归，215 项通过、1 项跳过；既有回归通过不关闭上面的独立复现问题。本轮核验了直接 CLI、CPU 合成模型、临时图片/标签、原生数据扫描及 Linux 子进程停止；未执行真实权重训练/完整 mAP、GPU/TensorRT 构建与推理、K230 编译/板端、桌面交互或安装包构建。没有修改用户数据、权重或外部 Fork 源码，也没有提交或发布。
