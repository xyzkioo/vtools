# vtools 项目交接

最后修改：2026-11-03

## 本轮更新（2026-11-03）：v0.4.9 正式 Release

- [v0.4.9 Release](https://github.com/xyzkioo/vtools/releases/tag/v0.4.9) 已正式发布并标记为 Latest，非草稿、非预发布；GitHub `releases/latest` 已返回 v0.4.9。附件为 `vtools_0.4.9_amd64.deb` 与 `.deb.sha256`，上传后逐项核对大小和 GitHub SHA-256 摘要，再由草稿转为正式版。
- 发布标签 `v0.4.9` 指向源码提交 `6a643cca359c103ea0b7518fa821c4e40dde66c6`，包含 v0.4.8 审查修复及后续标签解码提示。安装包为 amd64、252,338,232 字节，SHA-256 为 `4820c85d7473ce9d58c0f0e8093d1d524024c9ff7fbabb3a1f47223c83646767`。详情见 `docs/releases/v0.4.9.md`；二进制产物仍由 Git 忽略，仅作为 Release 附件交付。
- Python、前端 package.json/package-lock.json 与 VERSION.md 已同步至 0.4.9，按前端、PyInstaller 目录包、Debian 包的顺序重新构建。目录包 `--help` 返回 0，内置运行时确认版本 0.4.9 与二进制标签的具体路径提示；Debian 解压流可完整读取，包内诊断引擎、共享模块、前端入口与源码逐字节一致。语法、差异空白、发布文档链接和本地校验文件检查通过。
- 本轮沿用业务修复提交前的 232 项回归结果（231 通过、1 项按环境条件跳过）；版本和打包调整未改业务逻辑。未重新安装本机桌面包，已安装版本仍是 0.4.8；未执行真实训练、GPU/TensorRT、K230 端到端或桌面自动更新交互。
- 用户已将规则限定为默认分支，管理员仅在 PR 合并时可绕过审批；PR #4 已于本日合并，#2、#3 已关闭。此次版本与交接同步使用 [发布 PR #5](https://github.com/xyzkioo/vtools/pull/5)。下方 PR 状态为各记录写入时的历史。
- 现有 GitHub 连接器不支持创建 Release 和上传二进制附件，因此通过官方 GitHub CLI 的设备授权登录 `xyzkioo` 完成发布。CLI 已安装到 `~/.local/bin/gh`，Git 登录辅助程序指向该持久路径；不在仓库中记录密码或令牌。

### 本轮文件索引

- `VERSION.md`、`vtools_ui/__init__.py`、前端 `package.json` / `package-lock.json`：0.4.9 版本同步。
- `docs/releases/v0.4.9.md`、`docs/README.md`、`packaging/README.md`：发行说明、校验值与构建边界。
- `dist/vtools_0.4.9_amd64.deb`、`dist/vtools_0.4.9_amd64.deb.sha256`：本地发布产物，不纳入源码提交。

---

## 本轮更新（2026-11-03）：统一 PR 评审入口

- [PR #4](https://github.com/xyzkioo/vtools/pull/4) 已改为以 `main` 为目标，包含完整修复、测试、版本与前端资源、审查文档和交接记录，共 2 个提交、49 个文件。已确认其包含先前审查文档 PR #2 和修复 PR #3 的内容，并关闭 #2、#3，统一审核 #4。
- 仓库规则同时保护修复分支和交接分支，要求 PR 及至少 1 名有写权限的审阅者批准，导致追加交接文档的提交也被阻止。PR 作者为 `xyzkioo`，作者不能自行批准自己的 PR。后续应由协作者审核，或由仓库所有者明确调整适合个人维护的审批规则；不通过继续创建 PR 追加状态文档。
- 本节是 PR 整理后的本地状态补记，尚未推送；待规则允许更新现有 PR 分支后追加到 PR #4。下方 PR #3 的状态是整理前历史。

---

## 本轮更新（2026-11-03）：修复提交与诊断输入排查

- 修复已通过 GitHub 连接器推送到 `fix/review-fixes-0.4.8`，代码提交为 `9ce9db3bea2a2d38725963737d2a42f332b31c1a`，并创建 [PR #3](https://github.com/xyzkioo/vtools/pull/3)，状态为 open、非草稿，尚未合并。`main` 保护规则要求通过 PR；命令行 Git 没有 HTTPS 凭据。远端修复目录树 `06f6f678da07fa0019fa56e38397f0f45697d6f4` 与本地已验证代码提交完全一致。
- 提交范围包含下方两份审查报告中的 14 项修复、相关测试与文档、v0.4.8 版本和前端资源，以及后续补充的 YOLO 标签解码错误路径提示。`.deb`、运行目录、数据集和权重未纳入源码提交，未发布 GitHub Release。
- 本轮回归：通用 114 项、压缩 44 项、测速 58 项、诊断 16 项；合计 232 项中 231 项通过、1 项按环境条件跳过。全仓 `compileall`、`git diff --check` 和本轮 15 个 Markdown 文件的本地链接检查通过。真实权重训练、GPU/TensorRT、K230 端到端路径未验证。

### 已确认的现场输入问题

- ironball 诊断失败在读取 GT 标签，而非模型加载。`best.pt` 可由同一外部环境和 Fork 加载；验证标签 62/62 个、训练标签 253/254 个无法按 UTF-8 读取，首个失败标签为 `labels/val/IMG20260724155011.txt`，开头含二进制字节。附近的 `yolo_dataset.zip` 与这 316 个标签文件名没有匹配项，无法据此恢复标注。
- cherry 的 `cherry_tile640/weights/best.pt` ZIP 成员校验失败，`epoch0.pt` 也有成员损坏；`epoch50.pt`、`last.pt` 无法读取 ZIP 目录。另有 `yolo26s.pt`、`lichi_sahi.pt`，同一 Fork 实际加载均失败。新数据集验证标签 93 个均可按 UTF-8 读取；这只确认解码通过，不代表已完成标签语义或模型端到端检查。
- `model_diagnostics/diagnostics/engine.py` 的两种 YOLO GT 读取路径共用 `_read_yolo_label_lines()`，解码失败时报告具体标签路径并继续保持失败返回码；新增回归覆盖二进制标签。复现使用临时运行目录，未改写原始数据和权重。
- 已安装包仍为 0.4.8；标签解码提示是构建后的源码补充，未包含在现有 `.deb` 中，`docs/releases/v0.4.8.md` 已说明。

### 待办与文件索引

1. PR #3 按仓库保护规则完成评审后合并；较早的文档 PR #2 本轮未修改。
2. 现场诊断需要从原始标注重新导出 ironball 标签，或提供完整的 cherry 权重备份，再执行真实诊断。不要用跳过损坏标签或伪标签替代真值。
3. 如需把标签错误提示交付到桌面安装版，按版本规则递增补丁版本后重新构建与安装。

- `model_diagnostics/diagnostics/engine.py`、`model_diagnostics/tests/test_modules.py`：标签解码提示与回归。
- `model_diagnostics/README.md`：UTF-8 标签输入要求。
- `docs/releases/v0.4.8.md`：已安装包与后续源码的范围区别。

---

## 本轮更新（2026-11-02）：环境检验模型输入误填排查

- 用户提供的安装版环境检验日志中，ONNX 1.23.2 与 TensorRT 11.2.1.2 均可导入，唯一失败是 `模型构建/前向` 的 `KeyError: 'backbone'`。
- 最新任务日志 `~/.vtools_ui/logs/c20ad6772d794deda51bebe183f8fdff.log` 的启动命令把数据集 `ironball/dataset.yaml` 传给了 `--weights`。只读取该 YAML 的键，确认为 `names`、`nc`、`path`、`train`、`val`，不含模型结构所需的 `backbone`、`head`。`env_test/check_install.py` 会对 `--weights` 调用 `YOLO(weights_path)`，Fork 将 `.yaml` 作为模型结构处理并因此报错。
- 2026-10-30 的另一条日志 `~/.vtools_ui/logs/ef2e3283e29a4810b84e57ea02641201.log` 使用真实 `.pt`，同一环境的默认 `yolo26.yaml` 构建/前向和权重预测均通过；本轮另用临时缓存直接构建 Fork 的默认 `yolo26.yaml` 成功。故当前错误不能说明 ONNX、TensorRT 或 Fork 安装失败。
- 用户可在工作台“环境检查”清空“模型权重”以仅测试默认模型结构，或填写真实 `.pt`；数据集 YAML 应用于数据集相关工具。本轮只排查和解释，未改源码、重新打包或重新安装，已安装版本仍为 0.4.8。

---

## 本轮更新（2026-11-02）：v0.4.8 本机构建与安装

- 按仓库发布约定将 `vtools_ui/__init__.py`、前端 `package.json` / `package-lock.json` 和 `VERSION.md` 同步提升至 `0.4.8`，记录用户可见修复。前端、PyInstaller 桌面目录包和 Debian 安装包按顺序重新构建。
- 打包脚本只收集 Git 已跟踪工具源码，因此仅将上轮新增的 `vtools_runtime/module_selection.py` 加入暂存区，确保本次安装包包含它；其余上轮修复和文档仍留在本地工作树。未提交、推送或发布 GitHub Release。
- 包路径：`dist/vtools_0.4.8_amd64.deb`；Debian 元数据为 `Package: vtools`、`Version: 0.4.8`、`Architecture: amd64`；大小 252,338,156 字节；SHA-256 为 `33f1492e3bea49ed952b12684d9f2b0be7220f8edd4948d75762378b875b8fcf`。详情见 `docs/releases/v0.4.8.md`。
- 本机已通过系统图形管理员授权从已安装的 `0.4.7` 升级到 `0.4.8`，`dpkg-query` 显示 `install ok installed`。已确认安装目录包含共享模块与前端资源；安装版 `--help` 返回 0，短时间启动图形工作台时输出“vtools 工作台窗口已显示。”，15 秒后测试进程自动结束。
- 真实权重训练、GPU/TensorRT、K230 和完整桌面任务交互仍未执行。此前修复回归为 231 项中 230 项通过、1 项跳过；本轮版本与打包改动没有重新执行全部 Python 回归。

### 本轮文件索引

- `VERSION.md`、`vtools_ui/__init__.py`、`vtools_ui/webapp/frontend/package.json`、`package-lock.json`：版本同步。
- `docs/releases/v0.4.8.md`：包元数据、校验值、安装和启动证据。
- `dist/vtools/`：本地桌面目录包；`dist/vtools_0.4.8_amd64.deb`：本地安装包，均不属于源码提交内容。

---

## 本轮更新（2026-11-02）：修复两份审查报告中的确认问题

- 修复范围为 2026-11-01 两份报告的 R01–R09、S01–S05，共 14 项确认问题。静态扫描列出的无内部调用函数和未使用导入等仅为清理候选，本轮没有据此删除公开名称。
- 检测压缩现在把验证、结构化微调和蒸馏所需的标准 YOLO 图片与标签复制到本次 `runN/_input/`，不再用硬链接；无法隔离的布局会明确报错。分类默认 ImageFolder 加载器拒绝 train/val 类别编号不一致。
- 直接脚本入口在导入 `vtools_runtime` 前建立仓库路径。桌面停止任务等待整个 POSIX 进程组并对存活成员补发 SIGKILL。pipeline 测速缺真实输入时报错；F1 在 TP=0 且 FP/FN>0 时为 0；改名计划用一次计数查重，COCO JSON 支持 UTF-8 BOM。
- checkpoint 旧开关从默认 YAML 和 UI 字段说明移除，旧 YAML 字段会报错；检查只由 `modules.checkpoint.*` 控制。失败行保留到 CSV/汇总，`benchmark.fail_on_checkpoint_inspection: true` 停止当前后端及一键入口的后续阶段。诊断入口拒绝跨工具模块；TensorRT 后端删去不可达 PyTorch 计时分支与无效选项。四处模块选择基础规则归入 `vtools_runtime/module_selection.py`；层清单只写一次；移除 `mode_b.task` 残留。
- 同步更新根目录和子项目 README、`docs/modules/`、CSV 指南、前端源码及已跟踪的前端构建产物。两份 2026-11-01 审查报告是当日历史基准，不改写为当前问题清单。

### 已验证 / 证据

- 通用测试 114 项、压缩 44 项、测速 58 项（1 项环境条件跳过）、诊断 15 项；合计 231 项中 230 项通过、1 项跳过。新增回归覆盖输入副本、分类映射、进程组停止、checkpoint 失败、无输入 pipeline、跨工具模块、F1、BOM、重复改名及层清单写入。
- `python3 -m compileall -q .`、`git diff --check`、前端 `npm run build` 均通过。清空 `PYTHONPATH` 后的诊断、PyTorch 测速和 NDJSON 直接脚本实际运行进入业务错误路径，没有 `vtools_runtime` 导入错误。
- 本轮未执行真实权重的结构化微调/检测蒸馏、GPU/TensorRT、K230 或桌面交互。前端仅通过 TypeScript/Vite 构建，未构建安装包。

## 关键决策

- 不支持安全隔离的检测数据布局直接失败，避免原图修复或缓存扫描写入用户输入；独立拷贝增加磁盘占用，已在压缩文档说明。
- 分类映射不一致时拒绝验证，而非猜测缺失类别的映射；自定义 adapter DataLoader 不受默认 ImageFolder 校验影响。
- 当前工作树仍在 `main`，业务修复尚未本地提交、推送或发布；2026-11-01 交接记录中的 PR #2 是审查文档 PR，本轮未重新查询其远端状态。

## 待办（按优先级）

1. **现场验证**：用用户提供的真实标准 YOLO 数据与权重完成结构化微调/蒸馏和 GPU/TensorRT 路径，确认资源占用与报告；此项依赖相应数据和硬件。
2. **可选清理**：确认公开接口使用范围后再处理审查报告中无内部调用函数、未使用导入等清理候选。

## 文件索引（2026-11-02 状态）

- `docs/reviews/vtools-review-2026-11-01.md`、`docs/reviews/vtools-redundancy-2026-11-01.md`：历史审查证据与问题编号。
- `model_compression/core/detection.py`、`model_compression/core/model_runtime.py`：输入隔离与分类映射。
- `speed_test/backends/`、`speed_test/run_all.py`、`vtools_runtime/module_selection.py`：测速失败传播、后端分工与模块选择。
- `vtools_ui/webapp/tasks.py`、`model_diagnostics/diagnostics/modules.py`、`model_visualization/run_visualization.py`：进程组停止、诊断模块边界、层清单。
- `tests/`、`model_compression/tests/`、`speed_test/tests/`、`model_diagnostics/tests/`：相关回归。

---

## 历史记录（截至 2026-11-01）

## 本轮更新（2026-11-01）：GitHub 文档 PR

- 用户授权先向 GitHub 提交 PR。本轮提交两份审查报告、`docs/README.md` 导航及本交接文档；业务修复继续列为待办。
- 仓库：`xyzkioo/vtools`；目标分支 `main`；远程分支 `docs/project-audit-2026-11-01`。创建前确认远端 main 与本地审查基准均为 `cc39d05b9bf1f45b79744c1dbcf2ecc40f54e32f`。
- 两份报告合计记录 14 项问题（2 项 P1、9 项 P2、3 项 P3）及单独标注的清理候选，证据和硬件验证边界保留在报告中。
- 本轮文档空白、私有绝对路径和 47 个本地 Markdown 链接检查通过；业务代码未变，未重新执行上轮回归。
- GitHub 状态：已创建 [PR #2：记录项目功能缺陷与代码冗余审查](https://github.com/xyzkioo/vtools/pull/2)，状态为 open、非草稿，尚未合并。审查材料首个远程提交为 `8d237321750abcabc7b1767f7132c0fb36ede3b8`，随后补充本节的 PR 结果记录。
- 通过 GitHub 连接器操作远程分支与提交，本地工作树仍在 `main`，审查文档尚未提交到本地 Git 历史。

---

## 历史记录（同日的冗余补查）

## 本轮更新（2026-11-01）：冗余与失效代码补查

- 用户追问上轮是否检查代码冗余。上轮重点是功能缺陷，本轮补充应用 `review-agent`，完成 117 个 Python 文件的 AST 候选扫描、全仓引用检索和重点控制流验证。
- 新报告：[冗余补充审查](docs/reviews/vtools-redundancy-2026-11-01.md)，确认 2 项 P2、3 项 P3：旧 checkpoint 配置失效且检查失败可丢失；诊断接受不会执行的跨工具模块；TensorRT 后端 PyTorch 分支不可达；四份模块选择基础规则重复；层检查重复写 layers.csv。
- 另列 5 个仓库内无调用函数、13 个 Python 未使用导入名、前端 LoaderCircle 导入、两组精确函数体重复及 mode_b.task 残留。无内部调用不等于没有外部使用，尚未删除公开 API 或模板钩子。
- 实测/故障注入：诊断接受 speed.pytorch_call 后返回 0 且 aggregate 为空；fail_on_checkpoint_inspection=true 注入检查失败后仍返回模型信息行且没有失败行；层检查对 layers.csv 写入 2 次。业务代码未改，本轮没有重跑上轮 216 项回归。
- 本轮交付更新为补充报告、原报告的关联链接、docs 索引和本交接。上轮 9 项功能缺陷与本轮补充问题均尚未修复；未提交或发布。

## 待办（当前）

1. 优先处理原报告 R01/R02：输入隔离与直接脚本入口；其余原报告 R03–R09 仍待修复。
2. 处理补充报告 S01/S02 的失效配置与虚假成功，再清理已确认不可达分支、重复写入及无消费者旧模式字段。
3. 结合公开接口约定确认无调用函数的清理范围；共同规则可复用，领域默认值和依赖校验保留在 owner。

## 文件索引（当前）

- `docs/reviews/vtools-redundancy-2026-11-01.md`：本轮补充问题、清理候选和验证边界。
- `docs/reviews/vtools-review-2026-11-01.md`：上轮 9 项功能问题及回归结果。
- `docs/README.md`：两份审查报告入口。

---

## 历史记录（同日的功能审查）

## 本轮更新（2026-11-01）：项目代码审查

- 用户要求使用 skill 审查项目；本轮应用本地 `review-agent` 技能，以当前 HEAD `cc39d05`（v0.4.7）为基准检查当前代码，不限定单次提交差异。
- 完整清单见 [项目审查报告](docs/reviews/vtools-review-2026-11-01.md)：确认 9 项缺陷，2 项 P1、7 项 P2。业务代码尚未修复。
- P1：检测验证/蒸馏镜像图片使用硬链接，上游 JPEG 修复可写回原图；直接运行诊断、测速和 NDJSON 转换入口时，未预设 `PYTHONPATH` 会在共享模块导入处失败。均用临时样例复现。
- P2：结构化微调传入原始数据 YAML，原生扫描写源目录缓存；停止任务遗留忽略 SIGTERM 的子进程；分类 train/val 类别映射不一致；缺 source 的 pipeline-only 测速返回空结果且退出 0；无 TP 时 F1 错误为空；改名目标检测为 O(N²)；带 UTF-8 BOM 的 COCO JSON 读取失败。
- 四组回归合计 216 项：215 项通过、1 项跳过。压缩组首次因用户缓存目录只读而失败，经用户批准在沙箱外重跑，42 项通过。全仓 compileall 与 diff 空白检查通过。
- 本轮修改范围为审查报告、`docs/README.md` 和本交接文档；未提交、发布或修改输入数据、权重、外部 Fork。测试使用的子进程已清理。

## 待办（按优先级）

1. 修复审查报告 R01/R02：保证输入文件隔离，恢复未设置 `PYTHONPATH` 的公开直接脚本入口。
2. 处理 R03–R09，并按报告中的具体触发条件验证；现有测试通过不能作为这些问题已关闭的依据。
3. 真实训练/完整 mAP、GPU/TensorRT、K230 和桌面交互仍需现场验证，本轮未执行。

## 文件索引（当前）

- `docs/reviews/vtools-review-2026-11-01.md`：本轮缺陷、源码位置、独立复现结果与验证边界。
- `docs/README.md`：审查报告导航入口。
- 下方历史记录保留上轮运行日志查询的证据位置。

---

## 历史记录（此前的 2026-11-01 更新）

## 本轮更新（2026-11-01）：运行记录与日志查询

- 已通过代码与 README 确认：桌面工作台左侧“记录与设置 → 任务记录”提供历史状态、配置、结果入口和“查看日志”；底部日志面板支持复制、打开文件和打开目录。
- 源码运行的状态目录为仓库内 `.vtools_ui`；安装版默认为 `~/.vtools_ui`，可由 `VTOOLS_USER_DATA` 覆盖。
- 本次读取安装版默认目录的 `history.json` 与 4 份任务日志。最近记录为 2026-10-31 00:48；3 次数据集检查历史记录分别报告 316、4405、80 项问题，状态为 `issues_found`，不代表程序崩溃或数据集当前仍有同样的问题。
- 环境检查历史日志核心检查通过，存在 `No module named pip` 警告；该日志的“依赖冲突”提示本身不能证明存在实际依赖冲突。读取的任务日志没有崩溃堆栈。
- 本轮仅新增交接文档，未运行程序或复测模型；未提交、发布或修改输入数据。

## 文件索引

- `vtools_ui/webapp/frontend/src/App.tsx`：任务记录页面和日志面板。
- `vtools_ui/webapp/core.py`：状态目录、历史记录配置。
- `vtools_ui/webapp/tasks.py`：任务执行与日志落盘。
- `vtools_ui/README.md`：日志功能使用说明。
- 安装版本地证据：`~/.vtools_ui/history.json` 与 `~/.vtools_ui/logs/`，不属于源码提交内容。

## 待办与限制

- 当前问题为功能查询，无待实施代码变更；若继续排查故障，先定位具体任务与时间。
- 本轮结论来自历史文件和源码阅读，未执行桌面交互或硬件端到端验证。
