# vtools 版本记录

## 当前版本

- 当前版本：`0.5.10`
- 更新批次：`2026.11.08`
- 功能标识：`env-check-compression`
- 版本性质：环境检查新增压缩环境检查（警告级）
- 适用仓库：`xyzkioo/vtools`

### v0.5.10

- 环境检查新增「压缩环境」项：命令行 `--check-compression`，桌面工作台「环境检查」页面新增「检查压缩环境（torch-pruning）」勾选框。
- 该项检查模型压缩的可选依赖 `torch-pruning` 能否导入、版本是否满足 `>=1.4.1`，并报告实际文件与解释器，避免把其他环境的安装误判为可用。
- 属于警告级：未安装或版本过低只提示警告并说明「scale 结构缩放、量化和蒸馏不受影响」，不改变 Python、核心依赖、CUDA 和模型检查的结论；只有加 `--strict` 才判为失败。

### v0.5.9

- 修复 0.5.8 更新器在 GitHub 未认证接口限流（60 次/小时/IP）后直接报 403、两条版本行一起消失的问题。
- 发现版本改为 API 优先、失败回退 `releases.atom` + 资产 HEAD + `<资产>.sha256` 旁证文件；atom 与下载路径不走 REST 配额。
- 两条路都失败时使用 24 小时内的本地缓存结果，界面提示“显示 x 小时前缓存的结果”，不再直接中断。
- 回退路径命中即停，实测耗时从 30 秒降到约 7 秒；403/429 文案改为「GitHub 接口访问次数受限」并说明已切换来源。
- 界面新增来源提示行：使用旁证或缓存时明确告知；API 正常时不显示。

### v0.5.8

- 更新器不再只支持 Linux：同时解析 Windows 安装包与 `.deb`，两个平台各自比较版本，一个平台没有安装包不会阻塞另一个。
- “设置 → 软件更新”改为两行显示 Windows x64 与 Linux (.deb) 的最新可用版本及各自状态（已是最新 / 可更新 / 尚无安装包 / 校验信息缺失）。
- Windows 采用半自动安装：下载安装包、校验 SHA-256 与大小后调起 Inno Setup 安装向导，由用户确认完成，不再依赖 pkexec/apt。
- 版本比较改为纯 Python 数字段比较，Linux 端也不再调用 `dpkg --compare-versions`；资产选择继续要求 GitHub 提供的 SHA-256 摘要，缺失摘要的版本不提供安装。
- 跳过草稿版与预发布版；只采纳与平台命名完全一致的资产（Windows 用连字符、deb 用下划线）。

### v0.5.7

- 环境检查默认读取所选 Python 已安装的包和实际版本，不再固定 Ultralytics 8.4.128 或自动选择本地 Fork。
- 仅检查用户指定的模型 YAML 或权重，不再自动附加 YOLO26 模型检查；未执行模型检查时明确提示。
- 保留核心依赖导入、显式源码路径和模型运行失败的判定；精确版本比较由 --expected-version 显式启用。
- 同步环境检查页面说明与向导，保留 Windows 外部 Python 3.10/3.12 隔离和安装前页面验收。

### v0.5.6

- 修复安装版调用外部 Python 3.10 时误载内置 Python 3.12 扩展的 ABI 冲突。
- 工具脚本与配置改放 `_internal/tool-sources/`，外部 Python 仅导入纯源码，不再把二进制运行库目录加入模块路径。
- 启动外部任务和环境扫描时清理内置 Python/Qt 路径，临时恢复 Windows 默认 DLL 搜索并在创建子进程后恢复桌面设置。
- 正确识别 Windows Conda 根目录和 Library/bin；内置任务继续使用独立后台程序，任务日志保持 UTF-8。

### v0.5.5

- Windows 安装前读取所选目录的完整性标签，拒绝系统禁止渲染组件运行的低完整性位置；不修改标签、安全策略或用户所选路径。
- 安装器先在临时区解包并实际加载 React 页面，校验所选用户数据目录、本机 API 与 Qt 原生桥接，通过后才开始部署和创建快捷方式；验证失败在准备阶段中止。
- 直接启动低完整性程序时给出明确原因；渲染退出记录状态与退出码，保留同目录 user-data 及后台任务日志。
- Windows 仅提供当前用户安装，避免所有用户安装到只读目录后无法保存同目录数据。

### v0.5.4

- 桌面窗口改用直接调用的 PySide6 Qt WebEngine，保留 React 界面，移除 pywebview/pythonnet/.NET 依赖链。
- 文件选择与剪贴板通过 Qt WebChannel 在 GUI 线程处理；浏览器预览仍支持手工路径输入。
- Windows 默认将设置、日志、任务结果及网页缓存保存到安装目录旁的 user-data，可安装到可写的 D 盘目录；卸载保留用户数据。
- Windows 构建使用标准 CPython venv、固定依赖和明确的双入口 PyInstaller 配置，不再从 Conda 扫描补拷 DLL。

### v0.5.3

- 修复 Windows 启动时日志目录拒绝访问导致的未捕获异常。
- 默认用户目录不可写时，统一切换到当前进程的 LOCALAPPDATA/vtools，设置、日志和后台任务共用同一路径；原目录内容保留。
- 显式指定 VTOOLS_USER_DATA 时保留该选择；目录仍不可写会显示明确错误，日志初始化也受启动错误处理保护。

### v0.5.2

- Windows 工作台使用图形入口直接启动，不再显示命令行窗口。
- 内置工具使用独立后台任务程序，保留任务日志并隐藏子进程控制台。
- 启动失败显示错误提示，详细信息写入用户目录的 .vtools_ui/logs/desktop.log；受限安装目录建议重新安装到默认用户目录。
- 修复 Windows 桌面桥接暴露原生窗口对象导致的递归扫描及 COM 线程错误。

### v0.5.1

- 新增 Windows x64 安装程序，支持当前用户安装、开始菜单入口和可选桌面快捷方式。
- Windows 程序与安装包使用统一图标，补齐 Conda 打包所需的动态库。
- 保留 v0.5.0 的 NDJSON 下载配置功能；模型任务继续使用外部 Python 环境。

### v0.5.0

- NDJSON → YOLO 新增“下载配置”弹窗，可保存下载并发数、自动重试开关、失败后重试次数与单次超时。
- 工具默认并发降至 4，失败后自动重试 4 次，单次超时 120 秒；关闭自动重试后只尝试一次。
- 配合更新后的外部 Ultralytics Fork 使用可配置下载与临时文件写入，重跑复用已有图片；旧版 Fork 明确提示更新。
- 本地安装包和验证范围见 [v0.5.0 构建说明](docs/releases/v0.5.0.md)。Ultralytics Fork 仍由外部环境提供，不随安装包内置。

### v0.4.9

- YOLO 标签无法按 UTF-8 读取时，诊断明确报告具体标签路径和重新导出建议，避免只显示无文件上下文的解码异常。
- 标准数据集 YAML 与旧式 YOLO 标签目录共用标签读取逻辑，失败继续返回非零状态。
- 安装包包含 v0.4.8 的审查修复及上述后续补充，同步更新版本与发布说明。
- amd64 安装包信息、校验值与验证边界见 [v0.4.9 发行说明](docs/releases/v0.4.9.md)。

### v0.4.8

- 检测压缩验证、结构化微调和蒸馏使用独立图片与标签副本，避免上游修复或缓存扫描写回输入数据；分类训练与验证类别映射不一致时明确报错。
- 修复直接脚本导入、停止任务遗留子进程、无输入 pipeline 空跑、checkpoint 检查失败丢失及严格失败策略；一键测速按后端执行各自模块。
- 诊断拒绝跨工具模块，修正零 TP 的 F1，支持带 BOM 的 COCO 标注，优化批量改名冲突检查；清理 TensorRT 不可达分支与过时配置。
- 统一模块选择基础规则，修复层清单重复写出；前端配置说明与打包资源同步更新。
- 本机构建 amd64 `.deb` 并从 0.4.7 升级安装；图形工作台窗口启动验证通过。包大小、SHA-256 与验证边界见 [v0.4.8 构建记录](docs/releases/v0.4.8.md)。

### v0.4.7

- 检测蒸馏先校验教师与数据集的类别名称和顺序；通用预训练学生按目标类别重建检测头，同名类别输出按名称迁移。
- 适配后的初始化保存为 `student-initialized.pt`，学生基线、训练和产物血缘均使用该检查点；输入权重保持不变，无法忠实重建的已剪枝主干明确拒绝。
- 压缩页面改为按任务选择单一处理模式，过滤隐藏字段旧值；纯检测蒸馏无需公共模型权重，公共参数统计和基线标记为跳过，学生自身评估照常执行。
- 保留 CLI/YAML 的显式模块组合和空选择语义；结构化微调和蒸馏共用用户级 AMP 检查权重缓存。
- 验证：通用回归 109 项、压缩回归 42 项通过；源码和打包版页面均完成 80→16 类学生的一轮 CPU 蒸馏，输入哈希保持不变。前端、桌面目录包和 amd64 deb 构建及包内容核对通过；未执行完整 GPU 训练或 GPU AMP 数值比较。安装与验证边界见 [v0.4.7 说明](docs/releases/v0.4.7.md)。

### v0.4.6

- 新增检测知识蒸馏模块，使用支持 `distill_model/dis` 的外部 Ultralytics Fork 训练学生模型。
- 工作台提供教师和学生权重选择、检测蒸馏开关及训练参数说明；统一命令行转发权重参数。
- 校验教师/学生检测头、类别数和特征层兼容性；训练后重新加载独立学生模型，记录参数量与蒸馏前后精度。
- 数据集镜像及训练缓存保留在本次运行目录，避免改写输入数据缓存；分类任务禁用检测蒸馏。

### v0.4.5

- 更新 Linux 桌面安装包，集中交付“全部字段”按功能分组、剪枝与蒸馏配置独立显示、字段中文帮助和输入尺寸类型修复。
- 保留 YAML 原文编辑模式；图形化字段说明只显示在界面中，不写入 YAML 配置文件。

### v0.4.4

- 图形化配置移除“常用配置”，全部字段按功能分组（包含基线评估、非结构化剪枝、结构化剪枝、量化、知识蒸馏等），保留 YAML 原文视图。字段中文说明不写入 YAML。
- 修复输入 `640,640` 后尺寸被保存为字符串的问题；评估/测速尺寸保存为整数列表，微调训练和剪枝依赖图的方形尺寸保存为单个整数。非法尺寸会阻止保存并显示错误。
- 数字列表保持数值类型，文本列表保留原文本；切换配置视图、保存后重新读取保持一致。

### v0.4.3

- 模型压缩子进程异常退出且没有结果摘要时，工作台父进程补写失败报告，记录退出码、终止信号和最后执行阶段，并提供结果目录入口。

### v0.4.2

- Linux 桌面日志复制通过 Qt 主线程写入系统剪贴板并读回校验，解决网页剪贴板权限导致的复制失败。
- 日志面板显示保存路径，提供“打开文件”和“打开目录”；读取历史日志的提示明确说明内容显示在底部面板。

### v0.4.1

- 修复工作台运行详情的日志复制，任务记录可重新打开已保存日志；运行日志包含启动命令及完整子进程输出。
- 修复模型压缩 `--enable`、`--disable`、`--only` 覆盖后 `effective_config.json.modules` 与实际执行不一致的问题。
- 结构化剪枝低显存时提前提示风险，按评估、测速、依赖图追踪和微调阶段记录显存及失败位置，并及时释放每轮依赖图引用。

### v0.4.0

- 将读取已有预测文件的检测诊断纳入安装版内置运行时；保留 NDJSON 数据集转换及模型计算的外部依赖，并在代码、界面和文档中说明功能边界。
- 修复桌面启动器解析 uv 解释器符号链接，以及 uv 任务继承旧 Conda 环境标识的问题；模型任务不能误用工作台自身的轻量解释器。
- 首次生成可编辑配置副本时保留模板 `project.root` 的实际位置，避免复制后相对输入路径改变；已有用户配置不自动改写。
- JSON 配置经图形编辑后仍按 JSON 保存，保证按后缀选择解析器的诊断后端可读；修复压缩文档指向已删除 Spec 的链接。
- 修复诊断、测速、压缩和可视化的显式空模块选择；全部关闭可视化模块后不再隐式执行前向传播。
- 修复非对象空值 YAML 被当作默认配置、未实现验证模块静默成功、严格依赖检查执行异常仅发警告和更新检查失败后仍显示正在检查的问题。

### v0.3.29

- 安装版内置 Pillow 和 OpenCV，并用包内运行时执行图片缩放、视频抽帧、文件名转换和数据集质量检查，常用数据工具不再依赖另配 Python 环境。
- PyTorch、Ultralytics、TensorRT 和 nncase 仍按需使用外部模型环境，避免把多 GB、硬件相关依赖打进安装包。

### v0.3.28

- Linux 安装版首次启动时，若检测到 `~/python-envs/vtools/.venv`，自动将其设为模型任务解释器，省去首次手动切换环境。
- 保留已保存的解释器选择；没有可用 vtools uv 环境时，继续使用系统 Python 并可在设置中切换。

### v0.3.27

- 工作台环境选择器自动发现 `~/python-envs/<项目>/.venv` 中的 uv 环境，也支持通过 `VTOOLS_UV_PROJECTS_ROOT` 配置扫描目录。
- 运行任务时保留 uv 虚拟环境解释器路径，避免符号链接解析到系统 Python 后丢失环境依赖。
- 环境检查支持 uv 依赖一致性检查，并补充 uv 环境使用说明。

### v0.3.24

- 修复测速状态分类、可移植默认路径、NDJSON Fork 发现、普通目录可视化、压缩验证缓存隔离和 K230 模拟器临时文件污染。
- 完成 CUDA、TensorRT 和 K230 软件模拟器实测；TensorRT FP16 在完整验证集上的 mAP 与 PyTorch FP16 持平。
- 删除已被 React/pywebview 工作台取代的旧 PySide 界面、QProcess runner 和专属测试；设置、任务、历史与结果逻辑只保留当前 Web 工作台实现。
- 删除未被主流程使用的跨运行压缩注册表和 branch 接口；连续压缩只维护本次运行内的当前模型与产物血缘。
- K230 转换命令行只保留输出目录参数，自动生成同名 ONNX 和 kmodel。

### v0.3.25

- 工作台运行目录改为 `run1-askpk` 形式，保留递增序号并缩短随机后缀，方便查看结果。
- K230 校验支持选择图片目录批量比较，并为安装版验证进程自动发现 Conda 共用的 .NET hostfxr 运行时。

### v0.3.26

- 延续全项目审阅修复，发布批量 K230 校验、hostfxr 可移植性和短运行目录支持的正式安装包。

### v0.3.4

- 修复数据集质量检查：发现越界框等问题时，工作台显示“检查完成（发现问题）”，不再误报为任务执行失败。
- 更新器版本提升，确保修复版安装包可以被 0.3.3 正确识别并安装。

### v0.3.5

- 修复冻结版桌面包运行检测诊断时找不到 `vtools_runtime` 的问题。
- 将运行时发现模块随安装包展开到 `_internal`，外部 Python 环境可以正常加载模型适配器。

### v0.3.6

- 调整 WebView 根节点、主布局和运行详情栏的高度与背景绘制；长表单滚动导致的底部白框仍需后续版本修复。

### v0.3.7

- 打包版检测诊断会从所选数据集和权重所在工作区发现并加载同级 Ultralytics Fork。
- 工作台明确区分检测诊断与独立的数据集质量检查入口。
- 桌面窗口统一视口高度和宿主背景；长表单滚动仍可能带动整个页面，后续版本继续修复。

### v0.3.8

- 修复长表单点击运行后，整个网页被滚动并在日志面板下方露出大块白色区域的问题。
- 将外层工作台固定在窗口内，只允许表单内容和侧栏列表各自滚动。

### v0.3.9

- 修复安装版选择外部 Conda Python 后无法导入内置 `vtools_runtime` 的问题。
- 任务启动时同步所选环境的可执行文件、动态库和 Conda 前缀，并清理 PyInstaller 遗留的动态库搜索路径。

### v0.3.10

- 修复安装版从已保存环境切走后，Conda 环境从下拉列表消失的问题；即使桌面启动器的 `PATH` 中没有 `conda`，仍会读取 Conda 环境登记文件和常见安装位置。
- 环境列表不再把 `/opt/vtools/vtools` 工作台程序误当成可运行模型任务的 Python 环境。

### v0.3.11

- 安装版的可视化、测速和压缩工具会从权重、数据集等输入路径发现同一工作区中的 Ultralytics Fork，不再只在 `/opt/vtools` 附近查找。

### v0.3.12

- 性能测速会在 Fork 搜索前应用 `--model` 权重覆盖；安装版的 ONNX、TensorRT engine 和汇总报告统一写入本次可写运行目录，不再尝试写入 `/opt/vtools/runs-profile`。

### v0.3.13

- 性能测速执行正常但一致性检查发现数值或检测框差异时，工作台显示“已完成（有差异/问题）”，不再误显示为程序运行失败；真正的依赖、导入和写入错误仍显示失败。

### v0.3.14

- 修复安装版执行结构化模型压缩时，Ultralytics AMP 检查尝试向只读安装目录的 `weights/` 下载临时模型而失败的问题；缓存改为写入本次运行目录。

### v0.3.15

- 更正结构化压缩的 AMP 缓存重定向位置：同步覆盖 Ultralytics `utils.WEIGHTS_DIR` 与下载器内存配置，并在任务结束后恢复，解决 0.3.14 仍写入相对 `weights/` 的问题。
- Debian 包改为在临时路径完整构建后原子发布，避免中断或同名覆盖留下 APT 无法解析的半成品和尾部数据。

### v0.3.16

- 工作台路径输入框补充文件/目录选择说明：区分单文件、目录、文件或目录以及输出文件位置，并提示常见格式和目录要求；不改变任务运行逻辑。

### v0.3.17

- 为模型可视化、性能测速和模型压缩的路径字段补充具体输入规则：测速目录只需放图片，不要求完整 YOLO 数据集；检测压缩使用 `data.yaml`，分类训练/验证目录使用按类别分组的图片子目录。

### v0.3.18

- 修复安装版图形化配置保存权限错误：配置模板首次打开时复制到 `~/.vtools_ui/configs/`，编辑和后续运行使用用户可写副本，不再向 `/opt/vtools/_internal` 创建临时文件。

### v0.3.19

- 全面收口安装版的可写路径：数据集质量报告、K230 转换的默认产物和模型工具默认配置均改用用户目录；旧设置若仍指向安装目录会自动迁移到 `~/.vtools_ui/runs/`。
- 直接保存安装包内配置时也自动创建用户副本；显式输出路径指向安装目录会给出清晰提示，避免再次出现权限失败。

### v0.3.20

- K230 转换界面和命令行统一为一个输出目录，自动按权重文件名生成 `.onnx` 和 `.kmodel`；已删除重复的旧输出文件参数。
- K230 转换复用 vtools 的 Ultralytics Fork 发现机制，可自动加载同级 `ultralytics-cn` / `ultralytics-ezcn` 源码，不再要求所选 Python 环境另外安装 PyPI 版。

### v0.3.21

- 修复桌面包使用缺少 PySide6/pywebview 的 Python 构建后无法启动的问题；构建脚本现在会在依赖不完整时提前失败，并要求使用完整桌面环境重新构建。

### v0.3.22

- K230 转换自动探测 Conda 或系统 .NET 运行时并设置 `DOTNET_ROOT`，修复从桌面工作台启动时 nncase 报 `Failed to get hostfxr path` 的问题。

### v0.3.23

- 环境检查会从所选权重或数据集路径自动发现同级 `ultralytics-cn` Fork。
- 误把数据集 `data.yaml` 填入模型结构 YAML 时给出明确提示并继续检查权重，不再因 `KeyError: backbone` 误报模型失败。

### v0.3.3

- 新增独立的数据集质量检查工具，扫描坏图、标签问题、尺寸分布、重复图片和数据划分泄漏。
- 工作台新增“数据集质量检查”入口，可选择数据集 YAML、报告目录和抽样图数量。
- 结果统一写入独立运行目录，包含 `summary.json`、`issues.csv`、`report.md` 和抽样图。

## 更新批次：`2026.09.15`

- 持续验收：使用已安装 PyTorch、PySide6、Ultralytics 的 `yolo` 环境运行压缩与真实 Qt 控件测试；系统 Python 缺少这些包不再作为整个机器的环境结论。
- 修复：统一入口拒绝非法 `tools.yaml`，不再静默改跑诊断；K230 转换延迟加载可选依赖、校验数值参数、稳定选择校准样本并原子发布 Kmodel。
- 修复：图片/视频编码失败保留旧输出，批量格式转换预检目标冲突；COCO 与注册表写入、改名回滚和 UI 历史记录增强事务保护。
- 验证：当前通用回归测试 38 项、模型压缩测试 21 项、诊断测试 6 项通过；测速测试 44 项通过（其中 1 项按环境跳过）。全仓语法编译和差异格式检查通过；完整 GPU TensorRT 和 K230 板端流程仍需对应硬件。
- 修复：可视化图片写入采用原子替换，离线 HTML 转义外部名称；输入尺寸、检测阈值、特征通道模式和聚合方式会被严格校验。视频解帧拒绝把不支持的单文件当作视频。

- 第三轮修复：图片 ID 始终保留扩展名，自定义布局使用 input.dataset_root/--dataset-root；旧预测需重新生成。配置编辑器切页同步状态，手动修改配置路径刷新模块。新增隔离回归验证，真实 Qt 窗口仍待验收。

- 修复：一键测速接受一致性 `passed` 状态；新分支同轮检测压缩后可继续导出；checkpoint 检查 CSV 保留失败详情。
- 修复：FP16 ONNX 归一化后保持声明 dtype；旧 models 配置也必须服从 `predict.generate`；单图与目录可视化采用同一数据集相对 ID。
- 修复：UI YAML 高级页的后续修改不再被图形旧值覆盖；模块按钮在完整 YAML 模块集合上应用增量，保留未显示模块。
- 历史验证：使用系统 `/usr/bin/python3` 时 56 项通过、2 项因该解释器未安装 torch 失败；后续已改用含 PyTorch/PySide6 的 `yolo` 环境补跑相关测试，结果见本节最新条目。

## 更新批次：`2026.09.14`

- 修复：结构化检测剪枝保存时旧 EMA 覆盖、检测/分类分支权重血缘不一致、蒸馏教师/学生来源登记错误、注册表并发写入丢失。
- 修复：诊断重叠重复框误判、稳定图片 ID、相对路径解析、差图图片和阈值过滤、模块 allow-list 与 CLI 覆盖；可视化 CAM 类别/多目标选择、BF16 输入和缺失 source 覆盖。
- 修复：UI 两个配置页字段丢失、未知下拉值、差图依赖、TensorRT 首次构建、文件改名递归/确认、进程树停止；批处理改名回滚、COCO 精确匹配、视频/图片输出冲突和 Kmodel 严格校验。
- 修复：统一入口和测速入口正确传播失败退出码；默认检测压缩配置关闭分类动态 INT8，并同步 README/Spec。
- 验证：Python compileall、git diff --check；无额外依赖的 model_compression/model_diagnostics 11 项和 speed_test 43 项通过。2 项需要本环境未安装的 PyTorch，PySide6/TensorRT/K230 端到端路径未在当前环境执行；另用临时注册表、模块选择、图片 ID 和 Kmodel 数值样例做了回归验证。

## 更新批次：`2026.09.10`

- 功能：测速与目标检测诊断的 test 内部模块开关；差图清单、图片/HTML 输出和检测框重叠分析；可视化的特征图/CAM/阶段追踪开关；诊断默认不重复计算 Ultralytics 的 mAP/AP 与阈值扫描。
- 新增文件：`docs/specs/modular-tests-and-diagnostics.md`、`model_diagnostics/diagnostics/modules.py`、`model_diagnostics/diagnostics/overlap.py`、`model_diagnostics/diagnostics/bad_cases.py`、`speed_test/core/module_selection.py`、`run_tools.py`、`config/tools.yaml`。
- 修改文件：诊断引擎和桌面 UI/终端入口、PyTorch/TensorRT 测速入口、一致性检查、相关配置与 README。
- 配置变化：新增 `modules`；支持 `--only`、`--enable`、`--disable`、`--list-modules`；诊断 YAML 移除 AP 阈值旧参数，speed_test 移除 `test_model_call/test_pipeline/test_pytorch` 独立开关，统一由 `modules` 控制。重叠分析支持 IoU、交集占较小框比例、规则和类别范围配置。
- 验证方式：`python -m compileall -q model_diagnostics speed_test`；在 `speed_test/` 中运行 `python -m unittest discover -s tests -v`（43 项，1 项按环境跳过），在仓库根目录运行 `python -m unittest discover -s model_diagnostics/tests -v`（5 项）；各入口的 `--list-modules`；使用 canonical GT/预测进行重叠和差图离线冒烟。
- 已知限制：`validation.ultralytics` 仅作为预留开关，具体验证仍由 Ultralytics 原生入口执行；图片绘制需要 Pillow；TensorRT 模块仍需要 CUDA/TensorRT 环境。

这不是 Python 包的运行时版本号，而是仓库更新清单。以后每次增加功能时，继续在本文件追加一节即可。

## 本次更新目的

补充模型内部检查工具，用于回答以下问题：

1. 检测头输入的 P2/P3/P4/P5 特征是否有响应；
2. 候选框在筛选前、检测头 Top-K、最终检测结果三个阶段如何变化；
3. 候选框来自哪一个特征层，以及在哪一步被过滤；
4. 特征图、聚合图和 CAM 是否集中在目标区域；
5. TensorRT engine 当前由哪些参数和环境构建而来。

## 文件变更清单

### 新增：模型可视化与阶段追踪

新增目录：`model_visualization/`

| 文件 | 作用 |
| --- | --- |
| `model_visualization/run_visualization.py` | 命令行后端入口 |
| `model_visualization/adapters/ultralytics.py` | Ultralytics/YOLO 模型适配、特征捕获和检测阶段追踪 |
| `model_visualization/core/capture.py` | 激活值捕获、特征图、GradCAM/LayerCAM |
| `model_visualization/core/common.py` | 配置读取、图片预处理、绘图、统计和 HTML 索引 |
| `model_visualization/config/config.yaml` | YAML 配置模板 |
| `model_visualization/requirements.txt` | 可视化功能的额外依赖 |
| `model_visualization/README.md` | 使用说明和输出文件说明 |

主要输出包括：

- `layers.csv`：模型层清单；
- `features/`、`activations/`：特征图和聚合图；
- `cam/`：GradCAM/LayerCAM 结果；
- `raw_candidates.json`：筛选前候选框；
- `head_topk.json`：检测头 Top-K 候选；
- `final_detections.json`：最终检测框；
- `candidates.jsonl`、`stage_summary.csv`、`stage_events.jsonl`：阶段追踪明细；
- `canonical/`：可直接给 `model_diagnostics` 使用的标准预测文件；
- `index.html`：结果索引页面。

### 修改：检测诊断适配器

文件：`model_diagnostics/diagnostics/adapter_runtime.py`

- 保留并传递候选框的来源层、原始索引等 provenance 字段；
- 增加运行状态信息，便于确认实际加载的模型和设备。

### 新增和修改：TensorRT 构建记录

文件：

- `speed_test/core/build_metadata.py`（新增）；
- `speed_test/backends/pytorch_vision_tensorrt_benchmark_v2.py`；
- `speed_test/backends/tensorrt_engine_builder.py`；
- `speed_test/tests/test_build_metadata.py`（新增）。

作用：

- 在 engine 旁边记录模型、ONNX、TensorRT/CUDA、GPU 和构建参数；
- 构建成功后才发布新 engine；
- 构建失败时保留旧 engine；
- 不改变现有 `rebuild_engine` 缓存策略，也不强制引入哈希。

### 文档

- 根目录 `README.md`：补充项目结构、环境和可视化工具说明；
- `speed_test/README.md`：补充 engine 构建记录说明。

## 当前不需要修改的内容

- `ultralytics-cn/` 网络源码不需要重新复制；
- 训练脚本、训练权重和数据集不需要放入 vtools；
- `__pycache__/` 和 `*.pyc` 不属于更新内容；
- 不需要重新执行 `git am`；
- 不需要把模型权重复制到 vtools 内。

## 安装/更新方式

本次更新采用普通文件覆盖方式：

```bash
cd /path/to/vtools
unzip -o ~/下载/vtools-visualization-complete.zip
```

如果已经完成复制，只需要确认本文件放在：

```text
/path/to/vtools/VERSION.md
```

## 运行前配置

编辑：`model_visualization/config/config.yaml`

至少修改：

```yaml
model:
  weights: /absolute/path/to/best.pt

input:
  source: /absolute/path/to/image-or-directory

stage_trace:
  enabled: true
```

模型权重和数据可以继续放在 `lichiflower` 项目目录中；它们只是输入数据，不代表 vtools 属于该项目。

## 验证

```bash
cd /path/to/vtools
python -m compileall -q model_visualization speed_test/core
```

返回 `0` 表示语法检查通过。正式运行：

```bash
python model_visualization/run_visualization.py \
  --config model_visualization/config/config.yaml
```

结果默认位于：

```text
model_visualization/runs/runN/index.html
```

## 后续版本记录格式

以后新增功能时，在本文件顶部追加：

```markdown
## 更新批次：YYYY.MM.DD

- 功能：
- 新增文件：
- 修改文件：
- 配置变化：
- 验证方式：
- 已知限制：
```
