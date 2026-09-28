# vtools 依赖边界与项目审查记录

日期：2026-09-28。范围：内置/外部运行时分流、uv/Conda 子进程环境、工具入口、模块选择、配置读取与复制、错误状态、打包依赖声明及相关文档。

## 功能拆分与注释

- 已有预测文件的检测诊断复用现有诊断引擎，纳入内置运行时，支持匹配、错误分析、差图和 HTML 报告；无需导入 PyTorch 或 Ultralytics。
- 在 [运行时选择](../../vtools_ui/webapp/core.py)、[子进程环境](../../vtools_ui/process_environment.py)、[打包脚本](../../packaging/build_desktop.py) 中补充说明，按实际执行路径划分依赖。
- NDJSON GT 仍调用 Ultralytics 公共转换 API；checkpoint 检查仍需 PyTorch 读取权重；模型推理、CAM、压缩、TensorRT、K230 等仍使用对应外部环境。
- 界面标明轻量工具内置环境，解释器设置标为模型任务使用；[打包说明](../../packaging/README.md) 列出逐功能依赖表。

## 已确认并修复的问题

| 问题 | 原行为与影响 | 修复及验证 |
| --- | --- | --- |
| uv 桌面依赖查找解析解释器符号链接 | 将 `.venv/bin/python` 解析为共享基础 Python，检查错误的 site-packages，可能找不到已安装的 pywebview | 保留虚拟环境路径；符号链接回归测试验证探测参数 |
| uv 任务继承旧 Conda 标识 | 下游运行时可能发现旧环境并注入其动态库 | 清除不属于所选解释器的 Conda 标识，设置实际 VIRTUAL_ENV；跨环境回归验证 |
| 显式空模块集合恢复默认 | `--only ''` 被当作未指定，诊断、测速、压缩、可视化重新启用配置模块 | 区分 None 与空集合；四组选择器及一键测速回归通过 |
| 可视化全关闭仍执行前向 | 没有输出模块时仍加载权重、挂钩并执行模型 | visualize 模式记录 skipped/no_modules，避免加载和前向；list_layers 保留独立含义 |
| 模型任务误用内置解释器 | 没有外部解释器时可能重新进入轻量工作台，因框架缺失失败 | 启动前明确提示选择外部模型环境；轻量任务仍可运行 |
| engine/kmodel 被当作预测文件 | UI 分类与后端模型后缀规则不一致 | 统一使用 MODEL_SUFFIXES，模型文件走 weights 与外部环境 |
| 非对象的空值 YAML 被当作默认配置 | `[]`、false、0、空字符串因 `or {}` 绕过顶层对象检查 | 仅 None 视作空配置；入口、界面及模型配置读取回归通过 |
| 复制模板改变 project.root | 配置移动到用户目录后，相对输入位置随之改变 | 首次复制将相对 project.root 固定到原位置，原模板及已有用户副本不被覆盖 |
| JSON 图形编辑后保存为 YAML | JSON 后缀不变，按后缀选择解析器的诊断后端无法读取 | 复制及保存时保留 JSON 格式；复制、图形修改、保存回归验证 |
| 未实现的验证开关静默成功 | validation.ultralytics 没有执行验证，却可出现在成功报告中 | 显式启用时明确报错并指向原生 model.val()/yolo val |
| 严格依赖检查异常只发警告 | 检查器无法启动或超时仍可能使 --strict 通过 | strict 下记录 FAIL，非严格模式保持 WARN；异常及目标解释器回归通过 |
| 更新失败仍显示正在检查 | 网络/SSL 失败后错误与进行中提示并存 | 失败状态显示重试提示；前端类型检查与构建通过 |

同时修正可视化输入说明：特征图/CAM 需要可加载的 PyTorch 模型，不能把导出的 ONNX 当作等价输入。
修复压缩 README 指向已删除 Spec 的链接，改为实际存在的配置及依赖说明。

## 验证结果

使用既有 vtools uv 环境执行：

| 检查 | 结果 |
| --- | --- |
| `QT_QPA_PLATFORM=offscreen PYTHONPATH=.:speed_test:model_diagnostics python -m unittest discover -s tests -q` | 97 项通过 |
| `PYTHONPATH=. python -m unittest discover -s model_compression/tests -q` | 21 项通过 |
| `PYTHONPATH=.:speed_test python -m unittest discover -s speed_test/tests -q` | 51 项通过，1 项跳过 |
| `PYTHONPATH=.:model_diagnostics python -m unittest discover -s model_diagnostics/tests -q` | 13 项通过 |
| 系统 Python 单独执行缺少 torch 的 CLI 错误报告测试 | 1 项通过；补足主环境的跳过项 |
| 全仓 compileall、git diff --check | 通过 |
| 前端 TypeScript 检查及 Vite 构建 | 通过 |
| 18 个工具与打包入口 `--help` | 全部通过 |
| 本轮修改的 Markdown 本地链接 | 通过 |

预测文件诊断的端到端回归在禁止导入 torch/ultralytics 时，使用临时图片、标签、data.yaml 和预测文件生成漏检统计、差图及 HTML 报告。所有样例与产物在临时目录中，未使用或修改真实数据与权重。

主环境跳过项专门要求未安装 PyTorch；系统 Python 的独立补测成功。错误路径测试故意输出参数缺失、磁盘写入失败、依赖异常等日志，不代表套件失败。TensorRT 测试使用替身，不能当成真实硬件验收。

## 交付边界

- GPU/TensorRT、真实模型训练或压缩、nncase/.NET 和 K230 板端端到端流程未执行；干净目标系统的安装验收也未执行。
- 已保存的配置副本不自动迁移；若仍使用相对 project.root，应确认其实际位置。
- 后续按用户指定版本 0.4.0 重新构建前端与桌面目录包；冻结版实测在外部 Python 路径不可用时，仍可完成已有预测文件诊断，生成漏检统计、差图及 HTML 报告，且可加载内置 OpenCV。0.3.29 安装包保持原内容。新包 `dist/vtools_0.4.0_amd64.deb` 已构建完成，Debian 元数据版本为 0.4.0、架构 amd64，归档读取检查通过；大小 248302554 字节，SHA-256：`17164b33aade1598bdd8c2648065fd6321e5a09b6166f25ab28df220b0e0826a`。
- 原有 IDE 工作区变更保留，本轮未改写；生成的运行日志、缓存和测试产物不属于源码交付。
