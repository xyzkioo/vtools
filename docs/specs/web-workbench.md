# vtools 浅色工作台实现约定

## 入口与层次

- `python -m vtools_ui` 启动本机随机端口服务并打开 pywebview 桌面窗口；`--browser` 仅供开发预览。
- React 只负责表单、状态和预览。工具清单、字段、命令构造、配置读写、任务生命周期、历史与结果索引由 `vtools_ui/webapp/` 的 Python 模块负责。
- 模型类任务继续调用 `run_tools.py`；一键测速、checkpoint、环境与文件工具调用原有独立脚本。一次仅运行一个任务。

## 本地接口

所有 `/api/*` 接口需携带本次启动令牌。服务只监听 `127.0.0.1`，前端与服务同源。

| 接口 | 用途 |
| --- | --- |
| `GET /api/bootstrap`、`GET /api/environments` | 工具目录、设置、历史、当前任务和解释器候选 |
| `GET /api/config`、`POST /api/config/parse`、`POST /api/config/patch`、`POST /api/config/save` | 读取、校验、局部修改和原子保存 YAML |
| `POST /api/runs`、`POST /api/runs/stop`、`GET /api/runs/events` | 启动、停止和长轮询任务事件 |
| `GET /api/results`、`GET /api/results/preview`、`POST /api/results/open` | 列出结果、受限预览和打开所在目录 |
| `POST /api/settings`、`GET /api/history`、`POST /api/history/clear` | 设置和历史 |

启动请求只接受工具 ID、预定义变体、配置路径、解释器路径、结构化字段值和模块变更字典。模块变更只生成 `--enable` / `--disable`，不从可见按钮推断 `--only`，以保留隐藏模块。相对 CLI 路径相对于仓库根目录；YAML 内容由各工具按原规则解析。

## 配置与文件

- 三个配置视图共享一份 YAML 文本；表单字段通过后端局部修改，原文视图切出时重新校验。未知字段和点号模块 ID 保留。
- 保存配置前解析顶层映射并备份为 `.bak`；临时文件写入成功后替换目标文件。
- 结果文件路径必须解析后位于当前选定根目录内；只允许已知图片、文本和模型扩展名。前端按任意深度的相对路径构建可展开目录树；文本预览上限 256 KB。
- 设置与历史使用 `.vtools_ui/` 下的 JSON 文件；首次启动从旧 QSettings 读取已有值。任务日志按运行 ID 保存。

## 验收边界

Ubuntu 可在现有 `yolo` 环境运行无硬件检查。Windows 的桌面窗口、文件选择和子进程停止路径必须在 Windows 机器上另行实测；GPU、TensorRT、nncase 与真实权重路径也需对应环境，不能由静态测试代替。

## 图片预览

结果页选择图片后提供缩小、放大、适应窗口和重置比例。适应窗口是默认状态；放大后预览区域保留滚动条，不改变原始结果文件。

## 分发包

`packaging/build_desktop.py` 使用 PyInstaller 命令行生成目录包，前端静态资源、工具脚本、配置文件和 `vtools_runtime` 随包提供。PyInstaller 需要在 Windows、Ubuntu 分别执行，包内任务仍受目标机器的 GPU、TensorRT、nncase 和模型文件条件限制。
构建清单只收录版本控制中的工具源码及配置和前端构建产物，不收录本地权重、数据集和未跟踪文件。打包版默认将模型任务结果写到用户目录的 `.vtools_ui/runs/`；服务端生成每次独立的 `--run-dir`，结果浏览器使用同一根目录。

## 当前验收记录

| 范围 | 状态 |
|---|---|
| 浏览器模式、任务 API、配置编辑和结果预览 | 已通过：58 个 Python 测试，前端构建通过 |
| Ubuntu 无硬件桌面启动 | 已通过进程烟测；GBM/OpenGL 回退提示不影响启动 |
| 1280×800、1080×680、125%/150% 等效视口 | 已检查，无横向溢出，长表单和配置弹窗可滚动 |
| Windows 桌面窗口、原生文件选择、停止任务 | 待 Windows 机器实测 |
| CPU 环境、ONNX/TensorRT API 导入 | 已通过环境检查；没有执行 TensorRT 推理 |
| GPU、nncase、真实权重 | 待对应硬件和模型实测；当前 `torch.cuda.is_available()` 为 `False` |
| PyInstaller 最终目录包 | Ubuntu 目录包已重新生成；`--help`、包内工具入口及内容清单检查通过，早前完成浏览器和桌面启动烟测；Windows 需在 Windows 机器上构建 |
