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
