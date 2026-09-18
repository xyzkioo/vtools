# vtools 桌面工作台

vtools 使用浅色 React 界面、pywebview 桌面窗口和仅监听本机的 Python 服务。模型任务继续通过项目原有命令或独立脚本在子进程中运行；关闭窗口会停止当前任务。Windows 与 Ubuntu 是首版目标平台，macOS 尚未实测。

## 从仓库根目录启动

```bash
python -m pip install -r vtools_ui/requirements.txt
python -m vtools_ui
```

如果默认 Python 没有桌面依赖，启动器会自动寻找已安装 pywebview 的 conda 环境（例如 `yolo`）。也可以明确使用该环境：

```bash
conda run -n yolo python -m vtools_ui
# 或：conda activate yolo && python -m vtools_ui
```

不要直接双击 `frontend/index.html`；它是 Vite 开发源文件，没有本地 API。需要浏览器预览时使用 `python -m vtools_ui --browser`。

开发时可使用浏览器预览；仍由本地服务提供真实数据。浏览器模式中路径可手动填写，原生文件选择按钮只在桌面窗口中启用。允许文件或目录的输入框分别提供“选择文件”和“选择目录”按钮。

```bash
python -m vtools_ui --browser
```

前端构建产物已随源码提供，普通使用者不需要 Node。修改前端源码时，在仓库根目录运行：

```bash
cd vtools_ui/webapp/frontend
npm install
npm run build
```

需要生成可分发的桌面目录包或 Ubuntu `.deb` 安装包时，参见 [桌面包构建说明](../packaging/README.md)。安装 `.deb` 后可在应用菜单点击“vtools 工作台”启动。Windows 仍需在 Windows 机器上构建。打包版默认把模型运行结果放在 `~/.vtools_ui/runs/`，不会写入包内。

Linux 安装版会在启动时检查 GitHub Releases 的新版本。“设置 → 软件更新”可重新检查并下载安装；需要系统管理员授权。尚未发布到 GitHub 的构建不会被自动更新器发现。

Ubuntu 若缺少 Qt 平台插件所需系统库，可安装 `libxcb-cursor0`。pywebview 的 PySide6 适配依赖由 `requirements.txt` 安装；Windows 使用系统 WebView2 运行时。首次安装后若窗口仍显示旧内容，请完全关闭再启动。

## 功能

- 模型工具：检测诊断、模型可视化、PyTorch/TensorRT 测速、输出一致性、一键测速、checkpoint 检查和模型压缩。
- 转换和数据工具：独立的数据集质量检查、NDJSON 转 YOLO、PyTorch 转 K230 `.kmodel`、ONNX / `.kmodel` 校验、视频抽帧、图片缩放、图片 / YOLO / COCO 文件名转换。
- 数据集质量检查：在左侧“数据与转换 → 数据集质量检查”中选择 `data.yaml`，可设置报告目录和每个划分的标注抽样图数量；结果包含 `summary.json`、`issues.csv`、`report.md` 和抽样图，可直接跳转到结果查看。
- 检测诊断会加载模型；打包版会从所选数据集或权重所在工作区寻找同级 Ultralytics Fork。只检查图片和标签时使用独立的数据集质量检查入口。
- 环境检查：检查 Python、PyTorch、CUDA 和本地 Ultralytics 源码。
- 配置编辑：常用字段、全部标量字段与 YAML 原文共享编辑状态。YAML 是模型工具的完整参数来源；保存前校验并原子写入，不再生成 `.bak` 备份。对模块按钮的修改是本次运行的增量覆盖，不会关闭未显示的模块。
- 结果查看：扫描选定目录中的 runs、results、reports 和 outputs，以可展开的多级目录树浏览任意深度的子文件夹；预览诊断的 `summary.json`、`raw_data/`、差图，可视化单图 JSON、图片和 `index.html`，以及压缩的 `summary.json`、`effective_config.json` 和 `artifacts/`。图片支持缩小、放大、适应窗口和重置比例；文本最多预览前 256 KB；模型文件可打开所在目录。任务完成后可直接跳转至结果目录。
- 历史与设置：历史保存于 `.vtools_ui/history.json`，新增的任务日志保存在 `.vtools_ui/logs/`；结果目录、历史数量和任务解释器保存在 `.vtools_ui/settings.json`。首次启动会读取旧 Qt 工作台的 QSettings 值。

工作台保留模型任务的 YAML、CLI 覆盖项和独立子进程语义。路径输入中的相对路径按项目根目录解析；YAML 内部相对路径仍由各工具自身解释。

## 安全与运行限制

本地服务只绑定 `127.0.0.1` 的随机端口，API 要求每次启动随机生成的会话令牌。前端只提交结构化工具参数，服务端决定可执行的命令。结果预览限于当前选定目录，文本读取量受上限限制。一次只允许一个任务运行；启动失败、非零退出和用户停止会显示不同状态。

无 GPU、TensorRT、nncase 或真实权重时，界面仍可启动并运行无硬件任务；这些硬件路径需要在对应环境中分别验收。
