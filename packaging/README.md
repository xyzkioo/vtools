# vtools 桌面包

工作台默认使用 Python 和 PySide6 Qt WebEngine 运行。需要交付给没有 Node 环境的使用者时，可以用 PyInstaller 生成目录包；前端 `dist/`、已纳入 Git 的工具脚本和配置文件会一起放入包内。构建请在 Git 工作区中执行；本地权重、数据集、运行结果和未跟踪文件不会进入包。

桌面包包含工作台、前端资源和 Pillow、OpenCV 等图片/报告运行依赖。按功能路径而不是工具名称分流：

| 功能路径 | 安装版运行环境 | 依赖边界 |
| --- | --- | --- |
| 图片缩放、视频抽帧、文件名转换、数据集质量检查 | 内置 | Pillow、OpenCV、PyYAML |
| 检测诊断读取已有预测文件，输出匹配结果、差图和报告 | 内置 | 不加载模型；GT 为本地 YAML/JSON/YOLO 标签 |
| 检测诊断生成预测、模型可视化、测速、压缩、checkpoint 检查 | 所选外部 Python | PyTorch、Ultralytics 或对应后端 |
| NDJSON 数据集转换（包括诊断读取 NDJSON GT） | 所选外部 Python | 复用 Ultralytics 公共转换 API |
| K230 转换及 ONNX/kmodel 推理校验 | 所选外部 Python | 对应的模型、推理和 nncase/.NET 运行库 |

内置路径由 `vtools_ui/webapp/core.py` 明确选择；添加新路径前需验证其实际导入链，不应因为“不执行推理”就推断无需模型库。构建用 `opencv-python-headless` 避免引入额外的 Qt 界面库；构建环境应避免同时安装多个 OpenCV 发行包。PyTorch、TensorRT 和 nncase 不随主包复制。

桌面包将工具脚本、配置和 `vtools_runtime` 展开到独立的 `_internal/tool-sources/`。只有该纯源码目录会加入任务 Python 的模块搜索路径，不能添加含 `.pyd`/DLL 的 `_internal/` 根目录。启动外部任务时会清理冻结运行库的 Python/Qt 路径，Windows 在创建子进程期间暂时清除 PyInstaller 的 DLL 搜索目录并随即恢复；模型环境自己的解释器和动态库目录优先。外部 Python 因此可以加载 vtools 的运行时适配器及其自身的 PyTorch、TensorRT 等依赖，并自动发现 Ultralytics Fork。

## 构建

在仓库根目录执行：

```bash
python -m pip install -r vtools_ui/requirements.txt -r packaging/requirements-bundled-tools.txt pyinstaller
cd vtools_ui/webapp/frontend
npm install
npm run build
cd ../../..
python packaging/build_desktop.py --clean
```

构建结果位于 `dist/vtools/`：

- Windows：`dist/vtools/vtools.exe`
- Ubuntu：`dist/vtools/vtools`

目录包需要保留整个 `dist/vtools/` 目录，不能只复制可执行文件。首次启动仍然只监听本机随机端口；Linux 包的模型任务优先选择 `~/python-envs/vtools/.venv`，没有该环境时尝试系统 `python`/`python3`。已保存的解释器设置优先。GPU、TensorRT、nncase 和真实权重仍需要目标机器提供对应驱动或运行库。

## Windows `.exe` 安装包

Windows 安装包用 Inno Setup 6 封装 `dist/vtools/` 目录包，需要先完成上面的目录包构建，再在仓库根目录执行：

```powershell
winget install --id JRSoftware.InnoSetup --scope user   # 只需一次
python packaging/build_windows_installer.py
```

生成的 `dist/vtools_<版本>_windows-x64-setup.exe` 可以双击安装：

- 仅按当前用户安装，默认位置为 `%LOCALAPPDATA%\Programs\vtools`，不需要管理员权限，也可在安装向导选择 D 盘等可写目录；
- 安装后创建开始菜单快捷方式，桌面图标为可选任务；
- Windows 主程序使用无控制台的图形入口，双击快捷方式即可打开；内置工具由隐藏的 `vtools-worker.exe` 执行，任务输出仍显示在运行详情里；
- 数据统一保存在安装目录旁的 `user-data/`，包含设置、历史、任务结果、启动日志及 Qt WebEngine 缓存；请在安装目录页面选择当前用户可写的目录，如 `D:\Vtools`；
- 安装目录不可写时显示明确错误，不会悄悄改存 C 盘；也可通过 `VTOOLS_USER_DATA` 指定用户数据位置。旧版本数据保留，不自动迁移；
- 启动的 `vtools.exe` 与安装包共用 `packaging/vtools.ico`，该图标按 `packaging/vtools.svg` 的几何在构建时生成；
- 内置 Qt WebEngine，不依赖系统 WebView2、pythonnet/.NET、用户安装的 Python 或源码仓库；
- 卸载保留 `user-data/`，不删除用户设置、历史或任务结果。

`--iscc` 可以指定 `ISCC.exe` 路径；`--chinese-language` 可以传入 Inno Setup 中文语言文件（`.isl`），不提供时安装向导界面为英文。安装包的版本号来自 `vtools_ui/__init__.py`，构建脚本会打印安装包路径、大小和 SHA-256。

## Ubuntu 双击启动

在 Ubuntu 上构建完目录包后，从仓库根目录执行：

```bash
python packaging/build_deb.py
```

生成的 `dist/vtools_<版本>_<架构>.deb` 可以通过文件管理器双击安装。安装后在应用菜单搜索“vtools 工作台”，点击图标即可启动。安装包会把完整目录包放到 `/opt/vtools/`，并安装应用菜单入口和图标。卸载可通过系统软件管理器完成。

也可在仓库根目录执行 `sudo apt install ./dist/vtools_<版本>_<架构>.deb`。v0.5.11 的包信息、校验值和验证范围见[发行说明](../docs/releases/v0.5.11.md)。升级后完全退出旧工作台再重新打开。

更新安装包时先修改 `vtools_ui/__init__.py` 的版本，再重新构建目录包和 `.deb`。安装更高版本的 `.deb` 会升级已安装的工作台，无需卸载旧版。

从 0.5.8 起，安装版（Windows 与 Linux）会在启动时检查 [GitHub Releases](https://github.com/xyzkioo/vtools/releases)，也可在“设置 → 软件更新”手动检查。“软件更新”分两行列出 Windows x64 安装包（`vtools-<版本>-windows-x64-setup.exe`）与 Linux `.deb`（`vtools_<版本>_<架构>.deb`）各自的最新可用版本，两个平台互不阻塞：某个 release 只带了其中一个平台的资产时，另一个平台沿用自己最新的那一版。发布时标签使用 `v<版本>`，GitHub 提供的 SHA-256 摘要用于下载校验，缺少摘要的资产不会被提供安装。

Windows 的“下载并安装”是半自动流程：下载安装包 → 校验 SHA-256 与大小 → 调起 Inno Setup 安装向导，由用户在向导里确认并完成安装，随后重新打开工作台；安装前仍会执行 `--install-check`。Linux 的“下载并安装”下载并校验 `.deb` 后通过系统授权（pkexec + apt-get）安装。草稿版与预发布版不会被识别为最新版。

从 0.5.9 起，检查更新不再只依赖 GitHub REST API：接口限流（未认证每小时 60 次/IP，超限返回 403）或不可达时，改用 `releases.atom` 取版本号、对资产做 HEAD 取大小，并读取发布时上传的同名 `.sha256` 文件作为摘要；两条路都失败时使用 24 小时内的本地缓存并在界面提示。因此**发布时必须上传 `<资产>.sha256`**（本仓库的构建脚本与发布流程已经这样做了），否则回退路径无法校验该资产、会把它标为“校验信息缺失”。

已安装的 0.3.0 没有内置更新器，需手动安装 0.3.1 一次；此后才可在工作台里更新。当前构建脚本不会自动上传安装包。

如果只想试运行目录包，可直接执行 `./dist/vtools/vtools`。重新打包前应先重新运行上面的桌面目录包构建命令，以包含当前代码和配置。

打包版默认将模型任务结果写入用户目录下的 `~/.vtools_ui/runs/`，可在工作台设置中修改结果目录。每次运行分配新目录，避免向应用包内部写文件。设置、历史和日志也位于 `~/.vtools_ui/`；Windows 0.5.4 起统一使用安装目录旁的 `user-data/`，不使用上述 Linux 默认路径。转换与数据工具的显式输出路径仍以工具页面填写的路径为准。

## 平台说明

PyInstaller 需要在目标平台分别构建，不能在 Ubuntu 上直接生成 Windows 包。Windows 构建使用 Windows Python，Ubuntu 构建使用 Ubuntu Python。`.deb` 仅供 Debian/Ubuntu 系 Linux 系统使用；仓库尚未提供 `.AppImage` 安装器。

## Windows 可复现构建（0.5.4 起）

使用标准 CPython 3.12 x64，在仓库根目录创建独立环境；构建脚本会拒绝 Windows Conda 环境：

```powershell
python -m venv .venv-build
.venv-build\Scripts\python -m pip install -r packaging/requirements-windows-build.txt
npm.cmd --prefix vtools_ui/webapp/frontend ci
npm.cmd --prefix vtools_ui/webapp/frontend run build
.venv-build\Scripts\python packaging/build_desktop.py --clean
.venv-build\Scripts\python packaging/build_windows_installer.py
```

构建工具只在开发机器使用。交付的是整个 `dist/vtools/` 或安装程序，目标机器无需 Git、Node、Conda 或克隆源码。模型训练、推理等仍需要独立的模型运行环境。

## 安装完成判定（0.5.5 起）

安装器会先读取目标目录或其最近已存在父目录的 Windows 完整性标签。低于 Medium 的受限位置在复制文件前拒绝，静默安装返回非零退出码；不调整系统安全设置、标签或安装位置。

安装准备阶段先在临时区解包完整文件，指定实际目标用户数据目录，自动运行 `vtools-worker.exe --install-check`，实际加载本机 React 页面并等待 API 版本和 Qt WebChannel 桥接就绪，最长等待 30 秒。失败在准备阶段中止安装（静默安装返回非零退出码），不部署程序或创建快捷方式；成功才开始实际安装。验证程序不修改用户剪贴板或运行模型任务。错误详情记录在所选目录的 `user-data/logs/desktop.log`。

`--install-check` 也可用于交付验收，成功返回 0。它使用与正常启动相同的运行库、页面和数据目录，不依赖源码、Node 或外部 Python。
