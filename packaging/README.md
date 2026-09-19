# vtools 桌面包

工作台默认使用 Python 和 pywebview 运行。需要交付给没有 Node 环境的使用者时，可以用 PyInstaller 生成目录包；前端 `dist/`、已纳入 Git 的工具脚本和配置文件会一起放入包内。构建请在 Git 工作区中执行；本地权重、数据集、运行结果和未跟踪文件不会进入包。

桌面包只包含工作台、前端资源和轻量 API 依赖。模型、测速和转换任务继续使用工作台“设置”里选择的 Python 环境，这样不会把 torch、TensorRT 和 Ultralytics 的多 GB 运行库复制进每个桌面包。

桌面包同时展开 `vtools_runtime` 发现模块到 `_internal/`。启动任务时，工作台会把 `_internal/` 加入所选 Python 的模块搜索路径，把该环境的 `bin` 和动态库目录置于前面，并移除 PyInstaller 对外部 Python 不适用的动态库搜索路径。外部 Python 因此可以加载 vtools 的运行时适配器及其自身的 PyTorch、TensorRT 等依赖，并自动发现 Ultralytics Fork。

## 构建

在仓库根目录执行：

```bash
python -m pip install -r vtools_ui/requirements.txt pyinstaller
cd vtools_ui/webapp/frontend
npm install
npm run build
cd ../../..
python packaging/build_desktop.py --clean
```

构建结果位于 `dist/vtools/`：

- Windows：`dist/vtools/vtools.exe`
- Ubuntu：`dist/vtools/vtools`

目录包需要保留整个 `dist/vtools/` 目录，不能只复制可执行文件。首次启动仍然只监听本机随机端口；Linux 包默认优先使用系统的 `python`/`python3` 执行任务。GPU、TensorRT、nncase 和真实权重仍需要目标机器提供对应驱动或运行库。

## Ubuntu 双击启动

在 Ubuntu 上构建完目录包后，从仓库根目录执行：

```bash
python packaging/build_deb.py
```

生成的 `dist/vtools_<版本>_<架构>.deb` 可以通过文件管理器双击安装。安装后在应用菜单搜索“vtools 工作台”，点击图标即可启动。安装包会把完整目录包放到 `/opt/vtools/`，并安装应用菜单入口和图标。卸载可通过系统软件管理器完成。

更新安装包时先修改 `vtools_ui/__init__.py` 的版本，再重新构建目录包和 `.deb`。安装更高版本的 `.deb` 会升级已安装的工作台，无需卸载旧版。

从 0.3.1 起，工作台启动时会检查 [GitHub Releases](https://github.com/xyzkioo/vtools/releases) 的最新正式发布版，也可在“设置 → 软件更新”手动检查。发布时标签使用 `v<版本>`，上传同版本、同架构的 `vtools_<版本>_<架构>.deb`；GitHub 提供的 SHA-256 摘要用于下载校验。发现新版后，点击“下载并安装”，系统会弹出管理员授权窗口。安装完成后重新打开工作台。GitHub 尚未发布安装包时，界面会显示“GitHub 尚无发布版”。

已安装的 0.3.0 没有内置更新器，需手动安装 0.3.1 一次；此后才可在工作台里更新。当前构建脚本不会自动上传安装包。

如果只想试运行目录包，可直接执行 `./dist/vtools/vtools`。重新打包前应先重新运行上面的桌面目录包构建命令，以包含当前代码和配置。

打包版默认将模型任务结果写入用户目录下的 `~/.vtools_ui/runs/`，可在工作台设置中修改结果目录。每次运行分配新目录，避免向应用包内部写文件。设置、历史和日志也位于 `~/.vtools_ui/`；Windows 上对应当前用户的主目录。转换与数据工具的显式输出路径仍以工具页面填写的路径为准。

## 平台说明

PyInstaller 需要在目标平台分别构建，不能在 Ubuntu 上直接生成 Windows 包。Windows 构建使用 Windows Python，Ubuntu 构建使用 Ubuntu Python。`.deb` 仅供 Debian/Ubuntu 系 Linux 系统使用；仓库尚未提供 `.AppImage` 安装器。
