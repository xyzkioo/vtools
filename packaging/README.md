# vtools 桌面包

工作台默认使用 Python 和 pywebview 运行。需要交付给没有 Node 环境的使用者时，可以用 PyInstaller 生成目录包；前端 `dist/`、已纳入 Git 的工具脚本和配置文件会一起放入包内。构建请在 Git 工作区中执行；本地权重、数据集、运行结果和未跟踪文件不会进入包。

桌面包只包含工作台、前端资源和轻量 API 依赖。模型、测速和转换任务继续使用工作台“设置”里选择的 Python 环境，这样不会把 torch、TensorRT 和 Ultralytics 的多 GB 运行库复制进每个桌面包。

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

打包版默认将模型任务结果写入用户目录下的 `~/.vtools_ui/runs/`，可在工作台设置中修改结果目录。每次运行分配新目录，避免向应用包内部写文件。设置、历史和日志也位于 `~/.vtools_ui/`；Windows 上对应当前用户的主目录。转换与数据工具的显式输出路径仍以工具页面填写的路径为准。

## 平台说明

PyInstaller 需要在目标平台分别构建，不能在 Ubuntu 上直接生成 Windows 包。Windows 构建使用 Windows Python，Ubuntu 构建使用 Ubuntu Python。当前仓库提供的是目录包配置，还不是 `.deb` 或 `.AppImage` 安装器。
