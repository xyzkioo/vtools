# 桌面打包与公共运行时

[返回模块导航](../README.md)

把 React + Qt WebEngine 工作台和工具资源打成桌面目录包，再生成 Windows 或 Ubuntu/Debian 安装包；公共运行时帮助各模块发现独立 Ultralytics Fork。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python packaging/build_desktop.py --help
python packaging/build_deb.py --help
python packaging/build_windows_installer.py --help
```

## 输入与功能

构建流程见 [打包说明](../../packaging/README.md)：先准备依赖并构建前端，再构建桌面目录包，最后在对应平台构建安装器。Windows 使用标准 CPython 3.12 venv 和固定依赖清单，图形入口与后台任务入口共享资源。分发含新修复的安装包前须递增版本，并同步 Python、前端 package/package-lock 及 [VERSION.md](../../VERSION.md)。

[vtools_runtime](../../vtools_runtime/ultralytics.py) 负责共享的 Ultralytics 源码发现；通过 `project.ultralytics_repo`、`VTOOLS_ULTRALYTICS_REPO` 或同级目录定位 Fork，不是独立推理入口。

## 输出

桌面目录包输出 `dist/vtools/`，Debian 包输出 `dist/vtools_<版本>_<架构>.deb`。分发目录包必须保留完整目录；发布前核对包版本、架构、大小和 SHA-256。

Windows 安装程序输出 `dist/vtools-<版本>-windows-x64-setup.exe`。Windows 默认将设置、日志、任务结果和网页缓存保存在安装目录旁的 `user-data/`，该位置须可写；显式 `VTOOLS_USER_DATA` 设置优先，卸载保留用户数据。Windows 目标机器启动工作台无需源码、Git、Node、Conda、Python 或系统 WebView2。

## 使用边界

PyInstaller 须在目标平台分别构建，Linux 构建不能直接生成 Windows 包。主包不包含 PyTorch、TensorRT、nncase、GPU 驱动或真实权重；构建环境和模型任务环境分别准备。

详细说明：[原模块文档](../../packaging/README.md)。

安装器在复制前检查 Windows 目录完整性标签，在准备阶段解包并运行实际页面加载验收，通过后才开始部署与创建快捷方式；任一步失败均不视为成功安装。默认仅当前用户安装，不修改系统安全标签。

0.5.6 起工具资源位于 `_internal/tool-sources/`，该目录只含纯源码和配置；不得将含 ABI 相关模块的 `_internal/` 加入外部解释器的 PYTHONPATH。Windows 外部子进程启动期间恢复默认 DLL 搜索，之后恢复桌面配置。
