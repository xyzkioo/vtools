# 桌面打包与公共运行时

[返回模块导航](../README.md)

把工作台和工具资源打成桌面目录包，再生成 Ubuntu/Debian 安装包；公共运行时帮助各模块发现独立 Ultralytics Fork。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python packaging/build_desktop.py --help
python packaging/build_deb.py --help
```

## 输入与功能

构建流程见 [打包说明](../../packaging/README.md)：先准备依赖并构建前端，再构建桌面目录包，最后构建 .deb。分发含新修复的安装包前须递增版本，并同步 Python、前端 package/package-lock 及 [VERSION.md](../../VERSION.md)。

[vtools_runtime](../../vtools_runtime/ultralytics.py) 负责共享的 Ultralytics 源码发现；通过 `project.ultralytics_repo`、`VTOOLS_ULTRALYTICS_REPO` 或同级目录定位 Fork，不是独立推理入口。

## 输出

桌面目录包输出 `dist/vtools/`，Debian 包输出 `dist/vtools_<版本>_<架构>.deb`。分发目录包必须保留完整目录；发布前核对包版本、架构、大小和 SHA-256。

## 使用边界

PyInstaller 须在目标平台分别构建，Linux 构建不能直接生成 Windows 包。主包不包含 PyTorch、TensorRT、nncase、GPU 驱动或真实权重；构建环境和模型任务环境分别准备。

详细说明：[原模块文档](../../packaging/README.md)。
