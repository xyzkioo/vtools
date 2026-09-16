# vtools 桌面工作台

## 入口

```bash
python -m pip install -r vtools_ui/requirements.txt
python -m vtools_ui
```

启动后通过界面选择工具和配置；模型类工具的 YAML 是唯一参数来源。

这是 vtools 的第一版跨平台桌面 UI。它使用 PySide6/Qt Widgets，Windows 和 Ubuntu 都可以运行；模型推理、测速和诊断在独立子进程中执行，界面不会被长任务阻塞。

## 运行

在仓库根目录执行：

```bash
python -m pip install -r vtools_ui/requirements.txt
python -m vtools_ui
```

Ubuntu 若提示 Qt 无法加载 `xcb` 平台插件，先安装桌面运行库：

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

然后重新执行 `python -m vtools_ui`。PySide6 已经安装到当前 Conda 环境时，不需要重复安装 Python 依赖。

当前版本已经接入：

- 检测诊断、模型可视化、模型压缩；
- 模型压缩页面区分 YOLO 目标检测与图像分类：检测模式使用 `data.yaml`，提供基线、非结构化剪枝、结构化通道缩放和压缩前后 mAP/固定输入延迟评估；
- PyTorch、TensorRT、输出一致性、一次性测速和 checkpoint 检查；
- NDJSON 转 YOLO、PyTorch 转 K230 `.kmodel`、ONNX / `.kmodel` 校验；
- 视频抽帧、图片缩放、图片 / YOLO / COCO 文件名转换。
- 结果查看：扫描 runs、results、reports、outputs 和 store，只列出图片并提供图片预览；任务完成后会在状态栏、日志和提示框显示结果目录。

配置文件仍然是模型类工具的真实参数来源；独立脚本的常用参数已经映射到表单。模型类页面的配置文件右侧提供齿轮按钮，可用图形化表单修改常用字段，也可以在“全部配置”页编辑所有标量字段，或切换到 YAML 高级页编辑自定义结构。保存配置前会校验 YAML 和数值格式，并自动保留 `.bak` 备份。所有任务统一使用子进程、日志和历史记录。

如果窗口仍显示旧布局，请先关闭旧窗口，再从仓库根目录重新启动。PySide6 窗口不会自动热更新；下面的检查应当打印当前仓库里的 `vtools_ui` 路径：

```bash
cd /path/to/vtools
conda activate yolo
python -c "import vtools_ui; print(vtools_ui.__file__)"
python -m vtools_ui
```

诊断、可视化、测速和压缩页面的“配置文件”一行现在同时提供“图形配置”和右侧齿轮；打开后默认进入“全部配置”，YAML 中的点号模块名、列表、布尔值和空值都可以保存。诊断页面把模型权重、已有预测文件和数据集 YAML 分成三个独立输入，不会再把 `.pt` 权重误传给 `--predictions`。

顶部环境下拉框会在后台扫描当前 Python、Conda 环境（包含 `base`）以及系统 `python/python3`；未手动选择时继续使用启动 UI 的当前环境，手动选择会在下次启动恢复。

## 设计约定

- UI 与 CUDA、TensorRT、Ultralytics 依赖解耦。
- 子进程使用当前选择的 Python 解释器，路径通过 `pathlib` 处理。
- 任务历史保存到项目根目录的 `.vtools_ui/history.json`；“设置”页可以调整保留数量、默认结果目录和清空历史。
- 不依赖 shell 激活命令，因此 Windows 和 Ubuntu 的启动逻辑一致。
