# 通用目标检测诊断工具

## 入口

```bash
python model_diagnostics/run_model_diagnostics.py
python model_diagnostics/run_model_diagnostics.py --help
python run_tools.py --tool diagnostics --config model_diagnostics/config/pycharm_run.yaml
```

默认读取 `model_diagnostics/config/pycharm_run.yaml`；PyCharm 直接运行 `run_model_diagnostics.py` 等价。

本项目只评估目标检测质量与错误来源，不计算参数量、FLOPs、显存和延迟。适配器接口与 [vtools 的 `speed_test` adapter](https://github.com/xyzkioo/vtools/tree/main/speed_test/adapters) 对齐，因此可以复制模型适配器，再修改 YAML 完成测试；使用已有预测文件的核心诊断不依赖 Ultralytics，模式 B 或 NDJSON 转换才需要它。

当前诊断入口按模块执行：`diagnostics.missed`、`diagnostics.classification`、
`diagnostics.background`、`diagnostics.duplicate`、`diagnostics.localization`、
`diagnostics.overlap` 和 `output.bad_cases/images/html` 都可独立开关。mAP、AP50、
阈值扫描等常规评估交给 Ultralytics 原生验证，诊断流程不会重复计算。

## PyCharm 直接运行

建议先检查运行环境：

```bash
# 以下命令在 vtools 仓库根目录执行
python env_test/check_install.py --source ./ultralytics-cn
```

如果还没有安装诊断依赖，请先执行：

```bash
python -m pip install -r ./model_diagnostics/requirements.txt
```

环境检查脚本会检查 PyTorch、CUDA 和 `YOLO` 的训练、推理、验证接口。

然后：

1. 用 PyCharm 打开 `model_diagnostics` 文件夹。
2. 修改带中文注释的 `config/pycharm_run.yaml`。
3. 直接运行同目录的 `run_model_diagnostics.py`。
4. 每次结果写入 `runs/runN/<mode_name>/`，不会覆盖上一次运行。

入口配置使用 A/B/C 三种互斥模式，不再需要 `models[].enabled`：

- 模式 A：已有预测文件，只填写 `mode_a.predictions`，不加载模型；
- 模式 B：现成的 Ultralytics `.pt`，只填写 `mode_b.weights`；
- 模式 C：复制 `adapters/vision_adapter_template.py`，实现 `build_model`、预处理/推理和 `postprocess_detections`，再填写 `mode_c.weights` 与 `mode_c.adapter`。

三种模式共用 `dataset.data` 和 `dataset.split`。每次只把 `mode` 改成 `A`、`B` 或 `C` 之一；B/C 生成的预测会自动写入当前运行目录。

数据集优先使用 Ultralytics `data.yaml`（`path`、`train`、`val`、`test`、`names`），程序会自动定位对应的 `labels/*.txt` 和图片；Ultralytics Platform 的 `.ndjson` 也可通过官方兼容转换读取。同时兼容 canonical JSON/JSONL、COCO GT/预测结果和旧式 YOLO 标签目录。统一检测格式为原图像素坐标 `[x1, y1, x2, y2]`、`score`、`class_id`。

完整指标、数据格式、适配器钩子和确认方法见 [docs/README.md](docs/README.md)。

### 配置模块开关

`config/pycharm_run.yaml` 中的 `modules` 是诊断功能的唯一开关。常规 mAP、AP50
和阈值曲线由 Ultralytics 原生 `val` 提供，诊断入口不会重复计算；只有显式打开
`diagnostics.threshold_sweep` 等模块时才会执行对应分析。

```yaml
modules:
  diagnostics.missed: true
  diagnostics.classification: true
  diagnostics.overlap: true
  output.bad_cases: true
  output.images: false
  output.html: false
```

PyCharm 直接运行 `run_model_diagnostics.py` 即可；终端也可以临时覆盖同一组开关：

```bash
python model_diagnostics/run_model_diagnostics.py --only diagnostics.overlap
```
