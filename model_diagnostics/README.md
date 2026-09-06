# 通用目标检测诊断工具

本项目只评估目标检测质量与错误来源，不计算参数量、FLOPs、显存和延迟。适配器接口与 [vtools 的 `speed_test` adapter](https://github.com/xyzkioo/vtools/tree/main/speed_test/adapters) 对齐，因此可以复制模型适配器，再修改 YAML 完成测试；工具不依赖 Ultralytics。

## PyCharm 直接运行

推荐先运行安装向导：

```bash
python installer/install.py
```

安装向导会安装本仓库依赖并调用环境检查脚本。也可以单独运行检查：

```bash
python installer/check_environment.py --mode B
```

它会检查 PyTorch、CUDA 和 `YOLO` 的训练、推理、验证接口。

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
