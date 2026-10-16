# 模型压缩工具依赖

读取 YAML 和运行基础命令需要 `PyYAML`。真正加载模型、量化、剪枝和蒸馏时需要与本机匹配的 `PyTorch`；使用内置 `ImageFolder` 数据集时还需要配套的 `torchvision`。

YOLO 依赖感知结构化通道剪枝还需要 `torch-pruning>=1.4.1`。它是可选依赖：
只有将 `compression.structured.method` 设置为 `torch_pruning` 时才会加载；
默认的 `scale` 结构缩放不依赖它。

```bash
python -m pip install pyyaml
# 根据 CUDA/CPU 环境从 PyTorch 官方页面选择 torch 与 torchvision
```

工具不会在模块列表或查看配置时导入 PyTorch。动态 INT8 量化默认运行在 CPU，实际延迟和文件大小必须在目标设备上重新评测。

`distillation.detection` 另需支持 `distill_model/dis` 参数和 `DistillationModel` 的外部 Ultralytics Fork；vtools 同级的 `ultralytics-cn` 提供该接口。可用 `project.ultralytics_repo` 或 `VTOOLS_ULTRALYTICS_REPO` 指定路径。训练、验证与损失函数在 Fork 内执行，不随 vtools 的 deb 打包模型库。
