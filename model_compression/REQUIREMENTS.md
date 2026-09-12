# 模型压缩工具依赖

读取 YAML 和运行基础命令需要 `PyYAML`。真正加载模型、量化、剪枝和蒸馏时需要与本机匹配的 `PyTorch`；使用内置 `ImageFolder` 数据集时还需要配套的 `torchvision`。

```bash
python -m pip install pyyaml
# 根据 CUDA/CPU 环境从 PyTorch 官方页面选择 torch 与 torchvision
```

工具不会在模块列表或查看配置时导入 PyTorch。动态 INT8 量化默认运行在 CPU，实际延迟和文件大小必须在目标设备上重新评测。
