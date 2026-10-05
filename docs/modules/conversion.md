# 数据与 K230 转换

[返回模块导航](../README.md)

将 NDJSON 数据集转为本地 YOLO 格式，并连接 PyTorch/Ultralytics、ONNX 与 K230 .kmodel 转换校验。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python transform_tools/ndjson_to_yolo.py --help
python transform_tools/py2kmodel.py --help
python transform_tools/PY2KM_validate.py --help
```

## 输入与功能

| 入口 | 输入与用途 |
| --- | --- |
| `ndjson_to_yolo.py` | `--input` NDJSON、`--output` 数据集目录；复用 Ultralytics 转换 API |
| `py2kmodel.py` | `--pt` 权重、`--calib-dir` 校准图片、`--output-dir` 输出目录；生成 ONNX 和 INT8 kmodel |
| `PY2KM_validate.py` | `--onnx`、`--kmodel`、`--image` 单图或目录；主机输出对比 |

这些入口使用 CLI 参数而非统一模型 YAML，相对文件路径按当前工作目录解析。转换所需的 Ultralytics、ONNX、nncase 与校验所需的 ONNX Runtime 等依赖按用途准备。

## 输出

NDJSON 转换输出 YOLO 数据集；模型转换按权重名生成 `.onnx` 和 `.kmodel`。`--rebuild` 才允许覆盖已有模型输出。校验对图片逐张比较并在终端给出整体结论。

## 使用边界

校准图片、输入尺寸、letterbox 填充、RGB/CHW 顺序和归一化须与板端一致。nncase、nncase-kpu 与板端 SDK/runtime 应匹配。主机模拟器结果不能代替 K230 实机验收；数据下载需要网络及磁盘空间。

详细说明：[原模块文档](../../transform_tools/REQUIREMENT.md)。
