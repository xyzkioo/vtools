# 性能测速与一致性

[返回模块导航](../README.md)

比较固定输入下的 PyTorch 模型调用和 TensorRT engine 调用，检查权重恢复、显存和跨后端输出。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python speed_test/run_all.py --help
python speed_test/run_all.py --config speed_test/benchmark_config.yaml
```

## 输入与功能

配置见 [benchmark_config.yaml](../../speed_test/benchmark_config.yaml)。填写模型、adapter、设备、输入尺寸、batch、预热次数和重复次数。

| 入口 | 功能与模块 |
| --- | --- |
| `speed_test/benchmark_pytorch.py` | `speed.pytorch_call`、`speed.pytorch_pipeline`、`memory.pytorch_peak` |
| `speed_test/benchmark_tensorrt.py` | `export.onnx`、`build.tensorrt`、`speed.tensorrt_call` |
| `speed_test/check_consistency.py` | `consistency.tensor`、`consistency.detection` |
| `speed_test/inspect_checkpoint.py` | `checkpoint.inspect`、`checkpoint.load_check` |

`model.parameters`、`model.flops` 用于模型规模统计。内置 adapter 支持 Ultralytics 和完整 Module checkpoint，自定义模型使用 adapter 模板。相对输入路径按 `project.root` 解析。

## 输出

默认输出 `speed_test/runs/runN/`，保存后端 CSV、汇总 CSV、`run_info.json` 和按需生成的 `consistency.csv/json`。一键入口共用一个 run；ONNX/engine 使用配置指定的公共缓存。指标见 [CSV 指南](../../speed_test/CSV_METRICS_GUIDE.md)。

## 使用边界

`model_call`、`adapter_pipeline`、`engine_call` 的计时边界不同，比较时须对齐尺寸、batch、精度和边界。一致性通过不替代数据集精度评估。TensorRT 需要 CUDA GPU；修改模型或环境后应显式重建 engine。当前 PyTorch/TensorRT 单项测速和 checkpoint 入口连 --help 也需要安装 PyTorch。

详细说明：[原模块文档](../../speed_test/README.md)。
