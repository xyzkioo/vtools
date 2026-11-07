# 环境检验

[返回模块导航](../README.md)

核对实际 Python、依赖、CUDA 和 Ultralytics 源码来源，帮助定位导入路径与模型构建问题。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python env_test/check_install.py --help
python env_test/check_install.py --source ../ultralytics-ezcn
```

## 输入与功能

默认检查目标 Python 中已安装的包，报告实际版本，不要求本地 Fork、固定 Ultralytics 版本或 YOLO26。Python 3.10、3.12 等版本使用同一套检查规则，依赖能否导入和所选模型能否运行决定结果。

`--source` 显式指定独立 Fork；`--weights` 检查自己的权重；`--yaml` 指定模型结构 YAML，不能填写数据集 YAML。模型只检查指定的输入；没有指定时仅验证基础环境，并提示未验证模型运行。`--check-export` 增加 ONNX/TensorRT API 检查，`--check-compression` 增加模型压缩依赖 `torch-pruning`（要求 `>=1.4.1`）检查，`--skip-model` 跳过模型构建和推理。`--expected-version VERSION` 可选比较指定版本，`--strict` 将该比较的不一致、依赖冲突和可选依赖（导出、压缩）缺失或版本过低判为失败。

CLI 相对路径按当前工作目录解析。Fork 属于同级独立仓库，Python 导入名仍为 `ultralytics`；实际加载路径比安装列表更能说明使用了哪份源码。

## 输出

终端逐项打印环境检查及模型构建、前向传播结果。检查器不安装依赖或修改权重。

## 使用边界

基础导入通过不代表 GPU、导出和部署路径全部通过。跳过模型构建时没有验证模型可运行；可选依赖的警告应结合实际用途判断。

详细说明：[原模块文档](../../env_test/README.md)。
