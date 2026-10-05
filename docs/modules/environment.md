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

`--source` 指定独立 Fork；`--weights` 检查自己的权重；`--yaml` 指定模型结构 YAML，不能填写数据集 YAML。`--check-export` 增加 ONNX/TensorRT API 检查，`--skip-model` 跳过模型构建和推理，`--strict` 收紧检查判定。

CLI 相对路径按当前工作目录解析。Fork 属于同级独立仓库，Python 导入名仍为 `ultralytics`；实际加载路径比安装列表更能说明使用了哪份源码。

## 输出

终端逐项打印环境检查及模型构建、前向传播结果。检查器不安装依赖或修改权重。

## 使用边界

基础导入通过不代表 GPU、导出和部署路径全部通过。跳过模型构建时没有验证模型可运行；可选依赖的警告应结合实际用途判断。

详细说明：[原模块文档](../../env_test/README.md)。
