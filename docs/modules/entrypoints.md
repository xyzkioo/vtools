# 统一命令行与配置

[返回模块导航](../README.md)

把检测诊断、可视化、测速、一致性检查和压缩分发到已有后端，适合脚本调用和重复实验。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python run_tools.py --help
python run_tools.py --tool diagnostics --config model_diagnostics/config/config.yaml
```

## 输入与功能

`--tool` 支持 `diagnostics`、`pytorch`、`tensorrt`、`consistency`、`visualization`、`compression`。一键测速、数据集检查和转换工具使用各自入口。

默认配置映射见 [config/tools.yaml](../../config/tools.yaml)。传入这份配置时，子配置路径相对仓库根目录解析；直接传入子项目 YAML 时，由对应工具解释。

`--only`、`--enable`、`--disable` 覆盖模块选择，`--list-modules` 查询模块 ID。支持这些开关的模型入口中，`--only ''` 表示关闭全部模块，不能与增量开关混用。权重、设备等覆盖项是否适用取决于所选工具，具体以子入口帮助为准。

## 输出

运行结果由对应后端保存，统一入口本身不生成另一套报告。模型 YAML 的相对 `project.root` 按配置文件所在目录解析，权重和输出等路径再按该根目录解析。

## 使用边界

先确认子配置和运行环境；统一入口不会替代 PyTorch、TensorRT 或 nncase 的安装。

详细说明：[原模块文档](../../run_tools.py)。
