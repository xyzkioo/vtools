# 数据集质量检查

[返回模块导航](../README.md)

训练前只读扫描本地 YOLO 检测数据集，检查图片、标签和划分问题，无需加载模型。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python model_diagnostics/check_dataset.py --help
python model_diagnostics/check_dataset.py --data /path/to/dataset/data.yaml
```

## 输入与功能

输入为含 `train`、`val`、`test`、`names` 的数据集 YAML。检查损坏或单色图片、缺失或空标签、非法类别和框、字节重复图片、跨划分泄漏，并统计类别、尺寸、框宽高比和重叠。

当前支持五列 YOLO 检测框标签。`--sample-count 0` 可关闭标注抽样图。CLI 的相对路径按当前工作目录解析；YAML 的 `path` 优先相对 YAML 所在目录解析，划分路径再相对数据集根目录解析。

## 输出

默认输出 `model_diagnostics/runs/dataset_quality/runN/`，包含 `summary.json`、`issues.csv`、`report.md` 和 `samples/*.png`。error 级问题退出码为 1，扫描失败为 2；warning/info 进入报告但不使命令失败。

## 使用边界

重复检测使用文件 SHA-256，只能发现字节完全相同的图片。缺失标签可能是背景图，重叠框只能作为遮挡线索，仍需人工复查。

详细说明：[原模块文档](../../model_diagnostics/README.md)。
