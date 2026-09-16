# 图片数据集文件名转换工具

## 入口

```bash
python others/filename_transform/image_filename_converter.py --help
python others/filename_transform/image_filename_converter.py --dir /path/to/images
```

默认只预览，使用 `--apply` 才会执行改名。

这个工具可以批量修改图片文件名，并同步维护常见数据集标注：

- 普通图片文件夹：只修改图片文件名。
- YOLO：同步修改对应的 `labels/**/*.txt` 文件。
- COCO：同步修改 JSON 中 `images[].file_name`，不改变 `image_id` 和标注内容。
- 支持子文件夹、自然排序、编号模板、预览模式和 CSV 改名清单。

只使用 Python 标准库，不需要额外安装依赖。

## YOLO 数据集

典型目录：

```text
dataset/
├── images/
│   ├── train/
│   │   ├── IMG_001.jpg
│   │   └── IMG_002.jpg
│   └── val/
└── labels/
    ├── train/
    │   ├── IMG_001.txt
    │   └── IMG_002.txt
    └── val/
```

图片和标签必须保持同名主体。工具会执行：

```text
images/train/IMG_001.jpg  → images/train/flower_0001.jpg
labels/train/IMG_001.txt  → labels/train/flower_0001.txt
```

如果 `input_dir` 指向 `dataset` 根目录，工具会自动识别 `images` 和 `labels`；也可以直接把 `input_dir` 指向 `dataset/images`，再填写 `labels_dir`。

命令行示例：

```bash
python3 image_filename_converter.py \
  --dir /path/to/dataset/images \
  --dataset-format yolo \
  --labels-dir /path/to/dataset/labels \
  --prefix flower_ \
  --digits 4
```

如果有背景图没有 `.txt` 标签，工具会跳过不存在的标签文件，不会创建空标签。

## COCO 数据集

COCO 示例目录：

```text
coco_dataset/
├── images/
│   ├── train/000001.jpg
│   └── val/000002.jpg
└── annotations/
    └── instances_train.json
```

工具会同步更新：

```json
{
  "images": [
    {"id": 1, "file_name": "train/000001.jpg"}
  ]
}
```

COCO 原 JSON 在修改前会自动生成带时间戳的 `.bak` 备份文件。默认遇到没有被 JSON 引用的图片会停止，确认确实需要处理时再设置：

```python
"allow_unreferenced": True
```

## 命令行用法

普通图片预览：

```bash
python3 image_filename_converter.py --dir /path/to/images
```

普通图片执行改名：

```bash
python3 image_filename_converter.py \
  --dir /path/to/images \
  --prefix flower_ \
  --digits 4 \
  --apply
```

导出完整清单：

```bash
python3 image_filename_converter.py \
  --dir /path/to/dataset \
  --dataset-format yolo \
  --plan rename_plan.csv
```

## 模板变量

| 变量 | 含义 |
|---|---|
| `{index}` | 从 `start` 开始的编号 |
| `{index0}` | 从 0 开始的编号 |
| `{stem}` | 原文件名，不含扩展名 |
| `{old_name}` | 原完整文件名 |
| `{ext}` | 原扩展名，不含点号 |
| `{prefix}` | `prefix` 配置内容 |
| `{suffix}` | `suffix` 配置内容 |

例如：

```python
"template": "{index:04d}_{stem}",
```

会生成类似 `0001_IMG_001.jpg` 的名称。

## 安全机制

- 默认只预览，不修改数据集。
- 执行前检查目标文件是否重复或已存在，避免覆盖。
- 图片和标签采用临时文件中转，支持文件名循环交换。
- COCO JSON 修改前保留时间戳备份。
- COCO 图片引用不完整时默认停止，避免图片和 JSON 脱节。
