# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """将分类数据集拆分为新目录中的 train 和 val 目录。

    创建一个新的 '{source_dir}_split' 目录，其中包含 train/val 子目录，并保留原始类别结构，
    默认按 80/20 的比例拆分。仅复制扩展名匹配 IMG_FORMATS 的文件。

    目录结构：
        拆分前：
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        拆分后：
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    参数：
        source_dir (str | Path): 分类数据集根目录的路径。
        train_ratio (float): train 集所占比例，范围为 0 到 1。

    返回：
        (Path): 创建的拆分目录路径。

    示例：
        使用默认的 80/20 比例拆分数据集：
        >>> split_classify_dataset("path/to/caltech")

        使用自定义比例拆分：
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"

    # 创建目录结构
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # 处理类别目录
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len([f for f in d.glob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    for class_dir in class_dirs:
        # 创建 类别 目录
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # 拆分并复制文件
        image_files = [f for f in class_dir.glob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)

        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """自动将数据集拆分为 train/val/test 集，并将拆分结果保存到 autosplit_*.txt 文件。

    参数：
        path (Path): 图像目录路径。
        weights (tuple[float, float, float]): train、验证和测试集所占比例。
        annotated_only (bool): 为 True 时，仅使用存在对应 txt 文件的图像。

    示例：
        使用默认比例拆分图像：
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        使用自定义比例，仅拆分带标注的图像：
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)  # 图像目录
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # 仅保留图像文件
    n = len(files)  # 文件数量
    random.seed(0)  # 确保结果可复现
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 为每张图像分配数据集拆分

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt 文件
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # 删除已有文件

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 检查标签
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # 将图像路径写入 txt 文件


if __name__ == "__main__":
    split_classify_dataset("caltech101")
