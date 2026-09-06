# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from filelock import AsyncFileLock, Timeout
from PIL import Image

from ultralytics.utils import ASSETS_URL, DATASETS_DIR, LOGGER, NUM_THREADS, TQDM, YAML, clean_url
from ultralytics.utils.checks import check_file
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class() -> list[int]:
    """将 91 索引的 COCO 类别 ID 转换为 80 索引的 COCO 类别 ID。

    返回：
        (列表[int | None]): 长度为 91 的列表，索引表示 91 索引的类别 ID，值表示对应的 80 索引类别 ID；
            如果没有对应映射，则为 None。
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class() -> list[int]:
    r"""将 80 索引（val2014）的类别 ID 转换为 91 索引（论文）的类别 ID。

    返回：
        (list[int]): 80 个类别 ID 组成的列表，每个值都是对应的 91 索引类别 ID。

    示例：
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        将 darknet 格式转换为 COCO 格式
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        将 COCO 格式转换为 darknet 格式
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]

    参考：
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
    lvis: bool = False,
):
    """将 COCO 数据集标注转换为适用于训练 YOLO 模型的 YOLO 标注格式。

    参数：
        labels_dir (str, 可选): COCO 数据集标注文件所在目录的路径。
        save_dir (str, 可选): 保存结果的目录路径。
        use_segments (bool, 可选): 是否在输出中包含分割掩码。
        use_keypoints (bool, 可选): 是否在输出中包含关键点标注。
        cls91to80 (bool, 可选): 是否将 91 个 COCO 类别 ID 映射为对应的 80 个 COCO 类别 ID。
        lvis (bool, 可选): 是否按照 LVIS 数据集方式转换数据。

    示例：
        >>> from ultralytics.data.converter import convert_coco

        将 COCO 标注转换为 YOLO 格式
        >>> convert_coco("coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)

        将 LVIS 标注转换为 YOLO 格式
        >>> convert_coco("lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
    """
    # 创建数据集目录
    save_dir = increment_path(save_dir)  # 如果保存目录已存在，则递增目录名称
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # 创建目录

    # 转换类别
    coco80 = coco91_to_coco80_class()

    # 导入 json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # 文件夹名称
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            # 注意：预先为 train 和 val 创建文件夹，因为 LVIS 验证集除了包含 COCO 2017 验证集图像，
            # 还包含来自 COCO 2017 训练集的图像。
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 创建图像字典
        images = {f"{x['id']:d}": x for x in data["images"]}
        # 创建图像与标注的对应字典
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        image_txt = []
        dropped = False
        # 写入标签文件
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # COCO 边界框格式为 [左上角 x, 左上角 y, 宽度, 高度]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # 从左上角坐标转换为中心坐标
                box[[0, 2]] /= w  # 归一化 x 坐标
                box[[1, 3]] /= h  # 归一化 y 坐标
                if box[2] <= 0 or box[3] <= 0:  # 如果宽度或高度小于等于 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # 类别
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    if use_keypoints:
                        if ann.get("keypoints") is None:
                            continue
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )
                    bboxes.append(box)
                    if use_segments:
                        seg = ann.get("segmentation")
                        polygons = (
                            [
                                p
                                for p in seg or []
                                if isinstance(p, list)
                                and len(p) >= 6
                                and not len(p) % 2
                                and all(isinstance(c, (int, float)) for c in p)
                            ]
                            if isinstance(seg, list)
                            else []
                        )
                        if not polygons:
                            dropped = True
                            cx, cy, bw, bh = box[1:]
                            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                            segments.append([cls, x1, y1, x2, y1, x2, y2, x1, y2])
                        elif len(polygons) > 1:
                            s = merge_multi_segment(polygons)
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])
                        else:
                            s = [j for i in polygons for j in i]  # 拼接所有分割段
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])

            # 写入标签内容
            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, 边界框, 关键点
                    else:
                        line = (*(segments[i] if use_segments else bboxes[i]),)  # cls、边界框或分割段
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if dropped and not use_keypoints:  # 关键点拥有独立输出时，不使用分割段
            LOGGER.warning(
                f"{json_file}: annotations without a usable polygon, because the segmentation is missing, "
                "empty, or not a point list such as an RLE mask, use a segment shaped like their bounding box."
            )

        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


def convert_segment_masks_to_yolo_seg(masks_dir: str, output_dir: str, classes: int):
    """将分割掩码图像数据集转换为 YOLO 分割格式。

    此函数读取包含二值掩码图像的目录，并将其转换为 YOLO 分割格式。
    转换后的掩码会保存到指定的输出目录。

    参数：
        masks_dir (str): 保存所有掩码图像（png、jpg）的目录路径。
        output_dir (str): 保存转换后 YOLO 分割掩码的目录路径。
        classes (int): 数据集中的类别总数，例如 COCO 有 80 个类别。

    示例：
        >>> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        这里的 classes 是数据集中的类别总数，COCO 数据集有 80 个类别
        >>> convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)

    注意：
        掩码的目录结构应为：

            - 掩码
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        执行后，标签将整理为以下结构：

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(Path(masks_dir).iterdir()):
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取掩码图像
            img_height, img_width = mask.shape  # 获取 图像 维度
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # 获取表示不同类别的唯一像素值
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # 跳过背景
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # 为当前类别创建二值掩码并查找轮廓
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # 查找轮廓

                for contour in contours:
                    if len(contour) >= 3:  # YOLO 有效分割至少需要 3 个点
                        contour = contour.squeeze()  # 删除单维度条目
                        yolo_format = [class_index]
                        for point in contour:
                            # 归一化坐标
                            yolo_format.append(round(point[0] / img_width, 6))  # 四舍五入到 6 位小数
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # 将 Ultralytics YOLO 格式数据保存到文件
            output_path = output_dir / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")


def convert_dota_to_yolo_obb(dota_root_path: str):
    """将 DOTA 数据集标注转换为 YOLO OBB（定向边界框）格式。

    此函数处理 DOTA 数据集 'train' 和 'val' 文件夹中的图像。对于每张图像，函数从原始标签目录读取对应标签，
    再将新的 YOLO OBB 格式标签写入新目录。

    参数：
        dota_root_path (str): DOTA 数据集根目录的路径。

    示例：
        >>> from ultralytics.data.converter import convert_dota_to_yolo_obb
        >>> convert_dota_to_yolo_obb("path/to/DOTA")

    注意：
        DOTA 数据集的目录结构应为：

            - DOTA
                ├─ 图像
                │   ├─ train
                │   └─ val
                └─ 标签
                    ├─ train_original
                    └─ val_original

        执行后，函数会将标签整理为：

            - DOTA
                └─ 标签
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # 类别名称到索引的映射
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name: str, image_width: int, image_height: int, orig_label_dir: Path, save_dir: Path):
        """将单张图像的 DOTA 标注转换为 YOLO OBB 格式，并保存到指定目录。"""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ("train", "val"):
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """查找两个二维点数组之间距离最短的一对索引。

    参数：
        arr1 (np.ndarray): 形状为 (N, 2) 的 NumPy 数组，表示 N 个二维点。
        arr2 (np.ndarray): 形状为 (M, 2) 的 NumPy 数组，表示 M 个二维点。

    返回：
        (tuple[int, int]): 元组 (idx1, idx2)，其中 idx1 是 arr1 中的索引，idx2 是 arr2 中的索引，
            两者对应的点之间距离最短。
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """通过连接各分割段之间距离最短的坐标，将多个分割段合并为一个列表。

    此函数用细线连接这些坐标，将所有分割段合并为一个分割段。

    参数：
        segments (列表[列表]): COCO JSON 文件中的原始分割数据。每个元素都是坐标列表，例如
            [segmentation1, segmentation2, ...]。

    返回：
        (列表[np.ndarray]): 由 NumPy 数组表示的已连接分割段列表。
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # 记录每个分割段之间距离最短的索引
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # 分两轮连接所有分割段
    for k in range(2):
        # 正向连接
        if k == 0:
            for i, idx in enumerate(idx_list):
                # 中间分割段有两个索引；反转中间分割段的索引
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # 处理第一个和最后一个分割段
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_bbox2segment(im_dir: str | Path, save_dir: str | Path | None = None, sam_model: str = "sam_b.pt", device=None):
    """将现有的目标检测数据集（边界框）转换为 YOLO 格式的分割数据集。

    必要时使用 SAM 自动标注器生成分割数据。

    参数：
        im_dir (str | Path): 待转换图像目录的路径。
        save_dir (str | Path, 可选): 保存生成标签的路径。如果为 None，标签会保存到与 `im_dir` 同级的
            `labels-segment` 目录。
        sam_model (str): 用于生成中间分割数据的分割模型。
        device (int | str, 可选): 运行 SAM 模型的指定设备。

    注意：
        数据集的输入目录结构应为：

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - 标签
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy

    # 注意：添加占位类别，以通过类别索引检查
    dataset = YOLODataset(im_dir, data={"names": list(range(1000)), "channels": 3})
    if len(dataset.labels[0]["segments"]) > 0:  # 如果已经存在分割数据
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # 跳过空标签
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) < 3:  # 少于 3 个点不是多边形，写入后数据加载器也无法接受
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")


def create_synthetic_coco_dataset():
    """根据标签列表中的文件名创建包含随机图像的合成 COCO 数据集。

    此函数下载 COCO 标签，读取标签列表文件中的图像文件名，为 train2017 和 val2017 子集创建合成图像，
    并将其整理为 COCO 数据集结构。函数使用多线程高效生成图像。

    示例：
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    注意：
        - 下载标签文件需要网络连接。
        - 生成尺寸不同的随机 RGB 图像（480x480 到 640x640 像素）。
        - 删除不需要的现有 test2017 目录。
        - 从 train2017.txt 和 val2017.txt 文件读取图像文件名。
    """

    def create_synthetic_image(image_file: Path):
        """生成具有随机尺寸和颜色的合成图像，用于数据集增强或测试。"""
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)

    # 下载标签
    dir = DATASETS_DIR / "coco"
    download([f"{ASSETS_URL}/coco2017labels-segments.zip"], dir=dir.parent)

    # 创建合成图像
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)  # 删除不需要的 test2017 目录
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in ("train2017", "val2017"):
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            # 从标签列表文件读取图像文件名
            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file, encoding="utf-8") as f:
                    image_files = [dir / line.strip() for line in f]

                # 提交所有任务
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"Generating images for {subset}"):
                    pass  # 实际工作在后台完成
            else:
                LOGGER.warning(f"Labels file {label_list_file} does not exist. Skipping image creation for {subset}.")

    LOGGER.info("Synthetic COCO dataset created successfully.")


def convert_to_multispectral(path: str | Path, n_channels: int = 10, replace: bool = False, zip: bool = False):
    """通过在波长带之间进行插值，将 RGB 图像转换为多光谱图像。

    此函数对 RGB 图像进行插值，生成具有指定通道数量的多光谱图像。
    函数可以处理单张图像或图像目录。

    参数：
        path (str | Path): 待转换图像文件或包含待转换图像的目录路径。
        n_channels (int): 输出图像要生成的光谱通道数量。
        replace (bool): 是否使用转换后的文件替换原始图像文件。
        zip (bool): 是否将转换后的图像压缩为 zip 文件。

    示例：
        转换单张图像
        >>> convert_to_multispectral("path/to/image.jpg", n_channels=10)

        转换数据集
        >>> convert_to_multispectral("coco8", n_channels=10)
    """
    from ultralytics.data.utils import IMG_FORMATS

    path = Path(path)
    if path.is_dir():
        # 处理目录
        im_files = [f for ext in (IMG_FORMATS - {"tif", "tiff"}) for f in path.rglob(f"*.{ext}")]
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f"Error converting {im_path}: {e}")

        if zip:
            zip_directory(path)
    else:
        # 处理单张图像
        output_path = path.with_suffix(".tiff")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        # 使用线性插值和外推，一次性在 RGB 波长范围内处理所有像素
        rgb_wavelengths = np.array([650, 510, 475])  # R、G、B 波长（nm）
        target_wavelengths = np.linspace(450, 700, n_channels)
        order = np.argsort(rgb_wavelengths)  # 按升序排列波长，以便查找分段
        xp = rgb_wavelengths[order]
        seg = np.clip(np.searchsorted(xp, target_wavelengths) - 1, 0, len(xp) - 2)  # 每个目标波长对应的分段
        w = (target_wavelengths - xp[seg]) / (xp[seg + 1] - xp[seg])  # 权重（<0 或 >1 表示外推）
        img = img[..., order]
        multispectral = img[..., seg] * (1 - w) + img[..., seg + 1] * w
        cv2.imwritemulti(str(output_path), np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1))
        LOGGER.info(f"Converted {output_path}")


def _infer_ndjson_kpt_shape(image_records: list) -> list:
    """根据 NDJSON 姿态标注推断 kpt_shape [关键点数量, 维度数]。

    扫描图像记录中最多 50 条姿态标注。标注格式为 [classId, cx, cy, w, h, kp1_x, kp1_y, kp1_vis, ...]，
    因此关键点值从索引 5 开始。

    函数首先尝试维度数为 3（x、y、可见性），并验证可见性取值是否为 {0, 1, 2}；
    当值明确不能被 3 整除时，再回退到维度数为 2（仅 x、y）。
    """
    kpt_lengths = []
    samples = []  # 用于检查可见性的原始关键点值片段
    for record in image_records:
        for ann in record.get("annotations", {}).get("pose", []):
            kpt_len = len(ann) - 5  # 减去 classId 和边界框（4 个值）
            if kpt_len > 0:
                kpt_lengths.append(kpt_len)
                samples.append(ann[5:])
            if len(kpt_lengths) >= 50:
                break
        if len(kpt_lengths) >= 50:
            break

    if not kpt_lengths or len(set(kpt_lengths)) != 1:
        raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose")

    n = kpt_lengths[0]

    # 尝试维度数为 3：长度必须能被 3 整除，且每三个值中的第三个值（可见性）必须属于 {0, 1, 2}
    if n % 3 == 0 and all(v in (0, 1, 2) for s in samples for v in s[2::3]):
        return [n // 3, 3]

    # 尝试维度数为 2：仅当长度不能被 3 整除时使用，避免将维度数为 3 的数据误判
    if n % 2 == 0 and n % 3 != 0:
        return [n // 2, 2]

    raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose")


async def convert_ndjson_to_yolo(ndjson_path: str | Path, output_path: str | Path | None = None) -> Path:
    """将 NDJSON 数据集格式转换为 Ultralytics YOLO 数据集结构。

    此函数将以 NDJSON（按行分隔的 JSON）格式存储的数据集转换为标准 YOLO 格式。
    对于检测、分割、姿态、OBB 任务，会分别创建图像目录和标签目录；深度数据集使用平行的 images/ 和 depth/
    目录树，并保存经过缩放的 uint16 PNG 目标；分类任务使用 ImageNet 风格的 {split}/{class_name}/ 目录结构。
    文件下载会并发执行。

    NDJSON 格式由以下内容组成：
    - 第一行：包含类别名称、任务类型和配置的数据集元数据。
    - 后续各行：包含标注和可选 URL 的单张图像记录。

    参数：
        ndjson_path (str | Path): 包含数据集信息的输入 NDJSON 文件路径。
        output_path (str | Path | None, 可选): 保存转换后 YOLO 数据集的目录。如果为 None，则使用 DATASETS_DIR 目录。

    返回：
        (Path): 生成的 data.yaml 文件路径（检测任务），或数据集目录（分类任务）。

    示例：
        转换本地 NDJSON 文件：
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson")
        >>> print(f"Dataset converted to: {yaml_path}")

        使用自定义输出目录转换：
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson", output_path="./converted_datasets")

        用于 YOLO 训练：
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-ndjson.ndjson")
    """
    source = str(ndjson_path)
    output_path = Path(output_path or DATASETS_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    local = Path(source).is_file()
    source_id = str(Path(source).resolve()) if local else clean_url(source)
    source_hash = hashlib.sha256(source_id.encode()).hexdigest()[:8]
    cache_path = output_path / f".{Path(source_id).stem}-{source_hash}.cache"

    async def convert() -> Path:
        cache_path.unlink(missing_ok=True)
        result = await _convert_ndjson_to_yolo(Path(check_file(source)), output_path, local)
        cache_path.write_text(str(result.relative_to(output_path)))
        return result

    try:
        async with AsyncFileLock(cache_path.with_suffix(".lock"), timeout=0):
            return await convert()
    except Timeout:
        pass

    async with AsyncFileLock(cache_path.with_suffix(".lock")):
        if cache_path.is_file():
            result = output_path / cache_path.read_text()
            marker = result / ".ndjson.yaml" if result.is_dir() else result
            if marker.is_file():
                return result
        return await convert()


async def _convert_ndjson_to_yolo(ndjson_path: Path, output_path: Path, local: bool) -> Path:
    """在持有转换锁时，将已解析的 NDJSON 源转换为 YOLO 数据集。"""
    from ultralytics.utils.checks import check_requirements

    check_requirements("aiohttp")
    import aiohttp

    def read_records():
        with ndjson_path.open() as file:
            return [json.loads(line) for line in file if line.strip()]

    lines = await asyncio.get_running_loop().run_in_executor(None, read_records)
    dataset_record, image_records = lines[0], lines[1:]
    task = dataset_record.get("task", "detect")
    is_classification = task == "classify"
    is_depth = task == "depth"
    depth_scale = dataset_record.get("depth_scale", 1000)
    if is_depth and (
        not isinstance(depth_scale, (int, float))
        or isinstance(depth_scale, bool)
        or not math.isfinite(depth_scale)
        or depth_scale <= 0
    ):
        raise ValueError("Depth datasets require a positive finite depth_scale")
    class_names = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    classification_ids = set()

    local_path = dataset_record.pop("path", None) if local and not (is_classification or is_depth) else None

    # 对稳定内容和源标识进行哈希。排除查询字符串，因为签名 URL 在每次导出时都会变化。
    _h = hashlib.sha256()
    for i, r in enumerate(lines):
        if i:
            split, source_name = r.get("split"), r.get("file")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid NDJSON split: {split!r}")
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f"Invalid NDJSON image name: {source_name!r}")
            if local_path:
                if source_name != Path(source_name).name:
                    raise ValueError(f"Invalid NDJSON image name: {source_name!r}")
                r["url"] = (ndjson_path.parent / local_path / "images" / split / source_name).resolve()
            # 保留文件名或 URL 中已有的安全内容哈希，同时使用索引避免冲突。
            # 深度目标使用相同的主干名称，因此图像和目标 URL 遵循相同的输出机制。
            suffix = source_name.rsplit(".", 1)[-1]
            stems = (Path(clean_url(r.get("url") or "")).stem, Path(source_name).stem)
            content_hash = next(
                (s.lower() for s in stems if len(s) == 32 and all(c in "0123456789abcdef" for c in s)), None
            )
            stem = f"{content_hash}_{i}" if content_hash else i
            r["file"] = f"{stem}.{suffix}" if suffix.isalnum() and len(suffix) <= 10 else f"{stem}.jpg"
            if is_classification:
                ids = r.get("annotations", {}).get("classification", [])
                class_id = ids[0] if ids else 0
                if not isinstance(class_id, int):
                    raise ValueError(f"Invalid NDJSON classification ID: {class_id!r}")
                classification_ids.add(class_id)
        hash_record = {k: v for k, v in r.items() if k != "url"}
        if isinstance(r.get("depth"), dict):
            hash_record["depth"] = {k: v for k, v in r["depth"].items() if k != "url"}
            if r["depth"].get("url"):
                hash_record["depth"]["_source"] = clean_url(r["depth"]["url"])
        if r.get("file"):
            hash_record["_source"] = clean_url(r["url"]) if r.get("url") else str(ndjson_path.parent.resolve())
        _h.update(json.dumps(hash_record, sort_keys=True).encode())
    _hash = _h.hexdigest()[:8]
    class_dirs = {class_id: f"{i:06d}" for i, class_id in enumerate(sorted(classification_ids))}
    classification_names = {i: class_names.get(class_id, str(class_id)) for i, class_id in enumerate(class_dirs)}

    # 深度任务为每条图像记录增加一个同级 URL；文件命名、缓存和重试机制保持共用。
    if is_depth:
        for record in image_records:
            depth = record.get("depth")
            if not isinstance(depth, dict) or not isinstance(depth.get("url"), str) or not depth["url"]:
                raise ValueError(f"Depth record '{record.get('file', '<unknown>')}' is missing depth.url")

    # 带哈希的目录允许相同数据集复用下载结果，同时防止数据集变化时修改其他训练任务仍在读取的文件。
    dataset_dir = output_path / f"{ndjson_path.stem}-{_hash}"
    metadata_path = dataset_dir / (".ndjson.yaml" if is_classification else "data.yaml")
    if metadata_path.is_file():
        try:
            if (cached := YAML.load(metadata_path)).get("hash") == _hash and cached.get("complete") is True:
                return dataset_dir if is_classification else metadata_path
        except Exception:
            pass
    splits = {record["split"] for record in image_records}
    if not is_classification:
        if "train" not in splits:
            raise ValueError(f"Dataset missing required 'train' split. Found splits: {sorted(splits)}")
        if "val" not in splits:
            train_records = [r for r in image_records if r.get("split") == "train"]
            if len(train_records) < 2:
                raise ValueError(
                    f"Dataset has only {len(train_records)} image(s) and no 'val' split. "
                    f"Need at least 2 images to auto-split into train/val."
                )
            random.Random(0).shuffle(train_records)  # 使用本地随机数生成器，避免修改全局训练随机种子
            val_count = max(1, len(train_records) // 10)
            for r in train_records[:val_count]:
                r["split"] = "val"
            splits.add("val")
            LOGGER.warning(
                f"No 'val' split found in dataset. "
                f"Auto-splitting {len(train_records)} images into {len(train_records) - val_count} train, {val_count} val. "
                f"For best results, manually assign validation images in Platform dataset page."
            )

    inferred_nc = None

    if not is_classification:
        class_ids = {
            int(label[0])
            for record in image_records
            for labels in record.get("annotations", {}).values()
            for label in labels
            if label
        }
        if class_ids or class_names:
            max_class_id = max(class_ids | set(class_names))
            if class_names:
                for i in range(max_class_id + 1):
                    class_names.setdefault(i, f"class{i}")
            else:
                inferred_nc = max_class_id + 1
    if task == "pose" and "kpt_shape" not in dataset_record:
        dataset_record["kpt_shape"] = _infer_ndjson_kpt_shape(image_records)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = None

    if not is_classification:
        # 检测、分割、姿态、OBB、深度：准备 YAML 并创建基础目录结构
        if is_depth:
            data_yaml = {"task": "depth", "nc": 1, "names": {0: "depth"}, "depth_scale": depth_scale}
        else:
            data_yaml = dict(dataset_record)
            if class_names:
                data_yaml["names"] = class_names
            elif inferred_nc is not None:
                data_yaml["nc"] = inferred_nc
        data_yaml.pop("class_names", None)
        data_yaml.pop("type", None)  # 删除 NDJSON 专用字段
        for split in sorted(splits):
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / ("depth" if is_depth else "labels") / split).mkdir(parents=True, exist_ok=True)
            data_yaml[split] = f"images/{split}"

    async def ensure_file(session, path, url):
        """文件在本地存在时返回 True，否则按照重试策略从 URL 下载文件。"""
        if path.exists():
            return True
        if not url:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(url, Path):
            if not url.is_file():
                return False
            await asyncio.get_running_loop().run_in_executor(None, shutil.copy2, url, path)
            return True
        for attempt in range(3):
            error = None
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    response.raise_for_status()
                    path.write_bytes(await response.read())
                return True
            except aiohttp.ClientResponseError as e:
                error = e
                if e.status not in {408, 429} and e.status < 500:
                    LOGGER.warning(f"Failed to download {clean_url(url)}: HTTP {e.status}")
                    return False
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = e
            except Exception as e:  # OSError、磁盘空间不足或权限错误不是临时错误，不进行重试
                LOGGER.warning(f"Failed to save {clean_url(url)}: {e}")
                return False
            if attempt < 2:
                await asyncio.sleep(2**attempt)
            else:
                LOGGER.warning(
                    f"Failed to download {clean_url(url)} after 3 attempts: {type(error).__name__ if error else 'unknown'}"
                )
        return False

    async def process_record(session, semaphore, record):
        """使用异步会话处理单条图像记录。"""
        async with semaphore:
            split, original_name = record["split"], record["file"]
            annotations = record.get("annotations", {})

            if is_classification:
                # 分类任务：将图像放入 {split}/{class_name}/ 文件夹
                class_ids = annotations.get("classification", [])
                class_id = class_ids[0] if class_ids else 0
                class_name = class_dirs[class_id]
                image_path = dataset_dir / split / class_name / original_name
            else:
                image_path = dataset_dir / "images" / split / original_name
                if not is_depth:
                    stem = original_name.rsplit(".", 1)[0] or original_name
                    label_path = dataset_dir / "labels" / split / f"{stem}.txt"
                    lines_to_write = []
                    for key in annotations:
                        lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
                        break
                    label_path.write_text("\n".join(lines_to_write) + "\n" if lines_to_write else "")

            image_ok = await ensure_file(session, image_path, record.get("url"))
            if not is_depth:
                return image_ok

            stem = original_name.rsplit(".", 1)[0] or original_name
            depth_path = dataset_dir / "depth" / split / f"{stem}.png"
            depth_ok = await ensure_file(session, depth_path, record["depth"]["url"])
            if not image_ok or not depth_ok:
                image_path.unlink(missing_ok=True)
                depth_path.unlink(missing_ok=True)
                return False
            return True

    # 在大型数据集中保持较高下载并发度，同时避免为每条记录创建一个持续运行的协程。
    semaphore = asyncio.Semaphore(min(128, len(image_records)))
    async with aiohttp.ClientSession(trust_env=True) as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} → {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result

        success_count = 0
        for start in range(0, len(image_records), 1024):
            results = await asyncio.gather(*[tracked_process(record) for record in image_records[start : start + 1024]])
            success_count += sum(results)
        pbar.close()

    # 验证图像是否已成功下载
    if not image_records or success_count < len(image_records):
        raise RuntimeError(f"Downloaded {success_count}/{len(image_records)} images from {ndjson_path}")

    if is_classification:
        # 分类任务：返回数据集目录（check_cls_dataset 需要目录路径）
        # 保持类别路径安全，同时由 check_cls_dataset 恢复原始显示名称。
        YAML.save(metadata_path, {"names": classification_names, "hash": _hash, "complete": True})
        return dataset_dir
    else:
        # 检测任务：写入带哈希的数据.yaml，以便后续检测数据是否发生变化
        data_yaml.update(hash=_hash, complete=True)
        YAML.save(metadata_path, data_yaml)
        return metadata_path
