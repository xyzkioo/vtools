# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    DepthFormat,
    Format,
    LetterBox,
    RandomLoadText,
    SemanticFormat,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    load_depth,
    polygons2masks_overlap,
    save_dataset_cache_file,
    verify_image,
    verify_image_depth,
    verify_image_label,
    verify_image_mask,
)

# Ultralytics 数据集 *.cache 版本，适用于版本 >= 1.0.0 的 Ultralytics YOLO 模型。
DATASET_CACHE_VERSION = "1.0.4"


class YOLODataset(BaseDataset):
    """以 YOLO 格式加载对象检测和/或分割标签的数据集类。

    此类支持使用 YOLO 格式加载对象检测、实例分割、姿态估计和定向边界框（OBB）任务的数据。

    属性：
        format_class (type[Format]): 由 build_transforms 添加的格式化器，子类会按任务覆盖该属性。
        use_segments (bool): 是否使用分割掩码。
        use_keypoints (bool): 是否使用姿态估计关键点。
        use_obb (bool): 是否使用定向边界框。
        data (dict): 数据集配置字典。

    方法：
        cache_labels: 缓存数据集标签，检查图像并读取尺寸。
        get_labels: 返回用于 YOLO 训练的标签字典列表。
        get_label_files: 返回数据集图像对应的标签文件。
        verify_args: 返回逐图像验证函数及其参数。
        result_to_label: 将单个验证结果转换为标签字典。
        verify_labels: 检查已加载标签的边界框和分割一致性。
        build_transforms: 构建变换并追加到列表。
        close_mosaic: 禁用 mosaic、copy_paste、mixup 和 cutmix 增强，并重新构建变换。
        update_labels_info: 更新不同任务的标签格式。
        collate_fn: 将数据样本整理为批次。

    示例：
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    format_class = Format

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """初始化 YOLODataset。

        参数：
            data (dict, 可选): 数据集配置字典。
            task (str): 任务类型，可选 'detect'、'segment'、'pose' 或 'obb'。
            *args (Any): 传递给父类的其他位置参数。
            **kwargs (Any): 传递给父类的其他关键字参数。
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """缓存数据集标签，检查图像并读取尺寸。

        这是基于文件数据集的通用扫描框架。子类通过 `get_label_files`、`get_cache_hash`、`verify_args`、
        `result_to_label` 和 `scan_summary` 钩子进行定制，无需重复实现此方法。

        参数：
            path (Path): 保存缓存文件的路径。

        返回：
            (dict): 包含缓存标签及相关信息的字典。
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 缺失、找到、空、损坏的数量及消息
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            func, iterable = self.verify_args()
            results = pool.imap(func=func, iterable=iterable)
            pbar = TQDM(results, desc=desc, total=total)
            for result in pbar:
                label, nm_f, nf_f, ne_f, nc_f, msg = self.result_to_label(result)
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if label is not None:
                    x["labels"].append(label)
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {self.scan_summary(nf, nm, ne, nc)}"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            if self.augment:  # 训练需要标签；未标注的验证划分（例如 COCO test-dev）仅发出警告
                raise ValueError(f"{self.prefix}No labels found in {path}. {HELP_URL}")
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = self.get_cache_hash()
        x["results"] = nf, nm, ne, nc, total
        x["msgs"] = msgs  # warnings
        if x["labels"]:
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_label_files(self) -> list[str]:
        """返回数据集图像对应的标签文件，并将其保存到实例属性。

        返回：
            (列表[str]): 标签文件路径列表。
        """
        self.label_files = img2label_paths(self.im_files)
        return self.label_files

    def get_cache_hash(self) -> str:
        """返回用于将标签缓存与当前数据集文件进行校验的哈希值。

        返回：
            (str): 数据集缓存哈希值。
        """
        return get_hash(self.label_files + self.im_files)

    def scan_summary(self, nf: int, nm: int, ne: int, nc: int) -> str:
        """返回扫描计数器的单行摘要，用于进度条和缓存日志。

        参数：
            nf (int): 找到的图像数量。
            nm (int): 缺失标签的数量。
            ne (int): 空标签的数量。
            nc (int): 损坏图像的数量。

        返回：
            (str): 扫描摘要消息。
        """
        return f"{nf} 张图像，{nm + ne} 个背景，{nc} 张损坏图像"

    def verify_args(self) -> tuple:
        """返回 `cache_labels` 使用的逐图像验证函数及其参数可迭代对象。

        返回：
            (tuple): 供 ThreadPool.imap 使用的（验证函数、压缩参数可迭代对象）。
        """
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        return verify_image_label, zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(nkpt),
            repeat(ndim),
            repeat(self.single_cls),
        )

    def result_to_label(self, result: list) -> tuple[dict | None, int, int, int, int, str]:
        """将单个验证结果转换为标签字典，并返回扫描计数增量。

        参数：
            result (列表): `verify_args` 返回的验证函数结果。

        返回：
            (tuple):（标签字典或 None、缺失、找到、空、损坏、消息）。
        """
        im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = result
        label = (
            {
                "im_file": im_file,
                "shape": shape,
                "cls": lb[:, 0:1],  # n, 1
                "bboxes": lb[:, 1:],  # n, 4
                "segments": segments,
                "keypoints": keypoint,
                "normalized": True,
                "bbox_format": "xywh",
            }
            if im_file
            else None
        )
        return label, nm_f, nf_f, ne_f, nc_f, msg

    def verify_labels(self, labels: list[dict], cache_path: Path) -> None:
        """检查数据集是否完全由边界框或分割标签组成，并在需要时移除混合分割标签。

        参数：
            labels (列表[dict]): 标签字典列表。
            cache_path (Path): 数据集缓存文件路径，用于警告消息。
        """
        # 检查数据集是否完全由边界框或分割标签组成
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if self.use_segments and len_boxes != len_segments:
            raise ValueError(
                f"Segment dataset requires equal numbers of boxes and segments, but got len(segments) = "
                f"{len_segments}, len(boxes) = {len_boxes}. Please supply a segment dataset, not a detect dataset."
            )
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")

    def _load_or_scan_cache(self, cache_path: Path, cache_hash: str) -> tuple[dict, bool]:
        """如果数据集缓存文件匹配当前版本和哈希则加载，否则重新扫描并构建缓存。

        参数：
            cache_path (Path): 缓存文件路径。
            cache_hash (str): 数据集文件的预期哈希值。

        返回：
            (tuple):（缓存字典、是否加载了有效现有缓存文件）。
        """
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # 尝试加载 *.cache 文件
            assert cache["version"] == DATASET_CACHE_VERSION  # 匹配当前版本
            assert cache["hash"] == cache_hash  # 哈希值相同
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False  # 执行缓存操作
        return cache, exists

    def get_labels(self) -> list[dict]:
        """返回用于 YOLO 训练的标签字典列表。

        此方法从磁盘或缓存加载标签，验证其完整性，并为训练准备数据。

        返回：
            (列表[dict]): 标签字典列表，每个字典包含一张图像及其标注信息。
        """
        label_files = self.get_label_files()
        cache_path = Path(label_files[0]).parent.with_suffix(".cache")
        cache, exists = self._load_or_scan_cache(cache_path, self.get_cache_hash())

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")  # 找到、缺失、空、损坏、总数
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {self.scan_summary(nf, nm, ne, nc)}"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # 显示结果
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # 显示警告

        # 读取缓存
        labels = cache["labels"]
        if not labels:
            issues = "\n  ".join(sorted(set(cache["msgs"]))) or "no error details"
            raise RuntimeError(f"No valid images found in {cache_path}.\n  {issues}\n{HELP_URL}")
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # 删除这些项目
        self.im_files = [lb["im_file"] for lb in labels]  # 更新图像文件列表
        self.verify_labels(labels, cache_path)
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """构建变换并将其追加到列表中。

        参数：
            hyp (dict, 可选): 变换使用的超参数。

        返回：
            (Compose): 组合后的变换。
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            self.format_class(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # 仅影响训练
            )
        )
        return transforms

    def build_text_transforms(self, transforms: Compose, max_samples: int) -> Compose:
        """为提供 `category_freq` 的文本子类插入文本增强。

        参数：
            transforms (Compose): 由 build_transforms 组合的变换。
            max_samples (int): 每张图像的最大文本样本数量。

        返回：
            (Compose): 启用增强时在 Format 前插入 RandomLoadText 的变换组合。
        """
        if self.augment:
            # 注意：当前参数暂时采用硬编码。
            # 注意：此实现与官方 yoloe 不同，负样本选择策略限制在单个数据集中，
            # 官方实现则一次使用所有数据集预先保存的负样本嵌入。
            transform = RandomLoadText(
                max_samples=min(max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @staticmethod
    def _get_neg_texts(category_freq: dict) -> list[str]:
        """获取出现频率高于数据集阈值的负文本样本。"""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]

    def close_mosaic(self, hyp: dict) -> None:
        """将 mosaic、copy_paste、mixup 和 cutmix 增强的值设为 0.0，以禁用这些增强。

        参数：
            hyp (dict): 变换使用的超参数。
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """更新适用于不同任务的标签格式。

        参数：
            label (dict): 包含边界框、分割线、关键点等信息的标签字典。

        返回：
            (dict): 包含实例信息的更新后标签字典。

        注意：
            当前 cls 不与 bboxes 一起保存，分类和语义分割需要独立的 cls 标签。
            也可以通过添加或删除相应字典键来支持分类和语义分割。
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # 注意：不要对有向边界框重新采样
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # 如果原始长度大于 segment_resamples，确保分割线性插值正确
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # 列表[np.数组(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """将数据样本整理为批次。

        参数：
            batch (列表[dict]): 包含样本数据的字典列表。

        返回：
            (dict): 整理后、张量已堆叠的批次字典。
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # 确保所有样本的键顺序一致
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats", "semantic_mask", "sem_masks", "depth"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        if "batch_idx" in new_batch:
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # 为 build_targets() 添加目标图像索引
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class DepthDataset(YOLODataset):
    """加载配对 RGB 图像和深度图的单目深度估计数据集。

    此类继承 YOLODataset，在加载 RGB 图像的同时加载深度真值图。深度图以 PNG 或 NPY 文件保存，
    与图像目录采用平行结构（图像/train/*.jpg → depth/train/*.{png,npy}）。

    示例：
        >>> dataset = DepthDataset(img_path="/data/nyu/images/train", data={"nc": 1})
    """

    format_class = DepthFormat

    def _depth_path_for(self, im_file: str) -> str:
        """将图像路径映射到对应的 PNG 或 NPY 深度目标。"""
        parts = list(Path(im_file).parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "depth"
                break
        path = Path(*parts).with_suffix(".png")
        return str(path if path.is_file() else path.with_suffix(".npy"))

    def get_label_files(self) -> list[str]:
        """返回与数据集图像配对的深度路径。

        返回：
            (列表[str]): 深度文件路径列表。
        """
        self.depth_files_by_image = {f: self._depth_path_for(f) for f in self.im_files}
        self.depth_files = list(self.depth_files_by_image.values())
        return self.depth_files

    def get_cache_hash(self) -> str:
        """返回配对深度文件和图像文件的哈希值。

        返回：
            (str): 数据集缓存哈希值。
        """
        return get_hash(self.depth_files + self.im_files + [str(self.data.get("depth_scale", 1000))])

    def scan_summary(self, nf: int, nm: int, ne: int, nc: int) -> str:
        """返回图像和深度扫描计数器的单行摘要。"""
        return f"{nf} images, {nm} missing depth, {nc} corrupt"

    def verify_args(self) -> tuple:
        """返回深度验证函数及其参数可迭代对象。"""
        return verify_image_depth, zip(
            self.im_files, self.depth_files, repeat(self.prefix), repeat(self.data.get("depth_scale", 1000))
        )

    def result_to_label(self, result: tuple) -> tuple[dict | None, int, int, int, int, str]:
        """将单个 verify_image_depth 结果转换为标签字典，并返回扫描计数增量。"""
        im_file, shape, nf_f, nm_f, nc_f, msg = result
        label = (
            {
                "im_file": im_file,
                "shape": shape,
                "cls": np.array([], dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "normalized": True,
                "bbox_format": "xywh",
            }
            if im_file
            else None
        )
        return label, nm_f, nf_f, 0, nc_f, msg

    def verify_labels(self, labels: list[dict], cache_path: Path) -> None:
        """跳过边界框和分割检查；深度数据集不包含边界框或分割标注。"""

    def _load_depth(self, index):
        """返回图像的原始分辨率深度图。"""
        return load_depth(self.depth_files_by_image[self.im_files[index]], self.data.get("depth_scale", 1000))

    def get_image_and_label(self, index):
        """加载给定索引对应的图像、标签和深度图。"""
        label = super().get_image_and_label(index)
        h, w = label["resized_shape"]
        depth = self._load_depth(index)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        label["depth"] = depth
        return label

    def build_transforms(self, hyp=None):
        """构建深度估计所需的变换。

        参数：
            hyp (dict): 超参数。

        返回：
            (Compose): 组合后的变换。
        """
        # 注意：以下参数目前不受支持
        hyp.mosaic = hyp.mixup = hyp.cutmix = hyp.copy_paste = 0.0
        transforms = super().build_transforms(hyp)
        if not self.augment:
            # 拉伸图像，而不是进行填充
            transforms[-2] = LetterBox(new_shape=(self.imgsz, self.imgsz), scale_fill=True)
        return transforms


class YOLOMultiModalDataset(YOLODataset):
    """以 YOLO 格式加载对象检测和/或分割标签，并支持多模态输入的数据集类。

    此类扩展 YOLODataset，为多模态模型训练添加文本信息，使模型能够同时处理图像和文本数据。

    方法：
        update_labels_info: 为多模态模型训练添加文本信息。
        build_transforms: 使用文本增强改进数据变换。

    示例：
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """初始化 YOLOMultiModalDataset。

        参数：
            data (dict, 可选): 数据集配置字典。
            task (str): 任务类型，可选 'detect'、'segment'、'pose' 或 'obb'。
            *args (Any): 父类的其他位置参数。
            **kwargs (Any): 父类的其他关键字参数。
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """为多模态模型训练添加文本信息。

        参数：
            label (dict): 包含边界框、分割段、关键点等信息的标签字典。

        返回：
            (dict): 添加实例和文本后的标签字典。
        """
        labels = super().update_labels_info(label)
        # 注意：某些类别会通过 `/` 与其同义词拼接。
        # 注意：存在多个词时，`RandomLoadText` 会随机选择其中一个。
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """使用文本增强改进多模态训练的数据变换。

        参数：
            hyp (dict, 可选): 变换使用的超参数。

        返回：
            (Compose): 组合后的变换；如果适用则包含文本增强。
        """
        return self.build_text_transforms(super().build_transforms(hyp), self.data["nc"])

    @property
    def category_names(self):
        """返回数据集中的类别名称。

        返回：
            (set[str]): 类别名称集合。
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # 类别名称

    @property
    def category_freq(self):
        """返回数据集中每个类别出现的频率。"""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # 检查类别
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
            # 仅包含背景的数据集没有类别，因此每个类别都同样适合作为负样本
        return category_freq or dict.fromkeys((t.strip() for text in texts for t in text), 0)


class GroundingDataset(YOLODataset):
    """使用 grounding 格式 JSON 文件标注执行对象检测任务的数据集类。

    此数据集用于 grounding 任务，标注保存在 JSON 文件中，而不是标准的 YOLO 格式文本文件中。

    属性：
        json_file (str): 包含标注的 JSON 文件路径。

    方法：
        get_labels: 从 JSON 文件加载标注并为训练准备标签。
        build_transforms: 配置训练增强，并支持可选的文本加载。

    示例：
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # 带有标注的有效图像数量
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """初始化用于目标检测的 GroundingDataset。

        参数：
            json_file (str): 包含标注的 JSON 文件路径。
            task (str): GroundingDataset 必须使用 'detect' 或 'segment'。
            max_samples (int): 文本增强加载的最大样本数量。
            *args (Any): 父类的其他位置参数。
            **kwargs (Any): 父类的其他关键字参数。
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list[str]:
        """返回 `img_path` 下的所有图像；实际使用哪些图像由标注而非 `fraction` 决定。"""
        self.fraction = 1.0  # 截断后的清单会使后续图像被排除在缓存键之外
        self.scan_files = super().get_img_files(img_path)
        return self.scan_files

    def get_cache_hash(self) -> str:
        """返回由标注文件和已扫描图像共同生成的哈希值。"""
        return get_hash([self.json_file, *self.scan_files])

    def _verify_instance_counts(self, labels: list[dict[str, Any]]) -> None:
        """验证已知 grounding 数据集的实例数量。"""
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """从 JSON 文件加载标注，为每张图像过滤并归一化边界框。

        参数：
            path (Path): 保存缓存文件的路径。

        返回：
            (dict[str, Any]): 包含缓存标签及相关信息的字典。
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        dropped = False
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            bboxes = []
            segments = []
            segmented = False
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # 类别
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
                    raw_seg = ann.get("segmentation")
                    segmented |= raw_seg is not None
                    seg = raw_seg if isinstance(raw_seg, list) else []
                    polygons = [
                        p
                        for p in seg
                        if isinstance(p, list)
                        and len(p) >= 6
                        and not len(p) % 2
                        and all(isinstance(c, (int, float)) for c in p)
                    ]
                    dropped |= bool(raw_seg) and (not isinstance(raw_seg, list) or len(polygons) < len(seg))
                    if not polygons:  # 每个边界框保留一个分割段，使混合两种标注的图像保持对齐
                        cx, cy, bw, bh = box[1:]
                        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                        segments.append([cls, x1, y1, x2, y1, x2, y2, x1, y2])  # segments2boxes 返回边界框
                        continue
                    elif len(polygons) > 1:
                        s = merge_multi_segment(polygons)
                        s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                    else:
                        s = [j for i in polygons for j in i]  # 所有 分割段 concatenated
                        s = (
                            (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                            .reshape(-1)
                            .tolist()
                        )
                    segments.append([cls, *s])
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segmented:
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb[:, 1:] = segments2boxes(segments)  # 边界框跟随多边形
            else:
                segments = []  # 没有标注包含分割信息，因此不保存掩码

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        if dropped:
            LOGGER.warning(
                f"{self.json_file}: ignored segmentations that are not polygon point lists, such as RLE masks. "
                "Annotations left without a polygon use a segment shaped like their bounding box."
            )
        x["hash"] = self.get_cache_hash()
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """从缓存加载标签，或从 JSON 文件生成标签。

        返回：
            (列表[dict]): 标签字典列表，每个元素包含图像及其标注信息。
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        cache, _ = self._load_or_scan_cache(cache_path, self.get_cache_hash())
        [cache.pop(k) for k in ("hash", "version")]  # 移除无关项
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No images from {self.json_file} found in {self.img_path}. {HELP_URL}")
        if not any(label["texts"] for label in labels):  # category_freq 为空，无法构建负文本
            raise RuntimeError(
                f"No annotations in {self.json_file} survived filtering. Every one is iscrowd, resolves to an empty "
                f"caption span or has a zero-size box. {HELP_URL}"
            )
        self._verify_instance_counts(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """配置训练增强，并可选择加载文本。

        参数：
            hyp (dict, 可选): 变换使用的超参数。

        返回：
            (Compose): 组合后的变换；适用时包含文本增强。
        """
        return self.build_text_transforms(super().build_transforms(hyp), self.max_samples)

    @property
    def category_names(self):
        """返回数据集中的唯一类别名称。"""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """返回数据集中每个类别出现的频率。"""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq


class YOLOConcatDataset(ConcatDataset):
    """将多个数据集串联而成的数据集。

    此类用于组合多个现有数据集以进行 YOLO 训练，并确保它们使用相同的整理函数。

    方法：
        collate_fn: 使用 YOLODataset 整理函数将数据样本整理为批次的静态方法。

    示例：
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """将数据样本整理为批次。

        参数：
            batch (列表[dict]): 包含样本数据的字典列表。

        返回：
            (dict): 包含堆叠张量的整理后批次。
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """将 mosaic、copy_paste、mixup 和 cutmix 增强的值设为 0.0，以禁用这些增强。

        参数：
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


class SemanticDataset(YOLODataset):
    """使用 PNG 掩码标签的语义分割数据集。

    目录结构中每张图像都应有一个同名的 PNG 掩码文件。掩码中的像素值表示类别 ID，255 表示忽略标签。

    掩码目录通过数据集 YAML 中的 'masks_dir' 键指定，并镜像图像目录结构
   （例如，图像/train/ -> 掩码/train/）。

    属性：
        data (dict): 来自 YAML 的数据集配置。
        mask_files (列表[str]): 与图像对应的掩码文件路径列表。
        include_class (np.ndarray | None): 每个像素要保留的类别 ID（None 表示保留所有类别）。
    """

    format_class = SemanticFormat

    def __init__(self, *args, data: dict | None = None, **kwargs):
        """初始化 SemanticDataset.

        参数：
            *args (Any): 父类的其他位置参数。
            data (dict): 数据集配置字典。
            **kwargs (Any): 父类的其他关键字参数。
        """
        self.data = data or {}
        self.label_mapping = self._parse_label_mapping(self.data.get("label_mapping"))
        self.label_lut, self.inverse_lut = self._build_label_luts()
        self.mask_files = []
        self.include_class = None
        super().__init__(*args, data=data, **kwargs)

    def update_labels(self, include_class: list[int] | None) -> None:
        """更新标签，使其只包含指定类别。

        参数：
            include_class (列表[int], 可选): 要保留的类别列表。为 None 时保留所有类别。
        """
        if self.single_cls:
            raise NotImplementedError(
                "'single_cls=True' is not supported for semantic segmentation: it forces a single-channel "
                "model but cannot collapse multi-class masks. Use a dataset with 'nc: 1' for binary "
                "(foreground/background) segmentation instead."
            )
        self.include_class = None if include_class is None else np.asarray(include_class, dtype=np.int32).reshape(-1)
        if self.include_class is not None and int(self.data.get("nc", 0)) == 1:
            LOGGER.warning(
                "'classes' filtering is ignored for single-class (binary) semantic segmentation: keeping only "
                "the sole class would discard all background supervision."
            )
            self.include_class = None

    def _parse_label_mapping(self, mapping):
        """将数据集 YAML 中的 label_mapping 条目规范化为整数到整数的映射。"""
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            raise TypeError(f"Expected 'label_mapping' to be a dict in dataset YAML, but got {type(mapping).__name__}.")

        normalized = {}
        for src, dst in mapping.items():
            src = int(src)
            if isinstance(dst, str):
                dst = dst.strip()
                dst = 255 if dst == "ignore_label" else int(dst)
            elif dst is None:
                dst = 255
            else:
                dst = int(dst)
            normalized[src] = dst
        return normalized

    def _build_label_luts(self) -> tuple[np.ndarray, np.ndarray]:
        """为数据集标签映射构建包含 256 个条目的正向和反向查找表。"""
        forward, inverse = np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8)
        for k, v in self.label_mapping.items():  # 0-255 范围外的 ID 不会匹配 uint8 掩码像素
            if 0 <= k < 256:
                forward[k] = v
            if 0 <= v < 256:
                inverse[v] = k & 0xFF  # cityscapes 将类别映射为 -1；反向调用方会将结果转换为 uint8
        return forward, inverse

    def get_label_files(self) -> list[str]:
        """返回与数据集图像配对的掩码 PNG 路径。

        返回：
            (列表[str]): 掩码文件路径列表。
        """
        self.mask_files = img2label_paths(self.im_files, label_dir=self.data.get("masks_dir", "masks"), suffix=".png")
        return self.mask_files

    def get_cache_hash(self) -> str:
        """返回用于语义缓存验证的哈希值，其中还包含 label_mapping 的变化。

        返回：
            (str): 数据集缓存哈希值。
        """
        mapping = json.dumps(self.label_mapping, sort_keys=True, separators=(",", ":"))
        return get_hash(self.im_files + self.mask_files + [f"label_mapping:{mapping}", "mask_bit_depth"])

    def scan_summary(self, nf: int, nm: int, ne: int, nc: int) -> str:
        """返回图像-掩码扫描计数器的单行摘要。"""
        return f"{nf} images, {nm} missing masks, {nc} corrupt"

    def verify_args(self) -> tuple:
        """返回掩码验证函数及其参数迭代器。"""
        return verify_image_mask, zip(
            self.im_files,
            self.mask_files,
            repeat(self.prefix),
            repeat(int(self.data.get("nc", 0)) == 1),
        )

    def result_to_label(self, result: tuple) -> tuple[dict | None, int, int, int, int, str]:
        """将一次 verify_image_mask 结果转换为标签字典和扫描计数增量。"""
        im_file, mask_file, shape, is_1bit, nm_f, nf_f, nc_f, msg = result
        label = (
            {
                "im_file": im_file,
                "mask_file": mask_file,
                "shape": shape,
                "is_1bit": is_1bit,
                "cls": np.array([], dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "normalized": True,
                "bbox_format": "xywh",
            }
            if im_file
            else None
        )
        return label, nm_f, nf_f, 0, nc_f, msg

    def verify_labels(self, labels: list[dict], cache_path: Path) -> None:
        """跳过边界框和分割检查；语义掩码不包含边界框或分割标注。"""

    def get_labels(self) -> list[dict]:
        """从缓存加载语义标签，或扫描图像和掩码路径。

        返回：
            (列表[dict]): 包含掩码文件路径和图像尺寸的标签字典列表。
        """
        labels = super().get_labels()
        self.mask_files = [lb["mask_file"] for lb in labels]
        return labels

    def load_image(self, i, rect_mode=True):
        """加载用于语义分割的图像；rect_mode=True 时将短边缩放到 imgsz。"""
        return super().load_image(i, rect_mode=rect_mode, resize_short=self.augment)

    def load_mask(self, index: int, image_shape: tuple[int, int] | None = None) -> np.ndarray:
        """加载语义掩码，并应用可选的数据集标签映射。"""
        mask_file = self.labels[index]["mask_file"]
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Semantic mask not found or unreadable: {mask_file}")
        if int(self.data.get("nc", 0)) == 1 and self.labels[index]["is_1bit"]:
            mask[mask == 255] = 1  # cv2 会将 1 位 PNG 前景扩展为 255
        if self.label_mapping:
            mask = self.convert_label(mask, inverse=False)
        return mask.astype(np.uint8, copy=False)

    def convert_label(self, label, inverse=False):
        """使用数据集的标签映射转换标签值。

        参数：
            label (np.ndarray): 包含 0-255 整数 ID 的分割标签数组。
            inverse (bool): 为 True 时应用逆映射（映射值 -> 原始值），默认为 False。

        返回：
            (np.ndarray): 包含转换后值的新 uint8 数组。
        """
        lut = self.inverse_lut if inverse else self.label_lut
        return cv2.LUT(label, lut) if label.dtype == np.uint8 else lut[label]  # cv2.LUT 需要 uint8 输入

    def get_image_and_label(self, index):
        """获取给定索引对应的图像、标签和语义掩码。

        重写父类方法以包含语义掩码，使 Mosaic/CopyPaste 混合图像也能加载对应掩码。

        参数：
            index (int): 数据集索引。

        返回：
            (dict): 包含 'img'、'semantic_mask' 和元数据的标签字典。
        """
        label = super().get_image_and_label(index)
        h, w = label["img"].shape[:2]
        mask = self.load_mask(index, image_shape=(h, w))
        if self.include_class is not None:  # 仅保留选定类别，其余映射为忽略标签
            mask[~np.isin(mask, self.include_class)] = 255
        # 调整掩码尺寸，使其匹配缩放后图像的尺寸
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        label["semantic_mask"] = mask
        return label


class PolygonSemanticDataset(SemanticDataset, YOLODataset):
    """将 YOLO 多边形标签即时栅格化为掩码的语义分割数据集。

        当数据集 YAML 缺少 'masks_dir' 时使用。未被任何多边形覆盖的像素会成为独立的背景类别。
    使用前必须先调用 `add_polygon_background(data)`：当 nc > 1 时，将 `data['nc']` 增加到 user_nc + 1，
    并将背景类别放在 `nc - 1`；当 nc == 1 时保持 nc=1，并栅格化为 {0=背景，1=前景} 二值掩码供
    BCEWithLogitsLoss 使用。
    """

    def __init__(self, *args, data: dict | None = None, **kwargs):
        """初始化 PolygonSemanticDataset.

        参数：
            *args (Any): 传递给父类的其他位置参数。
            data (dict): 数据集配置字典。
            **kwargs (Any): 传递给父类的其他关键字参数。
        """
        nc = (data or {}).get("nc") or len((data or {}).get("names", {}))
        self.bg_class_idx = data.get("bg_class_idx", max(int(nc) - 1, 0))
        super().__init__(*args, data=data, **kwargs)

    # 将标签扫描重新绑定到 YOLODataset 的多边形 .txt 实现；否则 MRO（SemanticDataset、YOLODataset）
    # 会解析到 SemanticDataset 的 PNG 掩码钩子及其 get_labels，而多边形标签字典没有 mask_files。
    get_labels = YOLODataset.get_labels
    get_label_files = YOLODataset.get_label_files
    get_cache_hash = YOLODataset.get_cache_hash
    scan_summary = YOLODataset.scan_summary
    verify_args = YOLODataset.verify_args
    result_to_label = YOLODataset.result_to_label
    verify_labels = YOLODataset.verify_labels

    def load_mask(self, index: int, image_shape: tuple[int, int] | None = None) -> np.ndarray:
        """将当前图像的多边形栅格化为 (H, W) uint8 语义掩码，背景值为 self.bg_class_idx。"""
        h, w = image_shape
        label = self.labels[index]
        cls = label.get("cls")
        segments = label.get("segments") or []
        if cls is None or len(cls) == 0 or len(segments) == 0:
            return np.full((h, w), self.bg_class_idx, dtype=np.uint8)

        # 将多边形（以归一化 xy 保存）反归一化为 (h, w) 图像上的像素坐标。
        scale = np.array([w, h], dtype=np.float32)
        polys = [np.asarray(s, dtype=np.float32).reshape(-1, 2) * scale for s in segments]
        # 返回 (H, W) 实例索引图：0 表示无多边形，1..N 表示排序后的实例索引。
        inst, sorted_idx = polygons2masks_overlap((h, w), polys, downsample_ratio=1)
        out = np.full((h, w), self.bg_class_idx, dtype=np.uint8)
        fg = inst > 0
        if int(self.data.get("nc", 0)) == 1:  # 二值模式下，无论标签 cls 值为何，前景均为 1
            out[fg] = 1
        else:
            cls_arr = np.asarray(cls).reshape(-1).astype(np.int32)[sorted_idx]
            out[fg] = cls_arr[inst[fg] - 1].astype(np.uint8)
        return out


class ClassificationDataset:
    """封装 torchvision ImageFolder 功能、用于图像分类任务的数据集类。

    此类提供图像增强、缓存和校验等功能，旨在高效处理用于深度学习模型训练的大型数据集，
    并通过可选的图像变换和缓存机制加快训练。

    属性：
        cache_ram (bool): 是否启用内存缓存。
        cache_disk (bool): 是否启用磁盘缓存。
        samples (列表): 样本列表，每个元素包含图像路径、类别索引、.npy 缓存文件路径（磁盘缓存时），以及可选的已加载图像数组（内存缓存时）。
        torch_transforms (callable): 应用于图像的 PyTorch 变换。
        root (str): 数据集根目录。
        prefix (str): 日志和缓存文件名使用的前缀。

    方法：
        __getitem__: 返回给定样本索引对应的变换后图像和类别索引。
        __len__: 返回数据集中的样本总数。
        verify_images: 校验数据集中的所有图像。
        cache_images: 将图像解码到连续的内存缓存中。
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """使用根目录、参数、增强和缓存设置初始化 YOLO 分类数据集。

        参数：
            root (str): 数据集目录路径，图像按类别存放在对应文件夹中。
            args (Namespace): 包含数据集设置的配置，例如图像尺寸、增强参数和缓存设置。
            augment (bool, 可选): 是否对数据集应用增强。
            prefix (str, 可选): 日志和缓存文件名使用的前缀，便于识别数据集。
        """
        import torchvision  # 局部导入以加快 'import ultralytics'

        # 将基类作为属性而不是基类继承，以便局部导入速度较慢的 torchvision
        if TORCHVISION_0_18:  # 'allow_empty' 参数首次在 torchvision 0.18 中引入
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        is_ndjson = (Path(root).parent / ".ndjson.yaml").is_file()
        self.samples = self.base.samples
        self.root = self.base.root

        # 初始化属性
        if augment and args.fraction < 1.0:  # 减少训练数据比例
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # 将图像缓存到内存
        self.cache_disk = str(args.cache).lower() == "disk"  # 将图像作为未压缩 *.npy 文件缓存到硬盘
        self.samples = self.verify_images()  # 过滤损坏图像
        if is_ndjson:
            self.samples = [(f, int(Path(f).parent.name)) for f, _ in self.samples]
        if args.single_cls:
            self.samples = [(f, 0) for f, _ in self.samples]
        self.samples = [[*list(x), Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # 文件、索引、npy、图像
        if self.cache_ram:
            self.cache_images()
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> dict:
        """返回给定样本索引对应的变换后图像和类别索引。

        参数：
            i (int): 要获取的样本索引。

        返回：
            (dict): 包含图像及其类别索引的字典。
        """
        f, j, fn, im = self.samples[i]  # 文件名、索引、filename.with_suffix('.npy')、图像
        if self.cache_ram:
            im = self.img_cache[i]
        elif self.cache_disk:
            if not fn.exists():  # 加载 npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # 读取图像
            im = cv2.imread(f)  # BGR
        # 将 NumPy 数组转换为 PIL 图像
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.samples)

    def cache_images(self) -> None:
        """在 DataLoader 工作进程创建前，将所有图像一次性解码到连续的 uint8 缓冲区。

        按图像存储的 Python 数组列表会因写时复制的引用计数机制，被复制到每个派生工作进程中
        （https://github.com/ultralytics/ultralytics/issues/9824）；因此改用一个供所有工作进程只读的共享 NumPy 缓冲区，
        使内存占用保持稳定。变换所需的原始图像尺寸会被保留。
        """
        with ThreadPool(NUM_THREADS) as pool:
            ims = list(
                TQDM(
                    pool.imap(lambda s: cv2.imread(s[0]), self.samples),
                    total=len(self.samples),
                    desc=f"{self.prefix}Caching images",
                    disable=LOCAL_RANK > 0,
                )
            )
        self.img_cache = BaseDataset._ImageCache(ims)

    def verify_images(self) -> list[tuple]:
        """验证数据集中的所有图像。

        返回：
            (列表[tuple]): 验证后有效样本的列表。
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache 文件 路径

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # 检查图像读取速度
            cache = load_dataset_cache_file(path)  # 尝试加载 *.cache 文件
            assert cache["version"] == DATASET_CACHE_VERSION  # 与当前版本匹配
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # 哈希值相同
            nf, nc, n, samples = cache.pop("results")  # found, corrupt, total, samples
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # 显示警告
            return samples

            # 注意：捕获 ModuleNotFoundError，防止加载由其他 NumPy 版本创建的缓存文件时发生版本冲突
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            # *.cache 加载失败时执行扫描
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
