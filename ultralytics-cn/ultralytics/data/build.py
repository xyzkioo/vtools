# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from copy import copy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import (
    DepthDataset,
    GroundingDataset,
    PolygonSemanticDataset,
    SemanticDataset,
    YOLODataset,
    YOLOMultiModalDataset,
)
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_1_13, TORCH_2_0, TORCH_2_7, get_torch_device_backend


class InfiniteDataLoader(dataloader.DataLoader):
    """在无限迭代中复用工作进程的数据加载器。

    此数据加载器扩展 PyTorch DataLoader，支持无限复用工作进程，从而提高需要多次遍历数据集且无需重复创建工作进程的训练循环效率。

    属性：
        batch_sampler (_RepeatSampler): 无限重复的采样器。
        iterator (Iterator): 父 DataLoader 的迭代器。

    方法：
        __len__: 返回批次采样器中采样器的长度。
        __iter__: 从底层迭代器返回批次。
        __del__: 确保正确终止工作进程。
        close: 不再需要 DataLoader 时平稳关闭持久化工作进程。
        reset: 重置迭代器，适用于训练期间修改数据集设置。

    示例：
        创建用于训练的无限 DataLoader
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # 无限迭代
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """使用与 DataLoader 相同的参数初始化 InfiniteDataLoader。"""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # 早期版本不支持
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """返回批次采样器中采样器的长度。"""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """从持久化迭代器返回一个训练周期的批次。"""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """确保删除 DataLoader 时正确终止工作进程。"""
        try:
            for w in getattr(self.iterator, "_workers", ()):  # 强制终止
                if w.is_alive():
                    w.terminate()
            self.close()
        except Exception:
            pass

    def close(self):
        """关闭持久化工作进程，并在解释器退出前将其从 torch 的 SIGCHLD 监视器中注销。"""
        if hasattr(self.iterator, "_workers"):
            self.iterator._shutdown_workers()  # 等待工作进程结束，并调用 torch._C._remove_worker_pids

    def reset(self):
        """重置迭代器，以便在训练期间修改数据集。"""
        self.close()  # 创建新迭代器前释放旧工作进程管道
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """为无限迭代而无限重复的采样器。

    此采样器包装另一个采样器并无限返回其内容，使数据集可以无限迭代而无需重新创建采样器。

    属性：
        sampler (torch.utils.data.Sampler): 要重复的采样器。
    """

    def __init__(self, sampler: Any):
        """使用要无限重复的采样器初始化 _RepeatSampler。"""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """无限遍历采样器，并依次返回其中的内容。"""
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """将连续且按批次对齐的数据集块分配给每个 GPU 的分布式采样器。

    PyTorch 的 DistributedSampler 会以轮询方式分配样本（GPU 0 获取索引 [0,2,4,...]，GPU 1 获取 [1,3,5,...]），
    而此采样器会为每个 GPU 分配连续的数据集批次（GPU 0 获取批次 [0,1,2,...]，GPU 1 获取 [k,k+1,...] 等）。
    这样可以保留原始数据集中的顺序或分组；当样本按相似性组织时（例如按尺寸对图像排序，以便 rect=True 时高效地进行无填充批处理），这一点非常重要。

    当批次数量无法被进程数整除时，此采样器会将余数批次分配给前几个 rank，确保所有样本在所有 GPU 上恰好覆盖一次。

    参数：
        dataset (Dataset): 要采样的数据集，必须实现 __len__。
        num_replicas (int, 可选): 分布式进程数，默认为 world size。
        batch_size (int, 可选): 数据加载器使用的批次大小，默认为 dataset.batch_size 或 1。
        rank (int, 可选): 当前进程的 rank，默认为当前 rank。
        shuffle (bool, 可选): 是否在每个 rank 的数据块内打乱索引，默认为 False。为 True 时，打乱过程具有确定性，
            并由 set_epoch() 控制以保证可复现。

    示例：
        >>> # 对按尺寸分组的图像进行验证
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # 训练时启用打乱
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """使用数据集和分布式训练参数初始化采样器。"""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        # 当一个输入批次会跨越整个数据集时，使用大小为 1 的批次。
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """计算当前 rank 负责的数据块起始和结束样本索引。"""
        # 计算当前 rank 负责的批次
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas

        # 当 rank 小于余数时，当前 rank 多分配一个批次
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)

        # 计算起始批次：基础位置加上分配给前面 rank 的额外批次数量
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank

        # 将批次索引转换为样本索引
        start_idx = min(start_batch * self.batch_size, self.total_size)
        end_idx = min(end_batch * self.batch_size, self.total_size)

        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """生成当前 rank 负责的连续数据集块索引。"""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        return iter(indices)

    def __len__(self) -> int:
        """返回当前 rank 数据块中的样本数量。"""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """设置采样器周期，以确保不同周期使用不同的打乱模式。

        参数：
            epoch (int): 用作打乱随机种子的训练周期编号。
        """
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """设置数据加载器工作进程的随机种子，确保不同工作进程之间结果可复现。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
    fraction: float | None = None,
) -> Dataset:
    """根据配置参数构建并返回 YOLO 数据集。"""
    pad = 0.0 if mode == "train" else 0.5
    rect = cfg.rect or rect
    if cfg.task == "depth":
        dataset = DepthDataset
        pad, rect = 0.0, rect and mode == "train"  # 深度验证的 letterbox 会拉伸图像，因此忽略 pad 和 rect_shape
    elif cfg.task == "semantic":
        data_path = Path(data.get("path", ""))
        if "masks_dir" in data or (data_path / "masks").exists():
            dataset = SemanticDataset
        else:
            dataset = PolygonSemanticDataset
        pad = 0.0  # 语义分割不使用填充
    elif multi_modal:
        dataset = YOLOMultiModalDataset
    else:
        dataset = YOLODataset

    if fraction is None:
        fraction = cfg.fraction if mode == "train" else 1.0
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=pad,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    """根据配置参数构建并返回 GroundingDataset。"""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # 数据增强
        hyp=copy(cfg),
        rect=cfg.rect or rect,  # 矩形批次
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
    device: torch.device | str = "cuda",
) -> InfiniteDataLoader:
    """创建并返回用于训练或验证的 InfiniteDataLoader。

    参数：
        dataset (Dataset): 要加载数据的数据集。
        batch (int): 数据加载器的批次大小。
        workers (int): 数据加载使用的工作进程数。
        shuffle (bool, 可选): 是否打乱数据集。
        rank (int, 可选): 分布式训练中的进程 rank；单 GPU 训练时为 -1。
        drop_last (bool, 可选): 是否丢弃最后一个不完整批次。
        pin_memory (bool, 可选): 是否为数据加载器使用锁页内存。
        device (torch.device | str, 可选): 数据加载器使用方所在的设备。

    返回：
        (InfiniteDataLoader): 可用于训练或验证的数据加载器。

    示例：
        创建用于训练的数据加载器
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    dataset_len = len(dataset)
    batch = min(batch, dataset_len)
    seed = torch.initial_seed() - RANK - 1
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    samples = len(sampler) if sampler is not None else dataset_len
    drop_last = drop_last and bool(batch) and dataset_len % batch != 0
    batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
    device_type = getattr(device, "type", str(device).split(":")[0])
    nd = get_torch_device_backend(device).device_count() if device_type not in {"cpu", "mps"} else 0
    # 创建的 worker 进程数不超过最终加载器批次数。单批次加载器在进程内运行，
    # 避免持久化 DataLoader worker 池带来额外开销，并防止小型数据集在占用 CUDA 上下文时停滞。
    nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)  # worker 数量
    generator = torch.Generator()
    generator.manual_seed((6148914691236517205 + RANK + seed) % (1 << 64))
    pin_memory = nd > 0 and pin_memory
    pin_memory_device = (
        device_type if pin_memory and device_type in {"npu", "xpu"} and TORCH_1_13 and not TORCH_2_7 else None
    )
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # 从默认值 2 提高
        pin_memory=pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
        **({"pin_memory_device": pin_memory_device} if pin_memory_device else {}),
    )


def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """检查输入源类型，并返回对应的标志值。

    参数：
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): 要检查的输入源。

    返回：
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): 处理后的输入源。
        webcam (bool): 输入源是否为摄像头。
        screenshot (bool): 输入源是否为截图。
        from_img (bool): 输入源是否为图像或图像列表。
        in_memory (bool): 输入源是否为内存中的对象。
        tensor (bool): 输入源是否为 torch.Tensor。

    示例：
        检查文件路径输入源
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        检查摄像头输入源
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # 整数表示本地 USB 摄像头
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[-1] in (
            IMG_FORMATS | VID_FORMATS
        )
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # 下载文件
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # 将列表中的所有元素转换为 PIL 图像或 NumPy 数组
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("不支持的图像类型。支持的类型请参见 https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """加载对象检测的推理源，并应用必要的变换。

    参数：
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): 用于推理的输入源。
        batch (int, 可选): 数据加载器的批次大小。
        vid_stride (int, 可选): 视频源的帧间隔。
        buffer (bool, 可选): 是否缓冲视频流帧。
        channels (int, 可选): 模型输入通道数。

    返回：
        (Dataset): 指定输入源对应的数据集对象，并附带 source_type 属性。

    示例：
        加载图像源用于推理
        >>> dataset = load_inference_source("image.jpg", batch=1)

        加载视频流源
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # 数据加载器
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # 将输入源类型附加到数据集
    dataset.source_type = source_type

    return dataset
