# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    """继承 BaseTrainer 的训练器，用于训练图像分类模型。

    此训练器负责图像分类任务的训练流程，同时支持 YOLO 分类模型和 torchvision 模型，并提供完整的数据集处理与验证功能。

    属性：
        model (ClassificationModel): 要训练的分类模型。
        data (dict[str, Any]): 包含数据集信息的字典，包括类别名称和类别数量。
        loss_names (tuple): 损失项名称，来自损失函数返回的损失字典。
        validator (ClassificationValidator): 用于模型评估的验证器实例。

    方法：
        set_model_attributes：从加载的数据集中设置模型类别名称。
        get_model：返回配置好的修改版 PyTorch 分类模型。
        setup_model：为分类任务加载、创建或下载模型。
        build_dataset：创建 ClassificationDataset 实例。
        get_dataloader：返回包含图像预处理变换的 PyTorch 数据加载器。
        preprocess_batch：预处理图像和类别组成的批次。
        progress_string：返回显示训练进度的格式化字符串。
        get_validator：返回 ClassificationValidator 实例。
        final_eval：评估训练后的模型并保存验证结果。
        plot_training_samples：绘制带有标注的训练样本。

    示例：
        初始化并训练分类模型
        >>> from ultralytics.models.yolo.classify import ClassificationTrainer
        >>> args = dict(model="yolo26n-cls.pt", data="imagenet10", epochs=3)
        >>> trainer = ClassificationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化 ClassificationTrainer 对象。

        参数：
            cfg (dict[str, Any], 可选): 包含训练参数的默认配置字典。
            overrides (dict[str, Any], 可选): 用于覆盖默认配置的参数字典。
            _callbacks (dict, 可选): 训练期间执行的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """从加载的数据集中设置 YOLO 模型的类别名称。"""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回配置好的修改版 PyTorch YOLO 分类模型。

        参数：
            cfg (Any, 可选): 模型配置。
            weights (Any, 可选): 预训练模型权重。
            verbose (bool, 可选): 是否显示模型信息。

        返回：
            (ClassificationModel): 配置好的 PyTorch 分类模型。
        """
        model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if self.args.pretrained is False and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # 设置 dropout
        for p in model.parameters():
            p.requires_grad = True  # 用于训练
        return model

    def setup_model(self):
        """为分类任务加载、创建或下载模型。

        返回：
            (Any): 适用时返回模型检查点，否则返回 None。
        """
        import torchvision  # scope for faster 'import ultralytics'

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """根据图像路径和模式创建 ClassificationDataset 实例。

        参数：
            img_path (str): 数据集图像路径。
            mode (str, 可选): 数据集模式（'train'、'val' 或 'test'）。
            batch (Any, 可选): 批次信息（本实现未使用）。

        返回：
            (ClassificationDataset): 指定模式的数据集。
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """返回包含图像预处理变换的 PyTorch 数据加载器。

        参数：
            dataset_path (str): 数据集路径。
            batch_size (int, 可选): 每个批次的图像数量。
            rank (int, 可选): 分布式训练的进程排名。
            mode (str, 可选): 'train'、'val' 或 'test' 模式。

        返回：
            (torch.utils.data.DataLoader): 指定数据集和模式的数据加载器。
        """
        with torch_distributed_zero_first(rank):  # DDP 下仅初始化一次数据集 *.cache
            dataset = self.build_dataset(dataset_path, mode)

        if not dataset.samples:
            raise FileNotFoundError(
                f"No images found in '{mode}' split of {dataset_path}. "
                f"See https://docs.ultralytics.com/datasets/classify for cls dataset format."
            )

        # 过滤类别索引 >= nc 的样本（避免 CUDA 断言错误）
        nc = self.data.get("nc", 0)
        dataset_nc = max(x[1] for x in dataset.samples) + 1
        if nc and dataset_nc > nc:
            extra_classes = dataset.base.classes[nc:]
            original_count = len(dataset.samples)
            dataset.samples = [s for s in dataset.samples if s[1] < nc]
            skipped = original_count - len(dataset.samples)
            LOGGER.warning(
                f"{mode} split has {dataset_nc} classes but model expects {nc}. "
                f"Skipping {skipped} samples from extra classes: {extra_classes}"
            )
            if not dataset.samples:
                raise RuntimeError(
                    f"All {original_count} samples in '{mode}' split filtered out: every sample had class index >= "
                    f"model nc={nc}. Reset the model's class count or align dataset class indices."
                )
        drop_last = self.args.compile and mode == "train"
        loader = build_dataloader(
            dataset, batch_size, self.args.workers, rank=rank, drop_last=drop_last, device=self.device
        )
        # 附加推理变换
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """预处理图像和类别组成的批次。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        return batch

    def progress_string(self) -> str:
        """返回显示训练进度的格式化字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """返回用于验证的 ClassificationValidator 实例。"""
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        """绘制带有标注的训练样本。

        参数：
            batch (dict[str, torch.Tensor]): 包含图像和类别标签的批次数据。
            ni (int): 用于命名输出文件的批次索引。
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # 添加批次索引用于绘图
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
