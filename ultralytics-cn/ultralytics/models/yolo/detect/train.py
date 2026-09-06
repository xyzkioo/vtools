# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """用于训练检测模型的 BaseTrainer 子类。

    此训练器专门处理目标检测任务，负责 YOLO 模型训练所需的数据集构建、数据加载、预处理和模型配置。

    属性：
        model (DetectionModel)：正在训练的 YOLO 检测模型。
        data (dict)：包含数据集信息的字典，包括类别名称和类别数量。
        loss_names (tuple)：损失分量名称，来源于损失函数返回的损失字典。

    方法：
        build_dataset：构建用于训练或验证的 YOLO 数据集。
        get_dataloader：为指定模式构建并返回数据加载器。
        preprocess_batch：缩放图像并转换为浮点数，以预处理一个图像批次。
        set_model_attributes：根据数据集信息设置模型属性。
        get_model：返回 YOLO 检测模型。
        get_validator：返回用于模型评估的验证器。
        progress_string：返回格式化的训练进度字符串。
        plot_training_samples：绘制带有标注的训练样本。
        plot_training_labels：创建 YOLO 模型的标注训练图。
        auto_batch：根据模型的显存需求计算最佳批次大小。

    示例：
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化用于训练 YOLO 目标检测模型的 DetectionTrainer 对象。

        参数：
            cfg (dict，可选)：包含训练参数的默认配置字典。
            overrides (dict，可选)：用于覆盖默认配置的参数字典。
            _callbacks (dict，可选)：训练期间要执行的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集。

        参数：
            img_path (str)：包含图像的文件夹路径。
            mode (str)：模式，可选值为 ``'train'`` 或 ``'val'``；不同模式可以使用不同的数据增强设置。
            batch (int，可选)：批次大小，用于 ``'rect'`` 模式。

        返回：
            (Dataset)：为指定模式配置的 YOLO 数据集对象。
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """为指定模式构建并返回数据加载器。

        参数：
            dataset_path (str)：数据集路径。
            batch_size (int)：每个批次的图像数量。
            rank (int)：分布式训练的进程秩。
            mode (str)：模式；``'train'`` 表示训练数据加载器，``'val'`` 表示验证数据加载器。

        返回：
            (DataLoader)：PyTorch 数据加载器对象。
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # DDP 下只初始化一次数据集 *.cache 文件
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """缩放图像并转换为浮点数，以预处理一个图像批次。

        参数：
            batch (dict)：包含批次数据的字典，其中 ``'img'`` 为图像张量。

        返回：
            (dict)：包含归一化图像的预处理批次。
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
                )  # 新尺寸
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（扩展到步长的整数倍）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """根据数据集信息设置模型属性。"""
        # Nl = de_parallel(self.model).model[-1].nl  # 检测层数量（用于缩放超参数）
        # self.args.box *= 3 / nl  # 按层缩放
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 按类别数和层数缩放
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 按图像尺寸和层数缩放
        self.model.nc = self.data["nc"]  # 将类别数量写入模型
        self.model.names = self.data["names"]  # 将类别名称写入模型
        self.model.args = self.args  # 将超参数写入模型
        if getattr(self.model, "end2end", False):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_model_names_for_load(self, model):
        """在加载权重前设置目标数据集名称，以便分类头根据名称重新映射。"""
        if getattr(self.args, "cls_remap", True) and self.data.get("names"):
            model.names = self.data["names"]
        return model

    def get_class_counts(self):
        """返回训练数据集标签中每个类别的实例数量。"""
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def compute_class_weights(self, class_counts):
        """将类别计数转换为逆频率权重，并取 cls_pw 次幂。"""
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        return (1.0 / class_counts) ** self.args.cls_pw  # 直接应用幂运算

    def set_class_weights(self):
        """计算并设置类别权重，以处理类别不平衡。

        类别权重根据训练数据集中的类别逆频率计算，并取 cls_pw 次幂（0 < cls_pw <= 1 可减弱权重差异；取值
        范围限制为 [0, 1]）。最终权重会被归一化，使其平均值等于 1.0。
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        class_counts = self.get_class_counts()
        if not class_counts.any():  # 没有统计到任何类别（例如掩码缺失或无法读取），保留默认权重
            return
        weights = self.compute_class_weights(class_counts)
        weights = weights / weights.mean()  # 归一化，使均值等于 1.0
        model = self.model
        if hasattr(unwrap_model(model), "student_model"):
            model = unwrap_model(model).student_model  # 蒸馏时由学生模型构建损失函数
        model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """返回 YOLO 检测模型。

        参数：
            cfg (str，可选)：模型配置文件路径。
            weights (str，可选)：模型权重。
            verbose (bool)：是否显示模型信息。

        返回：
            (DetectionModel)：YOLO 检测模型。
        """
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def progress_string(self):
        """返回包含轮次、GPU 显存、损失、实例数和图像尺寸的格式化训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带有标注的训练样本。

        参数：
            batch (dict[str, Any])：包含批次数据的字典。
            ni (int)：用于命名输出文件的批次索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建 YOLO 模型的标注训练图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """通过计算模型的显存占用来获取最佳批次大小。

        返回：
            (int)：最佳批次大小。
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # Mosaic 增强使用 4 张图像
        n = len(train_dataset)
        del train_dataset  # 释放内存
        return super().auto_batch(max_num_obj, dataset_size=n)
