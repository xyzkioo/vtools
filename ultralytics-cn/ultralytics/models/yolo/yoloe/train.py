# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from ..world.train_world import WorldTrainerFromScratch
from .val import YOLOEDetectValidator, YOLOESegValidator


class YOLOETrainer(DetectionTrainer):
    """用于 YOLOE 目标检测模型的训练器。

    此类扩展 DetectionTrainer，为 YOLOE 模型提供专用训练功能，包括自定义模型初始化、验证以及支持多模态的数据集构建。

    属性：
        loss_names (tuple): 损失分量名称，来源于损失准则返回的损失字典。

    方法：
        get_model：初始化并返回适合当前任务的 YOLOE 模型。
        get_validator：返回适合当前任务的 YOLOE 验证器。
        build_dataset：构建支持多模态的 YOLO 训练数据集。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """使用指定配置初始化 YOLOE 训练器。

        参数：
            cfg (dict): 来自 DEFAULT_CFG 的默认训练设置配置字典。
            overrides (dict, 可选): 默认配置的参数覆盖字典。
            _callbacks (dict, 可选): 训练期间要执行的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """使用指定配置和权重初始化并返回适合当前任务的 YOLOE 模型。

        参数：
            cfg (dict | str, 可选): 模型配置，可以是包含 'yaml_file' 键的字典、YAML 文件路径或 None（使用默认配置）。
            weights (str | Path, 可选): 要加载到模型中的预训练权重文件路径。
            verbose (bool): 是否在初始化期间显示模型信息。

        返回：
            (YOLOEModel | YOLOESegModel): 初始化后的 YOLOE 模型。

        注意：
            - 根据官方配置，类别数量（nc）硬编码为最大 80。
            - 此处的 nc 参数表示单张图像中不同文本样本的最大数量，而不是实际类别数量。
        """
        # 注意：此处的 nc 是单张图像中不同文本样本的最大数量，而不是实际的 `nc`。
        # 注意：遵循官方配置，当前将 nc 硬编码为 80。
        model = (YOLOESegModel if self.args.task == "segment" else YOLOEModel)(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回适合当前任务的 YOLOE 验证器。"""
        validator = YOLOESegValidator if self.args.task == "segment" else YOLOEDetectValidator
        return validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建 YOLO Dataset.

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str): 'train' 或 'val' 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, 可选): 批次大小，用于矩形训练。

        返回：
            (Dataset): 配置完成的 YOLO 训练或验证数据集。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )


class YOLOEPETrainer(DetectionTrainer):
    """使用线性探测方法微调 YOLOE 模型。

    此训练器冻结大部分模型层，只训练特定投影层，从而在保留预训练特征的同时高效适配新数据集。

    方法：
        get_model：初始化除投影层外均被冻结的 YOLOE 模型。
        get_validator：返回适合当前任务的验证器。
    """

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """使用指定配置和权重初始化并返回适合当前任务的 YOLOE 模型。

        参数：
            cfg (dict | str, 可选): 模型配置。
            weights (str, 可选): 预训练权重路径。
            verbose (bool): 是否显示模型信息。

        返回：
            (YOLOEModel | YOLOESegModel): 初始化后的模型，除特定投影层外其余层均被冻结。
        """
        model = (YOLOESegModel if self.args.task == "segment" else YOLOEModel)(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # 注意：`get_text_pe` 与文本模型和 YOLOEDetect.reprta 相关，
        # 只要加载正确的预训练权重，就能得到正确结果。
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)  # 将文本嵌入融合到分类检测头
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].one2one_cv3[0][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].one2one_cv3[1][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].one2one_cv3[2][2]).requires_grad_(True)

        model.train()

        return model

    def get_validator(self):
        """返回不需要 YOLOE 验证器所要求文本提示的适合当前任务的验证器。"""
        validator = SegmentationValidator if self.args.task == "segment" else DetectionValidator
        return validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):
    """从头训练支持文本嵌入的 YOLOE 模型。

    此训练器结合 YOLOE 训练能力与 World 训练特征，支持使用文本嵌入和 grounding 数据集从头训练。

    方法：
        build_dataset：构建支持 grounding 的训练数据集。
        generate_text_embeddings：生成并缓存训练所需的文本嵌入。
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集。

        此方法根据模式和输入路径构建合适的数据集，同时处理标准 YOLO 数据集和不同格式的 grounding 数据集。

        参数：
            img_path (列表[str] | str): 包含图像的文件夹路径或路径列表。
            mode (str): 'train' 或 'val' 模式，可为每种模式自定义数据增强。
            batch (int, 可选): 批次大小，用于矩形训练或验证。

        返回：
            (YOLOConcatDataset | Dataset): 构建完成的训练或验证数据集。
        """
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        """为文本样本列表生成文本嵌入。

        参数：
            texts (列表[str]): 要编码的文本样本列表。
            batch (int): 处理批次大小。
            cache_dir (Path): 保存和加载缓存嵌入的目录。

        返回：
            (dict): 将文本样本映射到其嵌入的字典。
        """
        model = unwrap_model(self.model).text_model
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    """训练无提示的 YOLOE 模型。

    此训练器结合线性探测能力和从头训练能力，用于训练推理期间不需要文本提示的无提示 YOLOE 模型。

    方法：
        preprocess_batch：预处理不包含文本特征的批次。
        set_text_embeddings：为数据集设置文本嵌入（无提示模式下为空操作）。
    """

    def preprocess_batch(self, batch):
        """预处理 YOLOE 训练图像批次，并根据需要调整格式和维度。"""
        return DetectionTrainer.preprocess_batch(self, batch)

    def set_text_embeddings(self, datasets, batch: int):
        """无提示训练不需要文本嵌入，因此此处为空操作覆盖。

        参数：
            datasets (列表[Dataset]): 包含待处理类别名称的数据集列表。
            batch (int): 处理文本嵌入时的批次大小。
        """


class YOLOEVPTrainer(YOLOETrainerFromScratch):
    """使用视觉提示训练 YOLOE 模型。

    此训练器扩展 YOLOETrainerFromScratch，支持基于视觉提示的训练，在图像之外提供视觉线索以引导检测过程。

    方法：
        build_dataset：构建包含视觉提示加载变换的数据集。
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """构建包含视觉提示、用于训练或验证的 YOLO 数据集。

        参数：
            img_path (列表[str] | str): 包含图像的文件夹路径或路径列表。
            mode (str): 'train' 或 'val' 模式，可为每种模式自定义数据增强。
            batch (int, 可选): 批次大小，用于矩形训练或验证。

        返回：
            (YOLOConcatDataset | Dataset): 配置完成的训练或验证 YOLO 数据集，训练模式下包含视觉提示。
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def _close_dataloader_mosaic(self):
        """关闭马赛克增强，并向训练数据集添加视觉提示加载。"""
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, YOLOConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())
