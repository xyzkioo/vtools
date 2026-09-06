# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    """在预训练流程结束时设置模型类别和文本编码器。"""
    # 在所有 rank 上设置：验证会在每个 rank 上运行，但 txt_feats/nc 不是 DDP 缓冲区，因此不会自动同步
    names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
    unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


class WorldTrainer(DetectionTrainer):
    """用于在封闭集数据集上微调 YOLO World 模型的训练器。

    此训练器继承 DetectionTrainer，支持训练结合视觉和文本特征的 YOLO World 模型，以提升目标检测和理解能力。
    它负责生成和缓存文本嵌入，从而加速多模态数据训练。

    属性：
        text_embeddings (dict[str, torch.Tensor] | None): 按类别名称缓存的文本嵌入，用于加速训练。
        model (WorldModel): 正在训练的 YOLO World 模型。
        data (dict[str, Any]): 包含类别信息的数据集配置。
        args (Any): 训练参数和配置。

    方法：
        get_model：使用指定配置和权重返回初始化后的 WorldModel。
        build_dataset：构建用于训练或验证的 YOLO 数据集。
        set_text_embeddings：设置数据集的文本嵌入以加速训练。
        generate_text_embeddings：为文本样本列表生成文本嵌入。
        preprocess_batch：预处理 YOLOWorld 训练所需的图像和文本批次。

    示例：
        初始化并训练 YOLO World 模型
        >>> from ultralytics.models.yolo.world import WorldTrainer
        >>> args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        >>> trainer = WorldTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """使用给定参数初始化 WorldTrainer 对象。

        参数：
            cfg (dict[str, Any]): 训练器配置。
            overrides (dict[str, Any], 可选): 配置覆盖项。
            _callbacks (dict, 可选): 回调函数字典。
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WorldModel:
        """使用指定配置和权重返回初始化后的 WorldModel。

        参数：
            cfg (dict[str, Any] | str, 可选): 模型配置。
            weights (str, 可选): 预训练权重路径。
            verbose (bool): 是否显示模型信息。

        返回：
            (WorldModel): 初始化后的 WorldModel。
        """
        # 注意：这里的 `nc` 是一张图像中不同文本样本数量的上限，而不是实际的 `nc`。
        # 注意：按照官方配置，当前将 nc 硬编码为 80。
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        # 调用方的 Model 与此字典共享对象；直接追加会使回调超出当前训练器生命周期，并在后续训练器中不断叠加
        if on_pretrain_routine_end not in self.callbacks["on_pretrain_routine_end"]:
            self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集。

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str): `train` 或 `val` 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, 可选): 批次大小，用于 `rect` 模式。

        返回：
            (Any): 配置为用于训练或验证的 YOLO 数据集。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)  # 缓存文本嵌入以加速训练
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        """通过缓存类别名称的文本嵌入来加速数据集训练。

        此方法收集所有数据集中的唯一类别名称，然后为这些类别生成并缓存文本嵌入，以提高训练效率。

        参数：
            datasets (列表[Any]): 用于提取类别名称的数据集列表。
            batch (int | None): 处理时使用的批次大小。

        注意：
            此方法从具有 'category_names' 属性的数据集中收集类别名称，然后使用第一个数据集的图像路径确定生成文本嵌入的缓存位置。
        """
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        """为文本样本列表生成文本嵌入。

        参数：
            texts (列表[str]): 要编码的文本样本列表。
            batch (int): 处理时使用的批次大小。
            cache_dir (Path): 保存或加载缓存嵌入的目录。

        返回：
            (dict[str, torch.Tensor]): 文本样本到对应嵌入的映射字典。
        """
        model = "clip:ViT-B/32"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理 YOLOWorld 训练所需的图像和文本批次。"""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        # 添加文本特征
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type not in {"cpu", "mps"}
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
