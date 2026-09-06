# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
    """处理文本和视觉提示嵌入的 YOLOE 检测验证器。

    此类扩展 DetectionValidator，为 YOLOE 模型提供专用验证功能。
    它支持使用文本提示或从训练样本中提取的视觉提示嵌入进行验证，从而为基于提示的目标检测提供灵活的评估策略。

    属性：
        device (torch.device): 执行验证的设备。
        args (namespace): 验证配置参数。
        dataloader (DataLoader): 验证数据的数据加载器。

    方法：
        get_visual_pe：从训练样本中提取视觉提示嵌入。
        preprocess：预处理批次数据，确保视觉提示与图像位于同一设备。
        get_vpe_dataloader：为 LVIS 训练视觉提示样本创建数据加载器。
        __call__：使用文本或视觉提示嵌入运行验证。

    示例：
        使用文本提示进行验证
        >>> validator = YOLOEDetectValidator()
        >>> stats = validator(model=model, load_vp=False)

        使用视觉提示进行验证
        >>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
        """从训练样本中提取视觉提示嵌入。

        此方法使用 YOLOE 模型处理数据加载器，为每个类别计算视觉提示嵌入。
        它会归一化嵌入，并将没有样本的类别嵌入设为零。

        参数：
            dataloader (torch.utils.data.DataLoader): 提供训练样本的数据加载器。
            model (YOLOEModel): 用于提取视觉提示嵌入的 YOLOE 模型。

        返回：
            (torch.Tensor): 形状为 (1, num_classes, embed_dim) 的视觉提示嵌入。
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "从样本中提取视觉提示嵌入"

        # 统计每个类别的样本数量
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # 提取视觉提示嵌入
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        # 归一化有样本类别的嵌入，将其他类别设为零
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
        """为 LVIS 训练视觉提示样本创建数据加载器。

        此方法使用指定数据集准备视觉提示嵌入（VPE）数据加载器，并为验证应用包括 LoadVisualPrompt 在内的必要变换和配置。

        参数：
            data (dict): 包含路径和设置的数据集配置字典。

        返回：
            (torch.utils.data.DataLoader): 视觉提示样本的数据加载器。
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
            device=self.device,
        )

    @smart_inference_mode()
    def __call__(
        self,
        trainer: Any | None = None,
        model: YOLOEModel | str | None = None,
        refer_data: str | None = None,
        load_vp: bool = False,
    ) -> dict[str, Any]:
        """使用文本或视觉提示嵌入对模型运行验证。

        此方法根据 load_vp 标志使用文本提示或视觉提示验证模型。
        它支持训练期间验证（使用 trainer 对象）或使用给定模型独立验证。
        对于视觉提示，可以指定参考数据集，以从不同数据集中提取嵌入。

        参数：
            trainer (对象, 可选): 包含模型和设备的训练器对象。
            model (YOLOEModel | str, 可选): 要验证的模型；未提供 trainer 时必须指定。
            refer_data (str, 可选): 视觉提示参考数据路径。
            load_vp (bool): 是否加载视觉提示；为 False 时使用文本提示。

        返回：
            (dict): 包含验证期间计算指标的验证统计信息。
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.quantize = None
                # 直接使用训练期间提取视觉嵌入的同一个数据加载器
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device, verbose=False)

            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=self.device)  # 模型, ckpt
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if refer_data is not None:
                eval_data = check_det_dataset(self.args.data)
                eval_names = [name.split("/", 1)[0] for name in list(eval_data["names"].values())]
                if names != eval_names:
                    LOGGER.warning(
                        f"Class names from refer data {names} do not match evaluation dataset {eval_names}. "
                        f"This may lead to incorrect validation results."
                    )

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.quantize = None
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # prompt-free
                return super().__call__(trainer, model)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """支持文本和视觉提示嵌入的 YOLOE 分割验证器。"""
