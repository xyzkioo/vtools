# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """继承 DetectionValidator 的分割验证器，用于验证分割模型。

    此验证器同时处理边界框和掩码预测结果，并计算检测和分割任务的 mAP 等指标。

    属性：
        process (callable): 根据 save_json 和 save_txt 标志处理掩码的函数。
        args (SimpleNamespace): 验证器参数。
        metrics (SegmentMetrics): 分割任务指标计算器。
        stats (dict): 用于保存验证期间统计信息的字典。

    示例：
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """初始化分割验证器，并将任务设置为 'segment'、指标设置为 SegmentMetrics。

        参数：
            dataloader (torch.utils.data.DataLoader, 可选): 用于验证的数据加载器。
            save_dir (Path, 可选): 结果保存目录。
            args (dict, 可选): 验证器参数。
            _callbacks (dict, 可选): 回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理用于 YOLO 分割验证的图像批次。

        参数：
            batch (dict[str, Any]): 包含图像和标注的批次。

        返回：
            (dict[str, Any]): 预处理后的批次。
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化指标，并根据 save_json 标志选择掩码处理函数。

        参数：
            model (torch.nn.Module): 待验证的模型。
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # 在精度和速度之间选择：保存文件时使用更准确的原生处理
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """返回评估指标的格式化描述字符串。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """后处理 YOLO 预测结果，并返回包含 proto 的检测结果。

        参数：
            preds (列表[torch.Tensor]): 模型输出的原始预测结果。

        返回：
            (列表[dict[str, torch.Tensor]]): 包含掩码的后处理检测预测结果。
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # 从 proto 获取图像尺寸
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """处理图像和目标，准备验证所需的批次数据。

        参数：
            si (int): 样本在批次中的索引。
            batch (dict[str, Any]): 包含图像和标注的批次数据。

        返回：
            (dict[str, Any]): 包含处理后标注的批次数据。
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        super().gather_stats()  # 收集 DetectionValidator 的统计信息
        self._gather_image_metrics(self.metrics.seg)

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """根据边界框和可选掩码计算一个批次的正确预测矩阵。

        参数：
            preds (dict[str, torch.Tensor]): 包含预测结果的字典，例如 'cls' 和 'masks'。
            batch (dict[str, Any]): 包含批次数据的字典，例如 'cls' 和 'masks'。

        返回：
            (dict[str, np.ndarray]): 包含正确预测矩阵的字典，其中 'tp_m' 表示掩码 IoU。

        示例：
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)

        注意：
            - 此方法计算预测掩码与真实掩码之间的 IoU。
            - 根据 overlap_mask 参数设置处理重叠掩码。
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})  # 使用掩码 IoU 更新 tp
        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """绘制包含掩码和边界框的批次预测结果。

        参数：
            batch (dict[str, Any]): 包含图像和标注的批次数据。
            preds (列表[dict[str, torch.Tensor]]): 模型输出的预测结果列表。
            ni (int): 批次索引。
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"验证绘图最多显示 'max_det={self.args.max_det}' 个项目。")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # 绘制边界框

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """以指定格式将 YOLO 检测结果按归一化坐标保存到 txt 文件。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 和 'masks' 键的预测字典。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像尺寸。
            file (Path): 保存检测结果的文件路径。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """保存一条用于 COCO 评估的 JSON 结果。

        参数：
            predn (dict[str, torch.Tensor]): 包含边界框、掩码、置信度分数和类别的预测结果。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。
        """

        def to_string(counts: list[int]) -> str:
            """将 RLE 对象转换为紧凑的字符串表示。每个计数先进行差分编码，再编码为变长字符串。

            参数：
                counts (列表[int]): RLE 计数列表。
            """
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # 对第二个元素之后的所有计数应用差分编码
                if i > 2:
                    x -= int(counts[i - 2])

                # 对数值进行变长编码
                while True:
                    c = x & 0x1F  # 取 5 个比特
                    x >>= 5

                    # 如果符号位（0x10）被设置，则在 x != -1 时继续；否则在 x != 0 时继续
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # 设置续接位
                    c += 48  # 平移到 ASCII 范围
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """使用游程编码（RLE）转换多个二值掩码。

            参数：
                pixels (torch.Tensor): 二维张量，每一行表示一个展平后的二值掩码，形状为 [N, H*W]。

            返回：
                (列表[列表[int]]): 每个掩码对应的 RLE 计数列表。
            """
            num_masks, width = pixels.shape
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            n = len(row_idx)
            # 将每个（掩码、位置）对打包为一个整数，以便紧凑地完成一次设备到主机的数据传输。
            row_idx.mul_(width).add_(col_idx).add_(1)
            del col_idx
            packed = torch.cat((row_idx, pixels[:, 0].to(row_idx.dtype))).cpu().numpy()
            positions, starts = np.split(packed, (n,))
            boundaries = np.searchsorted(positions, np.arange(num_masks + 1) * width)

            # 计算游程长度
            counts = []
            for i in range(num_masks):
                mask_positions = positions[boundaries[i] : boundaries[i + 1]] - i * width
                if mask_positions.size:
                    count = np.diff(mask_positions).tolist()
                    count.insert(0, mask_positions[0])
                    count.append(width - mask_positions[-1])
                else:
                    count = [width]

                # 确保计数从背景（0）开始
                if starts[i] == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """返回 COCO 风格实例分割评估指标。"""
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # 标注
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
