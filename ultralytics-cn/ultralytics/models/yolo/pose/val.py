# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """继承 DetectionValidator 的验证器，用于验证姿态模型。

    此验证器专门处理姿态估计任务，负责关键点处理并实现姿态评估所需的专用指标。

    属性：
        sigma (np.ndarray): OKS 计算使用的 sigma 值，可以是 OKS_SIGMA 或 1/关键点数量。
        kpt_shape (列表[int]): 关键点形状，COCO 格式通常为 [17, 3]。
        args (dict): 验证器参数，其中 task 设置为 "pose"。
        metrics (PoseMetrics): 用于姿态评估的指标对象。

    方法：
        preprocess：将关键点数据转换为浮点数并移动到指定设备。
        get_desc：以字符串格式返回评估指标描述。
        init_metrics：初始化 YOLO 姿态估计指标。
        postprocess：后处理 YOLO 预测结果，提取并重塑姿态关键点。
        _prepare_batch：将关键点转换为浮点数并缩放到原始尺寸，准备处理批次。
        _process_batch：计算检测结果与真实标注之间的 IoU，返回正确预测矩阵。
        gather_stats：从所有 GPU 收集统计信息。
        scale_preds：将预测结果缩放到原始图像尺寸。
        save_one_txt：按归一化坐标将 YOLO 姿态检测结果保存到文本文件。
        pred_to_json：将 YOLO 预测结果转换为 COCO JSON 格式。
        eval_json：使用 COCO JSON 格式评估目标检测模型。

    示例：
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    注意：
        此类继承 DetectionValidator 并增加姿态专用功能，使用 sigma 值初始化 OKS 计算，并设置 PoseMetrics 进行评估。
        由于姿态模型存在已知问题，使用 Apple MPS 时会显示警告。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """初始化用于姿态估计验证的 PoseValidator 对象。

        此验证器专门处理姿态估计任务，负责关键点处理并实现姿态评估所需的专用指标。

        参数：
            dataloader (torch.utils.data.DataLoader, 可选): 用于验证的数据加载器。
            save_dir (Path | str, 可选): 结果保存目录。
            args (dict, 可选): 验证器参数，其中 task 设置为 "pose"。
            _callbacks (dict, 可选): 验证期间执行的回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """将关键点数据转换为浮点数并移动到指定设备。"""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """以字符串格式返回评估指标描述。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化 YOLO 姿态验证的评估指标。

        参数：
            model (torch.nn.Module): 待验证的模型。
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        if sigmas := self.data.get("kpt_oks_sigmas"):  # 从数据集 YAML 读取可选的自定义 OKS sigma
            self.sigma = np.array(sigmas, dtype=np.float32).flatten()
            if len(self.sigma) != nkpt or not np.all(self.sigma > 0):
                raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas}")
        else:
            self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """后处理 YOLO 预测结果，提取并重塑姿态估计所需的关键点。

        此方法继承父类后处理流程，从预测结果的 'extra' 字段提取关键点，并根据关键点形状配置重塑它们。
        关键点会从展平格式重塑为正确的维度结构（COCO 姿态格式通常为 [N, 17, 3]）。

        参数：
            preds (torch.Tensor): YOLO 姿态模型输出的原始预测张量，包含边界框、置信度分数、类别预测结果和关键点数据。

        返回：
            (列表[dict[str, torch.Tensor]]): 后处理预测字典列表，每个字典包含：
                - 'bboxes'：边界框坐标
                - 'conf'：置信度分数
                - 'cls'：类别预测结果
                - 'keypoints'：形状为 (-1, *self.kpt_shape) 的重塑关键点坐标

        注意：
            关键点从 'extra' 字段提取，该字段包含基础检测之外的任务专用数据。
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # 移除 extra 字段
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """将关键点转换为浮点数并缩放到原始尺寸，准备处理批次数据。

        参数：
            si (int): 样本在批次中的索引。
            batch (dict[str, Any]): 包含批次数据的字典，键包括 'keypoints'、'batch_idx' 等。

        返回：
            (dict[str, Any]): 包含已缩放关键点的批次数据，关键点对应模型输入（letterbox）图像尺寸。

        注意：
            此方法在父类 _prepare_batch 的基础上增加关键点处理，将关键点从归一化坐标缩放到模型输入（letterbox）图像尺寸。
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """计算检测结果与真实标注之间的交并比（IoU），并返回正确预测矩阵。

        参数：
            preds (dict[str, torch.Tensor]): 包含预测数据的字典，其中 'cls' 为类别预测，'keypoints' 为关键点预测。
            batch (dict[str, Any]): 包含真实数据的字典，其中 'cls' 为类别标签，'bboxes' 为边界框，'keypoints' 为关键点标注。

        返回：
            (dict[str, np.ndarray]): 包含正确预测矩阵的字典，其中 'tp_p' 表示 10 个 IoU 阈值下的姿态真正例。

        注意：
            面积计算中使用的 `0.53` 缩放因子参考自
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` 来自 https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # 使用关键点 IoU 更新 tp
        return tp

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        super().gather_stats()  # 收集 DetectionValidator 的统计信息
        self._gather_image_metrics(self.metrics.pose)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """按归一化坐标将 YOLO 姿态检测结果保存到文本文件。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 和 'keypoints' 键的预测字典。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像尺寸 (高度, 宽度)。
            file (Path): 保存检测结果的输出文件路径。

        注意：
            输出格式为：class_id x_center y_center 宽度 高度 置信度 关键点，其中每个关键点为归一化的 (x, y, visibility) 值。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果转换为 COCO JSON 格式。

        此方法接收预测张量和批次数据，将边界框从 YOLO 格式转换为 COCO 格式，并将包含关键点的结果追加到内部 JSON 字典（self.jdict）。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 和 'keypoints' 张量的预测字典。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。

        注意：
            此方法从文件名主干提取图像 ID（数字文件名转为整数，否则保留字符串），将边界框从 xyxy 转换为 xywh 格式，
            再将坐标从中心点调整为左上角，最后保存到 JSON 字典。
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["keypoints"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # 关键点

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return {
            **super().scale_preds(predn, pbatch),
            "keypoints": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """使用 COCO JSON 格式评估目标检测模型。"""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # 标注
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
