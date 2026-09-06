# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.plotting import plot_images


class OBBValidator(DetectionValidator):
    """继承 DetectionValidator 的验证器，用于验证旋转边界框（OBB）模型。

    此验证器专门评估预测旋转边界框的模型，常用于目标可能呈现不同方向的航空和卫星图像。

    属性：
        args (dict): 验证器配置参数。
        metrics (OBBMetrics): 用于评估 OBB 模型性能的指标对象。
        is_dota (bool): 指示验证数据集是否为 DOTA 格式的标志。

    方法：
        init_metrics：初始化 YOLO 评估指标。
        _process_batch：处理检测结果和真实边界框批次，计算 IoU 矩阵。
        _prepare_batch：准备 OBB 验证所需的批次数据。
        _prepare_pred：准备用于与真实标注比较的预测结果。
        plot_predictions：在输入图像上绘制预测旋转边界框。
        pred_to_json：将 YOLO 预测结果序列化为 COCO JSON 格式。
        save_one_txt：按归一化坐标将 YOLO 检测结果保存到 txt 文件。
        eval_json：评估 JSON 格式的 YOLO 输出并返回性能统计信息。

    示例：
        >>> from ultralytics.models.yolo.obb import OBBValidator
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml")
        >>> validator = OBBValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """初始化 OBBValidator，并将任务设置为 'obb'、指标设置为 OBBMetrics。

        此构造函数初始化用于验证旋转边界框（OBB）模型的 OBBValidator 实例。
        它继承 DetectionValidator，并针对 OBB 任务进行专门配置。

        参数：
            dataloader (torch.utils.data.DataLoader, 可选): 用于验证的数据加载器。
            save_dir (str | Path, 可选): 结果保存目录。
            args (dict, 可选): 包含验证参数的字典。
            _callbacks (dict, 可选): 验证期间调用的回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化 YOLO OBB 验证的评估指标。

        参数：
            model (torch.nn.Module): 待验证的模型。
        """
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation 路径
        self.is_dota = isinstance(val, str) and "DOTA" in val  # 检查数据集是否为 DOTA 格式
        self.confusion_matrix.task = "obb"  # 将混淆矩阵任务设置为 'obb'

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """计算检测结果与真实边界框批次的正确预测矩阵。

        参数：
            preds (dict[str, torch.Tensor]): 包含 'cls' 和 'bboxes' 键的预测字典，表示检测类别标签和边界框。
            batch (dict[str, torch.Tensor]): 包含 'cls' 和 'bboxes' 键的批次字典，表示真实类别标签和边界框。

        返回：
            (dict[str, np.ndarray]): 包含 'tp' 键的字典。正确预测矩阵为形状 (N, 10) 的 NumPy 数组，
                包含每个检测结果在 10 个 IoU 阈值下与真实标注的匹配情况。

        示例：
            >>> preds = {"cls": torch.randint(0, 5, (100,)), "bboxes": torch.rand(100, 5)}
            >>> batch = {"cls": torch.randint(0, 5, (50,)), "bboxes": torch.rand(50, 5)}
            >>> correct_matrix = validator._process_batch(preds, batch)
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """后处理 OBB 预测结果。

        参数：
            preds (torch.Tensor): 模型输出的原始预测结果。

        返回：
            (列表[dict[str, torch.Tensor]]): 后处理后的预测结果，角度信息已拼接到 bboxes。
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # 拼接角度
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """按正确的缩放和格式准备 OBB 验证批次数据。

        参数：
            si (int): 样本在批次中的索引。
            batch (dict[str, Any]): 包含批次数据的字典，键包括：
                - batch_idx：批次索引张量
                - cls：类别标签张量
                - bboxes：边界框张量
                - ori_shape：原始图像尺寸
                - img：图像批次
                - ratio_pad：缩放比例和填充信息

        返回：
            (dict[str, Any]): 包含缩放后边界框和元数据的批次数据。
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # 目标 边界框
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """在输入图像上绘制预测旋转边界框并保存结果。

        参数：
            batch (dict[str, Any]): 包含图像、文件路径和其他元数据的批次数据。
            preds (列表[dict[str, torch.Tensor]]): 批次中每张图像对应的预测字典列表。
            ni (int): 用于命名输出文件的批次索引。

        示例：
            >>> validator = OBBValidator()
            >>> batch = {"img": images, "im_file": paths}
            >>> preds = [{"bboxes": torch.rand(10, 5), "cls": torch.zeros(10), "conf": torch.rand(10)}]
            >>> validator.plot_predictions(batch, preds, 0)
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
        keys = preds[0].keys()
        batched_preds = {k: torch.cat([x[k] for x in preds], dim=0) for k in keys}
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果转换为包含旋转边界框信息的 COCO JSON 格式。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf' 和 'cls' 键的预测字典，分别表示边界框坐标、置信度分数和类别预测结果。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。

        注意：
            此方法处理旋转边界框预测结果，在写入 JSON 字典前将其同时转换为 rbox 格式
            (x, y, w, h, angle) 和多边形格式 (x1, y1, x2, y2, x3, y3, x4, y4)。
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = predn["bboxes"]
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for r, b, s, c in zip(rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "score": round(s, 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """按归一化坐标将 YOLO OBB 检测结果保存到文本文件。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf' 和 'cls' 键的预测字典，分别表示带角度的边界框坐标、置信度分数和类别预测结果。
            save_conf (bool): 是否在文本文件中保存置信度分数。
            shape (tuple[int, int]): 原始图像尺寸，格式为 (高度, 宽度)。
            file (Path): 保存检测结果的输出文件路径。

        示例：
            >>> validator = OBBValidator()
            >>> predn = {
            ...     "bboxes": torch.tensor([[100, 100, 50, 30, 45]]),
            ...     "conf": torch.tensor([0.9]),
            ...     "cls": torch.tensor([0]),
            ... }
            >>> validator.save_one_txt(predn, True, (640, 480), Path("detection.txt"))
        """
        import numpy as np

        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"], predn["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """评估 JSON 格式的 YOLO 输出，并以 DOTA 格式保存预测结果。

        参数：
            stats (dict[str, Any]): Performance statistics 字典.

        返回：
            (dict[str, Any]): Updated performance statistics.
        """
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # 预测结果
            pred_txt = self.save_dir / "predictions_txt"  # 预测结果
            pred_txt.mkdir(parents=True, exist_ok=True)
            with open(pred_json) as f:
                data = json.load(f)
            # 保存切分结果
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # 保存合并结果；与使用官方合并脚本相比，这可能导致 map 略低，
            # 这是由于 probiou 计算造成的。
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # 预测结果
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__", 1)[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # 类别
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 可以获得接近官方合并脚本的结果，甚至可能略好。
                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
