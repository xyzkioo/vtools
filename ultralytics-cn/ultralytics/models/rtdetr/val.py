# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # 元组或列表


class RTDETRDataset(YOLODataset):
    """继承基础 YOLODataset 的实时检测 Transformer（RT-DETR）数据集类。

    此专用数据集类用于 RT-DETR 对象检测模型，并针对实时检测和跟踪任务进行优化。

    属性：
        augment (bool): 是否应用数据增强。
        rect (bool): 是否使用矩形训练。
        use_segments (bool): 是否使用分割掩码。
        use_keypoints (bool): 是否使用关键点标注。
        imgsz (int): 训练目标图像尺寸。

    方法：
        load_image: 从数据集索引加载一张图像。
        build_transforms: 构建数据集变换流程。

    示例：
        初始化 an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """通过继承 YOLODataset 类初始化 RTDETRDataset。

        此构造函数在 YOLODataset 功能基础上，设置专门针对 RT-DETR（实时检测 Transformer）模型优化的数据集。

        参数：
            *args (Any): 传递给父类 YOLODataset 的可变长度参数列表。
            data (dict | None): 包含数据集信息的字典。为 None 时使用默认值。
            **kwargs (Any): 传递给父类 YOLODataset 的其他关键字参数。
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """从数据集索引 'i' 加载一张图像。

        参数：
            i (int): 要加载的图像索引。
            rect_mode (bool, 可选): 是否在批次推理中使用矩形模式。

        返回：
            im (np.ndarray): 加载后的图像 NumPy 数组。
            hw_original (tuple[int, int]): 原始图像尺寸，格式为（高度、宽度）。
            hw_resized (tuple[int, int]): 缩放后图像尺寸，格式为（高度、宽度）。

        示例：
            从数据集中加载图像
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)


class RTDETRValidator(DetectionValidator):
    """继承 DetectionValidator、为 RT-DETR（实时 DETR）对象检测模型提供专用验证能力的 RTDETRValidator。

    此类支持构建 RTDETR 专用验证数据集，对后处理应用置信度阈值，并相应更新评估指标。

    属性：
        args (Namespace): 验证配置参数。
        data (dict): 数据集配置字典。

    方法：
        build_dataset: 构建用于验证的 RTDETR 数据集。
        postprocess: 对预测输出应用置信度阈值。

    示例：
        初始化并运行 RT-DETR 验证
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    注意：
        有关属性和方法的更多细节，请参阅父类 DetectionValidator。
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建 RTDETR 数据集。

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str, 可选): `train` 或 `val` 模式，用户可以为每种模式定制不同增强。
            batch (int, 可选): 批次大小，用于 `rect`。

        返回：
            (RTDETRDataset): 配置用于 RT-DETR 验证的数据集。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # 不使用数据增强
            hyp=self.args,
            rect=False,  # 不使用矩形模式
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """返回未改变的预测结果，因为 RT-DETR 会在后处理中处理缩放。"""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """对预测输出应用后处理。

        Top-k 选择已在解码头内部完成。此方法将归一化 xywh 坐标转换为像素 xyxy 格式。

        参数：
            preds (torch.Tensor | 列表 | tuple): 模型预测结果，形状为 (batch_size, num_queries, 6)，
                最后一维为 [cx, cy, w, h, 分数, 类别]。

        返回：
            (列表[dict[str, torch.Tensor]]): 每张图像对应的字典列表，每个字典包含：
                - 'bboxes'：形状为 (N, 4) 的 xyxy 像素坐标边界框张量
                - 'conf'：形状为 (N,) 的置信度分数张量
                - 'cls'：形状为 (N,) 的类别索引张量
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores, labels = scores.squeeze(-1), labels.squeeze(-1)
        masks = [(score > self.args.conf).nonzero().squeeze(1)[: self.args.max_det] for score in scores]

        return [
            {"bboxes": bbox[m], "conf": score[m], "cls": label[m]}
            for bbox, score, label, m in zip(bboxes, scores, labels, masks)
        ]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果序列化为 COCO JSON 格式。

        参数：
            predn (dict[str, torch.Tensor]): 预测字典，包含 'bboxes'、'conf' 和 'cls' 键，分别对应边界框坐标、
                置信度分数和类别预测结果。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # 原生图像空间预测结果
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # 原生图像空间预测结果
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )
