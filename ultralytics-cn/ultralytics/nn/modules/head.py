# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模型检测头模块。"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "OBB",
    "Classify",
    "Depth",
    "Detect",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "SemanticSegment",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)


class Detect(nn.Module):
    """用于目标检测模型的 YOLO Detect 检测头。

    此类实现 YOLO 模型使用的检测头，用于预测边界框和类别概率。同时支持训练和推理模式，并可选用端到端检测。

    属性：
        dynamic (bool)：是否强制重建网格。
        export (bool)：导出模式标志。
        format (str)：导出格式。
        end2end (bool)：是否使用端到端检测模式。
        max_det (int)：每张图像允许的最大检测数量。
        shape (tuple)：输入形状。
        anchors (torch.Tensor)：锚点。
        strides (torch.Tensor)：特征图步长。
        legacy (bool)：兼容 v3/v5/v8/v9/v11 模型。
        xyxy (bool)：输出格式，可以是 xyxy 或 xywh。
        nc (int)：类别数量。
        nl (int)：检测层数量。
        reg_max (int)：DFL 通道数。
        no (int)：每个锚框的输出数量。
        stride (torch.Tensor)：构建期间计算得到的步长。
        cv2 (nn.ModuleList)：用于边界框回归的卷积层。
        cv3 (nn.ModuleList)：用于分类的卷积层。
        dfl (nn.Module)：分布式焦点损失层。
        one2one_cv2 (nn.ModuleList)：一对一边界框回归卷积层。
        one2one_cv3 (nn.ModuleList)：一对一分类卷积层。

    方法：
        forward：执行前向传播并返回预测结果。
        bias_init：初始化检测头偏置。
        decode_bboxes：从预测结果解码边界框。
        postprocess：对模型预测结果执行后处理。

    示例：
        创建一个用于 80 个类别的检测头。
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # 强制重建网格
    export = False  # 导出模式
    format = None  # 导出格式
    max_det = 300  # 最大检测数量
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # 兼容 v3/v5/v8/v9 模型
    xyxy = False  # 输出 xyxy 或 xywh 格式

    @staticmethod
    def _grouped_topk(x: torch.Tensor, k: int, groups: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """通过较小的分组选择精确选取 top-k 值。"""
        n = x.shape[1]
        while groups > 1 and (n % groups or n // groups < k):
            groups //= 2
        if groups == 1:  # 无法获得收益，例如轴较短或无法被均匀分组
            return x.topk(k, dim=1)
        size = n // groups
        values, index = x.reshape(x.shape[0], groups, size).topk(k, dim=-1)
        values, winners = values.flatten(1).topk(k, dim=1)
        return values, winners // k * size + index.flatten(1).gather(1, winners)

    def _gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """沿维度 1，从 x（形状为 batch、n、通道）中选择索引（batch、k）对应的行。"""
        return x.gather(1, index if x.ndim == 2 else index[..., None].expand(-1, -1, x.shape[-1]))

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """使用指定的类别数量和通道数初始化 YOLO 检测层。

        参数：
            nc (int)：类别数量。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：来自主干特征图的通道数元组。
        """
        super().__init__()
        self.nc = nc  # 类别数量
        self.nl = len(ch)  # 检测层数量
        self.reg_max = reg_max  # DFL 通道数
        self.no = nc + self.reg_max * 4  # 每个锚框的输出数量
        self.stride = torch.zeros(self.nl)  # 构建期间计算的步长
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 通道
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于兼容 v3/v5/v8/v9/v11。"""
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    @property
    def end2end(self):
        """检查模型是否包含 one2one 检测头，用于兼容 v3/v5/v8/v9/v11。"""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框和类别概率。"""
        if box_head is None or cls_head is None:  # 融合后的推理不需要检测头
            return {}
        bs = x[0].shape[0]  # 批次大小
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """拼接并返回预测的边界框和类别概率。"""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x] if self.training else x  # 分离张量，避免一对一分支影响主干网络
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """根据多层特征图解码预测的边界框和类别概率。

        参数：
            x (dict[str, torch.Tensor])：检测层输出的预测结果字典。

        返回：
            (torch.Tensor)：拼接后的张量，包含解码后的边界框和类别概率。
        """
        # 推理路径
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """根据锚点和步长获取解码后的边界框。"""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """初始化 Detect() 的偏置。注意：必须先计算步长。"""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0  # 边界框
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 对象, 80 类别, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # 一对一分支
                a[-1].bias.data[:] = 2.0  # 边界框
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 对象, 80 类别, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """根据预测结果解码边界框。"""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """对 YOLO 模型的预测结果执行后处理。

        参数：
            preds (torch.Tensor)：原始预测结果，形状为 (batch_size, num_anchors, 4 + nc + extra)，最后一维的格式为
                [x1, y1, x2, y2, class_probs, extra]。其中 extra 包含 Segment、Pose 和 OBB 检测头的掩码系数、
                关键点或角度；Detect 检测头不包含此部分。

        返回：
            (torch.Tensor)：处理后的预测结果，形状为 (batch_size, min(max_det, num_anchors), 6 + extra)，最后一维的格式为
                [x1, y1, x2, y2, max_class_prob, class_index, extra]。
        """
        # Segment、Pose 和 OBB 会在类别分数之后携带任务相关通道，Detect 不包含额外通道。
        boxes, scores, *extra = preds.split([s for s in (4, self.nc, preds.shape[-1] - 4 - self.nc) if s], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        return torch.cat([self._gather(boxes, idx), scores, conf, *(self._gather(e, idx) for e in extra)], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从分数中获取 top-k 索引。

        参数：
            scores (torch.Tensor)：分数张量，形状为 (batch_size, num_anchors, num_classes)。
            max_det (int)：每张图像允许的最大检测数量。

        返回：
            (torch.Tensor, torch.Tensor, torch.Tensor)：最高分数、类别索引和筛选后的索引。
        """
        anchors, nc = scores.shape[1:]  # i.e. 形状(16,8400,80)
        k = min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1)
            scores, index = self._grouped_topk(scores, k, 1)
            return scores[..., None], self._gather(labels[..., None].float(), index), index
        groups = 8 if self.export and self.format == "engine" and not self.dynamic else 1
        ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1]
        scores = self._gather(scores, ori_index)
        scores, index = self._grouped_topk(scores.flatten(1), k, groups)
        return scores[..., None], (index % nc)[..., None].float(), self._gather(ori_index, index // nc)

    def fuse(self) -> None:
        """移除一对多检测头，以优化推理过程。"""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """用于分割模型的 YOLO Segment 检测头。

    此类扩展 Detect 检测头，增加实例分割任务所需的掩码预测能力。

    属性：
        nm (int)：掩码数量。
        npr (int)：原型数量。
        proto (Proto)：原型生成模块。
        cv4 (nn.ModuleList)：用于生成掩码系数的卷积层。

    方法：
        forward：返回模型输出和掩码系数。

    示例：
        创建一个分割检测头。
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 分割检测头，包括掩码数量、原型数量和卷积层。

        参数：
            nc (int)：类别数量。
            nm (int)：掩码数量。
            npr (int)：原型数量。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型生成模块

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于保持向后兼容。"""
        return {"box_head": self.cv2, "cls_head": self.cv3, "mask_head": self.cv4}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3, "mask_head": self.one2one_cv4}

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """在训练时返回模型输出和掩码系数；在推理时同样返回模型输出和掩码系数。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # 掩码原型
        if isinstance(preds, dict):  # 训练和验证期间
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，并与掩码系数拼接。"""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和掩码系数。"""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def fuse(self) -> None:
        """移除一对多检测头，以优化推理过程。"""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """用于分割模型的 YOLO26 Segment 检测头。

    此类扩展 Segment 检测头，使用 Proto26 为实例分割任务生成掩码预测。

    属性：
        nm (int)：掩码数量。
        npr (int)：原型数量。
        proto (Proto26)：原型生成模块。
        cv4 (nn.ModuleList)：用于生成掩码系数的卷积层。

    方法：
        forward：返回模型输出和掩码系数。

    示例：
        创建一个分割检测头。
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO26 分割检测头，包括掩码数量、原型数量和卷积层。

        参数：
            nc (int)：类别数量。
            nm (int)：掩码数量。
            npr (int)：原型数量。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # 原型生成模块

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """在训练时返回模型输出和掩码系数；在推理时同样返回模型输出和掩码系数。"""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # 掩码原型
        if isinstance(preds, dict):  # 训练和验证期间
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """移除一对多检测头和原型模块的额外部分，以优化推理过程。"""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """用于旋转目标检测模型的 YOLO OBB 检测头。

    此类扩展 Detect 检测头，增加带旋转角度的有向边界框预测能力。

    属性：
        ne (int)：额外参数的数量。
        cv4 (nn.ModuleList)：用于预测角度的卷积层。
        angle (torch.Tensor)：预测的旋转角度。

    方法：
        forward：拼接并返回预测的边界框和类别概率。
        decode_bboxes：解码旋转边界框。

    示例：
        创建一个 OBB 检测头。
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """使用类别数量 `nc` 和各层通道数 `ch` 初始化 OBB 检测头。

        参数：
            nc (int)：类别数量。
            ne (int)：额外参数的数量。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # 额外参数的数量

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于保持向后兼容。"""
        return {"box_head": self.cv2, "cls_head": self.cv3, "angle_head": self.cv4}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3, "angle_head": self.one2one_cv4}

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，并与旋转角度拼接。"""
        # 方便 decode_bboxes 使用
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和角度。"""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # 批次大小
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB 角度 logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # 角度范围为 [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """解码旋转边界框。"""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def fuse(self) -> None:
        """移除一对多检测头，以优化推理过程。"""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """用于旋转目标检测模型的 YOLO26 OBB 检测头。

    此类扩展 OBB 检测头，修改角度处理方式，直接输出未经 sigmoid 变换的原始角度预测结果。

    属性：
        ne (int)：额外参数的数量。
        cv4 (nn.ModuleList)：用于预测角度的卷积层。
        angle (torch.Tensor)：预测的旋转角度。

    方法：
        forward_head：拼接并返回预测的边界框、类别概率和原始角度。

    示例：
        创建一个 OBB26 检测头。
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和原始角度。"""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # 批次大小
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB 角度 logits（未经 sigmoid 变换的原始输出）
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """用于关键点模型的 YOLO Pose 检测头。

    此类扩展 Detect 检测头，增加姿态估计任务所需的关键点预测能力。

    属性：
        kpt_shape (tuple)：关键点数量和维度（2 表示 x、y，3 表示 x、y、可见性）。
        nk (int)：关键点值的总数量。
        cv4 (nn.ModuleList)：用于预测关键点的卷积层。

    方法：
        forward：执行 YOLO 模型的前向传播并返回预测结果。
        kpts_decode：根据预测结果解码关键点。

    示例：
        创建一个姿态检测头。
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """使用默认参数和卷积层初始化 YOLO 姿态检测头。

        参数：
            nc (int)：类别数量。
            kpt_shape (tuple)：关键点数量和维度（2 表示 x、y，3 表示 x、y、可见性）。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # 关键点数量和维度（2 表示 x、y，3 表示 x、y、可见性）
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点值的总数量

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于保持向后兼容。"""
        return {"box_head": self.cv2, "cls_head": self.cv3, "pose_head": self.cv4}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3, "pose_head": self.one2one_cv4}

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，并与关键点拼接。"""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和关键点。"""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def fuse(self) -> None:
        """移除一对多检测头，以优化推理过程。"""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """根据预测结果解码关键点。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """用于关键点模型的 YOLO26 Pose 检测头。

    此类扩展 Pose 检测头，使用归一化流完成姿态估计任务中的关键点预测。

    属性：
        kpt_shape (tuple)：关键点数量和维度（2 表示 x、y，3 表示 x、y、可见性）。
        nk (int)：关键点值的总数量。
        cv4 (nn.ModuleList)：用于预测关键点的卷积层。

    方法：
        forward：执行 YOLO 模型的前向传播并返回预测结果。
        kpts_decode：根据预测结果解码关键点。

    示例：
        创建一个姿态检测头。
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """使用默认参数和卷积层初始化 YOLO26 姿态检测头。

        参数：
            nc (int)：类别数量。
            kpt_shape (tuple)：关键点数量和维度（2 表示 x、y，3 表示 x、y、可见性）。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # 每个关键点的 sigma_x 和 sigma_y
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于保持向后兼容。"""
        return {
            "box_head": self.cv2,
            "cls_head": self.cv3,
            "pose_head": self.cv4,
            "kpts_head": self.cv4_kpts,
            "kpts_sigma_head": self.cv4_sigma,
        }

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {
            "box_head": self.one2one_cv2,
            "cls_head": self.one2one_cv3,
            "pose_head": self.one2one_cv4,
            "kpts_head": self.one2one_cv4_kpts,
            "kpts_sigma_head": self.one2one_cv4_sigma,
        }

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和关键点。"""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # 批次大小
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """移除一对多检测头，以优化推理过程。"""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """根据预测结果解码关键点。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # 修复 NCNN 导出兼容性
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Depth(nn.Module):
    """用于单目深度估计的 YOLO Depth 检测头。

    这是一个稠密预测头，接收主干网络的多尺度特征，通过逐级上采样和融合生成单通道深度图。

    属性：
        nl (int)：金字塔层级数量。
        cal_a (torch.Tensor)：对数仿射校准的缩放缓冲区，默认值为恒等变换对应的 1.0。
        cal_b (torch.Tensor)：对数仿射校准的偏移缓冲区，默认值为恒等变换对应的 0.0。

    示例：
        >>> depth = Depth(ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> out = depth(x)  # 训练时：P2 分辨率（输入尺寸的 1/4），输出 {"depth": (1, 1, 160, 160)}
    """

    export = False  # 导出模式

    def __init__(self, c_mid: int = 256, ch: tuple = ()):
        """初始化 Depth 检测头。

        参数：
            c_mid (int)：融合解码器的中间通道数。
            ch (tuple)：主干网络特征图（P3、P4、P5）的输入通道数。
        """
        super().__init__()
        self.nl = len(ch)  # 检测层数量（金字塔层级数量）

        # 将每个金字塔层级投影到 c_mid 个通道
        self.proj = nn.ModuleList(Conv(c, c_mid, k=1) for c in ch)

        # 在 nl-1 次融合步骤之后分别使用细化模块（最粗层级不进行细化）
        self.refine = nn.ModuleList(nn.Sequential(Conv(c_mid, c_mid, k=3), Conv(c_mid, c_mid, k=3)) for _ in ch[:-1])

        self.head = nn.Sequential(
            Conv(c_mid, c_mid // 2, k=3),
            nn.ConvTranspose2d(c_mid // 2, c_mid // 2, kernel_size=2, stride=2, bias=True),
            Conv(c_mid // 2, c_mid // 4, k=3),
            nn.Conv2d(c_mid // 4, 1, kernel_size=1),
        )
        # 初始化为约 1.2 米，使训练早期的 exp() 输出保持良好的数值条件。
        self.head[-1].bias.data.fill_(0.182)

        # 仅缩放的对数仿射校准：d' = exp(a·log d + b)，默认使用恒等变换。
        self.register_buffer("cal_a", torch.ones(1))
        self.register_buffer("cal_b", torch.zeros(1))

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor] | torch.Tensor:
        """融合多尺度特征并预测深度。

        参数：
            x：来自主干网络或颈部网络的特征张量列表 [P3, P4, P5]。

        返回：
            训练：字典 {"depth": (B, 1, H/4, W/4)}，损失函数直接监督检测头的原始输出。
            评估：(B, 1, H/4, W/4)，应用校准后返回；预测器和验证器会将其调整到图像或真实标注尺寸。
            导出（self.export=True）：(B, 1, H, W)，上采样 4 倍至输入尺寸。输出没有范围限制。
        """
        # 将所有层级投影到相同的通道维度
        feats = [self.proj[i](x[i]) for i in range(self.nl)]

        out = feats[-1]
        for i in range(self.nl - 2, -1, -1):
            # 已发布的深度权重固定使用 align_corners=True。连续金字塔层级采用固定缩放比例，
            # 可使动态形状的 CoreML 导出保持静态上采样；输出尺寸保持不变。
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
            out = out + feats[i]
            out = self.refine[i](out)

        out = self.head(out)  # (B, 1, H/4, W/4)
        depth = torch.exp(out.clamp(-4.0, 5.0))

        if self.training:
            return {"depth": depth}

        depth = depth.pow(self.cal_a) * self.cal_b.exp()
        if self.export:
            depth = F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)
        return depth


class Classify(nn.Module):
    """YOLO 分类检测头，将 x(b,c1,20,20) 转换为 x(b,c2)。

    此类将特征图转换为类别预测结果。

    属性：
        export (bool)：导出模式标志。
        conv (Conv)：用于特征变换的卷积层。
        pool (nn.AdaptiveAvgPool2d)：全局平均池化层。
        drop (nn.Dropout)：用于正则化的 Dropout 层。
        linear (nn.Linear)：用于最终分类的线性层。

    方法：
        forward：对输入特征图执行前向传播。

    示例：
        创建一个分类检测头。
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # 导出模式

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """初始化 YOLO 分类检测头，将输入张量从 (b,c1,20,20) 转换为 (b,c2)。

        参数：
            c1 (int)：输入通道数。
            c2 (int)：输出类别数量。
            k (int)：卷积核尺寸。
            s (int)：步长。
            p (int，可选)：填充。
            g (int)：分组数量。
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 尺寸
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 输出形状为 x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # 输出形状为 x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """对输入特征图执行前向传播。"""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # 获取最终输出
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """将 YOLO 检测模型与文本嵌入的语义理解能力结合的检测头。

    此类扩展标准 Detect 检测头，引入文本嵌入，以增强目标检测任务中的语义理解能力。

    属性：
        cv3 (nn.ModuleList)：用于生成嵌入特征的卷积层。
        cv4 (nn.ModuleList)：用于文本与视觉对齐的对比学习检测头。

    方法：
        forward：拼接并返回预测的边界框和类别概率。
        bias_init：初始化检测头偏置。

    示例：
        创建一个 WorldDetect 检测头。
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """使用 nc 个类别和通道数 ch 初始化 YOLO 检测层。

        参数：
            nc (int)：类别数量。
            embed (int)：嵌入维度。
            with_bn (bool)：是否在对比学习检测头中使用批归一化。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """拼接并返回预测的边界框和类别概率。"""
        feats = list(x)  # 保存特征图引用，用于生成锚点；下面的循环会重新赋值 x[i]，不会修改原始列表
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # 使用不同文本推理时，self.nc 可能发生变化
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = {"boxes": boxes, "scores": scores, "feats": feats}
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """初始化 Detect() 的偏置。注意：必须先计算步长。"""
        m = self  # self.模型[-1]  # Detect() 模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 名义类别频率
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 来自各检测层
            a[-1].bias.data[:] = 1.0  # 边界框
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # 类别（.01 个目标，80 个类别，640 图像尺寸）


class LRPCHead(nn.Module):
    """用于高效目标检测的轻量级区域提议与分类检测头。

    此检测头将区域提议筛选与分类结合，从而支持使用动态词汇表进行高效检测。

    属性：
        vocab (nn.Module)：词汇表或分类模块。
        pf (nn.Module)：提议筛选模块。
        loc (nn.Module)：定位模块。
        enabled (bool)：是否启用该检测头。

    方法：
        conv2linear：将 1x1 卷积层转换为线性层。
        forward：处理分类特征和定位特征，生成检测提议。

    示例：
        创建一个 LRPC 检测头。
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """使用词汇表、提议筛选器和定位组件初始化 LRPCHead。

        参数：
            vocab (nn.Module)：词汇表或分类模块。
            pf (nn.Module)：提议筛选模块。
            loc (nn.Module)：定位模块。
            enabled (bool)：是否启用检测头功能。
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """将 1x1 卷积层转换为线性层。"""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels).requires_grad_(conv.weight.requires_grad)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """处理分类特征和定位特征，生成检测提议。"""
        if self.enabled:
            if not conf:  # 静态导出时，所有锚点都通过提议筛选器
                cls_feat = self.vocab(cls_feat.flatten(2).transpose(-1, -2))
                return self.loc(loc_feat), cls_feat.transpose(-1, -2), None
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """将 YOLO 检测模型与文本嵌入的语义理解能力结合的检测头。

    此类扩展标准 Detect 检测头，通过文本嵌入和视觉提示嵌入支持文本引导的目标检测，并增强语义理解能力。

    属性：
        is_fused (bool)：模型是否已融合以用于推理。
        cv3 (nn.ModuleList)：用于生成嵌入特征的卷积层。
        cv4 (nn.ModuleList)：用于文本与视觉对齐的对比学习检测头。
        reprta (Residual)：文本提示嵌入使用的残差模块。
        savpe (SAVPE)：空间感知视觉提示嵌入模块。
        embed (int)：嵌入维度。

    方法：
        fuse：将文本特征与模型权重融合，以提高推理效率。
        get_tpe：获取归一化后的文本提示嵌入。
        get_vpe：获取具有空间感知能力的视觉提示嵌入。
        forward_lrpc：在无提示模型中使用融合后的文本嵌入处理特征。
        forward：使用类别提示嵌入处理特征并生成检测结果。
        bias_init：初始化检测头偏置。

    示例：
        创建一个 YOLOEDetect 检测头。
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect([*x, cls_pe])
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """使用 nc 个类别和通道数 ch 初始化 YOLO 检测层。

        参数：
            nc (int)：类别数量。
            embed (int)：嵌入维度。
            with_bn (bool)：是否在对比学习检测头中使用批归一化。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # 使用新的 cv3 覆盖原分支
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode(False)  # 融合层仍保留在模型中，因此不能将其转换为推理张量
    def fuse(self, txt_feats: torch.Tensor = None):
        """将文本特征与模型权重融合，以提高推理效率。"""
        if txt_feats is None:  # 表示移除一对多分支
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(next(self.parameters()).dtype).squeeze(0)
        if self.cv3 and self.cv4:
            self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """将文本提示嵌入与模型权重融合，以提高推理效率。"""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device, conv.weight.dtype)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """获取归一化后的文本提示嵌入。"""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """获取具有空间感知能力的视觉提示嵌入。"""
        if vpe.shape[1] == 0:  # 没有视觉提示嵌入
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用类别提示嵌入处理特征并生成检测结果。"""
        if hasattr(self, "lrpc"):  # 用于无提示推理
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用融合后的文本嵌入处理特征，为无提示模型生成检测结果。"""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        # 无提示融合会移除一对多检测头。
        cv2 = self.one2one_cv2 if self.end2end or self.cv2 is None else self.cv2
        cv3 = self.one2one_cv3 if self.end2end or self.cv3 is None else self.cv3
        conf = 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](cls_feat, loc_feat, conf)
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = {
            "boxes": torch.cat(boxes, 2),
            "scores": torch.cat(scores, 2),
            "feats": x,
            "index": torch.cat(index) if conf else None,
        }
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """解码用于推理的预测边界框。"""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if x["index"] is None else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """返回一对多检测头组件，用于兼容 v3/v5/v8/v9/v11。"""
        return {"box_head": self.cv2, "cls_head": self.cv3, "contrastive_head": self.cv4}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3, "contrastive_head": self.one2one_cv4}

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """拼接并返回预测的边界框、类别概率和对比学习分数。"""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # 用于融合后的推理
            return {}
        bs = x[0].shape[0]  # 批次大小
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # 使用不同文本推理时，self.nc 可能发生变化
        return {"boxes": boxes, "scores": scores, "feats": x[:3]}

    def bias_init(self):
        """初始化 Detect() 的偏置。注意：必须先计算步长。"""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # 边界框
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # 边界框
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """带文本嵌入能力的 YOLO 分割检测头。

    此类扩展 YOLOEDetect，为具有文本引导语义理解能力的实例分割任务增加掩码预测功能。

    属性：
        nm (int)：掩码数量。
        npr (int)：原型数量。
        proto (Proto)：原型生成模块。
        cv5 (nn.ModuleList)：用于生成掩码系数的卷积层。

    方法：
        forward：返回模型输出和掩码系数。

    示例：
        创建一个 YOLOESegment 检测头。
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment([*x, text])
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """使用类别数量、掩码参数和嵌入维度初始化 YOLOESegment。

        参数：
            nc (int)：类别数量。
            nm (int)：掩码数量。
            npr (int)：原型数量。
            embed (int)：嵌入维度。
            with_bn (bool)：是否在对比学习检测头中使用批归一化。
            reg_max (int)：DFL 通道的最大数量。
            end2end (bool)：是否使用无需 NMS 的端到端检测。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """返回一对多检测头组件，用于兼容 v3/v5/v8/v9/v11。"""
        return {"box_head": self.cv2, "cls_head": self.cv3, "mask_head": self.cv5, "contrastive_head": self.cv4}

    @property
    def one2one(self):
        """返回一对一检测头组件。"""
        return {
            "box_head": self.one2one_cv2,
            "cls_head": self.one2one_cv3,
            "mask_head": self.one2one_cv5,
            "contrastive_head": self.one2one_cv4,
        }

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用融合后的文本嵌入处理特征，为无提示模型生成检测结果。"""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.one2one_cv2 if self.end2end or self.cv2 is None else self.cv2
        cv3 = self.one2one_cv3 if self.end2end or self.cv3 is None else self.cv3
        cv5 = self.one2one_cv5 if self.end2end or self.cv5 is None else self.cv5
        conf = 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](cls_feat, loc_feat, conf)
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index) if conf else None
        preds = {
            "boxes": torch.cat(boxes, 2),
            "scores": torch.cat(scores, 2),
            "feats": x,
            "index": index,
            "mask_coefficient": mc if index is None else mc[..., index],
        }
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """在训练时返回模型输出和掩码系数；在推理时同样返回模型输出和掩码系数。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # 掩码原型
        if isinstance(preds, dict):  # 训练和验证期间
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，并与掩码系数拼接。"""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和掩码系数。"""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def fuse(self, txt_feats: torch.Tensor = None):
        """将文本特征与模型权重融合，以提高推理效率。"""
        super().fuse(txt_feats)
        if txt_feats is None:  # 表示移除一对多分支
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """使用 Proto26 生成掩码的 YOLOE 风格分割检测头模块。

    此类扩展 YOLOESegment 的功能，通过集成 Proto26 原型生成模块和卷积层，为分割任务预测掩码系数。

    参数：
        nc (int)：类别数量，默认为 80。
        nm (int)：掩码数量，默认为 32。
        npr (int)：原型通道数量，默认为 256。
        embed (int)：嵌入维度，默认为 512。
        with_bn (bool)：是否使用批归一化，默认为 False。
        reg_max (int)：DFL 通道的最大数量，默认为 16。
        end2end (bool)：是否使用端到端检测模式，默认为 False。
        ch (tuple[int, ...])：每个尺度的输入通道数。

    属性：
        nm (int)：分割掩码数量。
        npr (int)：原型通道数量。
        proto (Proto26)：用于分割的原型生成模块。
        cv5 (nn.ModuleList)：根据特征生成掩码系数的卷积层。
        one2one_cv5 (nn.ModuleList，可选)：端到端检测分支使用的 cv5 深拷贝。
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """使用类别数量、掩码参数和嵌入维度初始化 YOLOESegment26。"""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # 原型生成模块

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """在训练时返回模型输出和掩码系数；在推理时同样返回模型输出和掩码系数。"""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semantic=False)  # 掩码原型

        if isinstance(preds, dict):  # 训练和验证期间
            if self.end2end and not hasattr(self, "lrpc"):  # 非无提示模式
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """用于目标检测的实时可变形 Transformer 解码器（RTDETRDecoder）模块。

    此解码器模块结合 Transformer 架构和可变形卷积，预测图像中目标的边界框和类别标签。它融合多个层级的特征，
    并依次通过多个 Transformer 解码器层，输出最终预测结果。

    属性：
        export (bool)：导出模式标志。
        hidden_dim (int)：隐藏层维度。
        nhead (int)：多头注意力的头数量。
        nl (int)：特征层级数量。
        nc (int)：类别数量。
        num_queries (int)：查询点数量。
        num_decoder_layers (int)：解码器层数量。
        input_proj (nn.ModuleList)：主干网络特征的输入投影层。
        decoder (DeformableTransformerDecoder)：Transformer 解码器模块。
        denoising_class_embed (nn.Embedding)：去噪使用的类别嵌入。
        num_denoising (int)：去噪查询数量。
        label_noise_ratio (float)：训练时使用的标签噪声比例。
        box_noise_scale (float)：训练时使用的边界框噪声缩放比例。
        learnt_init_query (bool)：是否学习初始查询嵌入。
        tgt_embed (nn.Embedding)：查询的目标嵌入。
        query_pos_head (MLP)：查询位置预测头。
        enc_output (nn.Sequential)：编码器输出层。
        enc_score_head (nn.Linear)：编码器分数预测头。
        enc_bbox_head (MLP)：编码器边界框预测头。
        dec_score_head (nn.ModuleList)：解码器分数预测头列表。
        dec_bbox_head (nn.ModuleList)：解码器边界框预测头列表。

    方法：
        forward：执行前向传播并返回边界框和分类分数。

    示例：
        创建一个 RTDETRDecoder。
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # 导出模式
    max_det = 300  # 每张图像的最大检测数量
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # 隐藏维度
        nq: int = 300,  # 查询数量
        ndp: int = 4,  # 解码器采样点数量
        nh: int = 8,  # 注意力头数量
        ndl: int = 6,  # 解码器层数量
        d_ffn: int = 1024,  # 前馈网络维度
        dropout: float = 0.0,
        act: nn.Module | None = None,
        eval_idx: int = -1,
        # 训练参数
        nd: int = 100,  # 去噪查询数量
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """使用给定参数初始化 RTDETRDecoder 模块。

        参数：
            nc (int)：类别数量。
            ch (tuple)：主干网络特征图的通道数。
            hd (int)：隐藏层维度。
            nq (int)：查询点数量。
            ndp (int)：解码器采样点数量。
            nh (int)：多头注意力的头数量。
            ndl (int)：解码器层数量。
            d_ffn (int)：前馈网络维度。
            dropout (float)：Dropout 概率。
            act (nn.Module)：激活函数。
            eval_idx (int)：评估时使用的层索引。
            nd (int)：去噪查询数量。
            label_noise_ratio (float)：标签噪声比例。
            box_noise_scale (float)：边界框噪声缩放比例。
            learnt_init_query (bool)：是否学习初始查询嵌入。
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # 层级数量
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # 主干网络特征投影
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 注意：这是简化版本，与 .pt 权重不一致。
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer 模块
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # 去噪部分
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # 解码器嵌入
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # 编码器检测头
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # 解码器检测头
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """执行模块的前向传播，为输入返回边界框和分类分数。

        参数：
            x (list[torch.Tensor])：来自主干网络的特征图列表。
            batch (dict，可选)：训练所需的批次信息。

        返回：
            outputs (tuple | torch.Tensor)：训练时返回包含边界框、分数和其他元数据的元组；推理时返回形状为
                (bs, num_queries, 6) 的张量，其中包含边界框、置信度分数和类别标签。
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # 输入投影和嵌入
        feats, shapes = self._get_encoder_input(x)

        # 准备训练所需的去噪输入
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 解码器
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            # 当批次中没有真实标注时，访问 denoising_class_embed，使 DDP 将其识别为已使用。
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """对预测结果执行后处理并选择 top-k 个检测结果。

        参数：
            boxes (torch.Tensor)：预测的边界框，形状为 (batch_size, num_queries, 4)，格式为 xywh。
            scores (torch.Tensor)：类别分数，形状为 (batch_size, num_queries, nc)。

        返回：
            (torch.Tensor)：处理后的预测结果，形状为 (batch_size, num_queries, 6)。导出时数量限制为 max_det，
                最后一维格式为 [cx, cy, w, h, max_class_prob, class_index]。
        """
        k = min(self.num_queries, self.max_det) if self.export else self.num_queries
        groups = 8 if self.export and self.format == "engine" and not self.dynamic else 1
        scores, index = Detect._grouped_topk(scores.flatten(1), k, groups)
        # CoreML MIL 不支持整数向下取整除法和取模下沉，因此使用 torch.div(rounding_mode="floor")，
        # 并通过（索引 - q*nc）计算类别索引。
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4).long())
        return torch.cat([boxes, scores[..., None], (index - query_idx * self.nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据给定形状和网格尺寸生成锚框，并验证其有效性。

        参数：
            shapes (list)：特征图尺寸列表。
            grid_size (float，可选)：网格单元的基础尺寸。
            dtype (torch.dtype，可选)：张量数据类型。
            device (str，可选)：创建张量所使用的设备。
            eps (float，可选)：用于保证数值稳定性的小值。

        返回：
            anchors (torch.Tensor)：生成的锚框。
            valid_mask (torch.Tensor)：锚框有效性掩码。
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """获取输入的投影特征并将其拼接，以生成编码器输入。

        参数：
            x (list[torch.Tensor])：来自主干网络的特征图列表。

        返回：
            feats (torch.Tensor)：处理后的特征。
            shapes (list)：特征图尺寸列表。
        """
        # 获取投影特征
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # 获取编码器输入
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据提供的特征和尺寸生成并准备解码器所需的输入。

        参数：
            feats (torch.Tensor)：编码器处理后的特征。
            shapes (list)：特征图尺寸列表。
            dn_embed (torch.Tensor，可选)：去噪嵌入。
            dn_bbox (torch.Tensor，可选)：去噪边界框。

        返回：
            embeddings (torch.Tensor)：解码器查询嵌入。
            refer_bbox (torch.Tensor)：参考边界框。
            enc_bboxes (torch.Tensor)：编码后的边界框。
            enc_scores (torch.Tensor)：编码后的分数。
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # 准备解码器输入
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # 选择查询
        # (bs*num_queries,)
        groups = 8 if self.export and self.format == "engine" and not self.dynamic else 1
        topk_ind = Detect._grouped_topk(enc_outputs_scores.max(-1).values, self.num_queries, groups)[1].view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # 动态锚框和静态内容
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """使用预定义的权重和偏置初始化或重置模型的各个组件。"""
        # 初始化类别和边界框检测头
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # 注意：使用 `linear_init` 初始化权重时，在自定义数据集上训练可能产生 NaN。
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """来自 https://arxiv.org/pdf/2405.14458 的 v10 检测头。

    此类实现 YOLOv10 检测头，采用双重分配训练和一致的双分支预测，以提升效率和性能。

    属性：
        end2end (bool)：端到端检测模式。
        max_det (int)：最大检测数量。
        cv3 (nn.ModuleList)：轻量级分类检测头层。
        one2one_cv3 (nn.ModuleList)：一对一分类检测头层。

    方法：
        __init__：使用指定的类别数量和输入通道初始化 v10Detect 对象。
        forward：执行 v10Detect 模块的前向传播。
        bias_init：初始化 Detect 模块的偏置。
        fuse：移除一对多检测头，以优化推理过程。

    示例：
        创建一个 v10Detect 检测头。
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """使用指定的类别数量和输入通道初始化 v10Detect 对象。

        参数：
            nc (int)：类别数量。
            ch (tuple)：主干网络特征图的通道数元组。
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # 通道
        # 轻量级分类检测头
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """移除一对多检测头，以优化推理过程。"""
        self.cv2 = self.cv3 = None


class SemanticSegment(nn.Module):
    """用于逐像素分类的 YOLO 语义分割检测头。

    此检测头生成稠密的逐像素类别预测结果。与实例分割不同，它不会生成边界框或实例掩码。

    属性：
        nc (int)：语义类别数量。
        nl (int)：输入特征层级数量。
        stride (torch.Tensor)：特征图步长。
        export (bool)：导出模式标志。
        format (str)：导出格式。
        classifier (nn.Sequential)：最终卷积分类检测头。
        aux_head (nn.Sequential | None)：位于 P4 上、用于深度监督的辅助分类器。
    """

    export = False  # 导出模式
    format = None  # 导出格式
    bake_argmax = False  # 导出：输出 [B, H, W] 类别图（TensorRT>=10 和多类别 Hailo-10/15）

    def __init__(self, nc=19, ch=()):
        """初始化语义分割检测头。

        参数：
            nc (int)：语义类别数量。
            ch (tuple)：颈部网络特征图（P3、P4）的通道数元组。
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)

        c_mid = ch[0]  # 使用 P3 的通道宽度作为中间维度
        # 最终分类器
        self.classifier = nn.Sequential(Conv(c_mid, c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        # P4（索引 1）上的辅助检测头，用于训练
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x):
        """前向传播：融合多尺度特征并预测逐像素类别。

        参数：
            x (list[torch.Tensor])：特征图列表 [P3, P4]。

        返回：
            (torch.Tensor | tuple)：训练和推理期间输出形状为 [B, nc, H/8, W/8] 的 logits；存在 aux_head 时，
                训练期间返回 (main, aux) 元组。ONNX、MNN、OpenVINO、TensorRT>=10 以及多类别 Hailo-10/15
                导出会内置类别归约，并返回形状为 [B, H, W] 的紧凑类别图（nc <= 256 时为 uint8，否则为 int32）。
                其他导出格式返回上采样后的 logits，形状为 [B, nc, H, W]。
        """
        # 分类
        logits = self.classifier(x[0])  # [B, nc, H/8, W/8]
        if self.training:
            if self.aux_head is not None:
                return logits, self.aux_head(x[1])  # 主输出 + 辅助输出（P4）
            return logits
        if self.export:
            y = F.interpolate(logits, scale_factor=8, mode="bilinear", align_corners=False)  # [B, nc, H, W]
            # 内置类别归约：输出 [B, H, W] 类别图，可将设备到主机的拷贝量缩小约 80 倍。ONNX/MNN/OpenVINO
            # 和多类别 Hailo-10/15 保留整数输出；TensorRT 仅在 TRT>=10 时支持 uint8 图输出，
            # 因此 engine 和 Hailo 的类别图生成由导出器控制。
            if self.format in {"onnx", "mnn", "openvino"} or (self.format in {"engine", "hailo"} and self.bake_argmax):
                cls = y.argmax(1) if self.nc > 1 else y.squeeze(1) > 0
                return cls.to(torch.uint8 if self.nc <= 256 else torch.int32)
            return y
        return logits
