# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import pickle
import re
import threading
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Depth,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    SemanticSegment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    LOGGER,
    SAFE_LOAD,
    SETTINGS,
    WINDOWS,
    YAML,
    IterableSimpleNamespace,
    colorstr,
    emojis,
)
from ultralytics.utils.checks import REMOTE_FILE_PREFIXES, check_file, check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    DepthLoss26,
    E2ELoss,
    PoseLoss26,
    SemanticSegmentationLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Ultralytics YOLO 系列所有模型的基类。

    此类为 YOLO 模型提供通用功能，包括前向传播处理、模型融合、信息显示和权重加载。

    属性：
        model (torch.nn.Sequential)：神经网络模型。
        save (list)：需要保存输出的层索引列表。
        stride (torch.Tensor)：模型步长值。

    方法：
        forward：执行训练或推理前向传播。
        predict：对输入张量执行推理。
        fuse：融合 Conv/BatchNorm 层并重新参数化以优化性能。
        info：打印模型信息。
        load：将权重加载到模型中。
        loss：计算训练损失。

    示例：
        创建 BaseModel 实例
        >>> model = BaseModel()
        >>> model.info()  # 显示模型信息
    """

    def forward(self, x, *args, **kwargs):
        """执行模型的训练或推理前向传播。

        如果 x 是字典，则计算并返回训练损失；否则返回推理预测结果。

        参数：
            x (torch.Tensor | dict)：用于推理的输入张量，或包含图像张量和标签的训练字典。
            *args (Any)：可变长度的位置参数。
            **kwargs (Any)：任意关键字参数。

        返回：
            (torch.Tensor)：当 x 为字典时返回损失（训练），否则返回网络预测结果（推理）。
        """
        if isinstance(x, dict):  # 用于训练过程中保存和验证的情况
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, augment=False, embed=None):
        """执行网络前向传播。

        参数：
            x (torch.Tensor)：模型的输入张量。
            profile (bool)：为 True 时打印每层计算耗时。
            augment (bool)：推理时是否对图像进行增强。
            embed (list, 可选)：需要返回嵌入特征的层索引列表。

        返回：
            (torch.Tensor)：模型的最后一层输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, embed)

    def _predict_once(self, x, profile=False, embed=None):
        """执行网络前向传播。

        参数：
            x (torch.Tensor)：模型的输入张量。
            profile (bool)：为 True 时打印每层计算耗时。
            embed (list, 可选)：需要返回嵌入特征的层索引列表。

        返回：
            (torch.Tensor)：模型的最后一层输出。
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从更早的层获取输入
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行前向传播
            y.append(x if m.i in self.save else None)  # 保存输出
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平特征
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行数据增强，并返回增强后的推理结果。"""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """统计模型单层在给定输入上的计算耗时和 FLOPs。

        参数：
            m (torch.nn.Module)：要统计的层。
            x (torch.Tensor)：该层的输入数据。
            dt (list)：用于保存该层计算耗时的列表。
        """
        try:
            import thop
        except ImportError:
            thop = None  # 兼容未安装 'ultralytics-thop' 的 Conda 环境

        c = m == self.model[-1] and isinstance(x, list)  # 最后一层为列表时复制输入，修复原地操作问题
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0
        device = next(self.parameters()).device
        t = time_sync(device)
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync(device) - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True, imgsz=640):
        """融合 Conv/ConvTranspose 与 BatchNorm 层，并重新参数化 RepConv/RepVGGDW 以提高效率。

        参数：
            verbose (bool)：是否在融合后打印模型信息。
            imgsz (int | list)：用于计算 FLOPs 的输入图像尺寸。

        返回：
            (torch.nn.Module)：返回融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                    delattr(m, "bn")  # 移除批归一化层
                    m.forward = m.forward_fuse  # 更新前向传播函数
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除批归一化层
                    m.forward = m.forward_fuse  # 更新前向传播函数
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播函数
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()  # 移除 one2many 检测头
            self.info(verbose=verbose, imgsz=imgsz)

        return self

    def is_fused(self, thresh=10):
        """检查模型中的归一化层数量是否低于指定阈值。

        参数：
            thresh (int, 可选)：归一化层数量阈值。

        返回：
            (bool)：模型中的归一化层数量低于阈值时返回 True，否则返回 False。
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # 模型中的 BatchNorm 层数量是否小于 thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        """打印模型信息。

        参数：
            detailed (bool)：为 True 时打印模型详细信息。
            verbose (bool)：为 True 时打印模型信息。
            imgsz (int)：用于计算模型信息的图像尺寸。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """将函数应用于模型中的所有张量，包括 Detect 检测头的步长和锚框等属性。

        参数：
            fn (function)：要应用于模型的函数。

        返回：
            (BaseModel)：更新后的 BaseModel 对象。
        """
        super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # 包含所有检测子类，例如 Segment、Pose、OBB、WorldDetect、YOLOEDetect、YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """将权重加载到模型中。

        参数：
            weights (dict | torch.nn.Module)：要加载的预训练权重。
            verbose (bool，可选)：是否记录权重传输进度。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision 模型不一定是字典
        csd = model.float().state_dict()  # 将检查点 state_dict 转为 FP32

        # 当类别数 nc 不同时，根据类别名称重新映射分类头行（例如 Obj365 -> COCO 微调）
        cls_remapped = self._remap_cls_by_names(csd, model, verbose=verbose)

        updated_csd = intersect_dicts(csd, self.state_dict())  # 求交集
        self.load_state_dict(updated_csd, strict=False)  # 加载权重
        len_updated_csd = len(updated_csd) + cls_remapped
        first_conv = "model.0.conv.weight"  # 当前 YOLO 模型中硬编码的首层卷积权重
        # 主要用于提升多通道训练能力
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def _remap_cls_by_names(self, csd: dict[str, torch.Tensor], src_model: torch.nn.Module, verbose: bool = True):
        """根据类别名称，将预训练分类头的行重新映射到当前类别顺序。

        当目标类别名称与源类别名称匹配时（忽略大小写并去除首尾空格），将预训练分类层中的对应行复制到当前
        模型的 state_dict 中。该功能适用于跨数据集微调：即使类别数量不同（例如 Objects365 与 COCO），或
        类别数量相同但顺序不同，也可以复用匹配类别的权重。方法会通过 state_dict 引用就地修改目标张量，
        并从 ``csd`` 中移除已匹配的分类张量，使后续 ``intersect_dicts`` 不再按源类别顺序复制它们。

        参数：
            csd (dict)：预训练检查点的 state_dict（会被修改）。
            src_model (torch.nn.Module)：预训练模块，用于读取 ``.names`` 和 ``.nc``。
            verbose (bool)：是否记录映射摘要。

        返回：
            (int)：重新映射的分类张量数量（计入“Transferred”日志）。
        """
        src_names = getattr(src_model, "names", None)
        tgt_names = getattr(self, "names", None)
        if not (isinstance(src_names, dict) and isinstance(tgt_names, dict)):
            return 0
        src_nc, tgt_nc = len(src_names), len(tgt_names)

        def _norm(s):
            return str(s).strip().lower()

        # 跳过默认占位名称 {0:"0", 1:"1", ...}（也会捕获空字典），因为没有内容可匹配
        if any(all(str(k) == str(v) for k, v in n.items()) for n in (src_names, tgt_names)):
            return 0

        src_lookup = {_norm(v): k for k, v in src_names.items()}
        idx = torch.tensor([src_lookup.get(_norm(tgt_names.get(k)), -1) for k in range(tgt_nc)], dtype=torch.long)
        n_match = int((idx >= 0).sum())
        # 如果没有匹配项，或类别名称的顺序和数量已经相同，则跳过（intersect_dicts 会直接复制）
        if n_match == 0 or (src_nc == tgt_nc and torch.equal(idx, torch.arange(tgt_nc))):
            return 0

        valid = idx >= 0
        state_dict = self.state_dict()
        # 仅选择检测头中精确的类别 logit 卷积权重和偏置键，避免误处理只是共享 nc 维度的其他张量
        # （例如主干模块以及边界框、掩码、姿态分支）。
        cls_keys = {
            f"{name}.{attr}.{i}.{len(seq) - 1}.{p}"
            for name, m in self.named_modules()
            if isinstance(m, Detect)
            for attr in ("cv3", "one2one_cv3")
            for i, seq in enumerate(getattr(m, attr, ()))
            if getattr(seq[-1], "out_channels", None) == tgt_nc
            for p in ("weight", "bias")
        }
        remapped = 0
        for k in cls_keys & csd.keys():
            v_src, v_tgt = csd[k], state_dict[k]
            if v_src.shape[1:] != v_tgt.shape[1:]:  # 不同 nc 的分类卷积输入宽度（c3）可能不同，此时只复制偏置
                continue
            v_tgt[valid] = v_src[idx[valid]].to(v_tgt.dtype)
            csd.pop(k)  # 防止 intersect_dicts 按源类别的错误顺序再次复制这些行
            remapped += 1
        if verbose and remapped:
            LOGGER.info(f"Remapped {n_match}/{tgt_nc} cls head rows from pretrained weights by class name")
        return remapped

    def loss(self, batch, preds=None):
        """计算损失。

        参数：
            batch (dict)：用于计算损失的批次数据。
            preds (torch.Tensor | list[torch.Tensor]，可选)：模型预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """初始化 BaseModel 的损失函数。"""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


def _initialize_yolo_model(model, cfg, ch, nc, verbose):
    """根据 YAML 配置初始化 YOLO 模型的通用属性。"""
    model.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典
    if model.yaml["backbone"][0][2] == "Silence":
        LOGGER.warning(
            "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
            "Please delete local *.pt file and re-download the latest model checkpoint."
        )
        model.yaml["backbone"][0][2] = "nn.Identity"

    model.yaml["channels"] = ch  # 保存通道数
    if nc and nc != model.yaml["nc"]:
        LOGGER.info(f"Overriding model.yaml nc={model.yaml['nc']} with nc={nc}")
        model.yaml["nc"] = nc  # 覆盖 YAML 中的类别数量
    model.model, model.save = parse_model(deepcopy(model.yaml), ch=ch, verbose=verbose)  # 模型和保存列表
    model.names = {i: f"{i}" for i in range(model.yaml["nc"])}  # 默认类别名称字典
    model.inplace = model.yaml.get("inplace", True)


class DetectionModel(BaseModel):
    """YOLO 目标检测模型。

    此类实现 YOLO 检测架构，负责目标检测任务的模型初始化、前向传播、增强推理和损失计算。

    属性：
        yaml (dict)：模型配置字典。
        model (torch.nn.Sequential)：神经网络模型。
        save (list)：需要保存输出的层索引列表。
        names (dict)：类别名称字典。
        inplace (bool)：是否使用原地操作。
        end2end (bool)：模型是否使用端到端检测。
        stride (torch.Tensor)：模型步长值。

    方法：
        __init__：初始化 YOLO 检测模型。
        _predict_augment：执行增强推理。
        _descale_pred：对增强推理后的预测结果进行反缩放。
        _clip_augmented：裁剪 YOLO 增强推理结果的尾部。
        init_criterion：初始化损失函数。

    示例：
        初始化一个检测模型。
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLO 检测模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__()
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        # 构建步幅
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包含所有检测子类，例如 Segment、Pose、OBB、YOLOEDetect、YOLOESegment
            s = 256  # 最小步幅的 2 倍
            m.inplace = self.inplace

            def _forward(x):
                """执行模型的前向传播，并根据不同 Detect 子类进行相应处理。"""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # 在训练开始前避免更改批次统计信息
            m.training = True  # 设置为 True，以正确返回步幅
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            self.model.train()  # 将模型恢复为训练（默认）模式
            m.bias_init()  # 只运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步幅，例如 RTDETR

        # 初始化权重和偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """返回模型是否使用无需 NMS 的端到端检测。"""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """设置模型检测头（最后一层）的属性。

        参数：
            **kwargs (Any)：表示待设置属性的任意关键字参数。
        """
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        """对输入图像 x 执行增强，并返回增强推理结果和训练输出。

        参数：
            x (torch.Tensor)：输入图像张量。

        返回：
            (tuple[torch.Tensor, None])：增强推理输出，以及表示训练输出的 None。
        """
        if getattr(self, "end2end", False) or type(self.model[-1]) is not Detect:
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度、宽度
        s = [1, 0.83, 0.67]  # 缩放比例
        f = [None, 3, None]  # 翻转方式（2-上下，3-左右）
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强推理的尾部
        return torch.cat(y, -1), None  # 增强推理输出，训练输出为 None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对增强推理后的预测结果进行反缩放（逆操作）。

        参数：
            p (torch.Tensor)：预测结果张量。
            flips (int | None)：翻转类型（None 表示不翻转，2 表示上下翻转，3 表示左右翻转）。
            scale (float)：缩放因子。
            img_size (tuple)：原始图像尺寸（高度、宽度）。
            dim (int)：进行拆分的维度。

        返回：
            (torch.Tensor)：反缩放后的预测结果。
        """
        p[:, :4] /= scale  # 取消缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 取消上下翻转
        elif flips == 3:
            x = img_size[1] - x  # 取消左右翻转
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理结果的尾部。

        参数：
            y (list[torch.Tensor])：检测张量列表。

        返回：
            (list[torch.Tensor])：裁剪后的检测张量列表。
        """
        nl = self.model[-1].nl  # 检测层数量（P3-P5）
        g = sum(4**x for x in range(nl))  # 网格点数量
        e = 1  # 排除的层数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大尺度
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小尺度
        return y

    def init_criterion(self):
        """初始化 DetectionModel 的损失函数。"""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO 有向边界框（OBB）模型。

    此类扩展 DetectionModel，用于处理有向边界框检测任务，并为旋转目标检测提供专用的损失计算。

    方法：
        __init__：初始化 YOLO OBB 模型。
        init_criterion：初始化 OBB 检测的损失函数。

    示例：
        初始化一个 OBB 模型。
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLO OBB 模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化模型的损失函数。"""
        return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO 实例分割模型。

    此类扩展 DetectionModel，用于处理实例分割任务，并为像素级目标检测和分割提供专用的损失计算。

    方法：
        __init__：初始化 YOLO 分割模型。
        init_criterion：初始化分割任务的损失函数。

    示例：
        初始化一个分割模型。
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 Ultralytics YOLO 分割模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化 SegmentationModel 的损失函数。"""
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)


class SemanticSegmentationModel(BaseModel):
    """YOLO 语义分割模型。

    此类实现生成逐像素类别预测结果的语义分割模型。与 SegmentationModel（实例分割）不同，此模型不会生成边界框。

    方法：
        __init__：初始化语义分割模型。
        init_criterion：初始化语义分割的损失函数。

    示例：
        初始化一个语义分割模型。
        >>> model = SemanticSegmentationModel("yolo26n-sem.yaml", ch=3, nc=19)
    """

    def __init__(self, cfg="yolo26n-sem.yaml", ch=3, nc=None, verbose=True):
        """初始化 YOLO 语义分割模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__()
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        # 构建步幅：跟踪所有层中的最小空间尺寸，以找到最深层的
        # backbone 步幅（例如 P5/32）。仅使用检测头输入是不够的：FPN 会在检测头前对 P5
        # 进行上采样，但编码器仍要求输入与该最深步幅对齐，否则 FPN 拼接会因舍入误差失败。
        m = self.model[-1]
        if isinstance(m, SemanticSegment):
            s = 256
            self.model.eval()
            m.training = True  # 获取训练输出（stride-4）
            min_h = [s]

            def _record(_m, _inp, out, _h=min_h):
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    _h[0] = min(_h[0], out.shape[-2])

            hooks = [layer.register_forward_hook(_record) for layer in self.model]
            try:
                self.forward(torch.zeros(1, ch, s, s))
            finally:
                for h in hooks:
                    h.remove()
            m.stride = torch.tensor([s / min_h[0]], dtype=torch.float32)  # 例如 256/8 = 32
            self.stride = m.stride
            self.model.train()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """初始化语义分割的损失函数。"""
        return SemanticSegmentationLoss(self)

    def _apply(self, fn):
        """将函数应用于模型中的所有张量。"""
        super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, SemanticSegment):
            m.stride = fn(m.stride)
        return self


class PoseModel(DetectionModel):
    """YOLO 姿态估计模型。

    此类扩展 DetectionModel，用于处理人体姿态估计任务，并为关键点检测和姿态估计提供专用的损失计算。

    属性：
        kpt_shape (tuple)：关键点数据的形状（关键点数量、维度数量）。

    方法：
        __init__：初始化 YOLO 姿态模型。
        init_criterion：初始化姿态估计的损失函数。

    示例：
        初始化一个姿态模型。
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """初始化 Ultralytics YOLO Pose 模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            data_kpt_shape (tuple)：关键点数据的形状。
            verbose (bool)：是否显示模型信息。
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # 加载 模型 YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化 PoseModel 的损失函数。"""
        loss = PoseLoss26 if isinstance(self.model[-1], Pose26) else v8PoseLoss
        return E2ELoss(self, loss) if self.end2end else loss(self)


class DepthModel(DetectionModel):
    """YOLO 单目深度估计模型。

    此类扩展 DetectionModel，用于单目深度估计，使用 YOLO 主干网络和 FPN，并搭配 DPT 风格的稠密深度解码头。
    该实现将 Depth Anything 方法适配到 YOLO 架构中。

    示例：
        >>> model = DepthModel("yolo26n-depth.yaml", ch=3)
        >>> results = model(image_tensor)
    """

    def __init__(self, cfg="yolo26n-depth.yaml", ch=3, nc=None, verbose=True):
        """初始化 YOLO Depth 模型。"""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化深度估计损失函数。"""
        return DepthLoss26(self)


class ClassificationModel(BaseModel):
    """YOLO 图像分类模型。

    此类实现用于图像分类任务的 YOLO 分类架构，并提供模型初始化、配置和输出重塑功能。

    属性：
        yaml (dict)：模型配置字典。
        model (torch.nn.Sequential)：神经网络模型。
        stride (torch.Tensor)：模型步长值。
        names (dict)：类别名称字典。

    方法：
        __init__：初始化 ClassificationModel。
        _from_yaml：设置模型配置并定义网络架构。
        reshape_outputs：将模型输出调整为指定的类别数量。
        init_criterion：初始化损失函数。

    示例：
        初始化一个分类模型。
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
        """使用 YAML、通道数、类别数量和详细输出标志初始化 ClassificationModel。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """设置 Ultralytics YOLO 模型配置并定义模型架构。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典

        # 定义模型
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # 输入通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # 覆盖 YAML 中的类别数量
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型和保存列表
        self.stride = torch.Tensor([1])  # 无步长约束
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认类别名称字典
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """在需要时将 TorchVision 分类模型调整为 `nc` 个类别。

        参数：
            model (torch.nn.Module)：待更新的模型。
            nc (int)：新的类别数量。
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # 最后一层模块
        if isinstance(m, Classify):  # YOLO Classify() 检测头
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet、EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # 最后一个 torch.nn.Linear 的索引
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # 最后一个 torch.nn.Conv2d 的索引
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """初始化 ClassificationModel 的损失函数。"""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR（基于 Transformer 的实时检测与跟踪）检测模型类。

    此类负责构建 RTDETR 架构、定义损失函数，并支持训练和推理流程。RTDETR 是一种目标检测与跟踪模型，
    继承自 DetectionModel 基类。

    属性：
        nc (int)：检测类别数量。
        criterion (RTDETRDetectionLoss)：训练使用的损失函数。

    方法：
        __init__：初始化 RTDETRDetectionModel。
        init_criterion：初始化损失函数。
        loss：计算训练损失。
        predict：执行模型的前向传播。

    示例：
        初始化一个 RTDETR 模型。
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """初始化 RTDETRDetectionModel。

        参数：
            cfg (str | dict)：配置文件名或路径。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：初始化期间是否打印详细信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _remap_cls_by_names(self, csd: dict[str, torch.Tensor], src_model: torch.nn.Module, verbose: bool = True):
        """根据类别名称重新映射 RT-DETR 解码器分类头中的行。

        此方法覆盖 BaseModel 中针对 YOLO 的实现：RT-DETR 的分类张量位于 RTDETRDecoder 内部的
        `score_head` 和 `class_embed` 下，而不是 `Detect.cv3`。这些张量都按类别逐行存储，包括仅在训练时使用的
        `denoising_class_embed` 嵌入，因此即使源模型和目标模型的 `nc` 不同，也能传递匹配类别的行；剩余的形状
        不匹配项会由 `intersect_dicts` 丢弃。

        参数：
            csd (dict)：预训练检查点的 state_dict（会被修改）。
            src_model (torch.nn.Module)：预训练模块，用于读取 `.names`。
            verbose (bool)：是否记录映射摘要。

        返回：
            (int)：重新映射的分类张量数量（计入“Transferred”日志）。
        """
        src_names = getattr(src_model, "names", None)
        tgt_names = getattr(self, "names", None)
        if not (isinstance(src_names, dict) and isinstance(tgt_names, dict)):
            return 0
        # 跳过默认占位名称 {0:"0", 1:"1", ...}（也会捕获空字典）
        if any(all(str(k) == str(v) for k, v in n.items()) for n in (src_names, tgt_names)):
            return 0

        src_lookup = {str(v).strip().lower(): k for k, v in src_names.items()}
        tgt_nc = len(tgt_names)
        idx = torch.tensor(
            [src_lookup.get(str(tgt_names[k]).strip().lower(), -1) for k in range(tgt_nc)], dtype=torch.long
        )
        n_match = int((idx >= 0).sum())
        # 如果没有匹配项，或类别名称的顺序和数量已经相同，则跳过（intersect_dicts 会直接处理）。
        if n_match == 0 or (len(src_names) == tgt_nc and torch.equal(idx, torch.arange(tgt_nc))):
            return 0

        valid = idx >= 0
        state_dict = self.state_dict()
        cls_keys = {k for k in csd if ("score_head" in k or "class_embed" in k) and k in state_dict}
        remapped = 0
        for k in cls_keys:
            v_src, v_tgt = csd[k], state_dict[k]
            if v_src.ndim != v_tgt.ndim or v_src.shape[1:] != v_tgt.shape[1:]:
                continue
            v_tgt[valid] = v_src[idx[valid]].to(v_tgt.dtype)
            csd.pop(k)  # 防止 intersect_dicts 按源类别的错误顺序再次复制这些行
            remapped += 1
        if verbose and remapped:
            LOGGER.info(f"Remapped {n_match}/{tgt_nc} decoder cls head rows from pretrained weights by class name")
        return remapped

    def _apply(self, fn):
        """将函数应用于模型中的所有张量，包括解码器锚框和有效性掩码。

        参数：
            fn (function)：要应用于模型的函数。

        返回：
            (RTDETRDetectionModel)：更新后的 RTDETRDetectionModel 对象。
        """
        super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """初始化 RTDETRDetectionModel 的损失函数。"""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """计算给定批次数据的损失。

        参数：
            batch (dict)：包含图像和标签数据的字典。
            preds (tuple，可选)：预先计算的模型预测结果。

        返回：
            (torch.Tensor)：总损失值。
            (dict)：包含三个主要损失的字典。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # 注意：将 gt_bbox 和 gt_labels 预处理为列表。
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # 注意：RTDETR 中大约有 12 个损失，反向传播会使用所有损失，但只显示主要的三个损失。
        return sum(loss.values()), {
            "giou_loss": loss["loss_giou"].detach(),
            "cls_loss": loss["loss_class"].detach(),
            "l1_loss": loss["loss_bbox"].detach(),
        }

    def predict(self, x, profile=False, batch=None, augment=False, embed=None):
        """执行模型的前向传播。

        参数：
            x (torch.Tensor)：输入张量。
            profile (bool)：为 True 时统计每层计算耗时。
            batch (dict，可选)：评估使用的真实标注数据。
            augment (bool)：为 True 时在推理期间执行数据增强。
            embed (list，可选)：需要返回嵌入特征的层索引列表。

        返回：
            (torch.Tensor)：模型输出张量。
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # 不包括检测头部分
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从更早的层获取输入
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行前向传播
            y.append(x if m.i in self.save else None)  # 保存输出
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平特征
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # 检测头推理
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World 模型。

    此类实现用于开放词汇目标检测的 YOLOv8 World 模型，支持通过文本指定类别，并集成 CLIP 模型实现零样本检测。

    属性：
        txt_feats (torch.Tensor)：类别的文本特征嵌入。
        clip_model (torch.nn.Module)：用于文本编码的 CLIP 模型。

    方法：
        __init__：初始化 YOLOv8 World 模型。
        set_classes：设置离线推理使用的类别。
        get_text_pe：获取文本位置嵌入。
        predict：使用文本特征执行前向传播。
        loss：使用文本特征计算损失。

    示例：
        初始化一个 World 模型。
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLOv8 World 模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # 特征占位符
        self.clip_model = None  # CLIP 模型占位符
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """预先设置类别，使模型无需 CLIP 模型即可执行离线推理。

        参数：
            text (list[str])：类别名称列表。
            batch (int)：处理文本 token 的批次大小。
            cache_clip_model (bool)：是否缓存 CLIP 模型。
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """使用 CLIP 模型获取文本位置嵌入。

        参数：
            text (list[str])：类别名称列表。
            batch (int)：处理文本 token 的批次大小。
            cache_clip_model (bool)：是否缓存 CLIP 模型。

        返回：
            (torch.Tensor)：文本位置嵌入。
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 兼容缺少 clip_model 属性的旧模型
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, txt_feats=None, augment=False, embed=None):
        """执行模型的前向传播。

        参数：
            x (torch.Tensor)：输入张量。
            profile (bool)：为 True 时统计每层计算耗时。
            txt_feats (torch.Tensor，可选)：文本特征；如果提供则使用该特征。
            augment (bool)：为 True 时在推理期间执行数据增强。
            embed (list，可选)：需要返回嵌入特征的层索引列表。

        返回：
            (torch.Tensor)：模型输出张量。
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:  # 遍历所有模块，包括检测头
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从更早的层获取输入
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # 执行前向传播

            y.append(x if m.i in self.save else None)  # 保存 输出
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平特征
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失。

        参数：
            batch (dict)：用于计算损失的批次数据。
            preds (torch.Tensor | list[torch.Tensor]，可选)：模型预测结果。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE 目标检测模型。

    此类实现 YOLOE 架构，支持使用文本提示和视觉提示进行高效目标检测，并支持有提示和无提示两种推理模式。

    属性：
        pe (torch.Tensor)：类别的提示嵌入。
        clip_model (torch.nn.Module)：用于文本编码的 CLIP 模型。

    方法：
        __init__：初始化 YOLOE 模型。
        get_text_pe：获取文本位置嵌入。
        get_visual_pe：获取视觉嵌入。
        set_vocab：为无提示模型设置词汇表。
        get_vocab：获取融合后的词汇表层。
        set_classes：设置离线推理使用的类别。
        get_cls_pe：获取类别位置嵌入。
        predict：使用提示执行前向传播。
        loss：使用提示计算损失。

    示例：
        初始化一个 YOLOE 模型。
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLOE 模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """使用 CLIP 模型获取文本位置嵌入。

        参数：
            text (list[str])：类别名称列表。
            batch (int)：处理文本 token 的批次大小。
            cache_clip_model (bool)：是否缓存 CLIP 模型。
            without_reprta (bool)：是否返回未经 reprta 模块处理的文本嵌入。

        返回：
            (torch.Tensor)：数据类型与模型参数一致的文本位置嵌入。
        """
        from ultralytics.nn.text_model import build_text_model

        assert len(text), f"Expected at least one class name, but got {text}"
        param = next(self.model.parameters())
        device = param.device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 兼容缺少 clip_model 属性的旧模型
            self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

        model = (
            self.clip_model
            if cache_clip_model
            else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
        )
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1]).to(param.dtype)  # CLIP 始终输出 float32
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # 运行辅助文本检测头

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """获取视觉位置嵌入。

        参数：
            img (torch.Tensor)：输入图像张量。
            visual (torch.Tensor)：视觉特征。

        返回：
            (torch.Tensor)：视觉位置嵌入。
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """为无提示模型设置词汇表。

        参数：
            vocab (nn.ModuleList)：词汇表模块列表。
            names (list[str])：类别名称列表。
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        names = check_class_names(names)  # 在重参数化之前验证名称，因为重参数化无法撤销
        assert len(vocab) == head.nl, f"Expected one vocabulary item per detection level ({head.nl}), got {len(vocab)}."

        # 缓存 anchors 用于 head
        with torch.no_grad():  # 被跟踪的预热会在主干网络中构建计算图
            self(next(self.parameters()).new_empty(1, 3, self.args["imgsz"], self.args["imgsz"]))  # 预热

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        cv2 = getattr(head, "one2one_cv2", head.cv2)

        # 为无提示模型执行重参数化
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
        )
        for loc_head, cls_head in zip(cv2, cv3):  # 这些分支用于构建 lrpc；端到端模式下使用 one2one 分支
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = names

    def get_vocab(self, names):
        """从模型中获取融合后的词汇表层。

        参数：
            names (list[str])：类别名称列表。

        返回：
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused
        names = list(check_class_names(names).values())  # 在融合检测头之前验证名称，因为融合无法撤销

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # 将提示嵌入融合到分类检测头

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        vocab = nn.ModuleList()
        for cls_head in cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """预先设置类别，使模型无需 CLIP 模型即可执行离线推理。

        参数：
            names (list[str])：类别名称列表。
            embeddings (torch.Tensor)：嵌入张量。
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.names = check_class_names(names)  # 在写入任何状态之前验证名称
        self.pe = embeddings
        self.model[-1].nc = len(names)

    def get_cls_pe(self, tpe, vpe):
        """获取类别位置嵌入。

        参数：
            tpe (torch.Tensor | None)：文本位置嵌入。
            vpe (torch.Tensor | None)：视觉位置嵌入。

        返回：
            (torch.Tensor)：类别位置嵌入。
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(self, x, profile=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False):
        """执行模型的前向传播。

        参数：
            x (torch.Tensor)：输入张量。
            profile (bool)：为 True 时统计每层计算耗时。
            tpe (torch.Tensor，可选)：文本位置嵌入。
            augment (bool)：为 True 时在推理期间执行数据增强。
            embed (list，可选)：需要返回嵌入特征的层索引列表。
            vpe (torch.Tensor，可选)：视觉位置嵌入。
            return_vpe (bool)：为 True 时返回视觉位置嵌入。

        返回：
            (torch.Tensor)：模型输出张量。
        """
        y, dt, embeddings = [], [], []  # 输出
        b = x.shape[0]
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:  # 遍历所有模块，包括检测头
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从更早的层获取输入
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x.append(cls_pe)  # 添加类别嵌入
            x = m(x)  # 执行前向传播

            y.append(x if m.i in self.save else None)  # 保存 输出
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平特征
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失。

        参数：
            batch (dict)：用于计算损失的批次数据。
            preds (torch.Tensor | list[torch.Tensor]，可选)：模型预测结果。
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # 待处理
            self.criterion = (
                (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
                if visual_prompt
                else self.init_criterion()
            )
        if preds is None:
            preds = self.forward(
                batch["img"],
                tpe=None if "visuals" in batch else batch.get("txt_feats", None),
                vpe=batch.get("visuals", None),
            )
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE 实例分割模型。

    此类扩展 YOLOEModel，用于处理带文本和视觉提示的实例分割任务，并为像素级目标检测和分割提供专用的损失计算。

    方法：
        __init__：初始化 YOLOE 分割模型。
        loss：使用提示计算分割任务的损失。

    示例：
        初始化一个 YOLOE 分割模型。
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLOE 分割模型。

        参数：
            cfg (str | dict)：模型配置文件路径或配置字典。
            ch (int)：输入通道数量。
            nc (int，可选)：类别数量。
            verbose (bool)：是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """计算损失。

        参数：
            batch (dict)：用于计算损失的批次数据。
            preds (torch.Tensor | list[torch.Tensor]，可选)：模型预测结果。
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # 待处理
            self.criterion = (
                (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
                if visual_prompt
                else self.init_criterion()
            )

        return super().loss(batch, preds)


class Ensemble(torch.nn.ModuleList):
    """模型集成容器。

    此类允许组合多个 YOLO 模型，并通过模型平均或其他集成方法提升性能。

    方法：
        __init__：初始化模型集成容器。
        forward：生成集成中所有模型的预测结果。

    示例：
        创建一个模型集成。
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """初始化模型集成。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False):
        """执行集成前向传播，并拼接所有模型的预测结果。

        参数：
            x (torch.Tensor)：输入张量。
            augment (bool)：是否增强输入。
            profile (bool)：是否统计模型耗时。

        返回：
            (torch.Tensor)：所有模型拼接后的预测结果。
            (None)：集成推理始终返回 None。
        """
        y = [module(x, augment, profile)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # 最大值集成
        # y = torch.stack(y).mean(0)  # 平均值集成
        y = torch.cat(y, 2)  # NMS 集成，y 的形状为 (B, HW, C*num_models)
        return y, None  # 推理输出，训练输出为 None


# 函数 ---------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """临时添加或修改 Python 模块缓存（`sys.modules`）的上下文管理器。

    此函数可在运行时修改模块路径，适用于代码重构场景：模块已从一个位置移动到另一个位置，但仍需支持旧的导入路径
    以保持向后兼容。

    参数：
        modules (dict，可选)：旧模块路径到新模块路径的映射字典。
        attributes (dict，可选)：旧模块属性到新模块属性的映射字典。

    示例：
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # 此时会导入 new.module
        >>> from old.module import attribute  # 此时会导入 new.module.attribute

    注意：
        这些修改仅在上下文管理器内部生效，退出上下文管理器后会撤销。
        注意：直接操作 `sys.modules` 可能导致不可预测的结果，尤其是在较大的应用程序或库中。请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 使用旧名称在 sys.modules 中设置属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 使用旧名称在 sys.modules 中设置模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 移除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class _SafeLoad:
    """可选的受限检查点加载器：仅重建已知模型类（`weights_only=True` 配合允许列表），并在不使用 `eval()` 的情况下构建模型。

    可通过进程级环境变量 `ULTRALYTICS_SAFE_LOAD` 或单次调用参数
    `torch_safe_load(..., safe_only=True)` 启用。默认加载方式（未启用标志时）保持不变。
    受限加载注册的全局对象会在当前进程中持续有效，因此也会应用于之后执行的其他
    `torch.load(weights_only=True)` 调用。
    """

    # 受限加载需要 torch 2.6+，因为需要检查点全局对象扫描和 `(obj, "module.Name")` 允许列表别名。
    # 在较旧版本的 torch 中，受限加载会降级为标准加载。
    SUPPORTED = hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint")
    _registry = None  # {"module.Name": 允许列表条目}，每个进程只构建一次
    _lock = (
        threading.Lock()
    )  # add_safe_globals 会重新绑定进程级设置；_build() 全程持锁，避免导入期间执行加载
    _local = threading.local()  # 线程局部标志，表示当前线程正在执行 weights_only 加载

    @classmethod
    def restricted(cls):
        """判断模型构建是否应使用不调用 eval() 的已知层路径（环境变量或正在进行的加载）。"""
        return cls.SUPPORTED and (SAFE_LOAD or getattr(cls._local, "active", False))

    @classmethod
    @contextlib.contextmanager
    def loading(cls, weight):
        """使用 `weights_only=True` 加载：注册检查点所需的全局对象，并将当前线程标记为受限模式，
        使进入模型构建（parse_model）的检查点同样使用不调用 eval() 的已知层路径。

        全局对象通过 `add_safe_globals` 注册，并在当前进程的整个生命周期内有效，而不是按单次加载设置作用域。
        这是因为 `safe_globals()` 上下文管理器退出时会从进程级集合中移除条目；并发加载时，一个线程退出可能会删除
        另一个线程正在反序列化时所需的允许列表。仅注册检查点实际引用的全局对象也能保持受限反序列化器的速度，
        因为 torch 会在每个 GLOBAL/NEWOBJ/REDUCE/BUILD 操作码处根据全部已注册对象重建查找表；包含 660 个条目的允许列表，
        对于只引用其中 20 个对象的检查点，加载时间几乎会翻倍。
        """
        try:
            needed = torch.serialization.get_unsafe_globals_in_checkpoint(weight)
        except ValueError:  # 不是 torch.save() zip 压缩包；torch.load 会报告格式错误，因此无需注册任何对象
            needed = []
        with cls._lock:
            if cls._registry is None:
                cls._registry = cls._build()
            if any(name.startswith("torchvision.transforms.") for name in needed):
                # 分类预处理变换；仅在检查点序列化了这些对象时导入。
                import torchvision.transforms.transforms as tvt
                from torchvision.transforms.functional import InterpolationMode

                for obj in (tvt.Compose, tvt.Normalize, tvt.Resize, tvt.CenterCrop, tvt.ToTensor, InterpolationMode):
                    cls._registry[f"{obj.__module__}.{obj.__qualname__}"] = obj
            entries = [cls._registry[name] for name in needed if name in cls._registry]
            if entries:
                torch.serialization.add_safe_globals(entries)
        cls._local.active = True
        try:
            yield
        finally:
            cls._local.active = False

    @staticmethod
    def activation(act):
        """在不使用 `eval()` 的情况下，将模型 YAML 中的 `activation` 配置解析为 `torch.nn` 模块实例。

        仅接受文档规定的 `[torch.]nn.<Class>(字面量参数)` 形式（例如 `nn.SiLU()`、
        `torch.nn.LeakyReLU(0.1)`），并拒绝其他形式。
        """
        import ast

        try:
            call = ast.parse(act.strip(), mode="eval").body
            assert isinstance(call, ast.Call)
            attrs = []
            node = call.func
            while isinstance(node, ast.Attribute):  # 展开，例如 torch.nn.SiLU -> ["SiLU", "nn", "torch"]
                attrs.append(node.attr)
                node = node.value
            assert isinstance(node, ast.Name)
            attrs.append(node.id)  # 例如 ["SiLU", "nn"] 或 ["SiLU", "nn", "torch"]
            assert attrs[1:] in (["nn"], ["nn", "torch"]), "activation must be a torch.nn class"
            klass = getattr(nn, attrs[0])
            assert isinstance(klass, type) and issubclass(klass, nn.Module)
            args = [ast.literal_eval(a) for a in call.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            return klass(*args, **kwargs)
        except Exception as e:
            raise TypeError(
                emojis(f"ERROR ❌️ unsupported activation '{act}' blocked during restricted model load.")
            ) from e

    @classmethod
    def _build(cls):
        """自动发现 `torch.nn` 和 Ultralytics 模型系列中的 `nn.Module` 子类。
        对每个可访问到这些类的命名空间路径进行注册（包括将 `block.RealNVP` 重新导出为
        `head.RealNVP` 的情况），并加入旧版别名。

        返回：
            (dict): `torch.serialization.add_safe_globals` 所需的条目，包括类和 `(obj, "module.Name")` 别名，
                以其服务的序列化 "module.Name" 路径为键。
        """
        import enum
        import importlib
        import inspect
        import pathlib
        import pkgutil

        import torch.nn.modules as torch_nn

        import ultralytics.nn.modules as ul_nn
        from ultralytics.nn import tasks as ul_tasks  # noqa: PLW0406

        allow = []

        def _scan(pkg):
            mods = [pkg]
            if hasattr(pkg, "__path__"):  # 对于包，包含所有子模块
                for info in pkgutil.iter_modules(pkg.__path__, f"{pkg.__name__}."):
                    try:
                        mods.append(importlib.import_module(info.name))
                    except Exception:  # noqa: S112  # 可选或异常子模块，跳过
                        continue
            for mod in mods:
                for name, klass in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(klass, nn.Module):
                        # 按该类可访问的路径注册，与检查点对其进行序列化时使用的路径保持一致
                        allow.append((klass, f"{mod.__name__}.{name}"))

        _scan(torch_nn)  # PyTorch nn 模块
        _scan(ul_nn)  # ultralytics 的 block/conv/head/transformer 模块
        _scan(ul_tasks)  # ultralytics 任务模型

        # 官方检查点中的非 nn.Module 数据全局对象，包括 8.0.44 之前使用的 `ultralytics.yolo.utils` 路径
        allow.append(IterableSimpleNamespace)
        allow.append((IterableSimpleNamespace, "ultralytics.yolo.utils.IterableSimpleNamespace"))

        # 旧版和跨平台别名（序列化路径不使用当前类的命名空间），与 temporary_modules() 保持一致
        from ultralytics.utils.loss import E2EDetectLoss

        def _getattr(obj, name):  # 检查点通过 getattr 序列化 `Detect.forward` 和 `InterpolationMode.BILINEAR`
            if isinstance(obj, type) and not name.startswith("__") and issubclass(obj, (nn.Module, enum.Enum)):
                return getattr(obj, name)
            raise pickle.UnpicklingError(f"unsafe getattr({obj!r}, {name!r}) blocked during restricted model load")

        allow += [
            (nn.Identity, "ultralytics.nn.modules.block.Silence"),  # YOLOv9e
            (DetectionModel, "ultralytics.nn.tasks.YOLOv10DetectionModel"),  # YOLOv10
            (E2EDetectLoss, "ultralytics.utils.loss.v10DetectLoss"),  # YOLOv10
            (_getattr, "builtins.getattr"),  # 非检测类 YOLOv8、YOLO11 检查点（限制为 nn.Module 属性）
        ]
        if WINDOWS:
            allow += [
                pathlib.WindowsPath,
                (pathlib.WindowsPath, "pathlib.WindowsPath"),
                (pathlib.WindowsPath, "pathlib.PosixPath"),
                (pathlib.WindowsPath, f"{pathlib.PosixPath.__module__}.{pathlib.PosixPath.__qualname__}"),
            ]
        else:
            allow += [
                pathlib.PosixPath,
                (pathlib.PosixPath, "pathlib.PosixPath"),
                (pathlib.PosixPath, "pathlib.WindowsPath"),
                (pathlib.PosixPath, f"{pathlib.WindowsPath.__module__}.{pathlib.WindowsPath.__qualname__}"),
            ]
        return {(e[1] if isinstance(e, tuple) else f"{e.__module__}.{e.__qualname__}"): e for e in allow}


def torch_safe_load(weight, safe_only=None):
    """使用 torch.load() 加载 PyTorch 模型。

    如果出现 ModuleNotFoundError，则捕获该错误、记录警告，并通过 check_requirements() 尝试安装缺失模块。
    安装完成后再次使用 torch.load() 尝试加载模型。

    参数：
        weight (str | Path): PyTorch 模型文件路径。
        safe_only (bool, 可选): 是否使用 `torch.load(weights_only=True)` 加载，只重建允许列表中的已知
            Ultralytics/PyTorch 模型类。默认取自 `ULTRALYTICS_SAFE_LOAD` 环境变量（关闭），因此不会改变标准用法；
            设置该环境变量即可启用。

    返回：
        (dict): 加载的模型检查点。
        (str): 加载的文件名。

    示例：
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

    if safe_only is None:
        safe_only = SAFE_LOAD
    if safe_only and not _SafeLoad.SUPPORTED:
        safe_only = False
    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # 如果本地缺失，则在线搜索

    def _load():
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
                # 解决跨平台 pathlib pickle 不兼容问题
                **(
                    {"pathlib.PosixPath": "pathlib.WindowsPath"}
                    if WINDOWS
                    else {"pathlib.WindowsPath": "pathlib.PosixPath"}
                ),
            },
        ):
            if safe_only:
                with _SafeLoad.loading(file):  # 使用已知类允许列表进行 weights_only 加载
                    return torch_load(file, map_location="cpu", weights_only=True)
            return torch_load(file, map_location="cpu")

    # weights_only=True 遇到 TorchScript archive 时会抛出异常；默认路径则返回 ScriptModule。
    torchscript_error = emojis(
        f"ERROR ❌️ {weight} is a TorchScript archive, not an Ultralytics PyTorch checkpoint.\n"
        f"Load the original .pt weights, or export again with format='torchscript' and load that file directly."
    )

    try:
        ckpt = _load()

    except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
        # 无法读取的文件会根据损坏方式，在该加载器中表现为三种内部错误之一：
        # RuntimeError 表示 zip 文件被截断，EOFError 表示文件为空，UnpicklingError 表示字节内容根本不是
        # pickle（例如将图像或其他压缩包重命名为 .pt）。它们对用户而言属于同一种情况，因此共用处理逻辑和提示信息。
        if isinstance(e, RuntimeError) and "TorchScript archive" in str(e):
            raise TypeError(torchscript_error) from e
        if isinstance(e, RuntimeError) and "PytorchStreamReader" not in str(e):
            raise  # 无关的 RuntimeError 表示真实执行失败，而不是文件损坏
        if safe_only and isinstance(e, pickle.UnpicklingError):
            # weights_only=True 拒绝了允许列表之外的全局对象：这是格式问题，不是文件损坏
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} references types outside the supported Ultralytics checkpoint format. "
                    f"Use an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        # 仅恢复通过裸名称请求的、缓存中的损坏官方资源；绝不修改用户主动提供的路径。
        name = Path(str(weight)).name
        if str(weight) != name or name not in GITHUB_ASSETS_NAMES:
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} is not a loadable checkpoint — the file is empty, truncated or corrupted "
                    f"({type(e).__name__}: {e}).\nRecommend fixes are to re-download or re-export the file, or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        LOGGER.warning(f"Corrupt cache {file}, re-downloading {weight}...")
        Path(file).unlink(missing_ok=True)
        file = attempt_download_asset(weight)
        ckpt = _load()

    except ModuleNotFoundError as e:  # e.name 是缺失模块的名称
        if e.name in {"models", "models.yolo", "models.common", "models.experimental"}:
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5. This model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        elif e.name and e.name.startswith("ultralytics."):
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires missing Ultralytics module '{e.name}'. "
                    "Train a new model using the latest 'ultralytics' package or run a command with an official "
                    "Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        if safe_only:
            # 在 weights_only 加载模式下，不要自动安装检查点指定的模块，也不要回退到 weights_only=False 重新加载。
            raise
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
        )
        check_requirements(e.name)  # 安装缺失模块
        ckpt = torch_load(file, map_location="cpu")

    if isinstance(ckpt, torch.jit.ScriptModule):
        raise TypeError(torchscript_error)  # 默认路径：torch.load 已转由 torch.jit.load 处理并成功返回

    if not isinstance(ckpt, dict):
        # 文件可能是直接使用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """加载单个模型权重。

    参数：
        weight (str | Path): 模型权重路径。
        device (torch.device, 可选): 加载模型的设备。
        inplace (bool): 是否执行原地操作。
        fuse (bool): 是否融合模型。

    返回：
        (torch.nn.Module): 加载的模型。
        (dict): 模型检查点字典。
    """
    if str(weight).lower().startswith(REMOTE_FILE_PREFIXES):
        weight = check_file(weight, download_dir=SETTINGS["weights_dir"])
    ckpt, weight = torch_safe_load(weight)  # 加载 ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 合并模型和默认参数，优先使用模型参数
    candidate = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(candidate, torch.nn.Module):
        raise TypeError(
            emojis(
                f"ERROR ❌️ {weight} references types outside the supported Ultralytics checkpoint format. "
                f"Use an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
            )
        )
    model = candidate.float()  # FP32 模型

    # 模型 compatibility 更新
    model.args = args  # 将参数附加到模型
    model.pt_path = str(weight)  # 将 *.pt 文件路径以字符串形式附加到模型（避免 WindowsPath 序列化问题）
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # 模型进入 eval 模式

    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # 返回 模型 和 ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """将 YOLO model.yaml 字典解析为 PyTorch 模型。

    参数：
        d (dict): 模型字典。
        ch (int): 输入通道数。
        verbose (bool): 是否打印模型详细信息。

    返回：
        (torch.nn.Sequential): PyTorch 模型。
        (list): 需要保存输出的层索引排序列表。
    """
    import ast

    # 参数
    legacy = True  # 用于兼容 v3/v5/v8/v9 模型
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"未提供模型规模，假定 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    restricted = _SafeLoad.restricted()
    if act:
        # 重新定义默认激活函数，即 Conv.default_act = torch.nn.SiLU()。
        # 在受限加载模式下，使用不调用 eval() 的方式解析该配置（参见 _SafeLoad.activation）。
        Conv.default_act = _SafeLoad.activation(act) if restricted else eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # 层, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # 使用 'repeat' 参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # 从, number, module, 参数
        m = (
            getattr(torch.nn, m[3:])
            if m.startswith("nn.")
            else getattr(__import__("torchvision").ops, m[16:])
            if m.startswith("torchvision.ops.")
            else globals()[m]
        )  # 获取模块
        if restricted and not (isinstance(m, type) and issubclass(m, torch.nn.Module)):
            # 在受限加载模式下，这里只能指定已知的模型层。
            raise TypeError(emojis(f"ERROR ❌️ module '{m}' is not a permitted model layer under restricted loading."))
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if m is not Classify:  # Classify() 输出必须保持为 nc；其他层都根据 width 进行缩放
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # 设置 1) embed channels 和 2) num heads
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
                hidden_channels = int(c2 * (args[6] if len(args) > 6 else 0.5))
                if hidden_channels % args[2]:
                    raise ValueError(
                        f"C2fAttn hidden channels {hidden_channels} (from c2={c2}) must be divisible by nh={args[2]}; "
                        "adjust width_multiple, nh, or C2fAttn expansion"
                    )
                args[1] = hidden_channels

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # 重复次数
                n = 1
            if m is C3k2:  # 用于 M/L/X 规模
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # 用于 L/X 规模
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is Depth:
            args = [*args[:1], [ch[x] for x in f]]  # c_mid、ch 元组；丢弃旧检查点保存的旧版模式参数
        elif m is SemanticSegment:
            args.append([ch[x] for x in f])  # nc 和 ch 元组
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # 将通道数作为第二个参数
        elif m is RTDETRDecoder:  # 特殊情况，通道参数必须传入索引 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、来源索引和类型
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 追加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLO 模型。

    参数：
        path (str | Path): YAML 文件路径。

    返回：
        (dict): 模型字典。
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 例如 yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # 模型 dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """从模型路径中提取模型规模对应的尺寸字符 n、s、m、l 或 x。

    参数：
        model_path (str | Path): YOLO 模型 YAML 文件路径。

    返回：
        (str): 模型规模的尺寸字符（n、s、m、l 或 x）；找不到时返回空字符串。
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """根据 PyTorch 模型的架构或配置推断其任务类型。

    参数：
        model (torch.nn.Module | dict | str | Path): PyTorch 模型、模型配置字典或模型文件路径。

    返回：
        (str): 模型任务类型（'detect'、'segment'、'classify'、'pose'、'obb'、'semantic' 或 'depth'）。
    """

    def cfg2task(cfg):
        """根据 YAML 字典推断任务类型。"""
        m = cfg["head"][-1][-2].lower()  # 输出 module 名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "semanticsegment" in m:
            return "semantic"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"
        if "depth" in m:
            return "depth"

    # 从模型配置推测任务
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # 从 PyTorch 模型推测任务
    if isinstance(model, torch.nn.Module):  # PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307：仅对已知属性路径执行安全求值
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307：仅对已知属性路径执行安全求值
        for m in model.modules():
            if isinstance(m, SemanticSegment):
                return "semantic"
            elif isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, Depth):
                return "depth"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    if isinstance(model, (str, Path)):
        from ultralytics.nn.backends.base import BaseBackend

        if task := BaseBackend.read_metadata(model).get("task"):  # 导出文件会嵌入任务信息，例如重命名后的 best.onnx
            return task

        # 从模型文件名推测任务
        model = Path(model)
        if "-sem" in model.stem or "semantic" in model.parts:
            return "semantic"
        elif "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "-depth" in model.stem or "depth" in model.parts:
            return "depth"
        elif "detect" in model.parts:
            return "detect"

    # 无法从模型确定任务
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', 'pose', 'obb' or 'semantic'."
    )
    return "detect"  # 默认使用检测任务
