# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """可序列化的前向钩子，将层输出保存到共享字典中。"""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """使用共享特征字典和用于保存输出的层索引初始化钩子。"""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """将层的前向输出按照索引保存到共享特征字典中。

        neck 层的输出是张量，而 Detect 检测头的输出是元组或字典，因此不为其指定具体类型。
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO 知识蒸馏模型。

    此类封装教师模型和学生模型，用于知识蒸馏训练。函数通过前向钩子从两个模型中提取特征，
    并据此计算蒸馏损失。

    属性：
        teacher_model (nn.Module): 已冻结、用于提供特征的教师模型。
        student_model (nn.Module): 待进行知识蒸馏训练的学生模型。
        feats_idx (列表): 用于提取特征的层索引。
        projector (nn.ModuleList): 将学生特征映射到教师特征维度的投影器。
        dis (float): 蒸馏损失的权重因子。

    方法：
        get_distill_layers: 从 Detect 检测头自动确定蒸馏特征层。
        forward: 运行学生模型，或在传入训练批次时计算组合损失。
        loss: 计算检测损失和蒸馏损失的组合损失。
        loss_sl2: 计算特征对的分数加权 L2 蒸馏损失。
        decouple_outputs: 统一训练和验证格式下教师模型与学生模型的检测头输出。
        fuse: 融合并返回用于推理和导出的学生模型。
        train: 设置训练模式，同时保持教师模型冻结。

    示例：
        使用较大的教师模型对学生模型进行知识蒸馏训练（设置 ``distill_model`` 参数后，
        训练器会在内部创建 DistillationModel）
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """使用教师模型、学生模型和特征提取钩子初始化蒸馏模型。

        参数：
            teacher_model (str | Path | nn.Module): 教师模型检查点路径或模型模块。
            student_model (nn.Module): 待训练的学生模型模块。
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        # 基于钩子捕获特征：教师模型和学生模型使用相同的方式
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # 通过虚拟前向传播获取特征维度（钩子会捕获输出）
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def __getstate__(self):
        """返回用于序列化的状态副本，不包含已捕获的特征或钩子句柄。

        这里会原地清空特征字典，而不是替换属性，因为已注册的 FeatureHook 共享这些字典对象；
        否则，对训练中途的模型执行 deepcopy 或 pickle 时，仍会访问钩子持有的张量，
        而这些张量带有 grad_fn，无法进行深度复制。
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """反序列化后清除过期特征和钩子，并重新注册前向钩子。"""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """移除之前注册的所有特征捕获钩子。"""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """从模块的前向钩子中移除所有 FeatureHook 实例。"""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """注册特征捕获钩子，并先移除过期的 FeatureHook 实例。"""
        self._remove_feature_hooks()
        for idx in self.feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """从模型的 Detect 检测头自动确定蒸馏特征层。

        返回 Detect 检测头的输入层索引，以及检测头自身的层索引。
        例如，YOLO26 -> [16, 19, 22, 23]，YOLOv8 -> [15, 18, 21, 22]。
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """在蒸馏过程中保持教师模型不变。"""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """设置模型的训练模式，同时保持教师模型处于冻结的评估模式。"""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """通过学生模型执行前向传播。"""
        if isinstance(x, dict):  # 处理训练期间训练和验证使用批次字典的情况
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640):
        """融合并返回学生模型，同时移除仅用于训练的蒸馏包装器。"""
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def loss(self, batch, preds=None):
        """计算损失。

        参数：
            batch (dict): 用于计算损失的数据批次。
            preds (torch.Tensor | 列表[torch.Tensor], 可选): 预测结果。
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # 训练期间进行验证时，仅计算常规损失
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # 前向传播前清空特征字典
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(batch["img"])  # 钩子捕获教师模型特征
        preds = self.student_model(batch["img"])  # 钩子捕获学生模型特征

        regular_loss, loss_items = self.student_model.loss(batch, preds)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        # neck 特征尺寸可能随批次变化（例如 multi_scale），因此根据当前教师特征拆分分数
        neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
            student_feat = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
            loss_distill += (
                self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
            )

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """计算特征对的分数加权 L2 蒸馏损失。

        参数：
            student_feat (torch.Tensor): 形状为 (N, C, H, W) 的学生特征张量。
            teacher_feat (torch.Tensor): 形状为 (N, C, H, W) 的教师特征张量。
            feat_idx (int): 用于选择教师分数的特征层索引。
            teacher_scores (tuple): 每个特征层对应的分数张量元组。

        返回：
            (torch.Tensor): 计算得到的分数加权 L2 损失。
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """获取学生模型的损失函数。"""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """设置学生模型的损失函数。"""
        self.student_model.criterion = value

    def init_criterion(self):
        """通过学生模型初始化损失函数。"""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """公开学生模型的端到端模式，供验证器或预测器控制。"""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """将端到端模式更新转发给学生模型。"""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """将检测头属性更新（例如 max_det、agnostic_nms、end2end）转发给学生模型。"""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """解耦教师模型和学生模型的输出。

        此方法处理 YOLO 模型的不同输出格式，包括训练或验证模式下的元组输出、
        带有 one2one/one2many 分支的字典输出，以及直接的张量输出。

        参数：
            preds (torch.Tensor | tuple | dict): 不同格式的模型预测结果。
            branch (str): 从字典输出中提取的分支（"one2one" 或 "one2many"）。

        返回：
            (torch.Tensor | dict): 解耦后的预测结果。
        """
        if isinstance(preds, tuple):  # 解耦验证模式下的输出
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
