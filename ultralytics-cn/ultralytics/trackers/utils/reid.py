# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""BoT-SORT、Deep OC-SORT 和 TrackTrack 共享的 ReID 编码器。

* `.pt` YOLO 检查点通过 `YOLO()` 加载；通过预测器的 `embed=[...]` 参数从倒数第二层提取嵌入
（适用于分类和 ReID 主干网络）。
* 其他扩展名（`.torchscript`、`.onnx`、`.engine`、`.openvino` 等）通过 `AutoBackend` 加载；
  模型应直接输出嵌入张量。
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")


class ReID:
    """ReID 编码器。`.pt` 文件通过 YOLO 预测器处理，其他格式通过 `AutoBackend` 处理。"""

    def __init__(self, model: str, imgsz: int = 224, device: str | torch.device | None = None, fp16: bool = False):
        """初始化用于重识别的编码器。

        参数：
            model (str): ReID 模型路径。`.pt` 通过 YOLO 预测器提取嵌入，其他扩展名通过 `AutoBackend` 处理。
            imgsz (int): AutoBackend 路径裁剪预处理使用的正方形输入尺寸；检测到模型静态输入尺寸时会覆盖该值。
            device (str | torch.device | None): 推理设备；默认在可用时使用 CUDA。
            fp16 (bool): 后端支持时是否使用半精度。
        """
        self.imgsz = imgsz
        self.batch_size = None
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.is_pt = str(model).endswith(".pt")

        if self.is_pt:
            from ultralytics import YOLO

            self.model = YOLO(model)
            # 使用 embed=[idx] 初始化预测器，使后续调用返回嵌入。
            self.model(embed=[len(self.model.model.model) - 2], device=self.device, verbose=False, save=False)
            self.fp16 = False
        else:
            from pathlib import Path

            if Path(str(model)).name in REID_ASSETS:
                from ultralytics.utils.downloads import attempt_download_asset

                model = attempt_download_asset(str(model))
            self.model = AutoBackend(str(model), device=self.device, fp16=fp16, verbose=False)
            self.fp16 = self.model.fp16

            # 获取模型输入尺寸，用于固定批次和裁剪尺寸，或检测动态批次和裁剪尺寸。
            session = getattr(self.model, "session", None)
            shape = session.get_inputs()[0].shape if session is not None else ()
            if len(shape) == 4:
                if isinstance(shape[0], int) and shape[0] > 0:
                    self.batch_size = shape[0]
                if isinstance(shape[2], int) and shape[2] > 0:
                    self.imgsz = shape[2]

    @staticmethod
    def _crop_detections(img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """从图像中裁剪检测区域，并先将 xywh 转换为 xyxy。

        参数：
            img (np.ndarray): BGR 图像。
            dets (np.ndarray): xywh 格式的检测结果（使用前 4 列）。

        返回：
            (列表[np.ndarray]): 裁剪后的图像块。
        """
        return [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]

    def _crops_to_tensor(self, crops: list[np.ndarray]) -> torch.Tensor:
        """将有效图像裁剪块列表堆叠为 self.imgsz 尺寸的归一化 BCHW 浮点张量。"""
        batch = torch.empty(len(crops), 3, self.imgsz, self.imgsz, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(
                t, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False
            )[0]
        batch = batch.to(self.device)
        return batch.half() if self.fp16 else batch

    @torch.no_grad()
    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray | None]:
        """提取检测对象的嵌入。"""
        crops = self._crop_detections(img, dets)
        valid = [bool(c.size) for c in crops]
        valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
        if not valid_crops:
            return [None] * len(crops)

        if self.is_pt:
            feats = self.model.predictor(valid_crops)
            if len(feats) != len(valid_crops) and feats[0].shape[0] == len(valid_crops):
                feats = feats[0]  # 非 PyTorch 后端的批量预测结果
            valid_feats = [f.cpu().numpy() for f in feats]
        else:
            batch = self._crops_to_tensor(valid_crops)
            bs, n = self.batch_size, batch.shape[0]
            if bs is None or n == bs:
                feats = self.model(batch)
            else:  # 固定批次模型（例如静态 ONNX）：按 bs 分块运行，并填充最后一个不完整批次
                outs = []
                for s in range(0, n, bs):
                    chunk = batch[s : s + bs]
                    if chunk.shape[0] < bs:
                        chunk = torch.cat([chunk, chunk[-1:].expand(bs - chunk.shape[0], *chunk.shape[1:])], 0)
                    outs.append(self.model(chunk))
                feats = torch.cat(outs, 0)[:n]
            valid_feats = [f.cpu().numpy() for f in feats]

        valid_feats = iter(valid_feats)
        return [next(valid_feats) if keep else None for keep in valid]


def build_encoder(with_reid: bool, model: str | None, device: str | torch.device | None = None):
    """返回 ReID 编码器、原生特征直通编码器或 None。

    参数：
        with_reid (bool): 是否启用 ReID。
        model (str | None): `"auto"` 返回将预提取主干特征转换为 NumPy 数组的可调用对象；
            其他值从对应路径加载 `ReID` 模型。`with_reid` 为 False 时忽略该参数。
        device (str | torch.device | None): ReID 模型推理设备；默认在可用时使用 CUDA。

    返回：
        (Callable | None): `(img, dets) -> list[np.ndarray | None]` 编码器；禁用 ReID 时返回 None。
    """
    if not with_reid:
        return None
    if model == "auto":

        def _auto_encoder(feats, _dets):
            if isinstance(feats, np.ndarray):
                return [f for f in feats]
            return [f.cpu().numpy() for f in feats]

        return _auto_encoder
    return ReID(model, device=device)


def smooth_feature(
    feat: np.ndarray, smooth: np.ndarray | None, alpha: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """对 `feat` 执行 L2 归一化，并通过指数移动平均将其融合到 `smooth`。

    参数：
        feat (np.ndarray): 新的未归一化外观特征。
        smooth (np.ndarray | None): 当前平滑特征，首次更新时为 None。
        alpha (float): 现有 `smooth` 的 EMA 权重（``1.0`` 表示保持不变）。

    返回：
        curr (np.ndarray | None): 归一化后的 float32 特征；`feat` 范数为零时返回 None。
        smooth (np.ndarray | None): 更新并归一化后的 float32 特征。
    """
    feat = np.asarray(feat, dtype=np.float32)  # 无论 ReID 后端返回何种类型，保存状态均为 float32
    norm = np.linalg.norm(feat)
    if norm < 1e-12:  # 零范数特征不包含外观信息，通知调用方保留当前特征
        return None, smooth
    feat = feat / norm
    if smooth is None:
        return feat, feat.copy()
    smooth = alpha * smooth + (1 - alpha) * feat
    return feat, smooth / np.linalg.norm(smooth)
