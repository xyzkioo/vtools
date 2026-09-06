# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from .byte_tracker import STrack
from .oc_sort import OCSORT, OCSortTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import parse_bboxes


class DeepOCSortTrack(OCSortTrack):
    """带有外观特征和以观测为中心状态管理的 Deep OC-SORT 跟踪对象。

    在 OCSortTrack 基础上增加 ReID 嵌入存储和指数移动平均平滑，并支持置信度自适应的
    embedding update rates.

    属性：
        smooth_feat (np.ndarray | None): Smoothed 特征 vector via EMA.
        curr_feat (np.ndarray | None): Current frame's 特征 vector.
        alpha_fixed_emb (float): Base EMA factor for embedding updates.
        det_thresh (float): Confidence 阈值 below which a new embedding is ignored rather than blended.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        score: float,
        cls: Any,
        delta_t: int = 3,
        feat: np.ndarray | None = None,
        alpha_fixed_emb: float = 0.95,
        det_thresh: float = 0.25,
    ):
        """使用可选的外观特征初始化 `DeepOCSortTrack`。

        参数：
            xywh (np.ndarray): Bounding 边界框 in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format.
            分数 (float): Detection 置信度 in `[0, 1]`.
            cls (Any): Class label for the detection.
            delta_t (int): Temporal window for OCM velocity direction computation.
            feat (np.ndarray | None): 此检测目标的可选外观特征向量。
            alpha_fixed_emb (float): Base EMA factor for embedding updates; higher = slower updates.
            det_thresh (float): 检测置信度阈值；低于该阈值时会替换嵌入，而不是
                blended.
        """
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        self.det_thresh = det_thresh
        if feat is not None:
            self.update_features(feat, score)

    def update_features(self, feat: np.ndarray, score: float | None = None) -> None:
        """使用置信度自适应 EMA 将新的外观特征融合到 `smooth_feat` 中。

        当 `score` 超过 `det_thresh` 时，EMA 因子为
        `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb) * (1 - trust)`，其中
        `trust = (score - det_thresh) / (1 - det_thresh)`，因此高置信度检测结果的融合权重更大。
        当 `score` 小于或等于 `det_thresh` 时，`alpha = 1.0`，保留现有 `smooth_feat` 并忽略新的低可信度特征，
        这与参考 Deep OC-SORT 实现的行为一致。

        参数：
            feat (np.ndarray): 新的（未归一化）外观特征向量。
            分数 (float | None): 用于调节 EMA 因子的检测置信度。
        """
        if score is not None and score > self.det_thresh:
            trust = (score - self.det_thresh) / max(1 - self.det_thresh, 1e-9)
            alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
        else:
            alpha = 1.0  # 对低可信度检测保留现有的 smooth_feat
        curr, smooth = smooth_feature(feat, self.smooth_feat, alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def update(self, new_track: STrack, frame_id: int) -> None:
        """使用匹配的检测结果更新轨迹状态，并刷新外观特征。

        参数：
        new_track (STrack): 当前帧匹配到的检测结果，可选包含 `curr_feat`。
            frame_id (int): Current frame id.
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """重新激活丢失的轨迹，并刷新外观特征。

        参数：
            new_track (STrack): Detection used to revive this track.
            frame_id (int): Current frame id.
            new_id (bool): 为 True 时分配新的跟踪 ID，而不是复用旧 ID。
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().re_activate(new_track, frame_id, new_id)

    @staticmethod
    def multi_gmc(stracks: list[DeepOCSortTrack], H: np.ndarray) -> None:
        """对 XYAH 卡尔曼状态正确应用全局运动补偿。

        `utils.stracks` 中的标准 `multi_gmc` 辅助函数会将 `(a, h)` 与 `(x, y)` 一起旋转，
        但这对宽高比维度是不正确的。本变体只旋转位置 `(x, y)` 和速度 `(vx, vy)` 块，
        保持 `(a, h)` 与 `(va, vh)` 不变，同时旋转已保存的 `last_observation`，以保持 OCR/ORU 一致。

        参数：
            stracks (列表[DeepOCSortTrack]): 要原地变换的轨迹。
            H (np.ndarray): 将前一帧映射到当前帧的 2x3 仿射单应矩阵。
        """
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        t = H[:2, 2]

        # 构建 8x8 变换：旋转 (x,y) 和 (vx,vy)，对 (a,h) 及 (va,vh) 使用单位变换
        R8x8 = np.eye(8)  # float64 可精确保存任意单应矩阵；更窄的类型会舍入 float64 计算结果
        R8x8[:2, :2] = R  # 旋转位置 (x, y)
        R8x8[4:6, 4:6] = R  # 旋转速度 (vx, vy)
        # 索引 2、3（a,h）和 6、7（va,vh）保持单位变换

        multi_mean = np.matmul(R8x8, multi_mean[..., None])[..., 0]
        multi_mean[:, :2] += t
        # 与 `utils.stracks` 中的实现一样，让右操作数保持 C 连续：F 连续数组会使矩阵乘法进入 BLAS 的转置 GEMM 内核，
        # 该内核在 macOS Accelerate 上即使面对有限输入也会留下浮点异常标志，导致 NumPy 错误报告除零 RuntimeWarning。
        multi_covariance = np.matmul(np.matmul(R8x8, multi_covariance), np.ascontiguousarray(R8x8.T))
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

                # 同时变换已保存的观测，以保持 OCR/ORU 一致性
            if stracks[i].last_observation[0] >= 0:
                obs = stracks[i].last_observation
                # 变换 xyxy 观测中心
                cx, cy = DeepOCSortTrack._xyxy_center(obs)
                w, h = obs[2] - obs[0], obs[3] - obs[1]
                new_c = R @ np.array([cx, cy]) + t
                stracks[i].last_observation = np.array(
                    [
                        new_c[0] - w / 2,
                        new_c[1] - h / 2,
                        new_c[0] + w / 2,
                        new_c[1] + h / 2,
                    ],
                    dtype=np.float32,
                )


class DeepOCSORT(OCSORT):
    """Deep OC-SORT：在 OC-SORT 基础上增加外观特征、GMC 和自适应权重。

    相较于直接集成的改进：
    - GMC 正确处理 XYAH 状态（仅旋转 x、y 位置，不旋转宽高比和高度）
    - 按照 BOTSORT 验证过的方法，代价组合使用 min(IoU, 外观相似度)
    - OCR 恢复阶段同样使用外观特征
    - 默认禁用 ByteTrack 风格的低置信度第二阶段匹配
    """

    def __init__(self, args: Any):
        """初始化 Deep OC-SORT 跟踪器。

        参数：
            args (Namespace | IterableSimpleNamespace): Parsed tracker config providing the OC-SORT keys plus
                `gmc_method`, `proximity_thresh`, `appearance_thresh`, `alpha_fixed_emb`, `with_reid`, and `model`.
        """
        super().__init__(args)

        # 用于相机运动补偿的 GMC
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))

        # 外观特征参数
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.75)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)

        self.encoder = build_encoder(
            getattr(args, "with_reid", False), getattr(args, "model", "auto"), getattr(args, "device", None)
        )

    def init_track(self, results, img: np.ndarray | None = None) -> list[DeepOCSortTrack]:
        """构建 `DeepOCSortTrack` 实例，并在启用时附加 ReID 特征。

        当 `with_reid=True` 且 `model="auto"` 时，`img` 应已经是原生骨干特征列表（每个检测结果对应一个特征）；
        对于其他 `model`，`img` 是源图像，配置的外部 ReID 编码器会对检测裁剪区域进行编码。

        参数：
            结果 (Any): 提供 `xywh`（或 `xywhr`）、`conf` 和 `cls` 属性的对象。
            img (np.ndarray | None): 根据 ReID 配置，可以是 BGR 帧或预提取的特征张量。

        返回：
            (列表[DeepOCSortTrack]): 每个检测结果对应一条轨迹；没有检测结果时返回空列表。
        """
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)

        if self.encoder is not None and img is not None:
            features = self.encoder(img, bboxes)
            return [
                DeepOCSortTrack(
                    xywh,
                    s,
                    c,
                    self.delta_t,
                    feat=f,
                    alpha_fixed_emb=self.alpha_fixed_emb,
                    det_thresh=self.args.track_high_thresh,
                )
                for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features)
            ]
        return [
            DeepOCSortTrack(
                xywh, s, c, self.delta_t, alpha_fixed_emb=self.alpha_fixed_emb, det_thresh=self.args.track_high_thresh
            )
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def _pre_first_associate(
        self,
        strack_pool: list[DeepOCSortTrack],
        unconfirmed: list[DeepOCSortTrack],
        img: np.ndarray | None,
        results_high: Any,
    ) -> None:
        """在第一阶段匹配前将 GMC 扭曲应用于卡尔曼状态。"""
        if img is None or self.gmc.method is None:
            return
        try:
            warp = self.gmc.apply(img, results_high.xyxy if len(results_high) else np.empty((0, 4)))
        except Exception:
            warp = np.eye(2, 3)
        DeepOCSortTrack.multi_gmc(strack_pool, warp)
        DeepOCSortTrack.multi_gmc(unconfirmed, warp)

    def _fuse_appearance(
        self,
        dists: np.ndarray,
        tracks: list[DeepOCSortTrack],
        detections: list[DeepOCSortTrack],
        iou_dists: np.ndarray | None = None,
    ) -> np.ndarray:
        """以 BoT-SORT 风格将外观距离以最小值方式融合到运动代价中。"""
        if self.encoder is None or not tracks or not detections:
            return dists
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        if iou_dists is not None:
            emb_dists[iou_dists > (1 - self.proximity_thresh)] = 1.0
        return np.minimum(dists, emb_dists)

    def reset(self) -> None:
        """重置 Deep OC-SORT 跟踪器，同时清除 GMC 扭曲状态。"""
        super().reset()
        self.gmc.reset_params()
