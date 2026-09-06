# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import deque
from functools import wraps
from typing import Any

import numpy as np
import torch

from ultralytics.utils.metrics import bbox_ioa

from ..utils import LOGGER
from .basetrack import TrackState
from .bot_sort import BOTrack
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.reid import smooth_feature
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes

# 角点索引数组（对应 (x1,y1,x2,y2) 边界框的 LT、LB、RT、RB），用于角度距离向量化。
_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])

_LOOSE_NMS_IOU = 0.95  # 较宽松的 NMS IoU，用于恢复严格 NMS 丢弃的检测结果
_LOOSE_NMS_DEDUP_IOU = 0.97  # 将检测结果视为“新结果”时使用的 IoU 阈值


def _hmiou_distance(tracks_a: list[TTSTrack], tracks_b: list[TTSTrack]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (iou_sim, 1 - HMIoU)，其中 HMIoU = HIoU * IoU，HIoU 为垂直重叠与垂直并集之比。"""
    n, m = len(tracks_a), len(tracks_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32), np.ones((n, m), dtype=np.float32)
    boxes_a = np.ascontiguousarray([track.xyxy for track in tracks_a], dtype=np.float32)
    boxes_b = np.ascontiguousarray([track.xyxy for track in tracks_b], dtype=np.float32)
    iou_sim = bbox_ioa(boxes_a, boxes_b, iou=True)
    h_over = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_union = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_iou = np.clip(h_over / (h_union + 1e-9), 0, 1)
    return iou_sim, 1.0 - h_iou * iou_sim


def _angle_distance(
    tracks: list[TTSTrack], dets: list[TTSTrack], frame_id: int, pairs: tuple[np.ndarray, np.ndarray], delta_t: int = 3
) -> np.ndarray:
    """仅为 IoU 支持的 `(track, det)` 索引 `pairs` 返回角度距离。

    `_cost_matrix` 会覆盖不受支持的配对，因此计算完整的 `(N, M)` 网格会产生无用开销。
    """
    track_idx, det_idx = pairs
    track_boxes = np.stack([tracks[i].get_history_box(frame_id, delta_t) for i in track_idx])  # (P, 4)
    det_boxes = np.stack([dets[i].xyxy for i in det_idx])  # (P, 4)
    deltas = det_boxes - track_boxes  # (P, 4)
    dx = deltas[:, _CORNER_DX_IDX]
    dy = deltas[:, _CORNER_DY_IDX]
    norms = np.sqrt(dx * dx + dy * dy) + 1e-5
    dx /= norms
    dy /= norms
    track_velocities = np.stack([tracks[i].velocity for i in track_idx])  # (P, 4, 2)
    dot = track_velocities[:, :, 0] * dx + track_velocities[:, :, 1] * dy
    dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi  # (P,)
    return dist * np.array([dets[i].score for i in det_idx])


def _confidence_distance(tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
    """计算每个跟踪目标的投影分数与每个检测结果置信度之间的绝对差值。"""
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_prev_scores = np.array([track.prev_score for track in tracks])
    track_curr_scores = np.array([track.score for track in tracks])
    track_proj_scores = track_curr_scores + (track_curr_scores - track_prev_scores)  # 一阶外推
    det_scores = np.array([det.score for det in dets])
    return np.abs(track_proj_scores[:, None] - det_scores[None])


def _iterative_associate(cost: np.ndarray, match_thr: float, reduce_step: float = 0.05) -> tuple[list]:
    """使用相互最近邻的贪心匹配，并在每次迭代中缩小阈值。

    返回 (matches, unmatched_tracks, unmatched_dets)。
    """
    matches = []
    cost = cost.copy()
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        nearest_det = np.argmin(cost, axis=1)
        nearest_track = np.argmin(cost, axis=0)
        new_matches = [
            [track_idx, nearest_det[track_idx]]
            for track_idx in range(cost.shape[0])
            if nearest_track[nearest_det[track_idx]] == track_idx
            and cost[track_idx, nearest_det[track_idx]] < match_thr
        ]
        if not new_matches:
            break
        matches.extend(new_matches)
        for track_idx, det_idx in new_matches:
            cost[track_idx, :] = np.inf
            cost[:, det_idx] = np.inf
        match_thr -= reduce_step
    matched_tracks = {track_idx for track_idx, _ in matches}
    matched_dets = {det_idx for _, det_idx in matches}
    unmatched_tracks = [i for i in range(cost.shape[0]) if i not in matched_tracks]
    unmatched_dets = [i for i in range(cost.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def _track_aware_nms(
    tracks: list[TTSTrack], dets: list[TTSTrack], tai_thr: float, new_track_thresh: float
) -> list[bool]:
    """TAI NMS：抑制与现有跟踪目标或更强检测结果高度重叠的检测结果。"""
    if not dets:
        return []
    scores = np.array([det.score for det in dets])
    allow = scores > new_track_thresh
    n_tracks, n_dets = len(tracks), len(dets)
    if n_tracks + n_dets < 2:
        return allow.tolist()
    boxes = np.ascontiguousarray([obj.xyxy for obj in tracks + dets], dtype=np.float32)
    iou = bbox_ioa(boxes, boxes, iou=True)

    if n_tracks:
        allow &= iou[n_tracks:, :n_tracks].max(axis=1) <= tai_thr

    det_iou = iou[n_tracks:, n_tracks:]
    order = scores.argsort()[::-1]
    for i in order:
        if not allow[i]:
            continue
        suppress = det_iou[i] > tai_thr
        suppress[i] = False
        allow[suppress] = False

    return allow.tolist()


def attach_raw_preds_hook(predictor) -> None:
    """包装 `predictor.postprocess`，捕获 NMS 前的原始预测结果和输入（操作幂等）。"""
    if hasattr(predictor, "_orig_postprocess"):
        return
    orig = predictor.postprocess

    @wraps(orig)
    def _wrapped(preds, img, orig_imgs, *args, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds  # PyTorch 模型返回 [推理结果, 额外数据]
        # 使用 clone()，避免原地 NMS 的 xywh->xyxy 转换修改捕获结果；保留源设备以供 box_iou 使用
        predictor._raw_preds = raw.detach().clone() if isinstance(raw, torch.Tensor) else raw
        predictor._postprocess_im = img
        predictor._postprocess_im0s = orig_imgs
        return orig(preds, img, orig_imgs, *args, **kwargs)

    predictor._orig_postprocess = orig
    predictor.postprocess = _wrapped


def compute_dets_del(predictor) -> list | None:
    """返回每个批次中被严格 NMS 丢弃的 `(xywh, conf, cls)` 元组；不可用时返回 None。"""
    raw = getattr(predictor, "_raw_preds", None)
    if raw is None or not isinstance(raw, torch.Tensor):
        return None
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou

    loose_results = predictor._orig_postprocess(
        raw, predictor._postprocess_im, predictor._postprocess_im0s, iou=_LOOSE_NMS_IOU
    )

    is_obb = predictor.args.task == "obb"
    out = []
    for loose, tight in zip(loose_results, predictor.results):
        tight_boxes = tight.obb if is_obb else tight.boxes
        loose_boxes = loose.obb if is_obb else loose.boxes
        if len(loose_boxes) == 0 or len(tight_boxes) == 0:
            out.append(None)
            continue
        max_iou = box_iou(loose_boxes.xyxy, tight_boxes.xyxy).max(dim=1).values
        mask = max_iou < _LOOSE_NMS_DEDUP_IOU
        if not mask.any():
            out.append(None)
            continue
        dels = loose_boxes.data[mask].cpu()
        if is_obb:
            xywh = dels[:, :5].numpy()  # xywhr 格式
            out.append((xywh, dels[:, 5].numpy(), dels[:, 6].numpy()))
        else:
            xywh = ops.xyxy2xywh(dels[:, :4]).numpy()
            out.append((xywh, dels[:, 4].numpy(), dels[:, 5].numpy()))

    predictor._raw_preds = None
    return out


def _cosine_distance(tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
    """计算 `[0, 1]` 范围内的跟踪目标与检测结果嵌入余弦距离；任一侧没有特征时返回 NaN。

    NaN 表示“该配对没有外观证据”，调用方会回退到运动信息，而不是将缺失或被遮挡抑制的嵌入视为最大不相似，
    从而避免错误惩罚真实匹配。
    """
    if not tracks or not dets:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    tfeat = [t.smooth_feat if t.smooth_feat is not None else t.curr_feat for t in tracks]
    dfeat = [d.curr_feat for d in dets]
    dim = next((f.shape[0] for f in (*tfeat, *dfeat) if f is not None), 128)
    zeros = np.zeros(dim, dtype=np.float32)
    # 与 `matching.embedding_distance` 一样固定使用 float32：不固定时，堆叠结果会继承编码器类型，
    # 半精度 ReID 后端（`quantize=16` 或 float16 ONNX 模型）会影响代价矩阵的精度。
    T = np.asarray([f if f is not None else zeros for f in tfeat], dtype=np.float32)
    D = np.asarray([f if f is not None else zeros for f in dfeat], dtype=np.float32)
    valid = np.array([f is not None for f in tfeat])[:, None] & np.array([f is not None for f in dfeat])[None, :]
    return np.where(valid, np.clip(1 - T @ D.T, 0, 1), np.nan).astype(np.float32)


class TTSTrack(BOTrack):
    """TrackTrack 使用的单目标跟踪对象，包含角点速度、分数历史和 ReID 特征。

    该类扩展 `BOTrack`（XYWH 卡尔曼状态和 EMA ReID 平滑），增加角点速度运动信息、分数历史和自适应分数的特征平滑。

    属性：
        min_track_len (int): 类级别默认值；由配置中的 TRACKTRACK 覆盖。
        kalman_filter (KalmanFilterXYWH): 激活后供该跟踪目标使用的卡尔曼滤波器。
        mean (np.ndarray): 均值状态向量。
        covariance (np.ndarray): 协方差矩阵。
        score (float): 当前检测置信度。
        prev_score (float): 上次更新时的置信度（用于分数投影）。
        tracklet_len (int): 激活后的成功更新次数。
        velocity (np.ndarray): 每个角点的 (4,2) 单位速度向量。
        smooth_feat (np.ndarray | None): 经过 EMA 平滑的 ReID 嵌入。
        curr_feat (np.ndarray | None): 当前帧的原始 ReID 嵌入。

    示例：
        创建并激活新的跟踪目标
        >>> track = TTSTrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls="person")
        >>> track.activate(KalmanFilterXYWH(), frame_id=1)
    """

    min_track_len = 3
    _alpha = 0.95
    _delta_t = 3

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, feat: np.ndarray | None = None):
        """根据检测边界框初始化 TTSTrack。

        参数：
            xywh (np.ndarray): `(x, y, w, h, idx)` 或 `(x, y, w, h, angle, idx)`，以中心点表示，并包含检测索引。
            score (float): 检测置信度。
            cls (Any): 类别标签。
            feat (np.ndarray | None): 可选的 ReID 特征向量。
        """
        super().__init__(xywh, score, cls)  # BOTrack 设置 smooth_feat/curr_feat 以及 XYWH 卡尔曼状态
        self.prev_score = score
        self.velocity = np.zeros((4, 2), dtype=np.float32)
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=self._delta_t + 1)
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        """归一化 `feat`，并通过分数自适应 EMA 将其融合到 `smooth_feat` 中。"""
        beta = self._alpha + (1 - self._alpha) * (1 - self.score)
        curr, smooth = smooth_feature(feat, self.smooth_feat, beta)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def get_history_box(self, frame_id: int, dt: int) -> np.ndarray:
        """返回 `dt` 帧之前的边界框；若不存在，则返回最近的边界框或当前边界框。"""
        target = frame_id - dt
        for fid, box in self._history:
            if fid == target:
                return box.copy()
        if self._history:
            return self._history[-1][1].copy()
        return self.xyxy

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        """初始化卡尔曼状态，并将跟踪目标提升为 New 状态。"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history.append((frame_id, self.xyxy))
        self.tracklet_len = 0
        self.state = TrackState.Tracked if self.min_track_len <= 1 else TrackState.New
        self.is_activated = frame_id == 1 or self.state == TrackState.Tracked
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int, new_id: bool = False) -> None:
        """通过 NSA-Kalman 将丢失的跟踪目标重新绑定到新的检测结果。"""
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score
        )
        self._history.append((frame_id, self.xyxy))
        self.score = new_track.score  # 在 update_features 前设置，使 EMA 权重使用当前置信度
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.cls, self.angle, self.idx = new_track.cls, new_track.angle, new_track.idx

    def update(self, new_track, frame_id: int) -> None:
        """使用新的检测结果更新已匹配目标；达到 min_track_len 后提升为 Tracked 状态。"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score
        )
        self._history.append((frame_id, new_track.xyxy))

        velocity = np.zeros((4, 2), dtype=np.float32)
        curr_box = new_track.xyxy
        for dt in range(1, self._delta_t + 1):
            delta = curr_box - self.get_history_box(frame_id, dt)
            dx, dy = delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX]
            norm = np.sqrt(dx * dx + dy * dy) + 1e-5
            velocity += np.stack([dx / norm, dy / norm], axis=-1) / dt
        self.velocity = velocity / self._delta_t

        self.score = new_track.score  # 在 update_features 前设置，使 EMA 权重使用当前置信度
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        if self.state == TrackState.Tracked or self.tracklet_len + 1 >= self.min_track_len:
            self.state = TrackState.Tracked
            self.is_activated = True
        self.cls, self.angle, self.idx = new_track.cls, new_track.angle, new_track.idx

    def __repr__(self) -> str:
        """返回跟踪目标的简短字符串表示。"""
        return f"TT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class TRACKTRACK:
    """实现基于轨迹视角关联和轨迹感知初始化的多目标跟踪器。

    检测结果被划分为高分、低分和删除集（由宽松 NMS 恢复），随后与已跟踪和已丢失目标的并集进行匹配。
    匹配代价融合 HMIoU、余弦 ReID、置信度和角度距离，并通过迭代分配求解。未匹配但仍处于 Lost 状态的目标可选地在
    第二轮宽松匹配中与剩余检测结果重新关联；通过轨迹感知 NMS 保留下来的剩余检测结果将生成新目标。

    属性：
        tracked_stracks (列表[TTSTrack]): 当前正在跟踪的目标。
        lost_stracks (列表[TTSTrack]): 丢失检测但仍在缓冲窗口内的目标。
        frame_id (int): 当前帧索引。
        args (Any): 解析后的跟踪器配置。
        max_time_lost (int): 删除丢失目标前允许的帧数（按源视频帧率缩放）。
        kalman_filter (KalmanFilterXYWH): 用于初始化新目标的卡尔曼滤波器。
        match_thr (float): 主迭代分配的代价门限。
        lost_match_thr (float): 可选丢失目标重新绑定过程的代价门限；为 0 时禁用。
        gmc (GMC): 用于相机运动变换的全局运动补偿器。
        encoder (Any): ReID 编码器；禁用 ReID 时为 None。

    方法：
        update: 推进跟踪器一帧并返回每个目标的跟踪结果。
        reset: 清除跟踪器的所有状态。

    示例：
        初始化并处理单帧图像
        >>> tracker = TRACKTRACK(args)
        >>> tracked_objects = tracker.update(yolo_results, img=image)
    """

    def __init__(self, args):
        """根据跟踪器配置初始化 TRACKTRACK（参见 `ultralytics/cfg/trackers/tracktrack.yaml`）。

        参数：
            args (Any): 解析后的跟踪器配置。所有参数均通过 `getattr(..., default)` 读取，因此缺少近期新增键的旧版 YAML
                文件仍可正常加载。
        """
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = args.track_buffer
        self.kalman_filter = KalmanFilterXYWH()

        self.match_thr = getattr(args, "match_thresh", 0.7)
        self.lost_match_thr = getattr(args, "lost_match_thr", 0.0)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.new_track_thresh = getattr(args, "new_track_thresh", 0.7)
        self.min_track_len = getattr(args, "min_track_len", 3)

        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))

        from .utils.reid import build_encoder

        self.encoder = build_encoder(
            getattr(args, "with_reid", False), getattr(args, "model", "auto"), getattr(args, "device", None)
        )

    @classmethod
    def setup_predictor(cls, predictor):
        """为轨迹感知初始化附加原始预测结果钩子（仅支持 detect/obb）。

        恢复的检测结果（由宽松 NMS 获取）仅包含边界框，并且在 NMS 后的 Results 中没有对应记录；
        因此在分割和姿态任务中无法携带掩码或关键点数据，还会导致下游索引错误。对于这些任务跳过恢复流程及其逐帧开销。
        """
        if predictor.args.task in {"detect", "obb"}:
            attach_raw_preds_hook(predictor)

    @classmethod
    def compute_frame_extras(cls, predictor):
        """返回每个批次中被严格 NMS 丢弃的 ``(xywh, conf, cls)`` 元组。"""
        return compute_dets_del(predictor)

    def _cost_matrix(self, tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
        """返回融合多种线索的代价矩阵（HMIoU + ReID + 置信度 + 角度），并由 IoU 支持关系进行门控。"""
        iou_sim, hmiou_dist = _hmiou_distance(tracks, dets)
        if self.encoder is not None:
            cos = _cosine_distance(tracks, dets)
            # 外观信息缺失时（NaN：新目标或被遮挡抑制的检测结果），该配对回退到纯运动代价，
            # 使嵌入既不会帮助匹配，也不会惩罚匹配。
            cost = np.where(np.isnan(cos), hmiou_dist, self.iou_weight * hmiou_dist + self.reid_weight * cos)
        else:
            cost = hmiou_dist
        cost += self.conf_weight * _confidence_distance(tracks, dets)
        if iou_sim.size > 0:
            supported = iou_sim > 0.10
            pairs = np.nonzero(supported)
            if len(pairs[0]):
                cost[pairs] += self.angle_weight * _angle_distance(tracks, dets, self.frame_id, pairs)
            cost[~supported] = 1.0
        return np.clip(cost, 0, 1)

    def _apply_gmc(self, img: np.ndarray, detections: list, pools: list[list[TTSTrack]]) -> None:
        """使用当前 GMC 仿射变换原地变换 `pools`。"""
        try:
            warp = self.gmc.apply(img, [det.xyxy for det in detections])
        except Exception as e:
            LOGGER.warning(f"GMC 失败，回退到单位变换：{e}")
            warp = np.eye(2, 3)
        for pool in pools:
            multi_gmc(pool, warp)

    def update(self, results, img: np.ndarray | None = None, dets_del=None, **kwargs) -> np.ndarray:
        """推进跟踪器一帧，并返回形状为 `(N, 8)` 的数组 `[x1, y1, x2, y2, id, score, cls, idx]`。"""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        scores = np.asarray(results.conf)  # 保持掩码为 numpy；numpy 会通过 __index__ 转换单元素 torch 布尔掩码
        boxes = parse_bboxes(results)
        high_mask = scores >= self.args.track_high_thresh
        low_mask = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        def _new_track(box, score, cls, feat=None):
            track = TTSTrack(box, score, cls, feat) if feat is not None else TTSTrack(box, score, cls)
            track.min_track_len = self.min_track_len
            return track

        high_boxes, high_scores, high_cls = boxes[high_mask], scores[high_mask], results.cls[high_mask]
        feats = kwargs.get("feats")
        use_native = getattr(self.args, "model", "auto") == "auto"
        encoder_input = None
        if self.encoder is not None and len(high_boxes) > 0:
            if use_native:
                encoder_input = feats[high_mask] if (feats is not None and len(feats)) else None
            elif img is not None:
                encoder_input = img

        if encoder_input is not None:
            features = self.encoder(encoder_input, high_boxes)
            dets_high = [_new_track(b, s, c, f) for b, s, c, f in zip(high_boxes, high_scores, high_cls, features)]
        else:
            dets_high = [_new_track(b, s, c) for b, s, c in zip(high_boxes, high_scores, high_cls)]
        dets_low = [_new_track(b, s, c) for b, s, c in zip(boxes[low_mask], scores[low_mask], results.cls[low_mask])]

        dets_recovered: list[TTSTrack] = []
        if dets_del is not None:
            del_xywh, del_conf, del_cls = dets_del
            mask = del_conf > self.args.track_high_thresh
            if mask.any():
                del_boxes = np.concatenate([del_xywh[mask], -np.ones((mask.sum(), 1))], axis=-1)
                dets_recovered = [_new_track(b, s, c) for b, s, c in zip(del_boxes, del_conf[mask], del_cls[mask])]

        # 按状态分流：第 1 帧的目标虽然可见（is_activated），但仍处于 New 状态，因此必须保持未确认
        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (tracked if track.state == TrackState.Tracked else unconfirmed).append(track)
        pool = joint_stracks(tracked, self.lost_stracks)

        if img is not None and self.gmc.method is not None:
            self._apply_gmc(img, dets_high, [pool, unconfirmed])
        TTSTrack.multi_predict(pool)
        TTSTrack.multi_predict(unconfirmed)

        # 主关联：将 pool 与（高分 + 低分 + 恢复）检测结果匹配，并为不同分组施加代价惩罚。
        all_dets = dets_high + dets_low + dets_recovered
        n_high, n_low = len(dets_high), len(dets_low)
        cost = self._cost_matrix(pool, all_dets)
        if cost.shape[1] > n_high:
            cost[:, n_high : n_high + n_low] += self.penalty_p
        if dets_recovered:
            cost[:, n_high + n_low :] += self.penalty_q
        cost = np.clip(cost, 0, 1)

        matches, unmatched_tracks, unmatched_dets = _iterative_associate(cost, self.match_thr, self.reduce_step)
        for track_idx, det_idx in matches:
            track, det = pool[track_idx], all_dets[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for track_idx in unmatched_tracks:
            track = pool[track_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # 第二次关联：将未确认目标与剩余的高置信度检测结果匹配。
        leftover = [all_dets[i] for i in unmatched_dets if i < n_high]
        if unconfirmed and leftover:
            uc_cost = self._cost_matrix(unconfirmed, leftover)
            uc_matches, uc_unmatched_tracks, uc_unmatched_dets = _iterative_associate(
                uc_cost, self.match_thr, self.reduce_step
            )
            for track_idx, det_idx in uc_matches:
                unconfirmed[track_idx].update(leftover[det_idx], self.frame_id)
                activated.append(unconfirmed[track_idx])
            for track_idx in uc_unmatched_tracks:
                unconfirmed[track_idx].mark_removed()
                removed.append(unconfirmed[track_idx])
            leftover = [leftover[i] for i in uc_unmatched_dets]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed.append(track)

        # 可选的宽松重新绑定：将仍处于 Lost 状态的目标与剩余检测结果匹配（lost_match_thr <= 0 时禁用）。
        if self.lost_match_thr > 0 and leftover:
            unmatched_lost = [t for t in pool if t.state == TrackState.Lost and t not in lost]
            if unmatched_lost:
                lost_cost = self._cost_matrix(unmatched_lost, leftover)
                lost_matches, _, lost_unmatched = _iterative_associate(lost_cost, self.lost_match_thr, self.reduce_step)
                for track_idx, det_idx in lost_matches:
                    unmatched_lost[track_idx].re_activate(leftover[det_idx], self.frame_id, new_id=False)
                    refind.append(unmatched_lost[track_idx])
                leftover = [leftover[i] for i in lost_unmatched]

        # TAI：从通过现有目标 NMS 的剩余检测结果中生成新目标。
        active = [track for track in self.tracked_stracks if track.state == TrackState.Tracked] + activated
        for det, ok in zip(leftover, _track_aware_nms(active, leftover, self.tai_thr, self.new_track_thresh)):
            if ok:
                det.activate(self.kalman_filter, self.frame_id)
                activated.append(det)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        merge_track_pools(self, activated, refind, lost, removed)
        return np.asarray(
            [track.result for track in self.tracked_stracks if track.is_activated and track.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def reset(self) -> None:
        """清除跟踪器的所有状态，包括 GMC 变换历史和全局 ID 计数器。"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        TTSTrack.reset_id()
        self.gmc.reset_params()
