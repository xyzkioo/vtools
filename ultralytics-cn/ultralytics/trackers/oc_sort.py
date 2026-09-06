# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.ops import xyxy2ltwh
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes


class OCSortTrack(STrack):
    """使用以观测为中心的状态管理方式表示 OC-SORT 跟踪对象。

    在 STrack 基础上增加真实检测观测存储和速度计算，从而支持 OC-SORT 的三个组件：ORU、OCM 和 OCR。

    属性：
        last_observation (np.ndarray): xyxy 格式的最近一次真实检测结果。
        observations (dict): 将 frame_id 映射到 xyxy 观测数组。
        velocity (np.ndarray | None): 以观测为中心的速度方向向量 (dx, dy)。
        delta_t (int): 用于计算速度的时间窗口。
    """

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3):
        """初始化 OCSortTrack，并创建观测存储。

        参数：
            xywh (np.ndarray): `(x, y, w, h, idx)` 或 `(x, y, w, h, angle, idx)` 格式的边界框。
            score (float): [0, 1] 范围内的检测置信度。
            cls (Any): 检测结果的类别标签。
            delta_t (int): 用于计算速度方向的时间窗口（帧数）。
        """
        super().__init__(xywh, score, cls)
        self.last_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None
        self.delta_t = delta_t
        self._saved_mean: np.ndarray | None = None
        self._saved_covariance: np.ndarray | None = None

    def activate(self, kalman_filter, frame_id: int) -> None:
        """激活新的轨迹，并初始化其观测历史。

        参数：
            kalman_filter (KalmanFilterXYAH): 共享的卡尔曼滤波器实例。
            frame_id (int): 创建轨迹时的帧编号。
        """
        super().activate(kalman_filter, frame_id)
        self._record_observation(self.xyxy.astype(np.float32), frame_id)  # 保持检测空间精度
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def update(self, new_track: STrack, frame_id: int) -> None:
        """使用匹配的检测结果更新轨迹，并记录此次观测。

        参数：
            new_track (STrack): 当前帧匹配到的检测结果。
            frame_id (int): 当前帧编号。
        """
        self._record_observation(new_track.xyxy.copy(), frame_id)
        super().update(new_track, frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """使用新的检测结果重新激活此前丢失的轨迹。

        参数：
            new_track (STrack): 用于恢复此轨迹的检测结果。
            frame_id (int): 当前帧编号。
            new_id (bool): 为 True 时分配新的跟踪 ID，而不是复用旧 ID。
        """
        self._record_observation(new_track.xyxy.copy(), frame_id)
        super().re_activate(new_track, frame_id, new_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    @staticmethod
    def _xyxy_center(xyxy: np.ndarray) -> np.ndarray:
        """返回 xyxy 格式边界框的 `(cx, cy)` 中心坐标。"""
        return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

    def _record_observation(self, obs: np.ndarray, frame_id: int) -> None:
        """保存 `frame_id` 对应的 `obs`，并丢弃超过 `delta_t + 2` 的历史以限制内存占用。

        保留的窗口始终覆盖 `_compute_velocity` 需要回溯的帧，因为 `(frame_id - delta_t, frame_id]` 内最多包含 `delta_t`
        个不同帧。
        """
        self.last_observation = obs
        self.observations[frame_id] = obs
        max_keep = self.delta_t + 2
        if len(self.observations) > max_keep:
            for frame in sorted(self.observations)[:-max_keep]:
                del self.observations[frame]

    def _compute_velocity(self) -> np.ndarray | None:
        """根据保存的观测计算以观测为中心的速度方向。

        返回：
            (np.ndarray | None): Normalized `(dx, dy)` direction vector, or None if there are fewer than two usable
                observations.
        """
        if len(self.observations) < 2:
            return None

        current_frame = max(self.observations.keys())
        current_center = self._xyxy_center(self.observations[current_frame])

        # 查找至少早于当前帧 delta_t 帧的最近观测结果
        prev_obs = None
        for frame in sorted(self.observations.keys(), reverse=True):
            if frame < current_frame - self.delta_t + 1:
                prev_obs = self.observations[frame]
                break

        # 回退：如果不存在 delta_t 帧之前的观测，则使用最早的观测
        if prev_obs is None:
            earliest_frame = min(self.observations.keys())
            if earliest_frame == current_frame:
                return None
            prev_obs = self.observations[earliest_frame]

        direction = current_center - self._xyxy_center(prev_obs)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def apply_oru(self, new_observation_xyxy: np.ndarray, current_frame_id: int) -> None:
        """通过在虚拟观测上重放预测和更新，修复遮挡间隔期间的卡尔曼状态。"""
        if self._saved_mean is None or not self.observations:
            return

        last_frame = max(self.observations.keys())
        gap = current_frame_id - last_frame
        if gap <= 1:
            return

        # 将卡尔曼状态恢复到最后一次观测位置
        self.mean = self._saved_mean.copy()
        self.covariance = self._saved_covariance.copy()

        last_obs = self.observations[last_frame]

        # 使用虚拟观测重放状态更新
        for t in range(1, gap):
            alpha = t / gap
            virtual_xyxy = (1 - alpha) * last_obs + alpha * new_observation_xyxy
            # 将 xyxy 转换为 tlwh，再转换为 xyah 作为卡尔曼测量值
            virtual_xyah = self.tlwh_to_xyah(xyxy2ltwh(virtual_xyxy))
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, virtual_xyah)

        # 最后执行一次预测，使状态到达当前帧
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)


class OCSORT(BYTETracker):
    """采用以观测为中心匹配策略的 OC-SORT 多对象跟踪器。

    此类在 BYTETracker 基础上实现三个关键组件：
    - 以观测为中心的更新（ORU）：在遮挡后修复卡尔曼状态。
    - 以观测为中心的动量（OCM）：计算速度方向一致性代价。
    - 以观测为中心的恢复（OCR）：使用最后一次观测位置重新匹配。

    属性：
        delta_t (int): 计算速度方向时使用的时间窗口。
        inertia (float): 匹配中速度一致性代价的权重。
        use_byte (bool): 是否使用 ByteTrack 风格的低置信度第二阶段匹配。
    """

    track_class = OCSortTrack

    def __init__(self, args: Any):
        """初始化 OC-SORT 跟踪器。

        参数：
            args (Namespace | IterableSimpleNamespace): 解析后的跟踪器配置，除 BYTE 配置项外还提供 `delta_t`、
                `inertia` 和 `use_byte`。
        """
        super().__init__(args)
        self.delta_t = getattr(args, "delta_t", 3)
        self.inertia = getattr(args, "inertia", 0.2)
        self.use_byte = getattr(args, "use_byte", False)

    def init_track(self, results, img: np.ndarray | None = None) -> list[OCSortTrack]:
        """根据类似 `Results` 的对象构建 `OCSortTrack` 实例。"""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [OCSortTrack(xywh, s, c, self.delta_t) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def _fuse_appearance(
        self,
        dists: np.ndarray,
        tracks: list[OCSortTrack],
        detections: list[OCSortTrack],
        iou_dists: np.ndarray | None = None,
    ) -> np.ndarray:
        """组合运动代价和外观代价的钩子。默认行为：直接透传（不使用 ReID）。"""
        return dists

    def get_dists(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        """代价矩阵 = IoU（可融合分数）+ 惯性·OCM（可通过钩子加入外观代价）。"""
        iou_dists = matching.iou_distance(tracks, detections)
        dists = matching.fuse_score(iou_dists, detections) if self.args.fuse_score else iou_dists.copy()
        dists = dists + self.inertia * self._velocity_direction_cost(tracks, detections)
        return self._fuse_appearance(dists, tracks, detections, iou_dists=iou_dists)

    def _ocr_associate(
        self,
        tracks: list[OCSortTrack],
        dets: list[OCSortTrack],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
    ) -> tuple[list[int], list[int]]:
        """执行一次 OCR（最后观测 IoU）匹配，并原地应用匹配结果。

        返回：
            (tuple[list[int], list[int]))：未匹配 ``tracks`` 和 ``dets`` 的局部索引。
        """
        if not tracks or not dets:
            return list(range(len(tracks))), list(range(len(dets)))
        iou_dists = self._ocr_distance(tracks, dets)
        ocr_dists = matching.fuse_score(iou_dists, dets) if self.args.fuse_score else iou_dists
        ocr_dists = self._fuse_appearance(ocr_dists, tracks, dets, iou_dists=iou_dists)
        matches, u_track, u_det = matching.linear_assignment(ocr_dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track, det = tracks[itracked], dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.apply_oru(det.xyxy, self.frame_id)
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        return list(u_track), list(u_det)

    def _post_first_association(
        self,
        strack_pool: list[OCSortTrack],
        detections: list[OCSortTrack],
        u_track: list[int],
        u_detection: list[int],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
    ) -> tuple[list[int], list[int]]:
        """在第一阶段匹配后执行以观测为中心的恢复（OCR）阶段。

        先对仍处于 Tracked 状态的未匹配轨迹执行 OCR，以保持活动轨迹的匹配优先级；
        再使用仍未匹配的检测结果处理 Lost 轨迹，避免最近丢失的轨迹抢占活动轨迹的匹配。
        """
        ocr_dets = [detections[i] for i in u_detection]
        if not ocr_dets:
            return u_track, u_detection

        tracked = [i for i in u_track if strack_pool[i].state == TrackState.Tracked]
        other = [i for i in u_track if strack_pool[i].state != TrackState.Tracked]

        u_t1, u_d1 = self._ocr_associate([strack_pool[i] for i in tracked], ocr_dets, activated, refind)
        remaining = [ocr_dets[j] for j in u_d1]
        u_t2, u_d2 = self._ocr_associate([strack_pool[i] for i in other], remaining, activated, refind)

        u_track = [tracked[i] for i in u_t1] + [other[i] for i in u_t2]
        u_detection = [u_detection[u_d1[j]] for j in u_d2]
        return u_track, u_detection

    def _second_association(
        self,
        strack_pool: list[OCSortTrack],
        u_track: list[int],
        detections_second: list[OCSortTrack],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
        lost: list[OCSortTrack],
    ) -> None:
        """仅在启用 ``use_byte`` 时执行 ByteTrack 风格的第二阶段匹配。"""
        if not self.use_byte:
            for i in u_track:
                track = strack_pool[i]
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost.append(track)
            return
        super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)

    def _velocity_direction_cost(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        """计算 OCM 速度方向一致性代价矩阵（向量化实现）。

        对每个轨迹与检测结果配对，计算轨迹历史运动方向与指向候选检测结果方向之间的角度差。

        参数：
            tracks (list[OCSortTrack]): 轨迹列表。
            detections (list[OCSortTrack]): 检测结果列表。

        返回：
            (np.ndarray): 形状为 (len(tracks), len(detections)) 的代价矩阵。
        """
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if cost.size == 0:
            return cost

        # 预先提取检测结果中心，形成 (M, 2) 数组
        det_centers = np.array([OCSortTrack._xyxy_center(det.xyxy) for det in detections], dtype=np.float32)

        for i, track in enumerate(tracks):
            if track.velocity is None or track.last_observation[0] < 0:
                continue
            track_center = OCSortTrack._xyxy_center(track.last_observation)
            directions = det_centers - track_center  # (M, 2)
            norms = np.linalg.norm(directions, axis=1)  # (M,)
            valid = norms > 1e-6
            if not valid.any():
                continue
            directions[valid] /= norms[valid, None]
            dots = np.clip(directions[valid] @ track.velocity, -1.0, 1.0)
            cost[i, valid] = np.arccos(dots) / np.pi

        return cost

    def _ocr_distance(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        """使用轨迹最后一次观测位置而非卡尔曼预测结果计算 IoU 距离。

        参数：
            tracks (list[OCSortTrack]): 带有 last_observation 属性的轨迹列表。
            detections (list[OCSortTrack]): 检测结果列表。

        返回：
            (np.ndarray): 基于最后观测（或 OBB 使用 xywha）计算的 IoU 代价矩阵。

        注意：
            `last_observation` 以 xyxy 格式保存。对于有向（OBB）轨迹，不保存有向的最后观测，
            因此该方法会回退到卡尔曼预测的 `xywha`，OCR 阶段也会退化为对预测边界框执行标准 IoU。
            标准（轴对齐）跟踪可以获得 OCR 的全部效果。
        """
        if tracks and tracks[0].angle is not None:
            atlbrs = [t.xywha for t in tracks]
            btlbrs = [d.xywha for d in detections]
        else:
            atlbrs = [t.last_observation if t.last_observation[0] >= 0 else t.xyxy for t in tracks]
            btlbrs = [d.xyxy for d in detections]
        return matching.iou_distance(atlbrs, btlbrs)
