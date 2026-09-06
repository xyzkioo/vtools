# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ultralytics.utils.metrics import bbox_ioa

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes


class FastSTrack(STrack):
    """具有遮挡感知状态的 FastTracker 单对象轨迹。

    此类在 `STrack` 基础上增加近期卡尔曼均值的有界环形缓冲区和每条轨迹的遮挡记录。当邻近对象突然遮挡目标时，
    历史缓冲区支持将卡尔曼状态回滚到遮挡前的帧。缓冲区使用固定大小的 `collections.deque`，因此无论轨迹持续多久，内存占用都保持有界。

    属性：
        mean_history (collections.deque): 近期 ``(mean, covariance)`` 快照的有界环形缓冲区，最新快照位于末尾，最多保存 ``history_len`` 条。
            同时回滚均值和协方差，可在遮挡间隔后恢复轨迹时保持卡尔曼状态内部一致。
        not_matched (int): 此轨迹连续未匹配到检测结果的帧数。
        is_occluded (bool): 轨迹被其他目标遮挡时为 True。
        occluded_len (int): 轨迹持续被遮挡的连续帧数。
        last_occluded_frame (int): 最近一次检测到遮挡时的帧编号；从未遮挡时为 -1。
        was_recently_occluded (bool): 在 ``occ_reappear_window`` 帧内保持的标志，供 `FASTTracker` 延长遮挡丢失轨迹的重新查找窗口。

    示例：
        >>> from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
        >>> t = FastSTrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls=0, history_len=8)
        >>> t.activate(KalmanFilterXYAH(), frame_id=1)
        >>> len(t.mean_history)
        1
    """

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, history_len: int = 16):
        """初始化 FastSTrack。

        参数：
            xywh (np.ndarray): ``(x, y, w, h, idx)`` 或 ``(x, y, w, h, angle, idx)`` 格式的边界框。
            score (float): [0, 1] 范围内的检测置信度。
            cls (Any): 检测结果的类别标签。
            history_len (int): 为遮挡回滚保留的历史卡尔曼均值向量最大数量。
        """
        super().__init__(xywh, score, cls)
        self.mean_history: deque = deque(maxlen=history_len)
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False

    def _push_history(self):
        """将当前卡尔曼状态的 `(mean, covariance)` 副本追加到有界缓冲区。"""
        if self.mean is not None:
            self.mean_history.append((self.mean.copy(), self.covariance.copy()))

    def activate(self, kalman_filter, frame_id: int):
        """激活轨迹，并初始化其均值历史。

        参数：
            kalman_filter (KalmanFilterXYAH): 共享的卡尔曼滤波器实例。
            frame_id (int): 创建轨迹时的帧编号。
        """
        super().activate(kalman_filter, frame_id)
        self._push_history()

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        """重新激活此前丢失的轨迹，并清除所有过期的遮挡记录。

        参数：
            new_track (FastSTrack): 用于恢复此轨迹的检测结果。
            frame_id (int): 当前帧编号。
            new_id (bool): 为 True 时分配新的跟踪 ID，而不是复用旧 ID。
        """
        super().re_activate(new_track, frame_id, new_id=new_id)
        self.is_occluded = False
        self.occluded_len = 0
        self.not_matched = 0
        self.was_recently_occluded = False
        self.last_occluded_frame = -1
        self._push_history()

    def update(self, new_track, frame_id: int):
        """使用新匹配的检测结果更新轨迹。

        参数：
            new_track (FastSTrack): 当前帧匹配到的检测结果。
            frame_id (int): 当前帧编号。
        """
        super().update(new_track, frame_id)
        self._push_history()


class FASTTracker(BYTETracker):
    """具有遮挡感知能力的 ByteTrack 风格多对象跟踪器。

    Adapted from the reference implementation in the FastTracker paper (arXiv:2508.14370). FastTracker extends
    `BYTETracker` with lightweight mechanisms that reduce ID switches through crowd occlusions without sacrificing
    throughput. Unmatched tracks whose area is strongly covered by an active neighbor are flagged as occluded and their
    Kalman state is rolled back to a pre-occlusion frame, with a one-shot bbox enlargement and dampened motion so they
    survive the occlusion. An occluded track is kept alive for an extra grace window before being marked lost, and once
    lost it stays re-findable for an extended window beyond the regular ``track_buffer``. New detections that strongly
    overlap an already-active track are suppressed at spawn time to prevent ghost IDs.

    All added work uses vectorized IoU / coverage matrices and only runs on unmatched tracks, so the per-frame overhead
    over `BYTETracker` stays on the order of a few hundred microseconds.

    属性：
        reset_velocity_offset_occ (int): 遮挡开始时恢复 Kalman 速度所回看的帧数。
        reset_pos_offset_occ (int): 遮挡开始时恢复 Kalman 位置所回看的帧数。
        enlarge_bbox_occ (float): 首次检测到遮挡时应用于边界框高度的一次性倍率。
        dampen_motion_occ (float): 遮挡期间应用于 Kalman 速度的 `[0, 1]` 范围倍率。
        active_occ_to_lost_thresh (int): 轨迹被标记为丢失前允许连续遮挡的最大帧数。
        init_iou_suppress (float): IoU 阈值，超过该值时阻止新检测生成新轨迹。
            设置为 1.0 可禁用抑制。
        occ_cover_thresh (float): 判定遮挡时，一条轨迹区域必须被另一条活动轨迹覆盖的比例。
        occ_reappear_window (int): 最近被遮挡的丢失轨迹在常规 ``track_buffer`` 之外仍可重新发现的帧数。

    方法：
        update: 处理一帧的检测结果，并返回当前跟踪的对象。
        init_track: 从类似 ``Results`` 的对象构建 `FastSTrack` 实例。

    示例：
        Plug FastTracker into a YOLO 模型 via the bundled config:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26s.pt")
        >>> model.track("video.mp4", tracker="fasttrack.yaml")

        Drive FastTracker directly with your own detections:
        >>> from ultralytics.trackers import FASTTracker
        >>> from ultralytics.utils import YAML, IterableSimpleNamespace
        >>> cfg = IterableSimpleNamespace(**YAML.load("ultralytics/cfg/trackers/fasttrack.yaml"))
        >>> tracker = FASTTracker(cfg)
        >>> tracks = tracker.update(detections)
    """

    track_class = FastSTrack

    def __init__(self, args):
        """使用从 ``args`` 读取的可调参数初始化 FastTracker。

        ``args`` 中缺失的 FastTracker 专用键会回退到合理默认值，因此也可以使用普通 ByteTrack 配置驱动 FastTracker。

        参数：
            args (Namespace | IterableSimpleNamespace): 解析后的跟踪器配置。必须提供 BYTETracker 所需的键
                （``track_high_thresh``、``track_low_thresh``、``new_track_thresh``、``track_buffer``、``match_thresh``、
                ``fuse_score``），也可以提供类文档字符串中说明的 FastTracker 专用键。
        """
        super().__init__(args)
        # 遮挡处理参数（args 中不存在时使用合理的默认值）
        self.reset_velocity_offset_occ = int(getattr(args, "reset_velocity_offset_occ", 5))
        self.reset_pos_offset_occ = int(getattr(args, "reset_pos_offset_occ", 3))
        self.enlarge_bbox_occ = float(getattr(args, "enlarge_bbox_occ", 1.1))
        self.dampen_motion_occ = float(getattr(args, "dampen_motion_occ", 0.5))
        self.active_occ_to_lost_thresh = int(getattr(args, "active_occ_to_lost_thresh", 10))
        self.init_iou_suppress = float(getattr(args, "init_iou_suppress", 0.7))
        self.occ_cover_thresh = float(getattr(args, "occ_cover_thresh", 0.7))
        self.occ_reappear_window = int(getattr(args, "occ_reappear_window", 40))
        # 将历史长度限制为所需最大回滚长度加少量余量。
        self._history_len = max(self.reset_velocity_offset_occ, self.reset_pos_offset_occ) + 4

    def init_track(self, results, img: np.ndarray | None = None) -> list[FastSTrack]:
        """根据类似 ``Results`` 的对象构建 `FastSTrack` 实例。

        参数：
            结果 (Any): Object exposing ``xywh`` (or ``xywhr``), ``conf``, and ``cls``.
            img (np.ndarray | None): Current BGR frame. Unused by FastTracker; accepted for signature parity with other
                trackers.

        返回：
            (列表[FastSTrack]): One `FastSTrack` per detection, empty if no detections.
        """
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [
            FastSTrack(xywh, s, c, history_len=self._history_len)
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def _apply_match(self, track: STrack, det: STrack, activated: list[STrack], refind: list[STrack]) -> None:
        """更新或重新激活轨迹，并在匹配成功后清除所有遮挡记录。"""
        super()._apply_match(track, det, activated, refind)
        track.is_occluded = False
        track.not_matched = 0
        track.occluded_len = 0

    def _second_association(
        self,
        strack_pool: list[STrack],
        u_track: list[int],
        detections_second: list[STrack],
        activated: list[STrack],
        refind: list[STrack],
        lost: list[STrack],
    ) -> None:
        """执行第二阶段匹配和遮挡处理（替代基类的标记丢失循环）。"""
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked_stracks and detections_second:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked_stracks)))
        self._handle_occlusions(r_tracked_stracks, u_track, activated, lost)

    def _init_new_tracks(
        self,
        u_detection: list[int],
        detections: list[STrack],
        activated: list[STrack],
        refind: list[STrack] | None = None,
    ) -> None:
        """激活新轨迹，并抑制与当前活动轨迹高度重叠的检测结果。"""
        active_boxes = [t.xyxy for t in activated if t.is_activated]
        if refind:
            active_boxes.extend(t.xyxy for t in refind if t.is_activated)
        active_boxes.extend(t.xyxy for t in self.tracked_stracks if t.state == TrackState.Tracked)
        suppress_on = self.init_iou_suppress < 1.0
        active_stack = (
            np.asarray(active_boxes, dtype=np.float32) if active_boxes else np.empty((0, 4), dtype=np.float32)
        )
        for inew in u_detection:
            det = detections[inew]
            if det.score < self.args.new_track_thresh:
                continue
            if (
                suppress_on
                and len(active_stack)
                and bbox_ioa(det.xyxy[None, :], active_stack, iou=True).max() >= self.init_iou_suppress
            ):
                continue
            det.activate(self.kalman_filter, self.frame_id)
            activated.append(det)
            active_stack = np.concatenate([active_stack, det.xyxy[None, :]], axis=0)

    def _remove_stale_lost(self, removed: list[STrack]) -> None:
        """移除丢失轨迹，并为最近发生遮挡的轨迹保留宽限期。"""
        for track in self.lost_stracks:
            recently_occluded = track.was_recently_occluded and (
                self.frame_id - track.last_occluded_frame <= self.occ_reappear_window
            )
            if not recently_occluded and (self.frame_id - track.end_frame > self.max_frames_lost):
                track.mark_removed()
                removed.append(track)

    def _format_output(self) -> np.ndarray:
        """仅输出当前帧更新过的轨迹，避免返回过期的 ``idx`` 值。"""
        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def _handle_occlusions(self, r_tracked, u_track, activated_stracks, lost_stracks):
        """当活动邻近轨迹覆盖未匹配的跟踪轨迹时，将其标记为遮挡。

        对每条未匹配轨迹，计算其区域被任意当前活动轨迹覆盖的比例。如果覆盖比例超过 ``occ_cover_thresh``，
        则将轨迹标记为遮挡，并使用环形缓冲区历史回滚其 Kalman 状态（速度取 ``reset_velocity_offset_occ`` 帧前的值，
        位置取 ``reset_pos_offset_occ`` 帧前的值），将边界框高度按 ``enlarge_bbox_occ`` 放大一次，
        并按 ``dampen_motion_occ`` 衰减速度。未匹配超过两帧的轨迹会转为 Lost，但处于
        ``active_occ_to_lost_thresh`` 遮挡宽限窗口内时除外。

        参数：
            r_tracked (列表[FastSTrack]): Candidate track pool.
            u_track (列表[int] | np.ndarray): Indices into ``r_tracked`` of tracks still unmatched.
            activated_stracks (列表[FastSTrack]): Tracks already matched this frame; used as the pool of potential
                occluders.
            lost_stracks (列表[FastSTrack]): 输出列表；转为 Lost 状态的轨迹会追加到其中。
        """
        if len(u_track) == 0:
            return

            # 一次性构建活动轨迹边界框数组（向量化覆盖检查）。
        active = [t for t in activated_stracks if t.is_activated and not t.is_occluded]
        if len(active):
            active_boxes = np.asarray([t.xyxy for t in active], dtype=np.float32)
            active_ids = np.asarray([t.track_id for t in active])
        else:
            active_boxes = np.empty((0, 4), dtype=np.float32)
            active_ids = np.empty((0,), dtype=np.int64)

        unmatched = [r_tracked[i] for i in u_track]
        unmatched_boxes = (
            np.asarray([t.xyxy for t in unmatched], dtype=np.float32)
            if unmatched
            else np.empty((0, 4), dtype=np.float32)
        )

        if active_boxes.size and unmatched_boxes.size:
            cov = bbox_ioa(active_boxes, unmatched_boxes)  # (A, U) = intersection / unmatched-track area
            # 避免与自身匹配：将对应相同轨迹 ID 的条目置零。
            unm_ids = np.asarray([t.track_id for t in unmatched])
            same = active_ids[:, None] == unm_ids[None, :]
            cov[same] = 0.0
            max_cov = cov.max(axis=0)  # 对每条未匹配轨迹，计算其被活动轨迹覆盖的最大面积比例
        else:
            max_cov = np.zeros(len(unmatched), dtype=np.float32)

        for i, track in enumerate(unmatched):
            track.not_matched += 1

            if max_cov[i] > self.occ_cover_thresh and not track.is_occluded and track.state == TrackState.Tracked:
                track.is_occluded = True
                track.occluded_len = 1
                track.last_occluded_frame = self.frame_id
                track.was_recently_occluded = True

                hist = track.mean_history
                if track.mean is not None and hist:
                    if len(hist) >= self.reset_velocity_offset_occ:
                        prev_mean, _ = hist[-self.reset_velocity_offset_occ]
                        track.mean[4:8] = prev_mean[4:8]
                    if len(hist) >= self.reset_pos_offset_occ:
                        prev_mean, prev_cov = hist[-self.reset_pos_offset_occ]
                        track.mean[0:4] = prev_mean[0:4]
                        track.covariance = prev_cov.copy()
                    # 放大高度以扩展搜索区域（XYAH 状态会固定 a，因此按比例缩放 h 也会通过 w = a * h 缩放 w）。
                    track.mean[3] *= self.enlarge_bbox_occ
                    track.mean[4:8] *= self.dampen_motion_occ
            elif track.is_occluded:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > self.occ_reappear_window):
                track.was_recently_occluded = False

            # 在将遮挡轨迹标记为丢失前，为其保留一段宽限期。
            if (
                track.state != TrackState.Lost
                and track.not_matched > 2
                and (not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh)
            ):
                track.mark_lost()
                lost_stracks.append(track)
