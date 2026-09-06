# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes


class STrack(BaseTrack):
    """使用卡尔曼滤波进行状态估计的单对象跟踪表示。

    此类负责保存单条轨迹的全部信息，并基于 Kalman 滤波器执行状态更新和预测。

    属性：
        shared_kalman (KalmanFilterXYAH): 所有 STrack 实例共享的 Kalman 滤波器，用于预测。
        _tlwh (np.ndarray): 保存边界框左上角坐标、宽度和高度的私有属性。
        kalman_filter (KalmanFilterXYAH): 此对象轨迹使用的 Kalman 滤波器实例。
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        分数 (float): Confidence 分数 of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): 对象的类别标签。
        idx (int): 对象的索引或标识符。
        frame_id (int): Current frame ID.
        start_frame (int): 首次检测到对象的帧编号。
        angle (float | None): 有向边界框的可选角度信息。

    方法：
        predict: 使用 Kalman 滤波器预测对象的下一状态。
        multi_predict: 预测多条轨迹的下一状态。
        activate: 激活新的轨迹。
        re_activate: 重新激活之前丢失的轨迹。
        update: 更新已匹配轨迹的状态。
        convert_coords: 将边界框转换为 x-y-aspect-高度格式。
        tlwh_to_xyah: 将 tlwh 边界框转换为 xyah 格式。

    示例：
        初始化并激活一条新轨迹
        >>> track = STrack(xywh=np.array([100, 200, 50, 80, 0]), score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        """初始化新的 STrack 实例。

        参数：
            xywh (np.ndarray): `(x, y, w, h, idx)` 或 `(x, y, w, h, angle, idx)` 格式的边界框，其中 (x, y) 为中心点，
                (w, h) 为宽度和高度，`idx` 为检测索引。
            分数 (float): 检测结果的置信度分数。
            cls (Any): 检测对象的类别标签。
        """
        super().__init__()
        # xywh+idx 或 xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """使用卡尔曼滤波预测对象的下一状态（均值和协方差）。"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """使用卡尔曼滤波对提供的 STrack 实例列表执行多对象状态预测。"""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """使用提供的卡尔曼滤波器激活新的轨迹，并初始化其状态和协方差。"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """使用新的检测数据重新激活此前丢失的轨迹，并更新其状态和属性。"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        """更新匹配轨迹的状态。

        参数：
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        示例：
            Update the state of a track with new detection information
            >>> track = STrack(np.array([100, 200, 50, 80, 0]), score=0.9, cls=0)
            >>> track.activate(KalmanFilterXYAH(), 1)
            >>> new_track = STrack(np.array([105, 205, 55, 85, 0]), score=0.95, cls=0)
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """将边界框的左上角-宽度-高度格式转换为 x-y-宽高比-高度格式。"""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """根据当前状态估计获取左上角-宽度-高度格式的边界框。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """将边界框从 (左上角 x, 左上角 y, 宽度, 高度) 转换为 (最小 x, 最小 y, 最大 x, 最大 y) 格式。"""
        ret = self.tlwh  # 已经是新的数组，可以安全修改
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """将边界框从 tlwh 格式转换为中心 x、中心 y、宽高比、高度（xyah）格式。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """以 (中心 x, 中心 y, 宽度, 高度) 格式获取当前边界框位置。"""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """以 (中心 x, 中心 y, 宽度, 高度, angle) 格式获取位置；缺少 angle 时发出警告。"""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """以适当的边界框格式获取当前跟踪结果。"""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """返回 STrack 对象的字符串表示，包括起始帧、结束帧和跟踪 ID。"""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker：基于 YOLO 构建的对象检测与跟踪算法。

    此类封装检测对象轨迹的初始化、更新和管理功能，
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    通过滤波预测对象的新位置，并执行数据关联。

    属性：
        tracked_stracks (列表[STrack]): List of successfully activated tracks.
        lost_stracks (列表[STrack]): List of lost tracks.
        removed_stracks (列表[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_frames_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter 对象.

    方法：
        update: 使用新的检测结果更新对象跟踪器。
        get_kalmanfilter: 返回用于跟踪边界框的 Kalman 滤波器对象。
        init_track: 初始化 对象 tracking with detections.
        get_dists: 计算 the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: 清除所有轨迹并重置跟踪器。
        joint_stracks: Combine two 列表 of stracks.
        sub_stracks: 从第一个列表中过滤掉存在于第二个列表中的轨迹。
        remove_duplicate_stracks: Remove duplicate stracks 基于 IoU.

    示例：
        初始化 BYTETracker，并使用检测结果更新它。
        >>> tracker = BYTETracker(args)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    track_class = STrack

    def __init__(self, args):
        """初始化用于对象跟踪的 BYTETracker 实例。

        参数：
            args (Namespace): Command-line arguments containing tracking 参数.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_frames_lost = args.track_buffer
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """使用新的检测结果更新跟踪器，并返回当前跟踪对象列表。"""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        results_high, results_low, mask_high, mask_low = self._split_detections(results)
        detections = self.init_track(results_high, self._input_for(img, feats, mask_high))
        detections_second = self.init_track(results_low, self._input_for(img, feats, mask_low))
        for tracks, mask in ((detections, mask_high), (detections_second, mask_low)):
            for track, i in zip(tracks, np.flatnonzero(mask)):
                track.idx = i  # idx 必须位于完整检测集合空间；parse_bboxes 只能看到子集

        unconfirmed, tracked_stracks = self._split_tracked()
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        self._pre_first_associate(strack_pool, unconfirmed, img, results_high)

        u_track, u_detection = self._first_association(strack_pool, detections, activated_stracks, refind_stracks)
        u_track, u_detection = self._post_first_association(
            strack_pool, detections, u_track, u_detection, activated_stracks, refind_stracks
        )
        self._second_association(
            strack_pool, u_track, detections_second, activated_stracks, refind_stracks, lost_stracks
        )
        u_detection, detections = self._unconfirmed_association(
            unconfirmed, u_detection, detections, activated_stracks, removed_stracks
        )
        self._init_new_tracks(u_detection, detections, activated_stracks, refind_stracks)
        self._remove_stale_lost(removed_stracks)

        merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
        return self._format_output()

    def _split_detections(self, results: Any) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        """将检测结果拆分为高置信度和低置信度子集，并丢弃退化边界框。

        参数：
            结果 (Any): Results-like 对象 with ``conf`` and ``xywh``/``xywhr`` attributes supporting boolean
                indexing.

        返回：
            (tuple[Any, Any, np.ndarray, np.ndarray]): High-置信度 结果, low-置信度 结果, high 掩码, and
                low 掩码.
        """
        scores = results.conf
        wh = (results.xywhr if hasattr(results, "xywhr") else results.xywh)[:, 2:4]
        valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)  # tlwh_to_xyah 要除以高度，因此 h=0 会使卡尔曼均值变为无穷大
        remain_inds = valid & (scores >= self.args.track_high_thresh)
        inds_low = valid & (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)
        return results[remain_inds], results[inds_low], remain_inds, inds_low

    def _input_for(self, img: np.ndarray | None, feats: np.ndarray | None, mask: np.ndarray) -> Any:
        """返回 ``init_track`` 所需的每个检测结果辅助输入。

        当提供 ``feats`` 时，会根据检测掩码对其切片。使用原生（``model="auto"``）ReID 编码器的跟踪器在缺少特征时
        （例如用户提供检测结果）会获得 None，因此 ``init_track`` 会回退到无编码路径，而不是将 BGR 帧传入自动编码器。
        外部 ReID 模型始终接收图像帧。

        参数：
            img (np.ndarray | None): 当前 BGR 图像帧。
            feats (np.ndarray | None): 可选的逐检测结果特征。
            mask (np.ndarray): 用于切片 ``feats`` 的布尔掩码。

        返回：
            (Any): 传递给 ``init_track`` 的辅助数据（特征、图像或 None）。
        """
        if feats is not None and len(feats):
            return feats[mask]
        if getattr(self, "encoder", None) is not None and getattr(self.args, "model", "auto") == "auto":
            return None
        return img

    def _split_tracked(self) -> tuple[list[STrack], list[STrack]]:
        """将 ``self.tracked_stracks`` 分为已确认和未确认列表。

        返回：
            (tuple[列表[STrack], 列表[STrack]]): ``(unconfirmed, tracked)`` where ``unconfirmed`` holds tracks whose
                ``is_activated`` flag is False.
        """
        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)
        return unconfirmed, tracked

    def _pre_first_associate(
        self, strack_pool: list[STrack], unconfirmed: list[STrack], img: np.ndarray | None, results_high: Any
    ) -> None:
        """在卡尔曼预测后、第一阶段分配前调用的钩子。默认行为：如果可用则使用 GMC。"""
        if hasattr(self, "gmc") and self.gmc.method is not None and img is not None:
            try:
                warp = self.gmc.apply(img, results_high.xyxy)
            except Exception as e:
                LOGGER.warning(f"GMC failed, falling back to identity: {e}")
                warp = np.eye(2, 3)
            multi_gmc(strack_pool, warp)
            multi_gmc(unconfirmed, warp)

    def _first_association(
        self, strack_pool: list[STrack], detections: list[STrack], activated: list[STrack], refind: list[STrack]
    ) -> tuple[list[int], list[int]]:
        """在轨迹池和高分检测结果之间执行第一阶段匹配。

        返回：
            (tuple[列表[int], 列表[int]]): 未匹配轨迹索引和未匹配检测索引。
        """
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        self._apply_matches(matches, strack_pool, detections, activated, refind)
        return u_track, u_detection

    def _post_first_association(
        self,
        strack_pool: list[STrack],
        detections: list[STrack],
        u_track: list[int],
        u_detection: list[int],
        activated: list[STrack],
        refind: list[STrack],
    ) -> tuple[list[int], list[int]]:
        """在第一阶段匹配结束后、第二阶段匹配开始前执行的钩子。

        返回：
            (tuple[列表[int], 列表[int]]): 可能已修改的未匹配轨迹索引和检测索引。
        """
        return u_track, u_detection

    def _apply_matches(
        self,
        matches: list[list[int]] | np.ndarray,
        pool: list[STrack],
        detections: list[STrack],
        activated: list[STrack],
        refind: list[STrack],
    ) -> None:
        """应用匹配阶段返回的 (track, detection) 配对列表。"""
        for itracked, idet in matches:
            self._apply_match(pool[itracked], detections[idet], activated, refind)

    def _apply_match(self, track: STrack, det: STrack, activated: list[STrack], refind: list[STrack]) -> None:
        """使用匹配的检测结果更新或重新激活单条轨迹。"""
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)

    def _second_association(
        self,
        strack_pool: list[STrack],
        u_track: list[int],
        detections_second: list[STrack],
        activated: list[STrack],
        refind: list[STrack],
        lost: list[STrack],
    ) -> None:
        """在剩余跟踪轨迹和低分检测结果之间执行第二阶段匹配。"""
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked_stracks and detections_second:
            # 按设计仅使用 IoU（ByteTrack 论文第 3.2 节）：融合低分会使代价超过 0.5 阈值
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked_stracks)))

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

    def _unconfirmed_association(
        self,
        unconfirmed: list[STrack],
        u_detection: list[int],
        detections: list[STrack],
        activated: list[STrack],
        removed: list[STrack],
    ) -> tuple[list[int], list[STrack]]:
        """将未确认轨迹与剩余的高分检测结果进行匹配。

        返回：
            (tuple[列表[int], 列表[STrack]]): Unmatched detection 索引 after association, and the filtered detection
                列表 those 索引 refer to.
        """
        detections = [detections[i] for i in u_detection]
        if not unconfirmed:
            return list(range(len(detections))), detections
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)
        return u_detection, detections

    def _init_new_tracks(
        self,
        u_detection: list[int],
        detections: list[STrack],
        activated: list[STrack],
        refind: list[STrack] | None = None,
    ) -> None:
        """根据通过所有匹配阶段的检测结果激活新轨迹。"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

    def _remove_stale_lost(self, removed: list[STrack]) -> None:
        """移除丢失时间超过允许最大帧数的轨迹。"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed.append(track)

    def _format_output(self) -> np.ndarray:
        """将当前跟踪对象格式化为输出数组。"""
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """返回使用 KalmanFilterXYAH 跟踪边界框的卡尔曼滤波器对象。"""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """使用给定的检测结果、分数和类别标签初始化对象跟踪，并创建 STrack 实例。"""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [self.track_class(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """使用 IoU 计算轨迹与检测结果之间的距离，并可选择融合分数。"""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """使用卡尔曼滤波预测多条轨迹的下一状态。"""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """重置 STrack 实例的 ID 计数器，确保不同跟踪会话中的跟踪 ID 唯一。"""
        STrack.reset_id()

    def reset(self):
        """清除所有跟踪中、丢失和已移除的轨迹，并重新初始化卡尔曼滤波器，从而重置跟踪器。"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
