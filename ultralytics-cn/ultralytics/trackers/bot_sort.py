# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import parse_bboxes


class BOTrack(STrack):
    """STrack 的扩展版本，为 YOLO 增加对象跟踪特征。

    此类扩展 STrack，增加目标跟踪所需的特征平滑、卡尔曼滤波预测和轨迹重新激活等功能。

    属性：
        shared_kalman (KalmanFilterXYWH): BOTrack 所有实例共享的卡尔曼滤波器。
        smooth_feat (np.ndarray): 平滑后的特征向量。
        curr_feat (np.ndarray): 当前特征向量。
        alpha (float): 特征指数移动平均的平滑因子。
        mean (np.ndarray): 卡尔曼滤波器的均值状态。
        covariance (np.ndarray): 卡尔曼滤波器的协方差矩阵。

    方法：
        update_features: 更新特征向量，并使用指数移动平均进行平滑。
        predict: 使用卡尔曼滤波器预测均值和协方差。
        re_activate: 使用更新后的特征重新激活轨迹，并可选地分配新 ID。
        update: 使用新的检测结果和帧 ID 更新轨迹。
        tlwh: 返回 tlwh 格式 `(左上角 x, 左上角 y, 宽度, 高度)` 的当前位置。
        multi_predict: 使用共享卡尔曼滤波器预测多个目标轨迹的均值和协方差。
        convert_coords: 将 tlwh 边界框坐标转换为 xywh 格式。
        tlwh_to_xywh: 将边界框从 tlwh 转换为 xywh 格式 `(中心 x, 中心 y, 宽度, 高度)`。

    示例：
        创建 BOTrack 实例并更新其特征
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.activate(KalmanFilterXYWH(), frame_id=1)
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None):
        """初始化 BOTrack 对象，设置特征平滑状态和卡尔曼滤波器。

        参数：
            xywh (np.ndarray): `(x, y, w, h, idx)` 或 `(x, y, w, h, angle, idx)` 格式的边界框，其中 (x, y) 为中心点，
                (w, h) 为宽度和高度，`idx` 为检测索引。
            score (float): 检测结果的置信度分数。
            cls (int): 检测目标的类别 ID。
            feat (np.ndarray, 可选): 与检测结果关联的特征向量。
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        """更新当前特征及其指数移动平均平滑特征。"""
        curr, smooth = smooth_feature(feat, self.smooth_feat, self.alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def predict(self) -> None:
        """使用卡尔曼滤波预测对象的未来状态，并更新其均值和协方差。"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        """使用更新后的特征重新激活轨迹，并可选择分配新的 ID。"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        """使用新的检测信息和当前帧 ID 更新轨迹。"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self) -> np.ndarray:
        """返回 `(左上角 x, 左上角 y, 宽度, 高度)` 格式的当前边界框位置。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        """使用共享的卡尔曼滤波器预测多个对象轨迹的均值和协方差。"""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """将 tlwh 边界框坐标转换为 xywh 格式。"""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """将边界框从 tlwh（左上角坐标、宽度、高度）转换为 xywh（中心坐标、宽度、高度）格式。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """BYTETracker 的扩展版本，用于 YOLO 目标跟踪，并支持 ReID 和 GMC 算法。

    属性：
        proximity_thresh (float): 轨迹与检测结果之间空间接近度（IoU）的阈值。
        appearance_thresh (float): 轨迹与检测结果之间外观相似度（ReID 嵌入）的阈值。
        encoder (Any): 处理 ReID 嵌入的对象；未启用 ReID 时为 None。
        gmc (GMC): 用于数据关联的 GMC 算法实例。
        args (Any): 包含跟踪参数的解析后命令行参数。

    方法：
        get_kalmanfilter: 返回用于目标跟踪的 KalmanFilterXYWH 实例。
        init_track: 使用检测结果初始化轨迹，并可选地使用图像提取 ReID 特征。
        get_dists: 使用 IoU 和可选的 ReID 计算轨迹与检测结果之间的距离。
        multi_predict: 使用共享卡尔曼滤波器预测多个目标轨迹的均值和协方差。
        reset: 将 BOTSORT 跟踪器重置为初始状态。

    示例：
        初始化 BOTSORT 并处理检测结果
        >>> bot_sort = BOTSORT(args)
        >>> bot_sort.init_track(results, img)
        >>> bot_sort.multi_predict(tracks)

    注意：
        此类用于 YOLO 目标检测模型；只有通过 args 启用后才支持 ReID。
    """

    def __init__(self, args: Any):
        """初始化 BOTSORT 对象，配置 ReID 模块和 GMC 算法。

        参数：
            args (Any): 包含跟踪参数的解析后命令行参数。
        """
        super().__init__(args)
        self.gmc = GMC(method=args.gmc_method)

        # ReID 模块
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = build_encoder(args.with_reid, args.model, getattr(args, "device", None))

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """返回用于跟踪过程中预测和更新目标状态的 KalmanFilterXYWH 实例。"""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        """使用检测边界框、分数、类别标签和可选的 ReID 特征初始化目标轨迹。"""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None and img is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """使用 IoU 和可选的 ReID 嵌入计算轨迹与检测结果之间的距离。"""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """使用共享的卡尔曼滤波器预测多个对象轨迹的均值和协方差。"""
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        """将 BOTSORT 跟踪器重置为初始状态，并清除所有跟踪对象和内部状态。"""
        super().reset()
        self.gmc.reset_params()
