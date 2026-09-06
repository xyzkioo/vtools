# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""用于在不同跟踪器之间操作跟踪对象列表的通用辅助函数。

这些函数保持通用性，只访问所有跟踪实现都会提供的属性（`track_id`、`frame_id`、`start_frame`、`xyxy`、
`mean` 和 `covariance`）。
"""

from __future__ import annotations

__all__ = ("joint_stracks", "merge_track_pools", "multi_gmc", "parse_bboxes", "remove_duplicate_stracks", "sub_stracks")

import numpy as np

from ..basetrack import TrackState
from . import matching


def merge_track_pools(
    tracker,
    activated: list,
    refind: list,
    lost: list,
    removed: list,
    removed_buffer: int = 1000,
) -> None:
    """在原地对跟踪器的持久化跟踪池执行标准的帧末整理。

    将新激活和重新找回的跟踪对象合并到 `tracker.tracked_stracks`，将状态发生转移的对象移动到
    `tracker.lost_stracks`，按 IoU 去重，将已移除对象追加到 `tracker.removed_stracks`，并将移除缓冲区裁剪为
    `removed_buffer` 个条目。

    参数：
        tracker (Any): 提供 `tracked_stracks`、`lost_stracks` 和 `removed_stracks` 列表的对象。
        activated (列表): 本帧从 Tracked 状态更新得到的跟踪对象。
        refind (列表): 本帧从 Lost 状态重新激活的跟踪对象。
        lost (列表): 本帧转为 Lost 状态的跟踪对象。
        removed (列表): 本帧转为 Removed 状态的跟踪对象。
        removed_buffer (int): 保留的历史移除跟踪对象的最大数量。

    示例：
        在跟踪器的 `update` 方法中执行帧末整理
        >>> merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
    """
    tracker.tracked_stracks = [t for t in tracker.tracked_stracks if t.state == TrackState.Tracked]
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, activated)
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, refind)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.tracked_stracks)
    tracker.lost_stracks.extend(lost)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.removed_stracks)
    tracker.tracked_stracks, tracker.lost_stracks = remove_duplicate_stracks(
        tracker.tracked_stracks, tracker.lost_stracks
    )
    tracker.removed_stracks_frame = removed
    tracker.removed_stracks.extend(removed)
    if len(tracker.removed_stracks) > removed_buffer:
        tracker.removed_stracks = tracker.removed_stracks[-removed_buffer:]


def parse_bboxes(results) -> np.ndarray:
    """从类似 Results 的对象中返回追加了原始索引的检测边界框。

    参数：
        results (Any): 提供 ``xywh``（或 ``xywhr``）、``conf`` 和 ``cls`` 属性的对象。

    返回：
        (np.ndarray): 对于 ``xywh``，返回形状为 ``(N, 5)`` 的数组；对于 ``xywhr``，返回形状为 ``(N, 6)`` 的数组。
            最后一列保存检测对象的原始索引。
    """
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    return np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)


def joint_stracks(atracks: list, btracks: list) -> list:
    """合并两个跟踪对象列表，并按 `track_id` 去重。

    参数：
        atracks (列表[STrack]): 第一个跟踪对象列表；发生 `track_id` 冲突时保留其中的对象。
        btracks (列表[STrack]): 第二个跟踪对象列表。

    返回：
        (列表[STrack]): 合并后的列表，其中重复的 `track_id` 已被移除。

    示例：
        将当前跟踪池与新激活的跟踪对象合并
        >>> merged = joint_stracks(tracked_stracks, activated_stracks)
    """
    a_ids = {t.track_id for t in atracks}
    return atracks + [t for t in btracks if t.track_id not in a_ids]


def sub_stracks(atracks: list, btracks: list) -> list:
    """过滤掉 `atracks` 中 `track_id` 出现在 `btracks` 里的跟踪对象。

    参数：
        atracks (列表[STrack]): 待过滤的源跟踪对象列表。
        btracks (列表[STrack]): 其 `track_id` 应从结果中排除的跟踪对象列表。

    返回：
        (列表[STrack]): `atracks` 中 `track_id` 不存在于 `btracks` 的对象。

    示例：
        从丢失池中移除重新跟踪到的对象
        >>> lost_stracks = sub_stracks(lost_stracks, tracked_stracks)
    """
    btrack_ids = {t.track_id for t in btracks}
    return [t for t in atracks if t.track_id not in btrack_ids]


def remove_duplicate_stracks(atracks: list, btracks: list, dup_thresh: float = 0.15) -> tuple[list, list]:
    """根据交并比（IoU）距离移除两个列表中的重复跟踪对象。

    当跟踪对象对的 IoU 距离小于 `dup_thresh`（即 IoU 大于 `1 - dup_thresh`）时，将其视为同一对象的重复跟踪。
    生命周期较短的跟踪对象（`frame_id - start_frame` 较小）会被丢弃；若时长相同，则从 `atracks` 中丢弃。

    参数：
        atracks (列表[STrack]): 第一个跟踪对象列表；对象必须提供 `xyxy`、`frame_id` 和 `start_frame` 属性。
        btracks (列表[STrack]): 第二个跟踪对象列表，属性要求与 `atracks` 相同。
        dup_thresh (float): 判定两个对象重复时允许的最大 IoU 距离，默认为 0.15（IoU > 0.85）。

    返回：
        resa (列表[STrack]): 移除重复对象后的 `atracks`。
        resb (列表[STrack]): 移除重复对象后的 `btracks`。

    示例：
        在帧末对活动池和丢失池去重
        >>> tracked, lost = remove_duplicate_stracks(tracked_stracks, lost_stracks)
    """
    pdist = matching.iou_distance(atracks, btracks)
    pairs = np.where(pdist < dup_thresh)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = atracks[p].frame_id - atracks[p].start_frame
        timeq = btracks[q].frame_id - btracks[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    dupa_set, dupb_set = set(dupa), set(dupb)
    resa = [t for i, t in enumerate(atracks) if i not in dupa_set]
    resb = [t for i, t in enumerate(btracks) if i not in dupb_set]
    return resa, resb


def multi_gmc(stracks: list, H: np.ndarray) -> None:
    """使用 2x3 仿射单应矩阵更新多个跟踪对象的位置和协方差。

    假设卡尔曼状态布局为 `(*box, *box_velocity)`，边界框中心 `(x, y)` 位于前两个维度。
    `R8x8` 以块对角形式旋转全部四个二维向量对，平移量 `t` 只应用于位置。
    该布局假设状态包含四组空间位置/速度向量（例如 XYWH）；XYAH 跟踪器必须重写此方法。

    参数：
        stracks (列表[STrack]): 要原地变换的跟踪对象；每个对象必须提供形状为 `(8,)` 的 `mean` 和形状为 `(8, 8)` 的
            `covariance`。
        H (np.ndarray): 将上一帧映射到当前帧的 2x3 仿射单应矩阵。

    示例：
        将相机运动补偿应用于活动跟踪池
        >>> warp = gmc.apply(frame, detection_boxes)
        >>> multi_gmc(tracked_stracks, warp)
    """
    if not stracks:
        return
    multi_mean = np.asarray([st.mean for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])

    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=np.float32), R)
    t = H[:2, 2]

    multi_mean = np.matmul(R8x8, multi_mean[..., None])[..., 0]
    multi_mean[:, :2] += t
    # 保持右操作数为 C 连续数组。F 连续数组会让 matmul 使用 BLAS 的转置 GEMM 内核；在 macOS Accelerate 上，
    # 即使输入值有限，该内核也可能留下浮点异常标志，导致 NumPy 报告虚假的除零、溢出或无效值警告，且速度更慢。
    multi_covariance = np.matmul(np.matmul(R8x8, multi_covariance), np.ascontiguousarray(R8x8.T))
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov
