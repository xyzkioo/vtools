# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # 用于线性分配

    assert lap.__version__  # 确认已导入的 lap 是软件包而不是目录
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """使用 lap.lapjv 或内置 NumPy 求解器执行线性分配。

    参数：
        cost_matrix (np.ndarray): 包含分配代价的矩阵，形状为 (N, M)。
        thresh (float): 判断分配有效性的阈值。
        use_lap (bool): 是否使用 lap.lapjv 执行分配。为 False 时使用 ops.linear_sum_assignment。

    返回：
        matched_indices (列表[列表[int]] | np.ndarray): 匹配索引，形状为 (K, 2)，K 为匹配数量。
        unmatched_a (tuple | 列表 | np.ndarray): 第一组中未匹配的索引。
        unmatched_b (tuple | 列表 | np.ndarray): 第二组中未匹配的索引。

    示例：
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # 使用 lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        from ultralytics.utils.ops import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)  # 行索引 x，列索引 y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """根据跟踪对象之间的交并比（IoU）计算代价。

    参数：
        atracks (列表[STrack] | 列表[np.ndarray]): 跟踪对象 'a' 或边界框列表。
        btracks (列表[STrack] | 列表[np.ndarray]): 跟踪对象 'b' 或边界框列表。

    返回：
        (np.ndarray): 根据 IoU 计算的代价矩阵，形状为 (len(atracks), len(btracks))。

    示例：
        计算两组跟踪对象之间的 IoU 距离。
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # 代价矩阵


def embedding_distance(tracks: list, detections: list) -> np.ndarray:
    """根据嵌入计算跟踪对象与检测对象之间的余弦距离。

    参数：
        tracks (列表[BOTrack]): 跟踪对象列表，每个对象包含嵌入特征。
        detections (列表[BOTrack]): 检测对象列表，每个对象包含嵌入特征。

    返回：
        (np.ndarray): 根据嵌入计算的代价矩阵，形状为 (N, M)，N 为跟踪对象数量，M 为检测对象数量。

    示例：
        使用余弦度量计算跟踪对象与检测对象之间的嵌入距离。
        >>> tracks = [BOTrack(...), BOTrack(...)]  # 带嵌入特征的轨迹对象列表
        >>> detections = [BOTrack(...), BOTrack(...)]  # 带嵌入特征的检测对象列表
        >>> cost_matrix = embedding_distance(tracks, detections)
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    # 上游会跳过零范数嵌入，使 curr_feat/smooth_feat 为 None。堆叠零占位符以保持数组规则，
    # 然后将缺失特征的任意对象对强制设为最大距离，使调用方忽略外观并回退到运动或 IoU，而不是崩溃或部分匹配。
    track_feats = [t.smooth_feat for t in tracks]
    det_feats = [d.curr_feat for d in detections]
    feat_dim = next((len(f) for f in (*track_feats, *det_feats) if f is not None), 0)
    zeros = np.zeros(feat_dim, dtype=np.float32)
    track_features = np.asarray([f if f is not None else zeros for f in track_feats], dtype=np.float32)
    det_features = np.asarray([f if f is not None else zeros for f in det_feats], dtype=np.float32)
    track_norm = np.linalg.norm(track_features, axis=1, keepdims=True)
    det_norm = np.linalg.norm(det_features, axis=1, keepdims=True).T
    cost_matrix = 1 - track_features @ det_features.T / np.maximum(
        track_norm * det_norm, np.finfo(track_features.dtype).eps
    )
    cost_matrix = np.maximum(0.0, cost_matrix)  # 归一化特征
    missing_t = [i for i, f in enumerate(track_feats) if f is None]
    missing_d = [j for j, f in enumerate(det_feats) if f is None]
    if missing_t:
        cost_matrix[missing_t] = 2.0  # 最大余弦距离 -> 调用方除以 2 后，外观门控会忽略该对象对
    if missing_d:
        cost_matrix[:, missing_d] = 2.0
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """融合代价矩阵和检测分数，生成单一代价矩阵。

    参数：
        cost_matrix (np.ndarray): 包含分配代价的矩阵，形状为 (N, M)。
        detections (列表[BaseTrack]): 检测对象列表，每个对象包含分数属性。

    返回：
        (np.ndarray): 融合后的代价矩阵，形状为 (N, M)。

    示例：
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 条轨迹和 10 个检测结果
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # 融合后的代价
