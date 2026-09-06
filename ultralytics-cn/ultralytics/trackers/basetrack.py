# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""定义 YOLO 对象跟踪所需的基类和数据结构。"""

from typing import Any


class TrackState:
    """表示被跟踪对象可能状态的枚举类。

    属性：
        New (int): 对象刚被检测到时的状态。
        Tracked (int): 对象在后续帧中被成功跟踪时的状态。
        Lost (int): 对象不再被跟踪时的状态。
        Removed (int): 对象从跟踪列表中移除时的状态。

    示例：
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        ...     print("对象刚被检测到。")
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """对象跟踪的基类，提供基础属性和方法。

    属性：
        _count (int): 用于生成唯一跟踪 ID 的类级计数器。
        track_id (int): 当前跟踪对象的唯一标识符。
        is_activated (bool): 指示当前跟踪是否处于激活状态。
        state (TrackState): 当前跟踪状态。
        score (float): 跟踪置信度分数。
        start_frame (int): 开始跟踪时的帧编号。
        frame_id (int): 当前跟踪处理的最近一帧编号。

    方法：
        end_frame: 返回对象最后被跟踪的帧编号。
        next_id: 递增并返回下一个全局跟踪 ID。
        activate: 激活跟踪对象的抽象方法。
        predict: 预测跟踪对象下一状态的抽象方法。
        update: 使用新数据更新跟踪对象的抽象方法。
        mark_lost: 将跟踪标记为丢失。
        mark_removed: 将跟踪标记为已移除。
        reset_id: 重置全局跟踪 ID 计数器。

    示例：
        初始化一个新的跟踪对象并将其标记为丢失：
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # 输出：2（TrackState.Lost）
    """

    _count = 0

    def __init__(self):
        """使用唯一 ID 和基础跟踪属性初始化新的跟踪对象。"""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        """返回对象最近一次被跟踪的帧编号。"""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """递增并返回对象跟踪使用的下一个唯一全局跟踪 ID。"""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """使用提供的参数激活跟踪对象，并初始化跟踪所需属性。"""
        raise NotImplementedError

    def predict(self) -> None:
        """根据当前状态和跟踪模型预测跟踪对象的下一状态。"""
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """使用新的观测和数据更新跟踪对象，并相应修改其状态和属性。"""
        raise NotImplementedError

    def mark_lost(self) -> None:
        """将状态更新为 TrackState.Lost，把跟踪对象标记为丢失。"""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """将状态设置为 TrackState.Removed，把跟踪对象标记为已移除。"""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        """将全局跟踪 ID 计数器重置为初始值。"""
        BaseTrack._count = 0
