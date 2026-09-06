# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.segment import SegmentationValidator


class FastSAMValidator(SegmentationValidator):
    """Ultralytics YOLO 框架中 FastSAM（Segment Anything Model）分割任务的自定义验证类。

    此类继承 SegmentationValidator，专门定制 FastSAM 的验证流程，将任务设置为 'segment'，
    使用 SegmentMetrics 进行评估，并禁用绘图功能。
    以避免验证期间出现错误。

    属性：
        dataloader (torch.utils.数据.DataLoader): 用于验证的数据加载器对象。
        save_dir (Path): 保存验证结果的目录。
        args (SimpleNamespace): 用于自定义验证流程的其他参数。
        _callbacks (dict): 验证期间调用的回调函数字典。
        metrics (SegmentMetrics): 用于评估的分割指标计算器。

    方法：
        __init__: 使用 FastSAM 自定义设置初始化 FastSAMValidator。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        """初始化 FastSAMValidator，将任务设置为 'segment'，并使用 SegmentMetrics 作为指标。

        参数：
            dataloader (torch.utils.数据.DataLoader, 可选): 用于验证的 DataLoader。
            save_dir (Path, 可选): 保存结果的目录。
            args (SimpleNamespace, 可选): 验证器配置。
            _callbacks (dict, 可选): 验证期间调用的回调函数字典。

        注意：
            此类禁用 ConfusionMatrix 和其他相关指标的绘图，以避免错误。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # 禁用 ConfusionMatrix 和其他绘图以避免错误
