# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # 验证集成已启用
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Ray 会话处于活动状态时，在周期结束向 Ray Tune 报告训练指标。

    从训练器对象获取指标，并将其与当前周期编号一起发送给 Ray Tune，以支持超参数优化。
    仅在活动的 Ray Tune 会话中执行。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含指标和周期信息的 Ultralytics 训练器对象。

    示例：
        >>> # 由 Ultralytics 训练循环自动调用
        >>> on_fit_epoch_end(trainer)

    参考：
        Ray Tune docs: https://docs.ray.io/en/latest/tune/index.html
    """
    if ray.train._internal.session.get_session():  # 检查 Ray Tune 会话是否处于活动状态
        metrics = trainer.metrics
        session.report({**metrics, "epoch": trainer.epoch + 1})


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
