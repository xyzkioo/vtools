# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    assert not TESTS_RUNNING  # 不记录 pytest 测试
    assert SETTINGS["dvc"] is True  # 确认已启用集成
    import dvclive

    assert checks.check_version("dvclive", "2.11.0", verbose=True)

    import os
    import re

    # DVCLive 日志记录器实例
    live = None
    _processed_plots = {}

    # `on_fit_epoch_end` 会在最终验证时调用（可能需要修复）；目前通过此变量区分最佳模型的最终评估和最后一个周期的验证
    _training_epoch = False

except (ImportError, AssertionError, TypeError):
    dvclive = None


def _log_images(path: Path, prefix: str = "") -> None:
    """使用 DVCLive 记录指定路径中的图像，并支持可选前缀。

    此函数会将给定路径中的图像记录到 DVCLive，并按批次组织图像，以便在界面中使用滑块查看。
    函数会从图像文件名中提取批次信息，并据此重新组织路径。

    参数：
        path (Path): 要记录的图像文件路径。
        prefix (str, 可选): 记录时添加到图像名称前的可选前缀。

    示例：
        >>> from pathlib import Path
        >>> _log_images(Path("runs/train/exp/val_batch0_pred.jpg"), prefix="validation")
    """
    if live:
        name = path.name

        # 按批次分组图像，以便在界面中启用滑块
        if m := re.search(r"_batch(\d+)", name):
            ni = m[1]
            new_stem = re.sub(r"_batch(\d+)", "_batch", path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)

        live.log_image(os.path.join(prefix, name), path)


def _log_plots(plots: dict, prefix: str = "") -> None:
    """如果绘图图像尚未处理，则记录它们以展示训练进度。

    参数：
        plots (dict): 包含绘图信息和时间戳的字典。
        prefix (str, 可选): 添加到已记录图像路径中的可选前缀。
    """
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator) -> None:
    """使用 DVCLive 记录验证器中的混淆矩阵。

    此函数处理验证器对象中的混淆矩阵，并将矩阵转换为目标标签和预测标签列表后记录到 DVCLive。

    参数：
        validator (BaseValidator): 包含混淆矩阵和类别名称的验证器对象。该对象必须具有
            `confusion_matrix.matrix`、`confusion_matrix.task` 和 `names` 属性。
    """
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task in {"detect", "obb"}:
        names += ["background"]

    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)

    live.log_sklearn_plot("confusion_matrix", targets, preds, name="cf.json", normalized=True)


def on_pretrain_routine_start(trainer) -> None:
    """在预训练流程期间初始化用于训练元数据的 DVCLive 日志记录器。"""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")
    except Exception as e:
        LOGGER.warning(f"DVCLive installed but not initialized correctly, not logging this run. {e}")


def on_pretrain_routine_end(trainer) -> None:
    """在预训练流程结束时记录与训练过程相关的绘图。"""
    _log_plots(trainer.plots, "train")


def on_train_start(trainer) -> None:
    """如果 DVCLive 日志处于启用状态，则记录训练参数。"""
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer) -> None:
    """在每个训练周期开始时将全局变量 _training_epoch 设置为 True。"""
    global _training_epoch
    _training_epoch = True


def on_fit_epoch_end(trainer) -> None:
    """在每个 fit 周期结束时记录训练指标和模型信息，并推进到下一步。

    此函数会在每个 fit 训练周期结束时调用，记录各种指标，包括训练损失项、验证指标和学习率。
    在第一个周期，它还会记录模型信息。此外，它会记录训练和验证绘图，并推进 DVCLive 步数计数器。

    参数：
        trainer (BaseTrainer): 包含训练状态、指标和绘图的训练器对象。

    注意：
        此函数仅在 DVCLive 日志启用且当前处于训练周期时执行记录操作。
        全局变量 _training_epoch 用于跟踪当前周期是否为训练周期。
    """
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)

        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "train")
        _log_plots(trainer.validator.plots, "val")

        live.next_step()
        _training_epoch = False


def on_train_end(trainer) -> None:
    """在训练结束时记录最佳指标、绘图和混淆矩阵。

    如果 DVCLive 日志处于启用状态，此函数会在训练过程结束时调用，以记录最终指标、可视化结果和模型资源。
    它会保存最佳模型性能指标、训练绘图、验证绘图和混淆矩阵，供后续分析使用。

    参数：
        trainer (BaseTrainer): 包含训练状态、指标和验证结果的训练器对象。

    示例：
        >>> # 在自定义训练循环内部
        >>> from ultralytics.utils.callbacks.dvc import on_train_end
        >>> on_train_end(trainer)  # 记录最终指标和产物
    """
    if live:
        # 最后记录最佳指标；内部会使用最佳模型运行验证器。
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "val")
        _log_plots(trainer.validator.plots, "val")
        _log_confusion_matrix(trainer.validator)

        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type="model")

        live.end()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)
