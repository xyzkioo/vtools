# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # 不记录 pytest 日志
    assert SETTINGS["neptune"] is True  # 验证集成已启用

    import neptune
    from neptune.types import File

    assert hasattr(neptune, "__version__")

    run = None  # NeptuneAI experiment logger instance

except (ImportError, AssertionError):
    neptune = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    """将标量记录到 NeptuneAI 实验日志记录器。

    参数：
        scalars (dict): 要记录到 NeptuneAI 的标量值字典。
        step (int, 可选): 当前日志记录步骤或迭代次数。

    示例：
        >>> metrics = {"mAP": 0.85, "loss": 0.32}
        >>> _log_scalars(metrics, step=100)
    """
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


def _log_images(imgs_dict: dict, group: str = "") -> None:
    """将图像记录到 NeptuneAI 实验日志记录器。

    当有效的 Neptune 运行处于活动状态时，此函数将图像数据记录到 Neptune.ai，图像按指定组名称组织。

    参数：
        imgs_dict (dict): 要记录的图像字典，键为图像名称，值为图像数据。
        group (str, 可选): Group 名称 to organize 图像 under in the Neptune UI.

    示例：
        >>> # Log validation images
        >>> _log_images({"val_batch": img_tensor}, group="validation")
    """
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))


def _log_plot(title: str, plot_path: str) -> None:
    """将绘图记录到 NeptuneAI 实验日志记录器。"""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)


def on_pretrain_routine_start(trainer) -> None:
    """在训练开始前初始化 NeptuneAI 运行并记录超参数。"""
    try:
        global run
        run = neptune.init_run(
            project=trainer.args.project or "Ultralytics",
            name=trainer.args.name,
            tags=["Ultralytics"],
        )
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}
    except Exception as e:
        LOGGER.warning(f"NeptuneAI installed but not initialized correctly, not logging this run. {e}")


def on_train_epoch_end(trainer) -> None:
    """在每个训练周期结束时记录训练指标和学习率。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")


def on_fit_epoch_end(trainer) -> None:
    """在每个拟合周期结束时记录模型信息和验证指标。"""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_val_end(validator) -> None:
    """在验证结束时记录验证图像。"""
    if run:
        # 记录验证标签和验证预测结果
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")


def on_train_end(trainer) -> None:
    """在训练结束时记录最终结果、绘图和模型权重。"""
    if run:
        # 记录最终结果、混淆矩阵和 PR 曲线
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        # 记录最终模型
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if neptune
    else {}
)
