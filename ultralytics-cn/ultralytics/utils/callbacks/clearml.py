# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # 不记录 pytest 日志
    assert SETTINGS["clearml"] is True  # 验证集成已启用
    import clearml
    from clearml import Task

    assert hasattr(clearml, "__version__")  # 确认导入的是有效软件包

except (ImportError, AssertionError):
    clearml = None


def _log_debug_samples(files, title: str = "Debug Samples") -> None:
    """将文件（图像）作为调试样本记录到 ClearML 任务中。

    参数：
        文件 (列表[Path]): A 列表 of 文件 路径 in PosixPath format.
        title (str): 将具有相同值的图像归为一组的标题。
    """
    import re

    if task := Task.current_task():
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                task.get_logger().report_image(
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )


def _log_plot(title: str, plot_path: str) -> None:
    """将图像作为绘图记录到 ClearML 的绘图区。

    参数：
        title (str): The title of the plot.
        plot_path (str | Path): 已保存图像文件的路径。
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
    ax.imshow(img)

    Task.current_task().get_logger().report_matplotlib_figure(
        title=title, series="", figure=fig, report_interactive=False
    )


def on_pretrain_routine_start(trainer) -> None:
    """在预训练流程开始时初始化并连接 ClearML 任务。"""
    try:
        if task := Task.current_task():
            # 警告：务必禁用 pytorch 和 matplotlib 的自动绑定！
            # 此集成会手动记录这些图表和模型文件
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib

            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            task = Task.init(
                project_name=str(trainer.args.project or "Ultralytics").lstrip("/") or "Ultralytics",
                task_name=trainer.args.name,
                tags=["Ultralytics"],
                output_uri=True,
                reuse_last_task_id=False,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
            )
            LOGGER.warning(
                "ClearML Initialized a new task. If you want to run remotely, "
                "please add clearml-init and connect your arguments before initializing YOLO."
            )
        task.connect(vars(trainer.args), name="General", ignore_remote_overrides=True)
    except Exception as e:
        LOGGER.warning(f"ClearML installed but not initialized correctly, not logging this run. {e}")


def on_train_epoch_end(trainer) -> None:
    """记录第一个周期的调试样本，并报告当前训练进度。"""
    if task := Task.current_task():
        # 仅记录第一个周期的调试样本
        if trainer.epoch == 1:
            _log_debug_samples(sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic")
        # 报告当前训练进度。
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)
        for k, v in trainer.lr.items():
            task.get_logger().report_scalar("lr", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer) -> None:
    """在周期结束时向日志记录器报告模型信息和指标。"""
    if task := Task.current_task():
        # 报告周期耗时和验证指标
        task.get_logger().report_scalar(
            title="Epoch Time", series="Epoch Time", value=trainer.epoch_time, iteration=trainer.epoch
        )
        for k, v in trainer.metrics.items():
            title = k.split("/")[0]
            task.get_logger().report_scalar(title, k, v, iteration=trainer.epoch)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)


def on_val_end(validator) -> None:
    """记录验证结果，包括标签和预测结果。"""
    if Task.current_task():
        # 记录验证标签和预测结果
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def on_train_end(trainer) -> None:
    """训练完成时记录最终模型和训练结果。"""
    if task := Task.current_task():
        # 记录最终结果、混淆矩阵和 PR 曲线
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        # 报告最终指标
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)
        # 记录最终模型
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
