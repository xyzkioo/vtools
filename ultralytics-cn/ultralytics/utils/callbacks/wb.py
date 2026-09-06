# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # 不记录 pytest 测试
    assert SETTINGS["wandb"] is True  # 确认已启用集成
    import wandb as wb

    assert hasattr(wb, "__version__")  # 确认导入的是有效软件包
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """创建并记录自定义指标可视化表。

    此函数制作自定义指标可视化，模拟默认 wandb 精确率-召回率曲线的行为，同时提供更强的定制能力，可用于监控模型在不同类别上的性能。

    参数：
        x (列表): x 轴数据，长度应为 N。
        y (列表): 对应的 y 轴数据，长度也应为 N。
        classes (列表): 用于标识每个数据点类别的标签，长度为 N。
        title (str, 可选): 绘图标题。
        x_title (str, 可选): x 轴标签。
        y_title (str, 可选): y 轴标签。

    返回：
        (wandb.Object): 适合记录并展示该指标可视化结果的 wandb 对象。
    """
    import polars as pl  # scope for faster 'import ultralytics'
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(cs.numeric().round(3))
    data = df.select(["class", "y", "x"]).rows()

    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """记录指标曲线可视化。

    此函数根据输入数据生成指标曲线并记录到 wandb。根据 'only_mean' 标志，曲线可以表示聚合数据（均值）或单个类别数据。

    参数：
        x (np.ndarray): x 轴数据，长度为 N。
        y (np.ndarray): 对应的 y 轴数据，形状为 (C, N)，其中 C 为类别数量。
        names (列表, 可选): 与 y 轴数据对应的类别名称，长度为 C。
        id (str, 可选): 在 wandb 中记录数据时使用的唯一标识符。
        title (str, 可选): 可视化图表标题。
        x_title (str, 可选): x 轴标签。
        y_title (str, 可选): y 轴标签。
        num_x (int, 可选): 用于可视化的插值数据点数量。
        only_mean (bool, 可选): 是否只绘制均值曲线。

    注意：
        此函数使用 '_custom_table' 函数生成实际可视化结果。
    """
    import numpy as np

    # 创建新的 x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # 创建用于日志记录的数组。
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # 添加新的 x 值
            y_log.extend(np.interp(x_new, x, yi))  # 将 y 插值到新的 x 值
            classes.extend([names[i]] * len(x_new))  # 添加类别名称
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """如果绘图尚未记录，则在指定步骤将其记录到 WandB。

    此函数将输入字典中的每个绘图与之前处理过的绘图进行比较，并在指定步骤将新增或更新的绘图记录到 WandB。

    参数：
        plots (dict): 要记录的图表字典，键为图表名称，值为包含图表元数据（包括时间戳）的字典。
        step (int): 在 WandB 运行中记录图表时对应的步骤或周期。

    注意：
        此函数使用绘图字典的浅拷贝，避免迭代期间被修改。绘图通过 stem 名称（不含扩展名的文件名）识别，每个绘图均作为 WandB Image 对象记录。
    """
    for name, params in plots.copy().items():  # 使用浅拷贝，避免迭代期间 plots 字典发生变化
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """如果 wandb 模块存在，则初始化并启动 wandb 项目。"""
    if not wb.run:
        from datetime import datetime
        from pathlib import Path

        name = str(trainer.args.name).replace("/", "-").replace(" ", "_")
        latest_run = Path(trainer.save_dir) / "wandb" / "latest-run"
        resuming = trainer.args.resume and latest_run.exists()
        wb.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
            name=name,
            config=vars(trainer.args),
            id=latest_run.resolve().name.split("-", 2)[2]
            if resuming
            else f"{name}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}",
            resume="allow" if resuming else None,
            dir=str(trainer.save_dir),
        )


def on_fit_epoch_end(trainer):
    """在周期结束时记录训练指标和模型信息。"""
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    wb.run.log(trainer.metrics, step=trainer.epoch + 1, commit=True)  # commit forces sync


def on_train_epoch_end(trainer):
    """在每个训练周期结束时记录指标并保存图像。"""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """将最佳模型保存为工件，并在训练结束时记录最终绘图。"""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # 检查是否确实存在要保存的绘图
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()  # required or run continues on dashboard


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
