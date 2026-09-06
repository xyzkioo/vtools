# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, env_bool, ops
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics

try:
    assert not TESTS_RUNNING  # 不记录 pytest 测试
    assert SETTINGS["comet"] is True  # 确认已启用集成
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # 确认导入的是有效软件包

    import os
    from pathlib import Path

    # 确保部分日志函数只在支持的任务上运行
    COMET_SUPPORTED_TASKS = ["detect", "segment"]

    # Ultralytics 创建并记录到 Comet 的绘图名称
    CONFUSION_MATRIX_PLOT_NAMES = "confusion_matrix", "confusion_matrix_normalized"
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve"
    LABEL_PLOT_NAMES = ["labels"]
    SEGMENT_METRICS_PLOT_PREFIX = "Box", "Mask"
    POSE_METRICS_PLOT_PREFIX = "Box", "Pose"
    DETECTION_METRICS_PLOT_PREFIX = ["Box"]
    RESULTS_TABLE_NAME = "results.csv"
    ARGS_YAML_NAME = "args.yaml"

    _comet_image_prediction_count = 0

except (ImportError, AssertionError):
    comet_ml = None


def _get_comet_mode() -> str:
    """从环境变量获取 Comet 模式，默认使用 `online`。"""
    comet_mode = os.getenv("COMET_MODE")
    if comet_mode is not None:
        LOGGER.warning(
            "The COMET_MODE environment variable is deprecated. "
            "Please use COMET_START_ONLINE to set the Comet experiment mode. "
            "To start an offline Comet experiment, use 'export COMET_START_ONLINE=0'. "
            "If COMET_START_ONLINE is not set or is set to '1', an online Comet experiment will be created."
        )
        return comet_mode

    return "online"


def _get_comet_model_name() -> str:
    """从环境变量获取 Comet 模型名称；未设置时默认为 `Ultralytics`。"""
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")


def _get_eval_batch_logging_interval() -> int:
    """从环境变量获取评估批次日志间隔；未设置时使用默认值 1。"""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", "1"))


def _get_max_image_predictions_to_log() -> int:
    """从环境变量获取要记录的图像预测结果最大数量。"""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", "100"))


def _scale_confidence_score(score: float) -> float:
    """按照环境变量指定的倍率缩放置信度分数。"""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", "100.0"))
    return score * scale


def _should_log_confusion_matrix() -> bool:
    """根据环境变量设置确定是否记录混淆矩阵。"""
    return env_bool("COMET_EVAL_LOG_CONFUSION_MATRIX", False)


def _should_log_image_predictions() -> bool:
    """根据环境变量确定是否记录图像预测结果。"""
    return env_bool("COMET_EVAL_LOG_IMAGE_PREDICTIONS", True)


def _resume_or_create_experiment(args: SimpleNamespace) -> None:
    """根据 args 恢复 CometML 实验，或创建新的实验。

    确保分布式训练期间只在一个进程中创建实验对象。

    参数：
        args (SimpleNamespace): 包含项目配置和其他参数的训练参数对象。
    """
    if RANK not in {-1, 0}:
        return

    # 如果用户未设置，则设置环境变量以配置 Comet 实验的在线模式。
    # 如果用户设置了 COMET_START_ONLINE，它将覆盖 COMET_MODE 的值。
    if os.getenv("COMET_START_ONLINE") is None:
        comet_mode = _get_comet_mode()
        os.environ["COMET_START_ONLINE"] = "1" if comet_mode != "offline" else "0"

    try:
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)
        experiment = comet_ml.start(project_name=_project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others(
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),
                "log_image_predictions": _should_log_image_predictions(),
                "max_image_predictions": _get_max_image_predictions_to_log(),
            }
        )
        experiment.log_other("Created from", "ultralytics")

    except Exception as e:
        LOGGER.warning(f"Comet installed but not initialized correctly, not logging this run. {e}")


def _fetch_trainer_metadata(trainer) -> dict:
    """返回 YOLO 训练元数据，包括周期和资源保存状态。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含训练状态和配置的 YOLO 训练器对象。

    返回：
        (dict): 包含当前周期、步数、是否保存资源以及是否为最后一个周期的字典。
    """
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return {"curr_epoch": curr_epoch, "curr_step": curr_step, "save_assets": save_assets, "final_epoch": final_epoch}


def _scale_bounding_box_to_original_image_shape(
    box, resized_image_shape, original_image_shape, ratio_pad
) -> list[float]:
    """将边界框从缩放图像坐标转换回原始图像坐标。

    YOLO 会在训练期间缩放图像，标签值以缩放后的图像形状为基准进行归一化。
    此函数会将边界框标签重新缩放到原始图像形状。

    参数：
        box (torch.Tensor): 归一化 xywh 格式的边界框。
        resized_image_shape (tuple): 缩放图像的形状（高度、宽度）。
        original_image_shape (tuple): 原始图像的形状（高度、宽度）。
        ratio_pad (tuple): 用于缩放的比例和填充信息。

    返回：
        (列表[float]): 经缩放并调整左上角坐标后的 xywh 格式边界框。
    """
    resized_image_height, resized_image_width = resized_image_shape

    # 将归一化 xywh 格式的预测结果转换为缩放尺寸下的 xyxy 格式
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # 将边界框预测结果从缩放图像尺寸还原到原始图像尺寸
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # 将边界框从 xyxy 格式转换为 xywh 格式，以便记录到 Comet
    box = ops.xyxy2xywh(box)
    # 调整 xy 中心点，使其对应左上角
    box[:2] -= box[2:] / 2
    box = box.tolist()

    return box


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None) -> dict | None:
    """整理目标检测任务的真实标注。

    此函数处理目标检测任务中一个图像批次的真实标注，提取指定图像的边界框、类别标签及其他元数据，
    并将其整理为可用于可视化或评估的格式。

    参数：
        img_idx (int): 要处理的图像在批次中的索引。
        image_path (str | Path): 图像文件路径。
        batch (dict): 包含检测数据的批次字典，键包括：
            - 'batch_idx'：批次索引张量
            - 'bboxes'：归一化 xywh 格式的边界框张量
            - 'cls'：类别标签张量
            - 'ori_shape'：原始图像形状
            - 'resized_shape'：缩放图像形状
            - 'ratio_pad'：比例和填充信息
        class_name_map (dict, 可选)：从类别索引映射到类别名称的字典。

    返回：
        (dict | None)：整理后的真实标注，包含 `name` 和 `data` 键；`data` 是标注字典列表，
            每个字典包含 `boxes`、`label` 和 `score` 键。如果图像中没有边界框，则返回 None。
    """
    indices = batch["batch_idx"] == img_idx
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes labels")
        return None

    cls_labels = batch["cls"][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    original_image_shape = batch["ori_shape"][img_idx]
    resized_image_shape = batch["resized_shape"][img_idx]
    ratio_pad = batch["ratio_pad"][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append(
            {
                "boxes": [box],
                "label": f"gt_{label}",
                "score": _scale_confidence_score(1.0),
            }
        )

    return {"name": "ground_truth", "data": data}


def _format_prediction_annotations(image_path, metadata, class_label_map=None, class_map=None) -> dict | None:
    """整理用于目标检测可视化的 YOLO 预测结果。

    参数：
        image_path (Path): 图像文件路径。
        metadata (dict): 包含边界框和类别信息的预测元数据。
        class_label_map (dict, 可选)：从类别索引映射到类别名称的字典。
        class_map (dict, 可选)：用于标签转换的其他类别映射。

    返回：
        (dict | None)：整理后的预测标注；不存在预测结果时返回 None。
    """
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes predictions")
        return None

    # 应用生成 JSON 时用于映射预测类别的映射表
    if class_label_map and class_map:
        class_label_map = {class_map[k]: v for k, v in class_label_map.items()}
    try:
        # 导入 faster_coco_eval 工具，以解压分割等任务的标注
        from faster_coco_eval.core.mask import decode
    except ImportError:
        decode = None

    data = []
    for prediction in predictions:
        boxes = prediction["bbox"]
        score = _scale_confidence_score(prediction["score"])
        cls_label = prediction["category_id"]
        if class_label_map:
            cls_label = str(class_label_map[cls_label])

        annotation_data = {"boxes": [boxes], "label": cls_label, "score": score}

        if decode is not None:
            # 只有能够解码时才处理分割数据
            segments = prediction.get("segmentation", None)
            if segments is not None:
                segments = _extract_segmentation_annotation(segments, decode)
            if segments is not None:
                annotation_data["points"] = segments

        data.append(annotation_data)

    return {"name": "prediction", "data": data}


def _extract_segmentation_annotation(segmentation_raw: str, decode: Callable) -> list[list[Any]] | None:
    """从压缩的分割数据中提取分割标注，并返回多边形列表。

    参数：
        segmentation_raw (str): 压缩格式的原始分割数据。
        decode (Callable)：用于解码压缩分割数据的函数。

    返回：
        (列表[列表[Any]] | None)：多边形点列表；提取失败时返回 None。
    """
    try:
        mask = decode(segmentation_raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        annotations = [np.array(polygon).squeeze() for polygon in contours if len(polygon) >= 3]
        return [annotation.ravel().tolist() for annotation in annotations]
    except Exception as e:
        LOGGER.warning(f"Comet Failed to extract segmentation annotation: {e}")
    return None


def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map, class_map) -> list | None:
    """如果真实标注和预测标注存在，则将二者合并。

    参数：
        img_idx (int): 图像在批次中的索引。
        image_path (Path): 图像文件路径。
        batch (dict): 包含真实标注的批次数据。
        prediction_metadata_map (dict): 按图像 ID 保存预测元数据的映射。
        class_label_map (dict): 从类别索引映射到类别名称的字典。
        class_map (dict): 用于标签转换的其他类别映射。

    返回：
        (列表 | None)：标注字典列表；不存在标注时返回 None。
    """
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map
    )
    prediction_annotations = _format_prediction_annotations(
        image_path, prediction_metadata_map, class_label_map, class_map
    )

    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None
    ]
    return [annotations] if annotations else None


def _create_prediction_metadata_map(model_predictions) -> dict:
    """按图像 ID 对模型预测结果分组，创建预测元数据映射。"""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)

    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch) -> None:
    """将混淆矩阵记录到 Comet 实验。"""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = [*list(trainer.data["names"].values()), "background"]
    experiment.log_confusion_matrix(
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )


def _log_images(experiment, image_paths, curr_step: int | None, annotations=None) -> None:
    """将图像及可选标注记录到实验中。

    此函数会将图像记录到 Comet ML 实验，并可选地附带用于可视化的边界框或分割掩码等标注数据。

    参数：
        experiment (comet_ml.CometExperiment): 要记录图像的 Comet ML 实验。
        image_paths (列表[Path]): 要记录的图像路径列表。
        curr_step (int | None): 用于实验时间线跟踪的当前训练步数或迭代次数。
        annotations (列表[列表[dict]], 可选)：每个图像对应的嵌套标注字典列表。每个标注包含边界框、标签和置信度等可视化数据。
    """
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step) -> None:
    """在模型验证期间将图像预测结果记录到 Comet ML 实验。

    此函数处理验证数据，并整理真实标注和预测标注，以便在 Comet 面板中可视化。
    函数会遵守配置的图像记录数量限制。

    参数：
        experiment (comet_ml.CometExperiment): 要记录数据的 Comet ML 实验。
        validator (BaseValidator): 包含验证数据和预测结果的验证器实例。
        curr_step (int): 用于记录时间线的当前训练步数。

    注意：
        此函数使用全局状态跟踪多次调用期间已记录的预测结果数量。
        它只记录 COMET_SUPPORTED_TASKS 中定义的支持任务的预测结果。
        记录图像的数量受 COMET_MAX_IMAGE_PREDICTIONS 环境变量限制。
    """
    global _comet_image_prediction_count

    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return

    jdict = validator.jdict
    if not jdict:
        return

    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names
    class_map = getattr(validator, "class_map", None)

    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue

        image_paths = batch["im_file"]
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= max_image_predictions:
                return

            image_path = Path(image_path)
            annotations = _fetch_annotations(
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
                class_map=class_map,
            )
            _log_images(
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            _comet_image_prediction_count += 1


def _log_plots(experiment, trainer) -> None:
    """将评估绘图和标签绘图记录到实验中。

    此函数会将各种评估绘图和混淆矩阵记录到实验跟踪系统。
    它处理不同类型的指标（SegmentMetrics、PoseMetrics、DetMetrics、OBBMetrics），并记录每种类型对应的绘图。

    参数：
        experiment (comet_ml.CometExperiment): 要记录绘图的 Comet ML 实验。
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含验证指标和保存目录信息的训练器对象。

    示例：
        >>> from ultralytics.utils.callbacks.comet import _log_plots
        >>> _log_plots(experiment, trainer)
    """
    plot_filenames = None
    if isinstance(trainer.validator.metrics, SegmentMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in SEGMENT_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, PoseMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in POSE_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, (DetMetrics, OBBMetrics)):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in DETECTION_METRICS_PLOT_PREFIX
        ]

    if plot_filenames is not None:
        _log_images(experiment, plot_filenames, None)

    confusion_matrix_filenames = [trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES]
    _log_images(experiment, confusion_matrix_filenames, None)

    if not isinstance(trainer.validator.metrics, ClassifyMetrics):
        label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]
        _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer) -> None:
    """将训练得到的最佳模型记录到 Comet.ml。"""
    model_name = _get_comet_model_name()
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)


def _log_image_batches(experiment, trainer, curr_step: int) -> None:
    """记录训练和验证图像批次的样本。"""
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)


def _log_asset(experiment, asset_path) -> None:
    """将指定资源文件记录到给定实验。

    此函数用于将文件等资源记录到指定实验，从而实现与实验跟踪平台的集成。

    参数：
        experiment (comet_ml.CometExperiment): 要记录资源的实验实例。
        asset_path (Path): 要记录的资源文件路径。
    """
    experiment.log_asset(asset_path)


def _log_table(experiment, table_path) -> None:
    """将表格记录到指定实验。

    此函数用于将表格文件记录到给定实验，表格由其文件路径标识。

    参数：
        experiment (comet_ml.CometExperiment): 要记录表格文件的实验对象。
        table_path (Path): 要记录的表格文件路径。
    """
    experiment.log_table(str(table_path))


def on_pretrain_routine_start(trainer) -> None:
    """在 YOLO 预训练流程开始时创建或恢复 CometML 实验。"""
    _resume_or_create_experiment(trainer.args)


def on_train_epoch_end(trainer) -> None:
    """在训练周期结束时记录指标并保存批次图像。"""
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]

    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)


def on_fit_epoch_end(trainer) -> None:
    """在每个训练周期结束时记录模型资源。

    此函数会在每个训练周期结束时，将指标、学习率和模型信息记录到 Comet ML 实验。
    它还会根据配置记录模型资源、混淆矩阵和图像预测结果。

    此函数获取当前 Comet ML 实验并记录各种训练指标。如果是第一个周期，还会记录模型信息。
    在指定的保存间隔内，它会记录模型、混淆矩阵（如果启用）以及图像预测结果（如果启用）。

    参数：
        trainer (BaseTrainer): 包含训练状态、指标和配置的 YOLO 训练器对象。

    示例：
        >>> # 在训练循环内部
        >>> on_fit_epoch_end(trainer)  # 将指标和资源记录到 Comet ML
    """
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    save_assets = metadata["save_assets"]

    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)

    if not save_assets:
        return

    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)


def on_train_end(trainer) -> None:
    """在训练结束时执行相关操作。"""
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    plots = trainer.args.plots

    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)

    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    _log_image_batches(experiment, trainer, curr_step)
    # 记录结果表格
    table_path = trainer.save_dir / RESULTS_TABLE_NAME
    if table_path.exists():
        _log_table(experiment, table_path)

    # 记录参数 YAML 文件
    args_path = trainer.save_dir / ARGS_YAML_NAME
    if args_path.exists():
        _log_asset(experiment, args_path)

    experiment.end()

    global _comet_image_prediction_count
    _comet_image_prediction_count = 0


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)
