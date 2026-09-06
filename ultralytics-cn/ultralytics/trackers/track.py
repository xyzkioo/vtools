# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from functools import partial
from pathlib import Path

import torch

from ultralytics.cfg import TASKS
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .deep_oc_sort import DeepOCSORT
from .fast_tracker import FASTTracker
from .oc_sort import OCSORT
from .track_tracker import TRACKTRACK

# 跟踪器类型到对应跟踪器类的映射
TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "tracktrack": TRACKTRACK,
    "fasttrack": FASTTracker,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """在预测过程中初始化对象跟踪器。

    参数：
        predictor (ultralytics.engine.predictor.BasePredictor): 要为其初始化跟踪器的预测器对象。
        persist (bool, 可选): 如果跟踪器已经存在，是否复用已有跟踪器。

    示例：
        为预测器对象初始化跟踪器
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    trackable = ("detect", "segment", "pose", "obb")  # 结果包含边界框的任务，按标准顺序排列
    if (task := predictor.args.task) in TASKS and task not in trackable:  # 对未知的第三方任务不做处理
        raise ValueError(f"❌ 任务 '{task}' 不支持 'mode=track'，有效任务为 {', '.join(trackable)}")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))
    cfg.device = predictor.device  # 在预测器所在设备上运行任意 ReID 编码器

    if cfg.tracker_type not in TRACKER_MAP:
        raise AssertionError(f"当前仅支持 {sorted(TRACKER_MAP)}，但得到的是 '{cfg.tracker_type}'")

    predictor._feats = None  # 重置 ReID 预钩子状态
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if hasattr(predictor, "_orig_postprocess"):  # 恢复之前 TRACKTRACK 运行留下的原始预测结果包装器
        predictor.postprocess = predictor._orig_postprocess
        del predictor._orig_postprocess
    if cfg.tracker_type in {"botsort", "tracktrack", "deepocsort"} and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo26n-cls.pt"
        else:
            # 注册钩子以提取 Detect 层的输入
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # 展开为新列表，避免在前向传播中被修改

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # 非流式模式复用单个跟踪器
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # 切换视频时用于重置跟踪器

    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if hasattr(tracker_cls, "setup_predictor"):
        tracker_cls.setup_predictor(predictor)


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """对检测到的边界框进行后处理，并更新对象跟踪结果。

    参数：
        predictor (对象): 包含预测结果的预测器对象。
        persist (bool, 可选): 如果跟踪器已经存在，是否保留并继续使用。

    示例：
        对预测结果进行后处理并更新跟踪信息
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"

    tracker_cls = type(predictor.trackers[0])
    dets_del_list = (
        tracker_cls.compute_frame_extras(predictor) if hasattr(tracker_cls, "compute_frame_extras") else None
    )

    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (src := result.obb if is_obb else result.boxes).cpu().numpy()
        kwargs = {"feats": getattr(result, "feats", None)}
        if dets_del_list is not None:
            kwargs["dets_del"] = dets_del_list[i]
        tracks = tracker.update(det, result.orig_img, **kwargs)
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1], device=src.data.device)}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """为模型注册或刷新预测期间进行对象跟踪所需的回调。

    之前的注册会原地替换，因此重复调用不会叠加回调，也不会保留过期的 `persist` 设置。

    参数：
        model (对象): 要注册跟踪回调的模型，必须提供事件映射 `callbacks`。
        persist (bool): 如果跟踪器已经存在，是否保留并继续使用。

    示例：
        为 YOLO 模型注册跟踪回调
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    for event, fn in (
        ("on_predict_start", on_predict_start),
        ("on_predict_postprocess_end", on_predict_postprocess_end),
    ):
        callbacks = model.callbacks[event]
        i = next((i for i, cb in enumerate(callbacks) if getattr(cb, "func", None) is fn), None)
        if i is None:
            model.add_callback(event, partial(fn, persist=persist))
        else:
            callbacks[i] = partial(fn, persist=persist)
