# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from ultralytics import SETTINGS, __version__
from ultralytics.cfg import MODES, TASKS
from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    GIT,
    IS_PIP_PACKAGE,
    ONLINE,
    PYTHON_VERSION,
    RANK,
    TESTS_RUNNING,
    TORCH_VERSION,
)
from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info, unwrap_model


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """发送一次性 JSON POST 请求。"""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # 紧凑 JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


def _arch(model):
    """返回模型所基于的架构（例如 'yolo11n-seg'）；如果无法确定则返回 None。

    配置会随检查点保存，因此微调模型可以报告其祖先架构，即使中间经过了多代模型派生。
    """
    desc = f"{getattr(model, 'description', '')}".split()  # 导出模型在此处保存架构名称，而不是 YAML 文件
    # 训练器将配置保存在模型自身，预测器则保存在下一层 backend 中；SAM 没有 .model 属性
    yaml = getattr(model, "yaml", None) or getattr(getattr(model, "model", None), "yaml", None) or {}
    stem = Path(yaml.get("yaml_file", "")).stem or (desc[1] if len(desc) > 1 else "")
    return stem.lower()[:100] or None  # 统一转小写，避免同一架构被拆成两项；100 是 GA4 的长度限制


class Events:
    """收集并按速率限制发送匿名使用分析数据。

    当设置中启用同步、当前进程 rank 为 -1 或 0、未运行测试、环境在线，且安装来源为 pip 或官方 Ultralytics GitHub
    仓库时，才会收集并发送事件。

    属性：
        url (str): 接收匿名事件的 Measurement Protocol 地址。
        events (列表[dict]): 等待发送的事件负载内存队列。
        rate_limit (float): 两次 POST 请求之间的最小间隔（秒）。
        t (float): 上次发送时间戳（自 Unix 纪元起的秒数）。
        metadata (dict): 描述运行时、安装来源和环境的静态元数据。
        enabled (bool): 指示是否启用分析数据收集。

    方法：
        __init__: 初始化事件队列、速率限制器和运行时元数据。
        __call__: 将事件加入队列，并在达到发送间隔后触发非阻塞发送。
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self) -> None:
        """使用队列、速率限制器和环境元数据初始化 Events 实例。"""
        self.events = []  # 待发送事件
        self.rate_limit = 30.0  # 速率限制（秒）
        self.t = 0.0  # 上次发送时间戳（秒）
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # 例如 3.13
            "torch": TORCH_VERSION,
            "CPU": get_cpu_info(),
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or GIT.origin == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg, device=None, run=None) -> None:
        """将事件加入队列，并在达到速率限制间隔后异步刷新队列。

        参数：
            cfg (IterableSimpleNamespace): 包含模式和任务信息的配置对象。
            device (torch.device | str, 可选): 设备类型（例如 'cpu'、'cuda'）。
            run (BasePredictor | BaseTrainer, 可选): 已完成的运行对象，用于读取对应模式的结果字段。
        """
        # 事件名称使用运行模式；否则任意模式都会生成任意事件名称，GA4 达到每个属性 500 个名称后会丢弃后续名称。
        if not self.enabled or cfg.mode not in MODES or cfg.task not in TASKS:
            return

        # 尝试加入新事件
        if len(self.events) < 25:  # 队列最多保存 25 个事件，以限制内存和流量
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": Path(str(cfg.model)).name[:100] if cfg.model else None,  # 仅保存文件名，不保存路径
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            elif cfg.mode == "train":
                # 与下方 predict 使用相同的保护和字段顺序，原因也相同
                try:
                    # 分组键：stem 统一 YAML 和目录，isinstance 避免字典表示中的路径混入
                    params["data"] = Path(cfg.data).stem[:100] if isinstance(cfg.data, (str, Path)) else None
                    params["imgsz"] = cfg.imgsz
                    # 本次会话的训练轮数，与 hours 保持一致；恢复训练时会还原绝对轮数
                    params["epochs_done"] = run.epoch + 1 - run.start_epoch
                    params["batch"] = run.batch_size  # 已解析，因为 autobatch 和 OOM 重试都会调整它
                    params["hours"] = round((time.time() - run.train_time_start) / 3600, 4)
                    params["n"] = len(run.train_loader.dataset)  # train split 尺寸, matching predict's n
                    if run.best_fitness is not None:  # 运行从未验证时为 None
                        # 按任务组合指标：detect 使用 mAP50-95，分割使用边界框+掩码，因此仅在同一任务内比较
                        params["fitness"] = round(float(run.best_fitness), 5)
                    # 两者均已解析：默认 'auto' 使用自身的优化器和 lr0，并忽略 cfg.lr0
                    params["optimizer"] = type(run.optimizer).__name__
                    # 取最小值，因为 MuSGD 会将每组拆成两组，并把微调 lr*3 的一半放在前面
                    params["lr0"] = min(g["initial_lr"] for g in run.optimizer.param_groups)
                    flags = {
                        "pretrained": bool(cfg.pretrained),
                        "cos_lr": cfg.cos_lr,
                        "amp": run.amp,  # 实际应用值：check_amp() 会在不支持的硬件上关闭请求的 True
                        "rect": cfg.rect,
                        "multi_scale": bool(cfg.multi_scale),
                        "freeze": bool(cfg.freeze),  # freeze=0 和 freeze=[] 都表示不冻结任何层
                        "dropout": cfg.dropout > 0,
                        "early_stop": run.epoch + 1 < run.epochs,  # .stop is also set on the last planned epoch
                        "resume": bool(cfg.resume),  # fitness 会延续，epochs 和 hours 不会延续
                    }
                    params["flags"] = ",".join(k for k, v in flags.items() if v) or None
                    params["arch"] = _arch(unwrap_model(run.model))  # DDP 和 EMA 都会包裹模型并隐藏 .yaml
                    params["ngpu"] = run.world_size if run.world_size > 1 else None  # 只有大于 1 的数量才有信息
                    if device.type == "cuda":  # makes hours comparable
                        params["GPU"] = get_gpu_info(device.index or 0).rsplit(", ", 1)[0]
                except Exception:
                    pass
            elif cfg.mode in {"predict", "track"}:  # track runs the predictor too, and is most of the video 推理
                # 在保护块内读取，避免任何异常影响用户运行；先读取成本最低的字段，顺序也是丢弃顺序
                try:
                    params["n"] = run.seen  # 由该文件的所有者设置预测器状态，因此不会抛出异常
                    params["pixels"] = run.pixels  # 平均推理面积，FLOPs 会随其变化；边长取平方根
                    for k, v in (run.speed or {}).items():  # 运行未处理图像时为空
                        params[f"{k}_ms"] = round(v, 3)
                    params["batch"] = min(getattr(run.dataset, "bs", 0), run.seen) or None
                    model = run.model
                    params["format"] = model.format
                    params["nc"] = len(getattr(model, "names", None) or ()) or None  # 决定检测头宽度和 NMS
                    # 记录影响推理时间的开关：compile 会替换 .模型，end2end 为三态值
                    flags = {
                        "compile": hasattr(model, "_orig_mod"),
                        "end2end": getattr(model, "end2end", False),
                        "augment": cfg.augment,
                    }
                    params["flags"] = ",".join(k for k, v in flags.items() if v) or None
                    params["arch"] = _arch(model)  # 读取模型 description 和模型 yaml
                    meta = getattr(model, "metadata", None) or {}
                    params["quantize"] = str(meta.get("args", {}).get("quantize") or cfg.quantize or 32)
                    if device.type == "cuda":  # CUDA 此时已初始化，因此无需额外开销
                        params["GPU"] = get_gpu_info(device.index or 0).rsplit(", ", 1)[0]
                    session = getattr(model, "session", None)  # ONNX Runtime provider, else OpenVINO device
                    ov = getattr(model, "ov_compiled_model", None)  # Arc GPU 运行不能被识别为 CPU
                    devices = session.get_providers() if session else ov.get_property("EXECUTION_DEVICES") if ov else []
                    params["provider"] = devices[0] if devices else None  # last: least reliable read
                except Exception:
                    pass
            # null 值最终会被丢弃，超过 25 个参数的事件会直接被拒绝，因此截断参数而不是丢弃整个事件
            params = dict([(k, v) for k, v in params.items() if v is not None][:25])
            self.events.append({"name": cfg.mode, "params": params})

        # 检查速率限制；未达到间隔时提前返回
        t = time.time()
        if (t - self.t) < self.rate_limit:
            return

        # 达到发送间隔：在线程后台发送队列快照
        payload_events = list(self.events)  # 复制快照，避免与队列重置产生竞态
        Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),  # SHA-256 anonymized
            daemon=True,
        ).start()

        # 重置队列和速率限制计时器
        self.events = []
        self.t = t


events = Events()


def on_train_end(trainer):
    """最终指标可用后记录匿名训练事件。"""
    events(trainer.args, trainer.device, trainer)


def on_val_start(validator):
    """记录匿名的独立验证事件。

    训练器的最终验证仍保留 mode=train，因此此处的判断可避免重复记录训练事件。
    """
    if validator.args.mode == "val":
        events(validator.args, validator.device)


def on_predict_end(predictor):
    """逐图像速度可用后记录匿名预测事件。"""
    events(predictor.args, predictor.device, predictor)


def on_export_start(exporter):
    """记录匿名导出事件。"""
    events(exporter.args, exporter.device)


callbacks = {
    "on_train_end": on_train_end,
    "on_val_start": on_val_start,
    "on_predict_end": on_predict_end,
    "on_export_start": on_export_start,
}
