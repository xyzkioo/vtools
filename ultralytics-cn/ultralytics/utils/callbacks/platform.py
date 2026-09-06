# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import time

from ultralytics.utils import (
    ENVIRONMENT,
    GIT,
    LOGGER,
    PLATFORM_API_URL,
    PLATFORM_URL,
    PYTHON_VERSION,
    SETTINGS,
    TESTS_RUNNING,
    Retry,
    colorstr,
)

PREFIX = colorstr("Platform: ")
_api_key = None
_executor = ThreadPoolExecutor(max_workers=10)


def slugify(text):
    """将文本转换为 URL 安全的 slug（例如 'My Project 1' -> 'my-project-1'）。"""
    if not text:
        return text
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9\s-]", "", str(text).lower()).replace(" ", "-")).strip("-")[:128]


def _interp_plot(plot, n=101):
    """将绘图曲线数据插值到 n 个点，以减少存储大小。"""
    import numpy as np

    if not plot.get("x") or not plot.get("y"):
        return plot  # No interpolation needed (e.g., confusion_matrix)

    x, y = np.array(plot["x"]), np.array(plot["y"])
    if len(x) <= n:
        return plot  # Already small enough

    # 新的 x 值（101 个点可得到整齐的 0.01 增量：0、0.01、0.02、...、1.0）
    x_new = np.linspace(x[0], x[-1], n)

    # 插值 y 值（同时处理一维和二维数组）
    if y.ndim == 1:
        y_new = np.interp(x_new, x, y)
    else:
        y_new = np.array([np.interp(x_new, x, yi) for yi in y])

    # 如果存在 ap，也进行插值（用于 PR 曲线）
    result = {**plot, "x": x_new.tolist(), "y": y_new.tolist()}
    if "ap" in plot:
        result["ap"] = plot["ap"]  # 保持 AP 值不变（每类别标量）

    return result


def _validation_payload(image_metrics, sample_limit=5_000, extremes_limit=100):
    """返回精确的 F1 极值和均匀排序的样本，用于相关性分析。"""
    ranked = sorted(image_metrics.items(), key=lambda item: (item[1]["f1"], item[0]))
    if len(ranked) > sample_limit:
        sample = [ranked[round(i * (len(ranked) - 1) / (sample_limit - 1))] for i in range(sample_limit)]
    else:
        sample = ranked

    def rows(items):
        return [[Path(name).stem.split("_", 1)[0], metric["tp"], metric["fp"], metric["fn"]] for name, metric in items]

    return {
        "population": len(ranked),
        "sampling": "f1_rank",
        "rows": rows(sample),
        "extremes": {"worst": rows(ranked[:extremes_limit]), "best": rows(reversed(ranked[-extremes_limit:]))},
    }


def _sanitize_json_value(value):
    """将负无穷、正无穷和 NaN 浮点数替换为 None，确保 requests JSON 编码成功。"""
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        return value if isfinite(value) else None  # 避免出现“超出范围的浮点值不符合 JSON 规范”警告
    return value


def _send(event, data, project, name, model_id=None, retry=2, timeout=30):
    """使用重试逻辑向 Platform 地址发送事件。"""
    if not _api_key:
        return None
    import requests  # scoped as slow import

    payload = {"event": event, "data": _sanitize_json_value(data)}
    if model_id:
        payload["modelId"] = model_id
    else:
        payload.update(project=project, name=name)

    def send_once():
        global _api_key
        r = requests.post(
            f"{PLATFORM_API_URL}/training/metrics",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout,
        )
        if 400 <= r.status_code < 500 and r.status_code not in {408, 429}:
            try:
                msg = r.json().get("error", r.reason)
            except Exception:
                msg = r.reason
            # 只有 401 与凭据相关；403/404 只影响单次运行，不应禁用整个进程。
            if r.status_code == 401:
                _api_key = None
            # 不得记录 console_output 失败：ConsoleLogger 会将警告作为下一批内容再次刷新回来，导致再次失败。
            # 401 是安全的，因为清除的键会让 _send 提前返回。
            if event != "console_output" or r.status_code == 401:
                LOGGER.warning(f"{PREFIX}{msg}")
            return None  # 不重试客户端错误（408 超时和 429 速率限制除外）
        r.raise_for_status()
        return r.json()

    # 与上面相同的循环，因此 console_output 发送在每个级别都保持静默，包括 Retry 的级别。
    # 每次尝试都记录警告。仍然必须重试：_flush_buffer 会在调用此函数前清空缓冲区。
    quiet = event == "console_output"
    try:
        return Retry(times=retry, delay=1, verbose=not quiet)(send_once)()
    except Exception as e:
        if not quiet:
            LOGGER.debug(f"{PREFIX}Failed to send {event}: {e}")
        return None


def _handle_control_response(trainer, ctx, response):
    """应用 centralized stop signals returned by Platform webhook responses.

    注意：
        ``ctx["cancelled"]`` is the durable cancellation signal. During startup, trainer setup later resets
        ``trainer.stop``, so early stop requests still rely on ``on_pretrain_routine_end()`` to reapply the flag after
        setup completes.
    """
    if response and response.get("cancelled"):
        ctx["cancelled"] = True
        trainer.stop = True
        LOGGER.info(f"{PREFIX}Training cancelled from Platform ⚠️")


def _upload_model(model_path, project, name, progress=False, retry=1, model_id=None, run_id=None):
    """将模型检查点发布到配置的 Platform 存储位置。"""
    from ultralytics.utils.uploads import safe_upload

    if not _api_key:
        return None
    model_path = Path(model_path)
    if not model_path.exists():
        LOGGER.warning(f"{PREFIX}Model file not found: {model_path}")
        return None
    model_size = model_path.stat().st_size
    if os.getenv("PLATFORM_API_URL"):
        return {"modelPath": str(model_path.resolve()), "modelSize": model_size}
    import requests  # scoped as slow import

    # 从 Platform 获取签名上传 URL（服务端会清理文件名以确保存储安全）
    @Retry(times=3, delay=2)
    def get_signed_url():
        payload = {"filename": model_path.name}
        if model_id:
            payload["modelId"] = model_id  # Direct lookup avoids slug mismatch from auto-increment
        else:
            payload.update(project=project, name=name)
        if run_id:
            payload["runId"] = run_id
        r = requests.post(
            f"{PLATFORM_API_URL}/models/upload",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    try:
        data = get_signed_url()
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to get upload URL: {e}")
        return None

    # 使用带重试逻辑的 safe_upload 上传到 GCS，并可选显示进度条
    if safe_upload(file=model_path, url=data["uploadUrl"], retry=retry, progress=progress):
        gcs_path = data.get("gcsPath")
        if gcs_path and run_id:
            saved = _send(
                "checkpoint_saved",
                {
                    "modelPath": gcs_path,
                    "runId": run_id,
                    "uploadPath": data.get("uploadPath"),
                },
                project,
                name,
                model_id,
                timeout=90,
            )
            return {"modelPath": gcs_path, "modelSize": model_size} if saved else None
        return {"modelPath": gcs_path, "modelSize": model_size}
    return None


def _get_environment_info():
    """使用现有 Ultralytics 工具收集完整的环境信息。"""
    import shutil

    import psutil
    import torch

    from ultralytics import __version__
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    # 获取内存和磁盘总量
    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage("/")

    env = {
        "ultralyticsVersion": __version__,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "environment": ENVIRONMENT,
        "pythonVersion": PYTHON_VERSION,
        "pythonExecutable": sys.executable,
        "cpuCount": os.cpu_count() or 0,
        "cpu": get_cpu_info(),
        "command": " ".join(sys.argv),
        "totalRamGb": round(memory.total / (1 << 30), 1),  # Total RAM in GB
        "totalDiskGb": round(disk_usage.total / (1 << 30), 1),  # Total disk in GB
    }

    # 使用缓存的 GIT 单例获取 Git 信息（不调用子进程）。
    try:
        if GIT.is_repo:
            if GIT.origin:
                env["gitRepository"] = GIT.origin
            if GIT.branch:
                env["gitBranch"] = GIT.branch
            if GIT.commit:
                env["gitCommit"] = GIT.commit[:12]  # Short hash
            if GIT.message:
                env["gitCommitMessage"] = GIT.message
    except Exception:
        pass

    # GPU 信息
    try:
        if torch.cuda.is_available():
            env["gpuCount"] = torch.cuda.device_count()
            env["gpuType"] = get_gpu_info(0) if torch.cuda.device_count() > 0 else None
    except Exception:
        pass

    return env


def _get_project_name(trainer):
    """从训练器参数获取转换为 slug 的项目和名称。"""
    raw = str(trainer.args.project)
    parts = raw.split("/", 1)
    project = f"{parts[0]}/{slugify(parts[1].replace('/', '-'))}" if len(parts) == 2 else slugify(raw)
    return project, slugify(str(trainer.args.name or "train"))


def on_pretrain_routine_start(trainer):
    """在训练开始时初始化 Platform 日志。"""
    global _api_key
    if TESTS_RUNNING or not trainer.args.project:
        return
    _api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not _api_key:
        return

    project, name = _get_project_name(trainer)
    LOGGER.info(f"{PREFIX}Streaming training metrics to Platform")

    from ultralytics.utils.logger import ConsoleLogger

    # 用单个字典保存所有平台回调状态。
    ctx = {
        "model_id": None,
        "run_id": None,
        "last_upload": time(),
        "checkpoint_upload": None,
        "cancelled": False,
        "console_logger": None,
        "system_logger": None,
    }
    trainer.platform = ctx

    # 创建将控制台输出发送到 Platform 的回调。
    def send_console_output(content, line_count, chunk_id):
        """将批量控制台输出发送到 Platform webhook。"""
        _executor.submit(
            _send,
            "console_output",
            {"chunkId": chunk_id, "content": content, "lineCount": line_count},
            project,
            name,
            ctx["model_id"],
        )

    # 按批次捕获控制台输出（5 行或 5 秒）。在此处创建，但只有 Platform 接受下面的运行任务后才启动：
    # 如果在 training_started 失败前就开始捕获，用户的 stdout 会一直重定向到失效的集成，
    # 最终刷新还会发送一批控制台内容。
    # 不携带 model_id。
    ctx["console_logger"] = ConsoleLogger(batch_size=5, flush_interval=5.0, on_flush=send_console_output)

    # 收集环境信息（W&B 风格的元数据）。
    environment = _get_environment_info()

    # 构建 trainArgs：回调在 get_dataset() 之前运行，因此 args.data 仍为原始值（例如 ul:// URI）。
    # 注意：model_info 会在模型真正加载后的 on_fit_epoch_end（第 0 个 epoch）中发送。
    train_args = {k: str(v) for k, v in vars(trainer.args).items()}

    # 同步发送以获取后续 Webhook 所需的 modelId（该步骤很关键，因此重试次数更多）。
    response = _send(
        "training_started",
        {
            "trainArgs": train_args,
            "epochs": trainer.epochs,
            "device": str(trainer.device),
            "environment": environment,
        },
        project,
        name,
        retry=4,
    )
    if response and response.get("modelId"):
        ctx["model_id"] = response["modelId"]
        ctx["run_id"] = response.get("runId")
        # 服务器返回实际 slug（由于自动递增可能与请求名称不同，例如“train”→“train-2”）。
        if response.get("modelSlug"):
            ctx["model_slug"] = response["modelSlug"]
            url = f"{PLATFORM_URL}/{project}/{ctx['model_slug']}"
            LOGGER.info(f"{PREFIX}View model at {url}")
        ctx["console_logger"].start_capture()  # 此时运行已被跟踪且 model_id 已知
        # 注意：trainer.stop 在 on_pretrain_routine_end 中设置（_setup_train 重置它之后）。
        _handle_control_response(trainer, ctx, response)
    else:
        LOGGER.warning(f"{PREFIX}Training will not be tracked on Platform")
        trainer.platform = None  # Disable further callbacks


def on_pretrain_routine_end(trainer):
    """应用 pre-start cancellation after _setup_train resets trainer.stop."""
    ctx = getattr(trainer, "platform", None)
    if ctx and ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Training cancelled from Platform before starting ✅")
        trainer.stop = True


def on_fit_epoch_end(trainer):
    """在周期结束时记录训练和系统指标。"""
    ctx = getattr(trainer, "platform", None)
    if not ctx:
        return

    project, name = _get_project_name(trainer)
    metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}

    if trainer.optimizer and trainer.optimizer.param_groups:
        metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]

    # 在第 0 个 epoch 提取模型信息（作为独立字段发送，而不是放入指标中）。
    model_info = None
    if trainer.epoch == 0:
        try:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            info = model_info_for_loggers(trainer)
            model_info = {
                "parameters": info.get("model/parameters", 0),
                "gflops": info.get("model/GFLOPs", 0),
                "speedMs": info.get("model/speed_PyTorch(ms)", 0),
            }
        except Exception:
            pass

    # 获取系统指标（在平台上下文中缓存 SystemLogger 以提高效率）。
    system = {}
    try:
        if not ctx["system_logger"]:
            from ultralytics.utils.logger import SystemLogger

            ctx["system_logger"] = SystemLogger(all_drives=True)
        system = ctx["system_logger"].get_metrics(rates=True)
    except Exception:
        pass

    payload = {
        "epoch": trainer.epoch,
        "metrics": metrics,
        "system": system,
        "fitness": trainer.fitness,
        "best_fitness": trainer.best_fitness,
    }
    if model_info:
        payload["modelInfo"] = model_info

    def _send_and_check_cancel():
        """发送 epoch_end，并检查响应中是否要求取消（在后台线程运行）。"""
        response = _send("epoch_end", payload, project, name, ctx["model_id"], retry=1)
        _handle_control_response(trainer, ctx, response)

    _executor.submit(_send_and_check_cancel)


def on_model_save(trainer):
    """上传模型检查点（速率限制为每 15 分钟一次）。"""
    ctx = getattr(trainer, "platform", None)
    if not ctx:
        return
    # 限制频率为每 15 分钟一次（900 秒）。
    if time() - ctx["last_upload"] < 900:
        return
    if ctx["checkpoint_upload"] and not ctx["checkpoint_upload"].done():
        return

    model_path = trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last
    if not model_path:
        return

    project, name = _get_project_name(trainer)
    ctx["checkpoint_upload"] = _executor.submit(
        _upload_model, model_path, project, name, model_id=ctx["model_id"], run_id=ctx["run_id"]
    )
    ctx["last_upload"] = time()


def on_train_end(trainer):
    """记录最终训练结果，并将最佳模型上传到 Platform。"""
    ctx = getattr(trainer, "platform", None)  # 仅由 on_pretrain_routine_start 设置，没有 API 密钥时为空
    if not ctx:
        return

    project, name = _get_project_name(trainer)

    if ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Uploading partial results for cancelled training")

    # 停止控制台捕获
    if ctx["console_logger"]:
        ctx["console_logger"].stop_capture()
        ctx["console_logger"] = None

    # 上传最佳模型（阻塞并显示进度条，以确保上传完成）。
    artifact = None
    if trainer.best and Path(trainer.best).exists():
        if ctx["checkpoint_upload"]:
            ctx["checkpoint_upload"].result()
        artifact = _upload_model(
            trainer.best,
            project,
            name,
            progress=True,
            retry=3,
            model_id=ctx["model_id"],
            run_id=ctx["run_id"],
        )
        if not artifact:
            LOGGER.warning(f"{PREFIX}Model will not be available for download on Platform (upload failed)")

    # 收集训练器和验证器生成的图，并按类型去重。
    plots_by_type = {}
    for info in getattr(trainer, "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type[info["data"]["type"]] = info["data"]
    for info in getattr(getattr(trainer, "validator", None), "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type.setdefault(info["data"]["type"], info["data"])  # Don't overwrite trainer plots
    plots = [_interp_plot(p) for p in plots_by_type.values()]  # 插值曲线以减少尺寸

    # 获取 类别 名称
    names = getattr(getattr(trainer, "validator", None), "names", None) or (trainer.data or {}).get("names")
    class_names = list(names.values()) if isinstance(names, dict) else list(names) if names else None

    # stopper.best_epoch 从 1 开始计数；减 1 后与从 0 开始计数的 `epoch` 字段对齐。
    best_epoch = max(0, getattr(getattr(trainer, "stopper", None), "best_epoch", trainer.epoch + 1) - 1)

    image_metrics = trainer.validator.metrics.box.image_metrics if trainer.args.task == "detect" else {}
    validation = _validation_payload(image_metrics)
    _send(
        "training_complete",
        {
            "results": {
                "metrics": {**trainer.metrics, "fitness": trainer.fitness},
                "bestEpoch": best_epoch,
                "bestFitness": trainer.best_fitness,
                **({"calibration": c} if (c := getattr(trainer, "depth_calibration", None)) else {}),
                **({"validation": validation} if validation["rows"] else {}),
                **(artifact or {}),
            },
            "classNames": class_names,
            "plots": plots,
            "runId": ctx["run_id"],
        },
        project,
        name,
        ctx["model_id"],
        retry=4,  # Critical, more retries
    )
    url = f"{PLATFORM_URL}/{project}/{ctx.get('model_slug', name)}"
    LOGGER.info(f"{PREFIX}View results at {url}")


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_model_save": on_model_save,
    "on_train_end": on_train_end,
}
