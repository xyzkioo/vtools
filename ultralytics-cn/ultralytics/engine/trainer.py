# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""在数据集上训练模型。

用法：
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import _YOLO_CLI_COMMAND, get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    TORCH_2_0,
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    get_torch_device_backend,
    init_seeds,
    one_cycle,
    parse_device,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """用于创建训练器的基类。

    此类为训练 YOLO 模型提供基础功能，负责训练循环、验证、检查点保存以及各种训练工具，
    同时支持单 GPU 和多 GPU 分布式训练。

    属性：
        args (SimpleNamespace)：训练器配置。
        validator (BaseValidator)：验证器实例。
        model (nn.Module)：模型实例。
        callbacks (defaultdict)：回调函数字典。
        save_dir (Path)：结果保存目录。
        wdir (Path)：权重保存目录。
        last (Path)：最新检查点路径。
        best (Path)：最佳检查点路径。
        save_period (int)：每隔多少个 epoch 保存检查点（小于 1 时禁用）。
        batch_size (int)：训练批次大小。
        epochs (int)：训练轮数。
        start_epoch (int)：训练起始轮次。
        device (torch.device)：训练使用的设备。
        amp (bool)：是否启用 AMP（自动混合精度）。
        scaler (torch.amp.GradScaler)：AMP 使用的梯度缩放器。
        data (dict)：包含路径和元数据的数据集字典。
        ema (ModelEMA)：模型的 EMA（指数移动平均）副本。
        resume (bool)：是否从检查点恢复训练。
        lf (callable)：学习率调度函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler)：学习率调度器。
        best_fitness (float)：当前达到的最佳适应度值。
        fitness (float)：当前适应度值。
        loss (torch.Tensor)：当前损失值。
        tloss (dict)：损失项的运行平均值。
        loss_names (tuple)：损失项名称，由第一个批次中损失函数返回的损失字典推导得到。
        csv (Path)：结果 CSV 文件路径。
        metrics (dict)：指标字典。
        plots (dict)：图表字典。

    方法：
        train：执行训练过程。
        validate：在验证集上执行验证。
        save_model：保存模型训练检查点。
        get_dataset：获取训练集和验证集。
        setup_model：加载、创建或下载模型。
        build_optimizer：为模型构建优化器。

    示例：
        初始化训练器并开始训练。
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 BaseTrainer。

        参数：
            cfg (str | dict | SimpleNamespace，可选)：配置文件路径或配置对象。
            overrides (dict，可选)：配置覆盖项。
            _callbacks (dict，可选)：回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        if getattr(self.args, "augmentations", None) and not isinstance(self.args.augmentations[0], dict):
            import albumentations as A

            self.args.augmentations = [A.to_dict(t) for t in self.args.augmentations]  # 兼容 YAML、pickle 和 DDP
        self.check_resume(overrides)
        self.args.device = parse_device(self.args.device)  # 规范化设备字符串，并只执行一次 '-1' 自动选择
        self.device = select_device(self.args.device)
        self.accelerator = get_torch_device_backend(self.device) if self.device.type not in {"cpu", "mps"} else None
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # 目录
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # 更新日志记录器使用的名称
        self.wdir = self.save_dir / "weights"  # 权重目录
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点 路径
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # 防止用户在计时训练中误传 epochs=None
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # 设备
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # CPU 训练的耗时主要由推理决定，而不是数据加载

        # 回调：提前初始化，使 on_pretrain_routine_start 能获取原始参数
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # 启动进程中的设备数量；这与生成的 DDP 工作进程中设置的 utils.WORLD_SIZE 不同
        if self.device.type in {"cpu", "mps"}:
            world_size = 0
        else:  # 例如 device='0'、'0,1,2,3'、'npu:0'，或自动选择单个 GPU 的空字符串
            world_size = len(self.args.device.split(",")) if self.args.device else 1

        self.ddp = world_size > 1 and LOCAL_RANK == -1  # 除非已经处于 DDP，否则启动 DDP 工作进程
        self.world_size = world_size
        # 在 get_dataset() 前运行 on_pretrain_routine_start，以获取原始参数（例如 ul:// URI）
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # 模型和数据集
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # 避免多次自动下载数据集
            self.data = self.get_dataset()

        self.ema = None

        # 初始化优化工具
        self.lf = None
        self.scheduler = None

        # epoch 级别指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ()
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """将给定回调添加到指定事件的回调列表中。"""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """使用给定回调覆盖指定事件的现有回调。"""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """运行与指定事件关联的所有现有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """执行训练过程：多 GPU 时使用 DDP 子进程，单 GPU 时直接训练。"""
        # DDP 训练时运行子进程，否则正常训练
        try:
            if self.ddp:
                # 参数检查
                if self.args.rect:
                    LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                    self.args.rect = False
                if self.args.batch < 1.0:
                    raise ValueError(
                        "AutoBatch with batch<1 not supported for Multi-GPU training, "
                        f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                    )

                # 命令
                cmd, file = None, None
                try:
                    cmd, file = generate_ddp_command(self)
                    LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                finally:
                    if file is not None:
                        ddp_cleanup(self, str(file))

            else:
                self._do_train()
        finally:
            unset_deterministic()  # 绝不让确定性状态持续存在，包括 DDP 父进程和失败的运行
        if not self.ddp:
            self.run_callbacks("teardown")

    def _setup_scheduler(self):
        """初始化训练学习率调度器。"""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 余弦调度 1 -> hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线性调度
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _get_warmup_iterations(self, num_batches):
        """返回预热迭代次数，并至少为常规训练保留最后一个 epoch。"""
        warmup_epochs = min(self.args.warmup_epochs, max(self.epochs - 1, 0))
        return round(warmup_epochs * num_batches) if warmup_epochs > 0 else 0

    def _setup_ddp(self):
        """初始化并设置训练使用的 DistributedDataParallel 参数。"""
        device_type = self.args.device.split(":", 1)[0]
        device_type = device_type if device_type in {"npu", "xpu"} else "cuda"
        devices = self.args.device.split(":", 1)[-1].split(",")
        index = int(devices[LOCAL_RANK])  # world_size > 1 保证这里是多设备字符串
        self.device = torch.device(device_type, index)
        self.accelerator = get_torch_device_backend(self.device)
        self.accelerator.set_device(index)
        if device_type == "cuda":
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 设置为 1 以强制执行超时
        elif device_type == "xpu" and not (hasattr(dist, "is_xccl_available") and dist.is_xccl_available()):
            raise RuntimeError("Multi-XPU training requires XCCL, which is not available in this PyTorch build.")
        dist.init_process_group(
            backend={"npu": "hccl", "xpu": "xccl"}.get(device_type, "nccl" if dist.is_nccl_available() else "gloo"),
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _build_train_pipeline(self):
        """为当前批次大小构建数据加载器、优化器和调度器。"""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        final_batch_size = len(self.train_loader.sampler) % self.train_loader.batch_size or self.train_loader.batch_size
        if self.args.imgsz < 2 * self.stride and not self.train_loader.drop_last and final_batch_size == 1:
            raise ValueError(
                f"final batch=1 training at imgsz={self.args.imgsz} gives BatchNorm a single value per channel; "
                f"change batch or use imgsz >= {2 * self.stride}"
            )
        # 注意：训练 DOTA 数据集时，将批次大小加倍可能导致包含超过 2000 个目标的图像发生显存溢出。
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task in {"obb", "semantic", "depth"} else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 缩放权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def _setup_train(self):
        """在训练循环开始前配置模型、优化器、数据加载器和训练工具。"""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # channels_last（NHWC）仅支持 CUDA：在 CUDA 上不会损失精度且有利于 Tensor Core，
        # 但在 MPS 上数值不正确，在 CPU 上也没有收益。
        if self.args.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        elif self.args.channels_last:
            LOGGER.warning(f"'channels_last=True' is only supported on CUDA, ignoring on '{self.device.type}'.")
        self.set_model_attributes()

        # 编译模型（知识蒸馏会立即运行封装后的模型，并依赖 DDP 中的 find_unused_parameters 处理冻结教师模型，
        # 因此蒸馏时禁用编译）
        if self.args.distill_model is not None and self.args.compile:
            LOGGER.warning("'compile' is not supported with knowledge distillation and will be disabled.")
            self.args.compile = False
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # 冻结层
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # 始终冻结这些层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        if isinstance(unwrap_model(self.model), DistillationModel):
            freeze_layer_names.append("teacher_model.")
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # 将 NaN 转为 0（因训练结果不稳定而注释）
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # 只有浮点张量可以计算梯度
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

        # 检查 AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True 或 False
        if self.amp and RANK in {-1, 0}:  # 单 GPU 或 DDP 主进程
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调，因为 check_amp() 会重置它们
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # 恢复回调
        if RANK > -1 and self.world_size > 1:  # DDP 模式
            self.amp = self.amp.int()  # gloo 不支持布尔值
            dist.broadcast(self.amp, src=0)  # 从 rank 0 广播到所有其他 rank
        self.amp = bool(self.amp)  # 转换为布尔值
        if self.device.type == "npu":
            import torch_npu

            self.scaler = torch_npu.npu.amp.GradScaler(enabled=self.amp)
        else:
            self.scaler = (
                torch.amp.GradScaler(self.device.type if self.device.type == "xpu" else "cuda", enabled=self.amp)
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp)
            )
        # 检查图像尺寸
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格尺寸（最大步长）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # 用于多尺度训练

        # 恢复训练会直接加载 DistillationModel，因此在此处检查
        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)
        if self.world_size > 1:
            # static_graph=True 允许在一次前向传播中多次使用参数（例如 torch.compile 下姿态损失的一对多和一对一分支中的
            # flow_model）。
            ddp_kwargs = {"static_graph": bool(self.args.compile)} if TORCH_1_11 else {}
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                find_unused_parameters=not bool(self.args.compile),
                **ddp_kwargs,
            )

        # 批次大小
        if self.batch_size < 1 and RANK == -1:  # 仅单 GPU 时估计最佳批次大小
            self.args.batch = self.batch_size = self.auto_batch()
        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        self.set_class_weights()  # 数据加载器就绪后计算类别权重
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # 不要移动
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """执行完整训练循环，包括初始化、epoch 迭代、验证和最终评估。"""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # 批次数量
        nw = self._get_warmup_iterations(nb)
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清零可能恢复的梯度，确保训练开始时稳定
        self._oom_retries = 0  # 第一个 epoch 的 OOM 自动降低批次计数器
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略“在 optimizer.step() 之前调用 lr_scheduler.step()”警告
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                if self.loss_names:
                    LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # 预热
                ni = i + nb * epoch
                if ni < nw:
                    xi = [0, nw]  # 插值区间
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # 偏置学习率从 0.1 降至 lr0，其他学习率从 0.0 升至 lr0
                        x["lr"] = float(
                            np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                        )
                        if "momentum" in x:
                            x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))

                # 前向传播
                try:
                    with autocast(self.amp, device=self.device.type):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # 分离推理和损失计算，以提升编译性能
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        if not self.loss_names:  # 根据第一个批次中损失函数返回的损失字典推导损失名称
                            self.loss_names = tuple(self.loss_items)
                            if RANK in {-1, 0}:
                                LOGGER.info(self.progress_string())
                                self.metrics.update(dict.fromkeys(self.label_loss_items(prefix="val"), 0.0))
                        self.tloss = (
                            self.loss_items
                            if self.tloss is None
                            else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
                        )

                    # Backward
                    self.scaler.scale(self.loss).backward()
                except RuntimeError as e:
                    is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError 需要 torch>=1.13
                    if not is_oom and not any(
                        s in str(e)
                        for s in (
                            "CUBLAS_STATUS_ALLOC_FAILED",
                            "CUDNN_STATUS_INTERNAL_ERROR",
                            "unable to find an engine",
                        )
                    ):
                        raise
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # 仅在单 GPU 的第一个 epoch 自动降低批次大小，最多重试 3 次
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    error = f"{self.device.type.upper()} out of memory" if is_oom else "CUDA backend memory error"
                    LOGGER.warning(
                        f"{error} with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    batch = loss = preds = None
                    self.loss = self.loss_items = self.tloss = None
                    self._clear_memory()
                    self._build_train_pipeline()  # 重建数据加载器、优化器和调度器
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = self._get_warmup_iterations(nb)
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # 使用降低后的批次大小重新开始 epoch 循环
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # 定时停止
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # DDP 训练时
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # 将“停止”广播到所有 rank
                            self.stop = broadcast_list[0]
                        if self.stop:  # 超过训练时间
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = len(self.tloss)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # GPU 内存使用量（GB）
                            *self.tloss.values(),  # 损失
                            batch.get("cls", batch["img"]).shape[0],  # 实例数量
                            batch["img"].shape[-1],  # 图像尺寸，例如 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # 允许在批次之间响应外部停止请求（例如平台取消）
            else:
                # for/else：仅当 for 循环未遇到 break（未触发 OOM 重试）时执行此代码块
                self._oom_retries = 0  # 第一个 epoch 成功后重置 OOM 计数器

            if self._oom_retries and not self.stop:
                continue  # OOM 恢复中断了 for 循环，使用降低后的批次大小重新开始

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 供日志记录器使用

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # 验证
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # 防止显存峰值过高
                self.metrics, self.fitness = self.validate()

            # NaN 恢复
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 保存模型
                if (self.args.save or final_epoch) and self.save_model():
                    self.run_callbacks("on_model_save")

            # 调度器
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                nw = self._get_warmup_iterations(nb)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # 不要移动
                self.stop |= epoch >= self.epochs  # 超过 epoch 数量时停止
            self.run_callbacks("on_fit_epoch_end")
            # 内存利用率超过 50% 时清理；由于存在内存泄漏，MPS 始终清理
            # https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # 提前停止
            if RANK != -1:  # DDP 训练时
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 将“停止”广播到所有 rank
                self.stop = broadcast_list[0]
            if self.stop:
                break  # 必须中断所有 DDP rank
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # 使用 best.pt 执行最终验证
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        for loader in (self.train_loader, self.test_loader):
            if hasattr(loader, "close"):
                loader.close()  # 关闭持久化数据加载器工作进程，避免其存活到解释器退出

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """根据模型和设备内存限制计算最佳批次大小。"""
        # 与真实多尺度最大尺寸对齐；金字塔检测头要求输入尺寸是步长的整数倍
        max_imgsz = math.ceil(self.args.imgsz * (1 + self.args.multi_scale) / self.stride) * self.stride
        return check_train_batch_size(
            model=self.model,
            imgsz=max_imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )  # 返回批次大小

    def _get_memory(self, fraction=False):
        """获取加速器内存使用量（GB）或占总内存的比例。"""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = self.accelerator.memory_reserved()
            if fraction:
                total = self.accelerator.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """调用垃圾回收器并清空缓存，以释放加速器内存。"""
        if threshold:
            assert 0 <= threshold <= 1, "threshold 必须介于 0 和 1 之间。"
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            self.accelerator.empty_cache()

    def read_results_csv(self):
        """使用 polars 将 results.csv 读取为字典。"""
        import polars as pl  # 局部导入，以加快 `import ultralytics`

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """将模型设置为训练模式。"""
        self.model.train()
        # 冻结 BN 统计量
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """保存模型训练检查点及附加元数据。"""
        import io

        # 瞬时 NaN/Inf 会永久污染 EMA 运行平均值（ema = decay*ema + (1-decay)*model），否则 save_model 会跳过每个
        # epoch，导致在有效输入上运行结束时仍没有检查点。对于有限的实时模型张量，重新同步每个受污染的 EMA 张量；
        # 如果某个张量在两者中都不是有限值，则交给下面的 nan_to_num_ 处理，确保始终写入可用检查点。
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])
        # 无论是否使用 channels_last 训练，都以 NCHW 格式序列化：已发布版本的融合逻辑使用 .view()，
        # 处理 NHWC 步幅的检查点权重时会崩溃；而 trainer/predictor 会在初始化时重新应用 channels_last。
        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None  # 从序列化快照中移除仅供训练使用的状态
        # 限制 fp16 序列化溢出，但不修改正在使用的 EMA。
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        # 只将检查点序列化到字节缓冲区一次（比重复调用 torch.save() 更快）
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # 恢复训练和最终检查点都从 EMA 获取模型
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # 保存为字典
                "train_metrics": {**self.metrics, "fitness": self.fitness},
                "train_results": self.read_results_csv(),
                "date": datetime.now().astimezone().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取待保存的序列化内容

        # 保存检查点
        self.wdir.mkdir(parents=True, exist_ok=True)  # 确保权重目录存在
        self.last.write_bytes(serialized_ckpt)  # 保存 last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存 best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存 epoch，例如 'epoch3.pt'
        return True

    def get_dataset(self):
        """从数据配置中获取训练集和验证集。

        返回：
            (dict)：包含训练集、验证集、测试集和类别名称的字典。
        """
        try:
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)

            # 根据任务检查数据集
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
                "depth",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 用于验证 'yolo train data=url.zip' 用法
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """为任意任务加载、创建或下载模型。

        返回：
            (dict | None)：用于恢复训练的检查点；如果未加载检查点则返回 None。
        """
        if isinstance(self.model, torch.nn.Module):  # 如果模型已提前加载，则无需设置
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.args.pretrained, (str, Path)) and not self.resume:
            weights, _ = load_checkpoint(self.args.pretrained)
        elif self.args.pretrained is False and not self.resume:
            weights = None

        # 从恢复训练的检查点重建 DistillationModel
        if isinstance(weights, DistillationModel):
            if RANK in {-1, 0}:
                LOGGER.info("Resuming training DistillationModel from checkpoint weights")
            student_model = self.get_model(cfg=cfg, weights=weights.student_model, verbose=RANK in {-1, 0})
            student_model.args = self.args
            # 为节省内存和磁盘空间，检查点中已移除教师模型；从 distill_model 路径重新构建
            teacher_model = weights.teacher_model if weights.teacher_model is not None else self.args.distill_model
            model = DistillationModel(student_model=student_model, teacher_model=teacher_model)
            if getattr(weights, "projector", None) is not None:
                model.projector.load_state_dict(weights.projector.state_dict())  # 恢复训练好的投影器
            model.criterion = None
            self.model = model
        else:
            self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})  # 调用 Model(cfg, 权重)
        return ckpt

    def optimizer_step(self):
        """执行一次训练优化器更新，包括梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)  # 取消梯度缩放
        if self.device.type == "npu" and TORCH_2_0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, foreach=False)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """允许根据任务类型自定义模型输入和真实标注的预处理。"""
        return batch

    def validate(self):
        """使用 self.validator 在验证集上执行验证。

        返回：
            (tuple): A tuple containing:
                - metrics (dict | None)：验证指标字典；跳过验证时为 None。
                - fitness (float | None)：验证适应度分数；跳过验证时为 None。
        """
        if self.ema and self.world_size > 1:
            # 将 EMA 缓冲区从 rank 0 同步到所有 rank
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 未找到适应度时使用损失的相反数
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型；加载配置文件时抛出 NotImplementedError。"""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """抛出 NotImplementedError（必须由子类实现）。"""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """抛出 NotImplementedError（子类必须返回 `torch.utils.data.DataLoader`）。"""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建数据集。"""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回带标签的训练损失字典；如果 loss_items 为 None，则返回损失名称列表。"""
        if loss_items is None:
            return [f"{prefix}/{x}" for x in self.loss_names]
        return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}

    def set_model_attributes(self):
        """在训练前设置或更新模型参数。"""
        self.model.names = self.data["names"]

    def set_class_weights(self):
        """计算并设置类别权重，以处理类别不平衡。子类可覆盖此方法。"""

    def build_targets(self, preds, targets):
        """构建 YOLO 模型训练所需的目标张量。"""

    def progress_string(self):
        """返回描述训练进度的字符串。"""
        return ""

    # TODO：后续可能需要将以下函数移入回调。
    def plot_training_samples(self, batch, ni):
        """在 YOLO 训练期间绘制训练样本。"""

    def plot_training_labels(self):
        """绘制 YOLO 模型的训练标签。"""

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 列数
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """根据 CSV 文件绘制指标。"""
        plot_results(file=self.csv, on_plot=self.on_plot)  # 保存 results.png

    def on_plot(self, name, data=None):
        """注册图表（例如供回调使用）。"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """对 YOLO 模型执行最终评估和验证。"""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # 仅在 GPU 0 上清理，其他 GPU 等待
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # 使用 last.pt 中的训练指标更新 best.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # 最终验证编译速度过慢，因此禁用
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.epoch += 1  # 在 epochs+1 步记录最佳指标，不覆盖最后一个 epoch
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1  # 恢复 epoch

    def check_resume(self, overrides):
        """检查恢复训练的检查点是否存在，并相应更新参数。"""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate 模型
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                    "distill_model",
                    "save_dir",
                ):  # 允许在恢复训练时更新参数，以减少内存或更换设备
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """从检查点加载优化器、scaler、EMA 和 best_fitness。"""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # 使用 EMA 验证会创建无法更新的推理张量
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness")

    def _handle_nan_recovery(self, epoch):
        """检测 NaN/Inf 损失，并通过加载最新检查点进行恢复。"""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        corrupted = RANK in {-1, 0} and (loss_nan or fitness_nan)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf"
        if RANK != -1:  # DDP：广播到所有 rank
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch:
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # 第一个 epoch 无法恢复，让训练继续
        if not self.last.exists():
            raise RuntimeError(f"{reason} detected but no valid last.pt is available for recovery")
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # 加载检查点前将模型设为训练模式，避免推理张量错误
        _, ckpt = load_checkpoint(self.last)
        ema = ckpt["ema"].float()
        ema_state = ema.state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            # 知识蒸馏：EMA 中已移除教师模型（从 distill_model 路径重建），因此只恢复学生模型和投影器；
            # 分别加载它们可以保持严格的键匹配。
            model.student_model.load_state_dict(ema.student_model.state_dict())
            model.projector.load_state_dict(ema.projector.state_dict())
        else:
            model.load_state_dict(ema_state)  # 将 EMA 权重加载到模型
        self._load_checkpoint_state(ckpt)  # 加载 优化器/scaler/EMA/best_fitness
        del ckpt, ema, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """从给定检查点恢复 YOLO 训练。"""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert 0 < start_epoch < self.epochs, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        self._load_checkpoint_state(ckpt)
        if getattr(unwrap_model(self.model), "end2end", False):
            # 初始化损失，并恢复一对一和一对多参数
            unwrap_model(self.model).criterion = unwrap_model(self.model).init_criterion()
            unwrap_model(self.model).criterion.updates = start_epoch - 1
            unwrap_model(self.model).criterion.update()
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

    def _close_dataloader_mosaic(self):
        """更新数据加载器，停止使用 mosaic 增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """为给定模型构建优化器。

        参数：
            model (torch.nn.Module)：要构建优化器的模型。
            name (str，可选)：要使用的优化器名称。为 'auto' 时，根据迭代次数自动选择优化器。
            lr (float，可选)：优化器的学习率。
            momentum (float，可选)：优化器的动量因子。
            decay (float，可选)：权重衰减系数。
            iterations (float，可选)：迭代次数；当 name 为 'auto' 时用于确定优化器。

        返回：
            (torch.optim.Optimizer)：构建的优化器。
        """
        g = [{}, {}, {}, {}]  # 优化器 参数 groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # 类别数量
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 拟合公式，保留 6 位小数
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # Adam 不超过 0.01

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim in {2, 4} and use_muon:  # muon 只对矩阵和卷积滤波器进行正交化
                    g[3][fullname] = param  # muon 参数
                elif "bias" in fullname:  # 偏置（不衰减）
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # 权重（不衰减）
                    # ContrastiveHead 和 BNContrastiveHead 在此包含 'logit_scale'
                    g[1][fullname] = param
                else:  # 权重（进行衰减）
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # 转换为参数列表

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
        elif name == "RMSprop":
            optim_args = {"lr": lr, "momentum": momentum}
        elif name == "SGD" or name == "MuSGD":
            optim_args = {"lr": lr, "momentum": momentum, "nesterov": True}
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # 参数数量
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # 更新参数数量
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            # 微调时为 MuSGD 的特定参数使用更高学习率
            target = unwrap_model(model)
            head = getattr(target, "student_model", target).model[-1]
            heads = (getattr(head, "cv3", None), getattr(head, "one2one_cv3", None))
            boosted = {id(p) for m in heads if m for p in m.parameters()}
            g_ = []  # 新的参数组
            for x in g:
                p = x.pop("params")
                p1, p2 = [], []
                for k, v in p.items():
                    (p1 if id(v) in boosted or "proto.semseg" in k or "SemanticSegment" in k else p2).append(v)
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = (partial(MuSGD, muon=muon, sgd=sgd) if use_muon else getattr(optim, name))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer


class MultiTrainer:
    """在多个数据集上微调同一个基础模型，并汇总每个数据集的结果。

    当 `data` 是列表或元组时，Model.train() 会自动使用此类，使一个基础模型能够在一次调用中基准测试多个数据集
    （例如 RF100 集合）。数据集会依次进行微调，每次运行都使用相同的基础权重副本，因此每次运行都从完全相同的模型开始。
    所有输出都归入一个统一的 sweep 目录（例如 runs/detect/multitrain）：每个数据集拥有独立的运行子目录，
    每个数据集的指标和平均指标会写入 multitrain_results.json，并与 multitrain_results.png 柱状图一起保存。
    基础模型对象保持不变；每个数据集微调后的权重保存在各自的运行目录中。

    属性：
        trainer (type[BaseTrainer] | None): Task trainer 类别 for Python runs, or None for CLI subprocess runs.
        args (dict): Training arguments shared across datasets; its `data` key holds the dataset collection.
        模型 (torch.nn.Module): Base 模型 whose 权重 seed 每个 per-dataset fine-tune.
        callbacks (dict | None): Callbacks forwarded to 每个 per-dataset trainer.
        trainers (列表[SimpleNamespace]): Completed per-dataset run records.
        指标 (dict): Mapping of 每个 run 名称 (e.g. coco8, coco8-2) to its 训练-指标 dict from the 检查点.
        save_dir (Path | None): Sweep 目录 holding the per-dataset runs and 结果 JSON/plot.

    示例：
        在多个数据集上微调一个基础模型，并读取每次运行的指标：
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=10)
        >>> results["coco8"]["fitness"]  # coco8 运行的最终适应度
    """

    def __init__(self, trainer, args, model, _callbacks: dict | None = None):
        """使用任务训练器类型、共享训练参数和基础模型初始化 MultiTrainer。

        参数：
            trainer (type[BaseTrainer] | None)：每个数据集运行一次的任务训练器类型；为 None 时使用 CLI 子进程。
            args (dict)：训练参数；`data` 键保存待微调数据集的列表或元组。
            model (torch.nn.Module)：为每个数据集微调提供初始权重的基础模型。
            _callbacks (dict，可选)：转发给每个数据集训练器的回调函数。
        """
        self.trainer = trainer
        self.args = args
        self.model = model
        self.callbacks = _callbacks
        self.trainers = []
        self.metrics = {}
        self.save_dir = None

    def train(self):
        """依次在每个数据集上微调基础模型，并返回 {dataset: 指标} 映射。"""
        from types import SimpleNamespace

        from ultralytics.utils.patches import torch_load, torch_save

        datasets = self.args["data"]
        # 将每个数据集的运行结果和汇总图表归入一个 sweep 目录，例如 runs/detect/multitrain
        sweep = SimpleNamespace(
            project=self.args.get("project"),
            task=self.args.get("task"),
            mode="train",
            exist_ok=self.args.get("exist_ok", False),
        )
        self.save_dir = get_save_dir(sweep, name="multitrain")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        base_model = self.save_dir / "multitrain_base.pt" if self.trainer is None else None
        if base_model:
            torch_save(
                {"model": deepcopy(self.model).half(), "train_args": getattr(self.model, "args", {})}, base_model
            )
        try:
            for i, data in enumerate(datasets):
                LOGGER.info(
                    f"\n{colorstr('blue', 'bold', f'MultiTrainer {i + 1}/{len(datasets)}:')} fine-tuning on {data}"
                )
                name = Path(str(data)).stem
                run_name = name
                try:
                    overrides = {
                        **self.args,
                        "data": data,
                        "project": str(self.save_dir),  # 将每个数据集的运行嵌套在 sweep 目录中
                        "name": name,
                        "resume": False,
                    }
                    run = SimpleNamespace(
                        project=overrides["project"],
                        name=overrides["name"],
                        task=overrides.get("task"),
                        mode="train",
                        exist_ok=overrides.get("exist_ok", False),
                        save_dir=None,
                    )
                    save_dir = get_save_dir(run)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    run_name = save_dir.name
                    overrides["save_dir"] = str(save_dir)
                    if self.trainer is None:
                        overrides["model"] = str(base_model)
                        overrides["pretrained"] = True
                        subprocess.run(
                            [
                                *_YOLO_CLI_COMMAND,
                                "train",
                                *(f"{k}={v}" for k, v in overrides.items()),
                            ],
                            check=True,
                        )
                    else:
                        trainer = self.trainer(overrides=overrides, _callbacks=self.callbacks)
                        trainer.model = trainer.get_model(weights=self.model, cfg=self.model.yaml)
                        trainer.train()
                    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
                    ckpt = best if best.exists() else last
                    metrics = None
                    if self.trainer is not None:
                        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
                        if metrics is not None:
                            metrics = metrics.results_dict
                    self.metrics[run_name] = metrics or (torch_load(ckpt)["train_metrics"] if ckpt.exists() else None)
                    self.trainers.append(SimpleNamespace(save_dir=save_dir, best=best, last=last))
                except Exception as e:  # 单个数据集失败不应中止整个 sweep
                    LOGGER.error(f"MultiTrainer: fine-tuning on {data} failed, skipping: {e}")
                    self.metrics[run_name] = None
        finally:
            if base_model:
                base_model.unlink(missing_ok=True)
        if RANK in {-1, 0} and self.trainers:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_results()  # 保存每个数据集和平均指标的 JSON，供程序后处理
            if self.args.get("plots", True):
                self.plot_results()
        return self.metrics

    def save_results(self):
        """将每个数据集的指标和平均指标写入 multitrain_results.json，供程序后处理。"""
        import json

        results = {run: ({k: float(v) for k, v in m.items()} if m else None) for run, m in self.metrics.items()}
        valid = [m for m in results.values() if m]
        keys = {k for m in valid for k in m}
        mean = {k: sum(m[k] for m in valid if k in m) / sum(k in m for m in valid) for k in keys}
        file = self.save_dir / "multitrain_results.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump({"results": results, "mean": mean}, f, indent=2)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', file)}")
        return file

    def plot_results(self):
        """保存跨数据集柱状图，其中包含每个数据集的指标和所有数据集的平均值。"""
        from ultralytics.cfg import TASK2METRIC
        from ultralytics.utils.plotting import plot_multitrain_results

        key = TASK2METRIC.get(self.args.get("task"))
        scores = {run: float(m.get(key, m.get("fitness", 0.0))) for run, m in self.metrics.items() if m}
        if not scores:
            return None
        fname = plot_multitrain_results(scores, key=key or "fitness", save_dir=self.save_dir)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', fname)}")
        return fname
