# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
检查模型在数据集测试集或验证集划分上的准确率。

用法：
    $ yolo mode=val model=yolo26n.pt data=coco8.yaml imgsz=640

用法 - 格式：
    $ yolo mode=val model=yolo26n.pt                 # PyTorch
                          yolo26n.torchscript        # TorchScript
                          yolo26n.onnx               # ONNX Runtime 或启用 dnn=True 的 OpenCV DNN
                          yolo26n_openvino_model     # OpenVINO
                          yolo26n.engine             # TensorRT
                          yolo26n.mlpackage          # CoreML（仅 macOS）
                          yolo26n_saved_model        # TensorFlow SavedModel
                          yolo26n.pb                 # TensorFlow GraphDef
                          yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo26n_paddle_model       # PaddlePaddle
                          yolo26n.mnn                # MNN
                          yolo26n_ncnn_model         # NCNN
                          yolo26n_imx_model          # Sony IMX
                          yolo26n_rknn_model         # Rockchip RKNN
                          yolo26n_executorch_model   # ExecuTorch
                          yolo26n_axelera_model      # Axelera AI
                          yolo26n_deepx_model        # DEEPX
                          yolo26n_qnn.onnx           # Qualcomm QNN
                          yolo26n.tflite             # LiteRT
                          yolo26n_ascend_model       # Huawei Ascend
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile, linear_sum_assignment
from ultralytics.utils.torch_utils import (
    attempt_compile,
    autocast,
    get_torch_device_backend,
    select_device,
    smart_inference_mode,
    torch_distributed_zero_first,
    unwrap_model,
)


class BaseValidator:
    """用于创建验证器的基类。

    此类为验证流程提供基础，负责模型评估、指标计算和结果可视化。

    属性：
        args (SimpleNamespace)：验证器配置。
        dataloader (DataLoader)：用于验证的数据加载器。
        model (nn.Module)：待验证的模型。
        data (dict)：包含数据集信息的数据字典。
        device (torch.device)：用于验证的设备。
        batch_i (int)：当前批次索引。
        training (bool)：模型是否处于训练模式。
        names (dict)：类别名称映射。
        seen (int)：验证过程中已处理的图像数量。
        stats (dict)：验证过程中收集的统计信息。
        confusion_matrix：用于分类评估的混淆矩阵。
        nc (int)：类别数量。
        iouv (torch.Tensor)：从 0.50 到 0.95、步长为 0.05 的 IoU 阈值。
        jdict (list)：用于保存 JSON 验证结果的列表。
        speed (dict)：包含 preprocess、inference、loss 和 postprocess 耗时的字典，单位为毫秒。
        save_dir (Path)：结果保存目录。
        plots (dict)：用于保存可视化图像的字典。
        callbacks (dict)：各种回调函数的字典。
        stride (int)：模型步长，用于填充计算。
        loss (dict)：验证训练期间累积的损失项。

    方法：
        __call__：在数据加载器上执行推理、计算性能指标并完成验证。
        match_predictions：使用 IoU 将预测结果与真实目标匹配。
        add_callback：为指定事件添加回调。
        run_callbacks：执行指定事件对应的所有回调。
        get_dataloader：根据数据集路径和批次大小创建数据加载器。
        build_dataset：根据图像路径构建数据集。
        preprocess：预处理输入批次。
        postprocess：对预测结果执行后处理。
        init_metrics：初始化 YOLO 模型的性能指标。
        update_metrics：根据预测结果和批次更新指标。
        finalize_metrics：完成并返回所有指标。
        get_stats：返回模型性能统计信息。
        print_results：输出模型预测结果。
        get_desc：获取 YOLO 模型描述。
        on_plot：注册用于可视化的图像。
        plot_val_samples：绘制验证期间的样本。
        plot_predictions：绘制批次图像上的 YOLO 模型预测结果。
        pred_to_json：将预测结果转换为 JSON 格式。
        eval_json：评估 JSON 格式的预测统计信息并返回结果。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        """初始化 BaseValidator 实例。

        参数：
            dataloader (torch.utils.data.DataLoader，可选)：用于验证的数据加载器。
            save_dir (Path，可选)：结果保存目录。
            args (SimpleNamespace，可选)：验证器配置。
            _callbacks (dict，可选)：用于保存各种回调函数的字典。
        """
        import torchvision  # noqa（在此处导入，避免将 torchvision 导入时间计入后处理耗时）

        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.01 if self.args.task == "obb" else 0.001  # 降低 OBB 验证的内存占用
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证流程，在数据加载器上运行推理并计算性能指标。

        参数：
            trainer (object，可选)：包含待验证模型的训练器对象。
            model (nn.Module，可选)：不使用训练器时要验证的模型。

        返回：
            (dict)：包含验证统计信息的字典。
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # 保持训练验证只读：输入可能是 fp16，但 autocast 下 EMA 和模型权重仍保持 fp32。
            self.args.quantize = 16 if (self.device.type != "cpu" and trainer.amp) else None
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # 使用未编译的原始模型进行验证，避免兼容性问题
            model = model.float()
            self.loss = {k: torch.zeros_like(v) for k, v in trainer.loss_items.items()}
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
            with torch_distributed_zero_first(LOCAL_RANK):
                self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)
            device_type = str(self.args.device).split(":", 1)[0]
            device_type = device_type if device_type in {"npu", "xpu"} else "cuda"
            model = AutoBackend(
                model=model or self.args.model,
                # DDP 各进程复用 trainer._setup_ddp() 中分配的设备
                device=select_device(self.args.device)
                if RANK == -1
                else torch.device(device_type, get_torch_device_backend(device_type).current_device()),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.quantize == 16,
            )
            self.device = model.device  # 更新设备
            self.args.quantize = 16 if model.fp16 else None  # 记录实际推理精度
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            if augment and not model.base_model:
                LOGGER.warning(f"'augment' is not supported by this model (format='{fmt}'), ignoring.")
                augment = False
            # 与 predictor.setup_model 使用相同的判断：只有 CUDA 上的原生 PyTorch 模型才能无损使用 NHWC。
            channels_last = self.args.channels_last and self.device.type == "cuda" and pt
            if self.args.channels_last and not channels_last:
                LOGGER.warning(
                    f"'channels_last=True' applies only to native PyTorch models on CUDA, ignoring for "
                    f"format='{fmt}' on '{self.device.type}'."
                )
            if channels_last:
                model.to(memory_format=torch.channels_last)
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(model, "dynamic", False):
                if hasattr(model, "imgsz"):
                    self.args.imgsz = imgsz = max(model.imgsz)  # 复用导出元数据中的方形图像尺寸
                self.args.batch = model.metadata.get("batch", 1)  # 导出模型默认批次大小为 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
                "depth",
            }:
                self.data = check_det_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # CPU 验证时推理占主要耗时，不使用数据加载线程反而更快
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False
            self.stride = model.stride  # get_dataloader() 用于计算填充
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device, mode=self.args.compile)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # 预热

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # 每次验证开始前清空
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # 预处理
            with dt[0]:
                batch = self.preprocess(batch)

            with autocast(self.training and self.args.quantize == 16, device=self.device.type):
                # 推理
                with dt[1]:
                    preds = model(batch["img"], augment=augment)

                # 损失
                with dt[2]:
                    if self.training:
                        for k, v in model.loss(batch, preds)[1].items():
                            self.loss[k] += v

            # 后处理
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            # 在所有 GPU 之间归约损失
            loss = {k: v.clone().detach() for k, v in self.loss.items()}
            if trainer.world_size > 1:
                for v in loss.values():
                    dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            loss = {k: v.cpu() / len(self.dataloader) for k, v in loss.items()}
            results = {**stats, **trainer.label_loss_items(loss, prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # 将结果保留 5 位小数
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # 展平并保存
                stats = self.eval_json(stats)  # 更新统计信息
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """使用 IoU 将预测结果与真实目标匹配。

        参数：
            pred_classes (torch.Tensor)：预测类别索引，形状为 ``(N,)``。
            true_classes (torch.Tensor)：目标类别索引，形状为 ``(M,)``。
            iou (torch.Tensor)：包含预测结果与真实目标两两 IoU 值的 ``N x M`` 张量。
            use_scipy (bool，可选)：是否使用更精确的匈牙利一对一匹配算法。

        返回：
            (torch.Tensor)：正确预测张量，形状为 ``(N, 10)``，对应 10 个 IoU 阈值。
        """
        # D×10 矩阵，其中 D 表示检测结果数量，10 表示 IoU 阈值数量
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # L×D 矩阵，其中 L 表示标签（行），D 表示检测结果（列）
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # 将类别错误的匹配项置零
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = linear_sum_assignment(-cost_matrix)  # 取负值以最大化 IoU
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU 大于阈值且类别匹配
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.from_numpy(correct)

    def add_callback(self, event: str, callback):
        """将给定回调添加到指定事件。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """运行指定事件对应的所有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """根据数据集路径和批次大小获取数据加载器。"""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """根据图像路径构建数据集。"""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """预处理输入批次。"""
        return batch

    def postprocess(self, preds):
        """对预测结果执行后处理。"""
        return preds

    def init_metrics(self, model):
        """初始化 YOLO 模型的性能指标。"""

    def update_metrics(self, preds, batch):
        """根据预测结果和批次更新指标。"""

    def finalize_metrics(self):
        """完成指标计算并返回所有指标。"""

    def get_stats(self):
        """返回模型的性能统计信息。"""
        return {}

    def gather_stats(self):
        """在 DDP 训练期间将所有 GPU 的统计信息汇总到 GPU 0。"""

    def print_results(self):
        """输出模型的预测结果。"""

    def get_desc(self):
        """获取 YOLO 模型的描述信息。"""

    @property
    def metric_keys(self):
        """返回 YOLO 训练和验证使用的指标键。"""
        return []

    def on_plot(self, name, data=None):
        """根据唯一路径注册用于可视化和日志记录的图像。"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        """绘制验证期间的样本。"""

    def plot_predictions(self, batch, preds, ni):
        """在批次图像上绘制 YOLO 模型的预测结果。"""

    def pred_to_json(self, preds, batch):
        """将预测结果转换为 JSON 格式。"""

    def eval_json(self, stats):
        """评估 JSON 格式的预测统计信息并返回结果。"""
