# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path
from shutil import which

import numpy as np
import torch

from ultralytics.nn.modules import Detect, Pose, Segment
from ultralytics.utils import IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE, IS_RASPBERRYPI, IS_UBUNTU, LOGGER, WINDOWS
from ultralytics.utils.checks import check_apt_requirements, check_requirements
from ultralytics.utils.patches import onnx_export_patch
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import copy_attr

# Model Compression Toolkit（MCT）的量化配置
MCT_CONFIG = {
    "YOLO11": {
        "detect": {
            "layer_names": ["sub", "mul_2", "add_14", "cat_19"],
            "weights_memory": 2585350.2439,
            "n_layers": {238, 239},
        },
        "pose": {
            "layer_names": ["sub", "mul_2", "add_14", "cat_21", "cat_22", "mul_4", "add_15"],
            "weights_memory": 2437771.67,
            "n_layers": {257, 258},
        },
        "classify": {"layer_names": [], "weights_memory": np.inf, "n_layers": {112}},
        "segment": {
            "layer_names": ["sub", "mul_2", "add_14", "cat_21"],
            "weights_memory": 2466604.8,
            "n_layers": {265, 266},
        },
    },
    "YOLOv8": {
        "detect": {
            "layer_names": ["sub", "mul", "add_6", "cat_15"],
            "weights_memory": 2550540.8,
            "n_layers": {168, 169},
        },
        "pose": {
            "layer_names": ["add_7", "mul_2", "cat_17", "mul", "sub", "add_6", "cat_18"],
            "weights_memory": 2482451.85,
            "n_layers": {187, 188},
        },
        "classify": {"layer_names": [], "weights_memory": np.inf, "n_layers": {73}},
        "segment": {
            "layer_names": ["sub", "mul", "add_6", "cat_17"],
            "weights_memory": 2580060.0,
            "n_layers": {195, 196},
        },
    },
}


class FXModel(torch.nn.Module):
    """用于兼容 torch.fx 的自定义模型类。

    此类继承 `torch.nn.Module`，用于确保 torch.fx 的跟踪和图操作兼容性。
    它会复制现有模型的属性，并显式设置模型属性以确保正确复制。

    属性：
        model (nn.Module): 原始模型的层。
        imgsz (tuple[int, int]): 输入图像尺寸 (高度, 宽度)。
    """

    def __init__(self, model, imgsz=(640, 640)):
        """初始化 FXModel。

        参数：
            model (nn.Module): 为兼容 torch.fx 而封装的原始模型。
            imgsz (tuple[int, int]): 输入图像尺寸 (高度, 宽度)，默认为 (640, 640)。
        """
        super().__init__()
        copy_attr(self, model)
        # 显式设置 `model`，因为 `copy_attr` 不会复制该属性。
        self.model = model.model
        self.imgsz = imgsz

    def forward(self, x):
        """执行模型前向传播。

        此方法执行模型前向传播，处理层之间的依赖关系并保存中间输出。

        参数：
            x (torch.Tensor): 传递给模型的输入张量。

        返回：
            (torch.Tensor): 模型输出张量。
        """
        y = []  # 中间输出
        for m in self.model:
            if m.f != -1:  # 如果不是从上一层获取输入
                # 来自较早的层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Detect):
                m._inference = types.MethodType(_inference, m)  # 将方法绑定到 Detect
                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )
            if type(m) is Pose:
                m.forward = types.MethodType(pose_forward, m)  # bind method to Pose
            if type(m) is Segment:
                m.forward = types.MethodType(segment_forward, m)  # bind method to Segment
            x = m(x)  # run
            y.append(x)  # 保存输出
        return x


def _inference(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """解码 IMX 目标检测的边界框和类别分数。"""
    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    return dbox.transpose(1, 2), x["scores"].sigmoid().permute(0, 2, 1)


def pose_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 IMX 姿态估计前向传播，包括关键点解码。"""
    bs = x[0].shape[0]  # 批次大小
    nk_out = getattr(self, "nk_output", self.nk)
    kpt = torch.cat([self.cv4[i](x[i]).view(bs, nk_out, -1) for i in range(self.nl)], -1)

    # 如果 Pose26 使用 5 个维度，则转换为 3 个维度后导出
    if hasattr(self, "nk_output") and self.nk_output != self.nk:
        spatial = kpt.shape[-1]
        kpt = kpt.view(bs, self.kpt_shape[0], self.kpt_shape[1] + 2, spatial)
        kpt = kpt[:, :, :-2, :]  # Remove sigma_x, sigma_y
        kpt = kpt.view(bs, self.nk, spatial)
    x = Detect.forward(self, x)
    pred_kpt = self.kpts_decode(kpt)
    return *x, pred_kpt.permute(0, 2, 1)


def segment_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 IMX 分割前向传播。"""
    p = self.proto(x[0])  # 掩码原型
    bs = p.shape[0]  # 批次大小
    mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # 掩码系数
    x = Detect.forward(self, x)
    return *x, mc.transpose(1, 2), p


class NMSWrapper(torch.nn.Module):
    """使用 edge-mdt-cl 的 multiclass_nms 层包装 PyTorch 模块。"""

    def __init__(
        self,
        model: torch.nn.Module,
        score_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        task: str = "detect",
    ):
        """使用 PyTorch 模块和 NMS 参数初始化 NMSWrapper。

        参数：
            model (torch.nn.Module): 模型实例。
            score_threshold (float): 非极大值抑制的分数阈值。
            iou_threshold (float): 非极大值抑制的交并比阈值。
            max_detections (int): 要返回的检测结果数量。
            task (str): 任务类型，可选 'detect'、'pose' 或 'segment'。
        """
        super().__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.task = task

    def forward(self, images):
        """执行模型推理和 NMS 后处理的前向传播。"""
        from edgemdt_cl.pytorch.nms.nms_with_indices import multiclass_nms_with_indices

        # 模型 推理
        outputs = self.model(images)
        boxes, scores = outputs[0], outputs[1]
        nms_outputs = multiclass_nms_with_indices(
            boxes=boxes,
            scores=scores,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        if self.task == "pose":
            kpts = outputs[2]  # (bs, max_detections, kpts 17*3)
            out_kpts = torch.gather(kpts, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, kpts.size(-1)))
            return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_kpts
        if self.task == "segment":
            mc, proto = outputs[2], outputs[3]
            out_mc = torch.gather(mc, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, mc.size(-1)))
            return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_mc, proto
        return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, nms_outputs.n_valid


def torch2imx(
    model: torch.nn.Module,
    output_dir: Path | str,
    conf: float,
    iou: float,
    max_det: int,
    metadata: dict | None = None,
    gptq: bool = False,
    dataset=None,
    prefix: str = "",
) -> str:
    """将 YOLO 模型导出为 IMX 格式，以部署到 Sony IMX500 设备。

    此函数使用 Model Compression Toolkit（MCT）量化 YOLO 模型，并将其导出为兼容 Sony IMX500 边缘设备的 IMX 格式。
    支持 YOLOv8n 和 YOLO11n 模型的检测、分割、姿态估计和分类任务。

    参数：
        model (torch.nn.Module): 要导出的 YOLO 模型，必须是 YOLOv8n 或 YOLO11n。
        output_dir (Path | str): 保存导出 IMX 模型的目录。
        conf (float): NMS 后处理的置信度阈值。
        iou (float): NMS 后处理的 IoU 阈值。
        max_det (int): 返回的最大检测数量。
        metadata (dict | None, 可选): 要嵌入 ONNX 模型的元数据，默认为 None。
        gptq (bool, 可选): 是否使用基于梯度的训练后量化；为 False 时使用标准训练后量化，默认为 False。
        dataset (可选): 用于量化校准的代表性数据集，默认为 None。
        prefix (str, 可选): 日志前缀字符串，默认为 ""。

    返回：
        (str): 导出 IMX 模型目录的路径。

    异常：
        ValueError: 模型不是受支持的 YOLOv8n 或 YOLO11n 变体时抛出。

    示例：
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> path = torch2imx(model, "output_dir/", conf=0.25, iou=0.7, max_det=300)

    注意：
        - 如果尚未安装，则自动安装 Java>=17、模型压缩工具包、imx500-converter 及相关软件包。
        - 仅支持 YOLOv8n 和 YOLO11n 模型（检测、分割、姿态和分类任务）。
        - 输出包含量化后的 ONNX 模型、IMX 二进制文件和标签.txt 文件。
    """
    # 安装 Java>=17
    try:
        java_output = subprocess.run(["java", "--version"], check=True, capture_output=True).stdout.decode()
        version_match = re.search(r"(?:openjdk|java) (\d+)", java_output)
        java_version = int(version_match.group(1)) if version_match else 0
        assert java_version >= 17, "Java version too old"
    except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
        if IS_UBUNTU or IS_DEBIAN_TRIXIE:
            LOGGER.info(f"\n{prefix} installing Java 21 for Ubuntu...")
            check_apt_requirements(["openjdk-21-jre"])
        elif IS_RASPBERRYPI or IS_DEBIAN_BOOKWORM:
            LOGGER.info(f"\n{prefix} installing Java 17 for Raspberry Pi or Debian ...")
            check_apt_requirements(["openjdk-17-jre"])

    check_requirements(
        (
            "model-compression-toolkit>=2.4.1",
            "edge-mdt-cl<1.1.0",
            "edge-mdt-tpc>=1.2.0",
            "pydantic<2.12",
        )
    )
    check_requirements("imx500-converter[pt]>=3.17.3")
    dataset = dataset() if callable(dataset) else dataset  # resolve lazy dataloader

    import model_compression_toolkit as mct
    import onnx
    from edgemdt_tpc import get_target_platform_capabilities

    LOGGER.info(f"\n{prefix} starting export with model_compression_toolkit {mct.__version__}...")

    def representative_dataset_gen(dataloader=dataset):
        for batch in dataloader:
            img = batch["img"]
            img = img / 255.0
            yield [img]

    # 注意：IMX500 姿态估计模型需要 tpc_version 为 "4.0"
    tpc = get_target_platform_capabilities(tpc_version="4.0", device_type="imx500")

    bit_cfg = mct.core.BitWidthConfig()
    mct_config = MCT_CONFIG["YOLO11" if "C2PSA" in model.__str__() else "YOLOv8"][model.task]

    # 检查模型是否包含预期数量的层
    if len(list(model.modules())) not in mct_config["n_layers"]:
        raise ValueError("IMX export only supported for YOLOv8n and YOLO11n models.")

    for layer_name in mct_config["layer_names"]:
        bit_cfg.set_manual_activation_bit_width([mct.core.common.network_editors.NodeNameFilter(layer_name)], 16)

    config = mct.core.CoreConfig(
        mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
        quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
        bit_width_config=bit_cfg,
    )

    resource_utilization = mct.core.ResourceUtilization(weights_memory=mct_config["weights_memory"])

    quant_model = (
        mct.gptq.pytorch_gradient_post_training_quantization(  # 执行基于梯度的训练后量化
            model=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            gptq_config=mct.gptq.get_pytorch_gptq_config(
                n_epochs=1000, use_hessian_based_weights=False, use_hessian_sample_attention=False
            ),
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
        if gptq
        else mct.ptq.pytorch_post_training_quantization(  # 执行训练后量化
            in_module=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
    )

    if model.task != "classify":
        quant_model = NMSWrapper(
            model=quant_model,
            score_threshold=conf or 0.001,
            iou_threshold=iou,
            max_detections=max_det,
            task=model.task,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_model = output_dir / "model_imx.onnx"

    with onnx_export_patch():
        mct.exporter.pytorch_export_model(
            model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
        )

    model_onnx = onnx.load(onnx_model)  # 加载 ONNX 模型
    for k, v in (metadata or {}).items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, onnx_model)

    # 查找 imxconv-pt 二进制文件：先检查虚拟环境 bin 目录，再检查 PATH。
    bin_dir = Path(sys.executable).parent
    imxconv = bin_dir / ("imxconv-pt.exe" if WINDOWS else "imxconv-pt")
    if not imxconv.exists():
        imxconv = which("imxconv-pt")  # fallback to PATH
    if not imxconv:
        raise FileNotFoundError("imxconv-pt not found. Install with: pip install imx500-converter[pt]")

    subprocess.run(
        [str(imxconv), "-i", str(onnx_model), "-o", str(output_dir), "--no-input-persistency", "--overwrite-output"],
        check=True,
    )

    # imx 模型所需配置。
    with open(output_dir / "labels.txt", "w", encoding="utf-8") as labels_file:
        labels_file.writelines([f"{name}\n" for _, name in model.names.items()])

    return str(output_dir)
