# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class HailoBackend(BaseBackend):
    """用于 Ultralytics Hailo HEF 模型的 HailoRT 推理后端。"""

    def load_model(self, weight: str | Path) -> None:
        """加载 Hailo 导出目录及其 Ultralytics 元数据。"""
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as e:
            raise ImportError(
                "Hailo inference requires HailoRT. "
                "See https://docs.ultralytics.com/integrations/hailo#run-hailo-inference"
            ) from e

        w = Path(weight)
        hef_file = next(w.rglob("*.hef"), None)
        if hef_file is None or not hef_file.is_file():
            raise FileNotFoundError(f"No .hef file found in: {w}")

        LOGGER.info(f"Loading {hef_file} for Hailo inference...")
        self.apply_metadata(self.read_metadata(hef_file))
        if self.task and self.task not in {"detect", "segment", "pose", "obb", "classify", "semantic", "depth"}:
            raise ValueError(
                f"Hailo inference only supports detect, segment, pose, obb, classify, semantic and depth tasks, "
                f"not task='{self.task}'."
            )

        self.hef = HEF(str(hef_file))
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        with ExitStack() as stack:
            target = stack.enter_context(VDevice())
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(self.hef, configure_params)[0]
            stack.enter_context(network_group.activate(network_group.create_params()))
            input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            self.model = stack.enter_context(InferVStreams(network_group, input_params, output_params))
            self._stack = stack.pop_all()
        self._anchors = None
        if self.task in {"segment", "pose", "obb"}:
            from ultralytics.nn.modules import DFL

            self._dfl = DFL()
        # 分割、姿态和 OBB 返回供预测器 NMS 使用的密集张量；检测和分类任务不返回密集张量
        self.end2end = self.task not in {"segment", "pose", "obb"}

    def __del__(self):
        """释放 Hailo 流程和设备。"""
        if stack := getattr(self, "_stack", None):
            stack.close()

    def forward(self, im: torch.Tensor) -> np.ndarray | list[torch.Tensor]:
        """执行 Hailo 推理并返回解码后的检测结果，或返回用于分割的密集输出和原型。"""
        im = np.ascontiguousarray(np.clip(im.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8))
        results = self.model.infer({self.input_info.name: im})
        outputs = [results[x.name] for x in self.output_infos]
        if self.task == "segment":
            return self._decode_segment(outputs)
        if self.task == "pose":
            return self._decode_pose(outputs)
        if self.task == "obb":
            return self._decode_obb(outputs)
        if self.task == "classify":
            return torch.from_numpy(outputs[0]).reshape(outputs[0].shape[0], -1)  # on-chip softmax probabilities
        if self.task == "semantic":
            out = torch.from_numpy(outputs[0])
            if self.metadata.get("semantic_baked"):
                # 多类别 Hailo-10/15 已在芯片上固化上采样和 argmax，直接返回类别图。
                return out.reshape(out.shape[0], out.shape[1], out.shape[2])
            # Hailo-8/8L 和单类别头返回原始步长为 8 的 logits；将其交给预测器现有的双线性上采样、
            # 去除 letterbox 和类别归约逻辑，使结果与 PyTorch 模型完全一致。
            return out.permute(0, 3, 1, 2)
        if self.task == "depth":
            return self._decode_depth(outputs[0])
        return self._decode_raw(outputs) if not self.metadata.get("nms", False) else self._decode_nms(outputs[0])

    def _decode_nms(self, output: list) -> np.ndarray:
        """将 Hailo 逐类别 NMS 输出从归一化的 ``yxyx`` 转换为像素 ``xyxy`` 坐标。"""
        height, width = self.input_info.shape[:2]
        scale = np.array([width, height, width, height], dtype=np.float32)
        frames = []
        for detections in output:
            rows = []
            for class_id, class_detections in enumerate(detections):
                if len(class_detections):
                    class_detections = np.asarray(class_detections)
                    boxes = class_detections[:, [1, 0, 3, 2]] * scale
                    classes = np.full((len(boxes), 1), class_id, dtype=np.float32)
                    rows.append(np.concatenate((boxes, class_detections[:, 4:5], classes), axis=1))
            frame = np.concatenate(rows) if rows else np.empty((0, 6), dtype=np.float32)
            frames.append(frame[np.argsort(-frame[:, 4])[:300]])
        count = max(map(len, frames), default=0)
        predictions = np.zeros((len(frames), count, 6), dtype=np.float32)
        for i, frame in enumerate(frames):
            predictions[i, : len(frame)] = frame
        return predictions

    def _decode_boxes(self, reg_maps: list[torch.Tensor], angle: torch.Tensor | None = None) -> torch.Tensor:
        """对缓存的锚框执行 DFL 和边界框解码，返回 (B, A, 4) 的 xywh 边界框（提供 angle 时返回旋转框）。"""
        from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

        if self._anchors is None:
            strides = [self.input_info.shape[0] / m.shape[2] for m in reg_maps]
            self._anchors = make_anchors(reg_maps, strides)
        anchors, stride_tensor = self._anchors
        dist = self._dfl(torch.cat([m.flatten(2) for m in reg_maps], 2)).transpose(1, 2)
        if angle is not None:
            return dist2rbox(dist, angle, anchors) * stride_tensor
        return dist2bbox(dist, anchors, xywh=True) * stride_tensor

    def _decode_segment(self, outputs: list[np.ndarray]) -> list[torch.Tensor]:
        """解码原始分割张量（每个尺度包含 reg、cls、coeff 以及原型），供预测器的 NMS 使用。"""
        proto = torch.from_numpy(outputs[-1]).permute(0, 3, 1, 2)
        k = len(outputs) - 1
        reg_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[0:k:3]]
        cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[1:k:3]]
        cof_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[2:k:3]]
        boxes = self._decode_boxes(reg_maps)
        cls = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
        cof = torch.cat([x.flatten(2) for x in cof_maps], 2).transpose(1, 2)
        return [torch.cat((boxes, cls, cof), 2).transpose(1, 2), proto]

    def _decode_pose(self, outputs: list[np.ndarray]) -> torch.Tensor:
        """将原始姿态张量（每个尺度包含 reg、cls、kpt）解码为预测器 NMS 所需的密集输出。"""
        maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
        reg_maps, cls_maps, kpt_maps = maps[0::3], maps[1::3], maps[2::3]
        boxes = self._decode_boxes(reg_maps)
        anchors, stride_tensor = self._anchors
        cls = torch.cat([m.flatten(2) for m in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
        kpts = torch.cat([m.flatten(2) for m in kpt_maps], 2).transpose(1, 2)  # (B, A, nk) raw
        n_kpt, ndim = self.kpt_shape
        b, a, _ = kpts.shape
        y = kpts.view(b, a, n_kpt, ndim)
        # Pose.kpts_decode：xy = (raw * 2 + (anchor - 0.5)) * stride；可见性 sigmoid 在主机端执行
        xy = (y[..., :2] * 2.0 + (anchors.view(a, 1, 2) - 0.5)) * stride_tensor.view(a, 1, 1)
        kpts = torch.cat((xy, y[..., 2:3].sigmoid()), -1) if ndim == 3 else xy
        return torch.cat((boxes, cls, kpts.view(b, a, -1)), 2).transpose(1, 2)

    def _decode_obb(self, outputs: list[np.ndarray]) -> torch.Tensor:
        """将原始 OBB 张量（每个尺度包含 reg、cls、angle）解码为旋转 NMS 所需的密集输出。"""
        maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
        reg_maps, cls_maps, ang_maps = maps[0::3], maps[1::3], maps[2::3]
        angle = torch.cat([m.flatten(2) for m in ang_maps], 2).transpose(1, 2)  # (B, A, 1) raw
        angle = (angle.sigmoid() - 0.25) * math.pi  # OBB head angle squash, applied on the host
        boxes = self._decode_boxes(reg_maps, angle)  # rotated xywh
        cls = torch.cat([m.flatten(2) for m in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
        return torch.cat((boxes, cls, angle), 2).transpose(1, 2)

    def _decode_raw(self, outputs: list[np.ndarray]) -> np.ndarray:
        """解码分支优先排列的 YOLO26 回归和类别输出。"""
        from ultralytics.utils.tal import dist2bbox, make_anchors

        split = len(outputs) // 2
        box_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[:split]]
        cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[split:]]
        if self._anchors is None:
            strides = [self.input_info.shape[0] / x.shape[2] for x in box_maps]
            self._anchors = make_anchors(box_maps, strides)
        anchors, stride_tensor = self._anchors
        boxes = torch.cat([x.flatten(2) for x in box_maps], 2).transpose(1, 2)
        boxes = dist2bbox(boxes, anchors, xywh=False) * stride_tensor
        scores = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2).sigmoid()
        classes = scores.shape[2]
        anchor_index = scores.amax(-1).topk(min(300, scores.shape[1]), dim=1).indices[..., None]
        boxes = boxes.gather(1, anchor_index.expand(-1, -1, 4))
        scores = scores.gather(1, anchor_index.expand(-1, -1, classes))
        scores, index = scores.flatten(1).topk(min(300, scores.shape[1] * classes), dim=1)
        boxes = boxes.gather(1, (index // classes)[..., None].expand(-1, -1, 4))
        return torch.cat((boxes, scores[..., None], (index % classes)[..., None].float()), 2).numpy()

    def _decode_depth(self, output: np.ndarray) -> torch.Tensor:
        """将原始深度 logit 解码为度量深度图，并在主机端复现 ``Depth.forward`` 的处理。

        HEF 在检测头的最终 logit 卷积处截断，因此检测头后续的 clamp/exp 和学习得到的对数仿射校准在此处执行。
        深度图保持检测头分辨率（H/4, W/4）；``DepthPredictor.postprocess`` 使用 ``scale_masks`` 将其调整到图像尺寸，
        与 PyTorch 模型推理采用相同路径。
        """
        logit = torch.from_numpy(output).permute(0, 3, 1, 2)  # (B, H/4, W/4, 1) -> (B, 1, H/4, W/4)
        depth = logit.clamp(-4.0, 5.0).exp()
        return depth.pow(self.metadata.get("cal_a", 1.0)) * math.exp(self.metadata.get("cal_b", 0.0))
