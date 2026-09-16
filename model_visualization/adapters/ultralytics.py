#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ultralytics/YOLO26 adapter used by the standalone visualization runner."""

from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..core.common import (
    image_tensor,
    letterbox,
    safe_name,
    write_csv,
    write_image,
    write_json,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class UltralyticsAdapter:
    """Load a local Ultralytics checkout and expose raw model boundaries."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.project = _mapping(config.get("project"))
        self.model_values = _mapping(config.get("model"))
        self.device: Any = None
        self.precision = str(self.model_values.get("precision", "fp32")).lower()
        self.fuse = bool(self.model_values.get("fuse", False))
        self.yolo: Any = None
        self.model: Any = None
        self.head_name: Optional[str] = None
        self.runtime_state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        import torch

        repo = self.project.get("ultralytics_repo")
        if repo:
            path = Path(str(repo)).expanduser().resolve()
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from ultralytics import YOLO

        from ..core.common import resolve_device

        self.device = resolve_device(self.model_values.get("device", "auto"))
        weights = Path(str(self.model_values.get("weights", ""))).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"模型权重不存在：{weights}")
        self.yolo = YOLO(str(weights), task=str(self.model_values.get("task", "detect")))
        self.model = getattr(self.yolo, "model", None)
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO 对象没有可直接前向的 model")
        self.model.to(self.device)
        self.model.eval()
        if self.fuse:
            fuse_method = getattr(self.model, "fuse", None)
            if callable(fuse_method):
                fuse_method()
        requested_precision = self.precision
        modules = self.config.get("modules") if isinstance(self.config.get("modules"), Mapping) else {}
        cam_enabled = bool(modules.get("visualization.cam", False))
        effective_precision = "fp32"
        if cam_enabled:
            # CAM requires stable FP32 gradients.  The requested precision is
            # retained in runtime_state while all visualization forwards use FP32.
            self.model.float()
            effective_precision = "fp32"
        elif requested_precision in {"fp16", "half"} and self.device.type == "cuda":
            self.model.half()
            effective_precision = "fp16"
        elif requested_precision in {"bf16", "bfloat16"} and self.device.type == "cuda" and hasattr(self.model, "bfloat16"):
            self.model.bfloat16()
            effective_precision = "bf16"
        self.head_name = self._find_head_name()
        self.precision = effective_precision
        self.runtime_state = {
            "adapter": "ultralytics",
            "weights": str(weights),
            "device": str(self.device),
            "requested_precision": requested_precision,
            "effective_precision": effective_precision,
            "cam_forces_fp32": cam_enabled,
            "fuse_requested": self.fuse,
            "fuse_effective": bool(self.fuse and callable(getattr(self.model, "fuse", None))),
            "model_class": type(self.model).__name__,
            "head_name": self.head_name,
            "torch_version": getattr(torch, "__version__", "unknown"),
        }

    def _find_head_name(self) -> Optional[str]:
        found = None
        for name, module in self.model.named_modules():
            class_name = type(module).__name__.lower()
            if class_name in {"detect", "segment", "pose", "obb", "worlddetect"} or "detect" in class_name:
                found = name
        return found

    @property
    def head(self) -> Any:
        if not self.head_name:
            return None
        return dict(self.model.named_modules()).get(self.head_name)

    def named_modules(self):
        return self.model.named_modules()

    def list_layers(self) -> list[dict[str, Any]]:
        rows = []
        for index, (name, module) in enumerate(self.model.named_modules()):
            if not name:
                continue
            parameters = sum(int(parameter.numel()) for parameter in module.parameters(recurse=False))
            rows.append({
                "index": index,
                "name": name,
                "type": type(module).__name__,
                "parameters": parameters,
                "is_detect_head": name == self.head_name,
            })
        return rows

    def default_layer_names(self, preset: str = "detect_inputs") -> list[str]:
        if str(preset).lower() == "detect_inputs" and self.head_name:
            return [self.head_name]
        names = []
        for name, module in self.model.named_modules():
            if name and type(module).__name__.lower() in {"conv", "c2f", "c3k2", "sppf", "concat"}:
                names.append(name)
        return names[-3:] or ([self.head_name] if self.head_name else [])

    def prepare(self, source: str | Path, size: tuple[int, int]) -> tuple[np.ndarray, Any, Any]:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图片：{source}")
        resized, metadata = letterbox(image, size, str(source))
        tensor = image_tensor(resized, self.device, self.precision)
        return image, metadata, tensor

    def forward(self, tensor: Any) -> Any:
        return self.model(tensor)

    def effective_layers(self, layers_config: Mapping[str, Any]) -> tuple[list[str], str]:
        values = _mapping(layers_config)
        modules = [str(item) for item in values.get("modules", []) if str(item)]
        preset = str(values.get("preset", "detect_inputs"))
        return modules or self.default_layer_names(preset), preset

    @staticmethod
    def _branch(predictions: Mapping[str, Any], requested: str) -> tuple[str, Optional[Mapping[str, Any]]]:
        branch = str(requested or "auto").lower()
        if branch == "auto":
            branch = "one2one" if "one2one" in predictions else "one2many" if "one2many" in predictions else "self"
        selected = predictions.get(branch, predictions)
        return branch, selected if isinstance(selected, Mapping) else None

    @staticmethod
    def _restore_box(box: list[float], metadata: Any) -> list[float]:
        ratio = float(getattr(metadata, "ratio", 1.0) or 1.0)
        left = float(getattr(metadata, "pad_left", 0))
        top = float(getattr(metadata, "pad_top", 0))
        width = float(getattr(metadata, "original_width", 0))
        height = float(getattr(metadata, "original_height", 0))
        x1, y1, x2, y2 = [(float(value) - offset) / ratio for value, offset in ((box[0], left), (box[1], top), (box[2], left), (box[3], top))]
        return [max(0.0, min(width, x1)), max(0.0, min(height, y1)), max(0.0, min(width, x2)), max(0.0, min(height, y2))]

    @staticmethod
    def _xywh_to_xyxy(boxes: Any) -> Any:
        result = boxes.clone()
        result[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        result[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        result[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        result[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
        return result

    @staticmethod
    def _xyxy_to_xywh(boxes: Any) -> Any:
        result = boxes.clone()
        result[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
        result[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
        result[..., 2] = boxes[..., 2] - boxes[..., 0]
        result[..., 3] = boxes[..., 3] - boxes[..., 1]
        return result

    def _source_info(self, index: int, features: list[Any], head: Any) -> dict[str, Any]:
        sizes = [int(feature.shape[-2] * feature.shape[-1]) for feature in features]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        level = max(0, min(len(sizes) - 1, bisect.bisect_right(offsets, index) - 1)) if sizes else -1
        local = index - offsets[level] if level >= 0 else index
        width = int(features[level].shape[-1]) if level >= 0 else 0
        grid_y, grid_x = (local // width, local % width) if width else (None, None)
        stride = None
        try:
            stride_value = head.stride[level]
            stride = float(stride_value.detach().cpu().item() if hasattr(stride_value, "detach") else stride_value)
        except Exception:
            pass
        if stride and stride > 0 and abs(math.log2(stride) - round(math.log2(stride))) < 1e-5:
            level_name = f"P{int(round(math.log2(stride)))}"
        else:
            level_name = f"level_{level}" if level >= 0 else "unknown"
        return {
            "source_level": level_name,
            "source_level_index": level,
            "source_index": local,
            "stride": stride,
            "grid_x": grid_x,
            "grid_y": grid_y,
        }

    @staticmethod
    def _row(
        image_stem: str,
        branch: str,
        stage: str,
        raw_index: int,
        box_input: list[float],
        score: float,
        class_id: int,
        source: Mapping[str, Any],
        metadata: Any,
        rank: Optional[int] = None,
    ) -> dict[str, Any]:
        row = {
            "image_id": image_stem,
            "candidate_id": f"{stage}:{raw_index}" if rank is None else f"{stage}:{rank}",
            "stage": stage,
            "branch": branch,
            "raw_index": int(raw_index),
            "bbox_input": [float(value) for value in box_input],
            "bbox": UltralyticsAdapter._restore_box(box_input, metadata),
            "score": float(score),
            "class_id": int(class_id),
            **dict(source),
        }
        if rank is not None:
            row["rank"] = int(rank)
        return row

    def trace_stage(
        self,
        tensor: Any,
        config: Mapping[str, Any],
        *,
        image_bgr: Optional[np.ndarray] = None,
        transform_meta: Any = None,
        run_dir: Optional[Path] = None,
        image_stem: str = "image",
        image_id: str | None = None,
    ) -> dict[str, Any]:
        """Trace raw candidates to the final rows while preserving exact indices."""
        import torch

        image_id = image_stem if image_id is None else image_id

        head = self.head
        if head is None:
            return {"status": "unsupported", "reason": "no_detection_head"}
        head_inputs: list[Any] = []

        def prehook(_module: Any, inputs: Any) -> None:
            nonlocal head_inputs
            if inputs and isinstance(inputs[0], (list, tuple)):
                head_inputs = list(inputs[0])
            else:
                head_inputs = [item for item in inputs if torch.is_tensor(item)]

        hook = head.register_forward_pre_hook(prehook)
        try:
            with torch.no_grad():
                model_output = self.forward(tensor)
        finally:
            hook.remove()
        predictions: Optional[Mapping[str, Any]] = None
        if isinstance(model_output, tuple):
            predictions = next((item for item in model_output if isinstance(item, Mapping)), None)
        elif isinstance(model_output, Mapping):
            predictions = model_output
        if predictions is None:
            return {"status": "unsupported", "reason": "model_output_has_no_raw_predictions"}
        stage_values = _mapping(config.get("stage_trace"))
        branch, raw_branch = self._branch(predictions, str(stage_values.get("branch", "auto")))
        if raw_branch is None or "scores" not in raw_branch or "boxes" not in raw_branch:
            return {"status": "unsupported", "reason": "selected_branch_missing_boxes_or_scores", "branch": branch}
        if not head_inputs and isinstance(raw_branch.get("feats"), (list, tuple)):
            head_inputs = list(raw_branch["feats"])
        raw = head._inference(raw_branch)
        raw_boxes = raw[0, :4].transpose(0, 1)
        end2end = branch == "one2one"
        try:
            decoded_boxes_are_xyxy = bool(getattr(head, "end2end", False)) or bool(getattr(head, "xyxy", False))
        except Exception:
            decoded_boxes_are_xyxy = end2end
        boxes_xyxy = raw_boxes if decoded_boxes_are_xyxy else self._xywh_to_xyxy(raw_boxes)
        class_count = int(getattr(head, "nc", max(1, int(raw.shape[1] - 4))))
        probabilities = raw[0, 4 : 4 + class_count].transpose(0, 1)
        raw_scores, raw_labels = probabilities.max(dim=1)
        detection = _mapping(config.get("detection"))
        conf = float(detection.get("conf", 0.001))
        iou = float(detection.get("iou", 0.70))
        max_det = int(detection.get("max_det", getattr(head, "max_det", 300)))
        features = [item.detach() for item in head_inputs if torch.is_tensor(item) and item.ndim == 4]
        raw_rows = []
        for index in range(int(raw_scores.shape[0])):
            source = self._source_info(index, features, head)
            row = self._row(image_id, branch, "raw_candidate", index, boxes_xyxy[index].float().cpu().tolist(), float(raw_scores[index].item()), int(raw_labels[index].item()), source, transform_meta)
            row["class_scores"] = probabilities[index].float().cpu().tolist()
            raw_rows.append(row)

        final_rows: list[dict[str, Any]] = []
        topk_rows: list[dict[str, Any]] = []
        final_raw_indices: list[int] = []
        try:
            scores_nca = probabilities.unsqueeze(0)
            top_scores, top_labels, top_indices = head.get_topk_index(scores_nca, max_det)
            for rank in range(int(top_indices.shape[1])):
                score = float(top_scores[0, rank, 0].item())
                raw_index = int(top_indices[0, rank].item())
                label = int(top_labels[0, rank, 0].item())
                source = self._source_info(raw_index, features, head)
                row = self._row(image_id, branch, "head_topk", raw_index, boxes_xyxy[raw_index].float().cpu().tolist(), score, label, source, transform_meta, rank)
                topk_rows.append(row)
        except Exception as error:
            # Older/custom heads may not expose get_topk_index; final NMS can
            # still be traced for one-to-many outputs.
            if end2end:
                return {"status": "unsupported", "reason": f"topk_failed:{type(error).__name__}:{error}", "branch": branch}
            topk_rows.clear()
            top_scores = top_labels = top_indices = None
        if end2end:
            for row in topk_rows:
                if float(row["score"]) > conf:
                    final_rows.append({**row, "stage": "final", "candidate_id": f"final:{len(final_rows)}"})
                    final_raw_indices.append(int(row["raw_index"]))
        else:
            try:
                from ultralytics.utils.nms import non_max_suppression

                nms_input = raw if not decoded_boxes_are_xyxy else raw.clone()
                if decoded_boxes_are_xyxy:
                    nms_input[0, :4] = self._xyxy_to_xywh(nms_input[0, :4].transpose(0, 1)).transpose(0, 1)
                nms_output, keep_indices = non_max_suppression(
                    nms_input,
                    conf_thres=conf,
                    iou_thres=iou,
                    agnostic=bool(getattr(head, "agnostic_nms", False)),
                    max_det=max_det,
                    nc=int(getattr(head, "nc", probabilities.shape[1])),
                    end2end=False,
                    return_idxs=True,
                )
                kept = keep_indices[0].view(-1).tolist()
                nms_rows = nms_output[0]
                for rank, (item, raw_index) in enumerate(zip(nms_rows, kept)):
                    raw_index = int(raw_index)
                    score = float(item[4].item())
                    label = int(item[5].item())
                    source = self._source_info(raw_index, features, head)
                    final_rows.append(self._row(image_id, branch, "final", raw_index, item[:4].float().cpu().tolist(), score, label, source, transform_meta, rank))
                    final_raw_indices.append(raw_index)
            except Exception as error:
                return {"status": "unsupported", "reason": f"nms_failed:{type(error).__name__}:{error}", "branch": branch}

        topk_index_set = {int(row["raw_index"]) for row in topk_rows}
        final_index_set = set(final_raw_indices)
        for row in raw_rows:
            raw_index = int(row["raw_index"])
            row["in_head_topk"] = raw_index in topk_index_set
            row["in_final"] = raw_index in final_index_set
            if row["in_final"]:
                row["filter_reason"] = "kept_final"
            elif float(row["score"]) <= conf:
                row["filter_reason"] = "below_conf"
            elif branch == "one2many":
                row["filter_reason"] = "removed_by_nms"
            elif row["in_head_topk"]:
                row["filter_reason"] = "removed_after_head_topk"
            else:
                row["filter_reason"] = "removed_by_topk"

        target_config = _mapping(_mapping(config.get("cam")).get("target"))
        target_kind = str(target_config.get("kind", "raw_candidate"))
        target_class_id = target_config.get("class_id")
        raw_rows_for_target = raw_rows
        target_index = None
        final_rows_for_target = (
            [row for row in final_rows if target_class_id is None or int(row.get("class_id", -1)) == int(target_class_id)]
        )
        if target_kind == "final_detection" and final_rows_for_target:
            target_index = int(final_rows_for_target[0]["raw_index"])
        elif target_kind != "final_detection" and raw_rows_for_target:
            if target_class_id is not None:
                target_index = max(
                    raw_rows_for_target,
                    key=lambda row: float((row.get("class_scores") or [0.0])[int(target_class_id)] if int(target_class_id) < len(row.get("class_scores") or []) else 0.0),
                )["raw_index"]
            else:
                target_index = max(raw_rows_for_target, key=lambda row: float(row["score"]))["raw_index"]
        summary_rows = [
            {"image_id": image_id, "branch": branch, "stage": "raw_candidate", "count": len(raw_rows), "conf_threshold": conf, "iou_threshold": iou},
            {"image_id": image_id, "branch": branch, "stage": "head_topk", "count": len(topk_rows), "conf_threshold": conf, "iou_threshold": iou},
            {"image_id": image_id, "branch": branch, "stage": "final", "count": len(final_rows), "conf_threshold": conf, "iou_threshold": iou},
        ]
        result: dict[str, Any] = {
            "status": "ok",
            "branch": branch,
            "raw_rows": raw_rows,
            "topk_rows": topk_rows,
            "final_rows": final_rows,
            "target_index": target_index,
            "summary": summary_rows,
            "target": {"kind": target_kind, "candidate_index": target_index},
            "links": [],
            "canonical_raw_record": {
                "image_id": image_id,
                "file_name": getattr(transform_meta, "source_path", image_stem),
                "predictions": raw_rows,
            },
            "canonical_final_record": {
                "image_id": image_id,
                "file_name": getattr(transform_meta, "source_path", image_stem),
                "predictions": final_rows,
            },
        }
        if run_dir is not None:
            stage_root = run_dir / "stage_trace" / safe_name(image_stem)
            stage_root.mkdir(parents=True, exist_ok=True)
            image_metadata = {
                "output_id": image_stem,
                "image_id": image_id,
                "source_image_path": str(getattr(transform_meta, "source_path", "")),
                "branch": branch,
            }
            raw_path = stage_root / "raw_candidates.json"
            topk_path = stage_root / "head_topk.json"
            final_path = stage_root / "final_detections.json"
            events_path = stage_root / "stage_events.jsonl"
            summary_path = stage_root / "stage_summary.csv"
            candidates_path = stage_root / "candidates.jsonl"
            write_json(raw_path, {**image_metadata, "candidates": raw_rows})
            write_json(topk_path, {**image_metadata, "detections": topk_rows})
            write_json(final_path, {**image_metadata, "predictions": final_rows})
            write_csv(summary_path, summary_rows)
            with events_path.open("w", encoding="utf-8") as handle:
                for event in summary_rows:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            with candidates_path.open("w", encoding="utf-8") as handle:
                for row in raw_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            result["links"].extend(str(path.relative_to(run_dir)) for path in (raw_path, topk_path, final_path, summary_path, events_path, candidates_path))
            if bool(stage_values.get("save_raw", True)):
                np.savez_compressed(
                    stage_root / "raw_arrays.npz",
                    boxes=boxes_xyxy.detach().float().cpu().numpy(),
                    scores=raw_scores.detach().float().cpu().numpy(),
                    labels=raw_labels.detach().cpu().numpy(),
                    class_scores=probabilities.detach().float().cpu().numpy(),
                    source_level=np.asarray([row["source_level_index"] for row in raw_rows], dtype=np.int32),
                    source_index=np.asarray([row["source_index"] for row in raw_rows], dtype=np.int32),
                )
                result["links"].append(str((stage_root / "raw_arrays.npz").relative_to(run_dir)))
            if bool(stage_values.get("export_canonical", True)):
                canonical_root = run_dir / "canonical"
                canonical_root.mkdir(parents=True, exist_ok=True)
                raw_canonical = canonical_root / f"raw_{safe_name(image_stem)}.json"
                final_canonical = canonical_root / f"final_{safe_name(image_stem)}.json"
                write_json(raw_canonical, {"records": [result["canonical_raw_record"]], "metadata": {**image_metadata, "stage": "raw_candidate"}})
                write_json(final_canonical, {"records": [result["canonical_final_record"]], "metadata": {**image_metadata, "stage": "final"}})
                result["links"].extend(str(path.relative_to(run_dir)) for path in (raw_canonical, final_canonical))
            if image_bgr is not None:
                overlay = image_bgr.copy()
                display_conf = float(stage_values.get("display_conf", 0.25))
                for row in raw_rows:
                    if float(row["score"]) < display_conf:
                        continue
                    x1, y1, x2, y2 = [int(round(value)) for value in row["bbox"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 1)
                for row in final_rows:
                    if float(row["score"]) < display_conf:
                        continue
                    x1, y1, x2, y2 = [int(round(value)) for value in row["bbox"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(overlay, f"{row['class_id']}:{row['score']:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
                overlay_path = stage_root / "stages_overlay.jpg"
                write_image(overlay_path, overlay)
                result["links"].append(str(overlay_path.relative_to(run_dir)))
        return result


__all__ = ["UltralyticsAdapter"]
