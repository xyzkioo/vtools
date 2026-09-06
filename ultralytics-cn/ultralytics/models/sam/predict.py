# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate predictions using the Segment Anything Model (SAM).

SAM is an advanced image segmentation model offering features like promptable segmentation and zero-shot performance.
This module contains the implementation of the prediction logic and auxiliary utilities required to perform segmentation
using SAM. It forms an integral part of the Ultralytics framework and is designed for high-performance, real-time image
segmentation tasks.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.metrics import box_iou, mask_iou
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)

if TYPE_CHECKING:
    from .sam3.geometry_encoders import Prompt


class Predictor(BasePredictor):
    """用于 SAM 的预测器，支持实时图像分割和可提示分割能力。

    此类继承 BasePredictor，实现 Segment Anything Model（SAM）的高级图像分割任务。
    它支持点、边界框和掩码等多种输入提示，以精细控制分割结果。

    属性：
        args (SimpleNamespace): 预测器配置参数。
        model (torch.nn.Module): 已加载的 SAM 模型。
        device (torch.device): 加载模型的设备（CPU 或 GPU）。
        im (torch.Tensor): 预处理后的输入图像。
        features (torch.Tensor): 提取的图像特征。
        prompts (dict[str, Any]): 保存各种提示（例如边界框、点和掩码）的字典。
        segment_all (bool): 指示是否执行整图分割的标志。
        mean (torch.Tensor): 图像归一化的均值。
        std (torch.Tensor): 图像归一化的标准差。

    方法：
        preprocess：为模型推理准备输入图像。
        pre_transform：对输入图像执行初始变换。
        inference：根据输入提示执行分割推理。
        prompt_inference：执行基于提示的内部图像分割推理。
        generate：生成整张图像的分割掩码。
        setup_model：初始化用于推理的 SAM 模型。
        get_model：构建并返回 SAM 模型。
        postprocess：后处理模型输出以生成最终结果。
        setup_source：设置推理数据源。
        set_image：设置并预处理单张图像。
        get_im_features：使用 SAM 图像编码器提取图像特征。
        set_prompts：设置后续推理使用的提示。
        reset_image：重置当前图像及其特征。
        remove_small_regions：从掩码中移除较小的断开区域和孔洞。

    示例：
        >>> predictor = Predictor()
        >>> predictor.setup_model(model_path="sam_model.pt")
        >>> predictor.set_image("image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)
    """

    stride = 16

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置、覆盖项和回调函数初始化预测器。

        设置 SAM（Segment Anything Model）预测器对象，并应用提供的配置覆盖项或回调函数。
        初始化 SAM 的任务专用设置，例如将 retina_masks 设置为 True 以获得更佳结果。

        参数：
            cfg (dict): 包含默认设置的配置字典。
            overrides (dict | None): 用于覆盖默认配置的值字典。
            _callbacks (dict | None): 用于自定义行为的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides.update({"task": "segment", "mode": "predict", "batch": 1})
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """预处理用于模型推理的输入图像。

        此方法通过执行变换和归一化准备输入图像。它同时支持 torch.Tensor 和 np.ndarray 列表作为输入格式。
        对于通过 OpenCV 加载的图像，输入通常为 BGR，并会在预处理期间转换为 RGB。

        参数：
            im (torch.Tensor | list[np.ndarray]): BCHW 张量格式的输入图像，或 HWC NumPy 数组列表。
                NumPy 数组应为 BGR 顺序（OpenCV 返回的格式），并会被转换为 RGB。

        返回：
            (torch.Tensor): 归一化并转换为适当数据类型的预处理图像张量。

        示例：
            >>> predictor = Predictor()
            >>> image = torch.rand(1, 3, 640, 640)
            >>> preprocessed_image = predictor.preprocess(image)
        """
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        if not_tensor:
            im = (im - self.mean) / self.std
        im = im.half() if self.model.fp16 else im.float()
        return im

    def pre_transform(self, im):
        """对输入图像执行预处理所需的初始变换。

        此方法执行调整大小等变换，为后续预处理准备图像。目前不支持批量推理，因此列表长度应为 1。

        参数：
            im (list[np.ndarray]): 包含一张 HWC NumPy 数组格式图像的列表。

        返回：
            (list[np.ndarray]): 包含变换后图像的列表。

        异常：
            AssertionError: 输入列表包含多张图像时抛出。

        示例：
            >>> predictor = Predictor()
            >>> predictor.imgsz = [1024, 1024]  # normally set by setup_source()
            >>> image = np.random.rand(480, 640, 3)  # 单张 HWC 图像
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM 模型当前不支持批量推理"
        letterbox = LetterBox(self.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """使用当前加载的图像，根据给定输入提示执行图像分割推理。

        此方法利用由图像编码器、提示编码器和掩码解码器组成的 SAM（Segment Anything Model）架构，
        执行实时且可提示的分割任务。

        参数：
            im (torch.Tensor): 张量格式的预处理输入图像，形状为 (N, C, H, W)。
            bboxes (np.ndarray | list | None): 形状为 (N, 4) 的 XYXY 格式边界框。
            points (np.ndarray | list | None): 表示目标位置的像素点，形状为 (N, 2)。
            labels (np.ndarray | list | None): 点提示标签，形状为 (N)，1 表示前景，0 表示背景。
            masks (np.ndarray | None): 之前预测得到的低分辨率掩码，形状为 (N, H, W)；SAM 中 H=W=256。
            multimask_output (bool): 是否返回多个掩码的标志，适用于有歧义的提示。
            *args (Any): 其他位置参数。
            **kwargs (Any): 其他关键字参数。

        返回：
            pred_masks (torch.Tensor): 输出掩码，形状为 (C, H, W)，其中 C 是生成的掩码数量。
            pred_scores (torch.Tensor): 长度为 C 的数组，包含模型为每个掩码预测的质量分数。

        示例：
            >>> predictor = Predictor()
            >>> predictor.setup_model(model_path="sam_model.pt")
            >>> predictor.set_image("image.jpg")
            >>> results = predictor(bboxes=[[0, 0, 100, 100]])
        """
        # 如果 self.prompts 中保存了提示，则覆盖传入的提示
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        labels = self.prompts.pop("labels", labels)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """使用 SAM 专用架构，根据输入提示执行图像分割推理。

        此内部函数利用 Segment Anything Model（SAM）执行基于提示的实时分割，
        处理边界框、点和掩码等输入提示以生成分割掩码。

        参数：
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | list | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2) or (N, num_points,
                2), in pixels.
            labels (np.ndarray | list | None): Point prompt labels with shape (N) or (N, num_points). 1 for foreground,
                0 for background.
            masks (np.ndarray | None): Low-res masks from previous predictions with shape (N, H, W). For SAM, H=W=256.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        返回：
            pred_masks (torch.Tensor): Output masks with shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): Quality scores predicted by the model for each mask, with length C.

        示例：
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores = predictor.prompt_inference(im, bboxes=bboxes)
        """
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_prompts(im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks)
        return self._inference_features(features, *prompts, multimask_output)

    def _inference_features(
        self,
        features,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """使用 SAM 模型对图像特征执行推理。

        参数：
            features (torch.Tensor): Extracted image features with shape (B, C, H, W) from the SAM model image encoder.
            bboxes (np.ndarray | list[list[float]] | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list[list[float]] | None): Object location points with shape (N, 2), in pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N,). 1 = foreground, 0 = background.
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        返回：
            pred_masks (torch.Tensor): Output masks with shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): Quality scores for each mask, with length C.
        """
        points = (points, labels) if points is not None else None
        # 编码提示
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)

        # 预测掩码
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可能为 1 或 3，取决于 `multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """根据目标形状准备并变换输入提示。

        参数：
            dst_shape (tuple[int, int]): The target shape (height, width) for the prompts.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            bboxes (np.ndarray | list | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2) or (N, num_points,
                2), in pixels.
            labels (np.ndarray | list | None): Point prompt labels with shape (N) or (N, num_points). 1 for foreground,
                0 for background.
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array with
                shape (H, W).

        返回：
            bboxes (torch.Tensor | None): Transformed bounding boxes.
            points (torch.Tensor | None): Transformed points.
            labels (torch.Tensor | None): Transformed labels.
            masks (torch.Tensor | None): Transformed masks.

        异常：
            AssertionError: If the number of points don't match the number of labels, in case labels were passed.
        """
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # 转换输入提示
        if points is not None:
            points = torch.as_tensor(points, dtype=self.torch_dtype, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # 用户未传入标签时，默认所有标签均为正样本。
            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert points.shape[-2] == labels.shape[-1], (
                f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            points *= r
            if points.ndim == 2:
                # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
                points, labels = points[:, None, :], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = np.asarray(masks, dtype=np.uint8)
            masks = masks[None] if masks.ndim == 2 else masks
            letterbox = LetterBox(dst_shape, auto=False, center=False, padding_value=0, interpolation=cv2.INTER_NEAREST)
            masks = np.stack([letterbox(image=x).squeeze() for x in masks], axis=0)
            masks = torch.tensor(masks, dtype=self.torch_dtype, device=self.device)
        return bboxes, points, labels, masks

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        """使用 Segment Anything Model（SAM）执行图像分割。

        此方法利用 SAM 的先进架构和实时性能能力，将整幅图像分割为不同区域。
        也可以选择对图像裁剪区域进行处理，以获得更精细的分割结果。

        参数：
            im (torch.Tensor): Input tensor representing the preprocessed image with shape (N, C, H, W).
            crop_n_layers (int): Number of layers for additional mask predictions on image crops.
            crop_overlap_ratio (float): Overlap between crops, scaled down in subsequent layers.
            crop_downscale_factor (int): Scaling factor for sampled points-per-side in each layer.
            point_grids (list[np.ndarray] | None): Custom grids for point sampling normalized to [0,1].
            points_stride (int): Number of points to sample along each side of the image.
            points_batch_size (int): Batch size for the number of points processed simultaneously.
            conf_thres (float): Confidence threshold [0,1] for filtering based on mask quality prediction.
            stability_score_thresh (float): Stability threshold [0,1] for mask filtering based on stability.
            stability_score_offset (float): Offset value for calculating stability score.
            crop_nms_thresh (float): IoU cutoff for NMS to remove duplicate masks between crops.

        返回：
            pred_masks (torch.Tensor): Segmented masks with shape (N, H, W).
            pred_scores (torch.Tensor): Confidence scores for each mask with shape (N,).
            pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 4).

        示例：
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)  # 示例输入图像
            >>> masks, scores, boxes = predictor.generate(im)
        """
        import torchvision  # scope for faster 'import ultralytics'

        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(points_stride, crop_n_layers, crop_downscale_factor)
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # w, h
            # 裁剪图像并插值到输入尺寸
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False)
            crop_features = self.get_im_features(crop_im)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                prompts = self._prepare_prompts(crop_im.shape[2:], self.batch[1][0].shape[:2], points=points)
                pred_mask, pred_score = self._inference_features(crop_features, *prompts, multimask_output=True)
                # 将预测掩码插值到输入尺寸
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode="bilinear", align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # 布尔类型的内存效率更高。
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask], pred_mask[keep_mask], pred_score[keep_mask]

                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)

            # 在此裁剪区域内执行 NMS
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(crop_masks.shape[0]))

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)

        # 移除不同裁剪区域之间的重复掩码
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[keep], pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes

    @smart_inference_mode(False)  # 模型生命周期超过此次调用，因此其权重不能是推理张量
    def setup_model(self, model=None, verbose=True):
        """初始化用于推理的 Segment Anything Model（SAM）。

        此方法会将 SAM 模型分配到合适的设备，并初始化图像归一化及其他 Ultralytics 兼容性设置所需的参数。

        参数：
            model (torch.nn.Module | None): 预训练 SAM 模型。为 None 时根据配置构建新模型。
            verbose (bool): 为 True 时打印所选设备信息。

        示例：
            >>> predictor = Predictor()
            >>> predictor.setup_model(model=sam_model, verbose=True)
        """
        device = select_device(self.args.device, verbose=verbose)
        if self.args.channels_last:
            LOGGER.warning("'channels_last=True' is not supported for SAM predictors, ignoring.")
        if model is None:
            model = self.get_model()
        # 先将模型移动到设备，再转换数据类型，最后设置为评估模式，确保评估阶段缓存创建在目标设备上。
        model = model.to(device)
        model = model.half() if self.args.quantize == 16 else model.float()
        model.eval()
        self.model = model
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # Ultralytics 兼容性设置
        self.model.format = "sam"
        self.model.base_model = False  # SAMModel 不是 Ultralytics BaseModel，不支持 `augment` 或 `embed`
        self.model.stride = 32
        self.model.fp16 = self.args.quantize == 16
        self.done_warmup = True
        self.torch_dtype = torch.float16 if self.model.fp16 else torch.float32

    def get_model(self):
        """获取或构建用于图像分割任务的 Segment Anything Model（SAM）。"""
        from .build import build_sam  # slow import

        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        """后处理 SAM 的推理输出，生成目标检测掩码和边界框。

        此方法会将掩码和边界框缩放到原始图像尺寸，并对掩码预测结果应用阈值，
        从而利用 SAM 的架构完成实时、可提示的分割任务。

        参数：
            preds (tuple): SAM 模型推理输出，包含：
                - pred_masks (torch.Tensor)：形状为 (N, 1, H, W) 的预测掩码。
                - pred_scores (torch.Tensor)：形状为 (N, 1) 的每个掩码置信度分数。
                - pred_bboxes (torch.Tensor, optional)：segment_all 为 True 时返回的预测边界框。
            img (torch.Tensor)：形状为 (C, H, W) 的预处理输入图像张量。
            orig_imgs (list[np.ndarray] | torch.Tensor)：原始的未处理图像。

        返回：
            (list[Results])：Results 对象列表，包含每张处理后图像的检测掩码、边界框和其他元数据。

        示例：
            >>> predictor = Predictor()
            >>> preds = predictor.inference(img)
            >>> results = predictor.postprocess(preds, img, orig_imgs)
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # 转换为布尔值
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # 注意：SAM 模型不会返回 cls 信息，此处的 `cls` 仅作为占位符以保持接口一致。
                cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
                idx = pred_scores > self.args.conf
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)[idx]
                masks = masks[idx]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # 重置全分割模式。
        self.segment_all = False
        return results

    def set_image(self, image):
        """预处理并设置单张图像以执行推理。

        此方法会在模型尚未初始化时完成模型设置，配置数据源并预处理图像以提取特征，
        确保一次只设置一张图像，并提取后续推理所需的图像特征。

        参数：
            image (str | np.ndarray)：图像文件路径字符串，或由 cv2 读取的 NumPy 图像数组（BGR 通道顺序）。

        异常：
            AssertionError：尝试设置多张图像时引发。

        示例：
            >>> predictor = Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(cv2.imread("path/to/image.jpg"))

        注意：
            - 对新图像执行推理前应调用此方法。
            - 提取的特征会保存到 `self.features` 属性，供后续使用。
        """
        if self.model is None:
            self.setup_model()
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def setup_source(self, source):
        """为 SAM 推理设置数据源。"""
        if source is None:  # 处理提前设置 imgsz 的情况
            return
        super().setup_source(source, self.stride)
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM models only support square image size, but got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)

    def get_im_features(self, im):
        """使用 SAM 模型的图像编码器提取特征，供后续掩码预测使用。"""
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        """设置供后续推理操作使用的提示。"""
        self.prompts = prompts

    def reset_image(self):
        """重置当前图像及其特征，为后续推理清除已有数据。"""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """从分割掩码中移除较小的非连通区域和孔洞。

        此函数对 Segment Anything Model（SAM）生成的分割掩码执行后处理：
        从输入掩码中移除较小的非连通区域和孔洞，然后执行非极大值抑制（NMS）以消除新产生的重复边界框。

        参数：
            masks (torch.Tensor)：待处理的分割掩码，形状为 (N, H, W)，其中 N 为掩码数量，H 为高度，W 为宽度。
            min_area (int)：移除非连通区域和孔洞的最小面积阈值，小于此值的区域会被移除。
            nms_thresh (float)：NMS 算法用于移除重复边界框的 IoU 阈值。

        返回：
            new_masks (torch.Tensor)：移除小区域后的掩码，形状为 (N, H, W)。
            keep (list[int])：NMS 后保留的掩码索引，用于筛选对应边界框。

        示例：
            >>> masks = torch.rand(5, 640, 640) > 0.5  # 5 random binary masks
            >>> new_masks, keep = remove_small_regions(masks, min_area=100, nms_thresh=0.7)
            >>> print(f"Original masks: {masks.shape}, Processed masks: {new_masks.shape}")
            >>> print(f"Indices of kept masks: {keep}")
        """
        import torchvision  # scope for faster 'import ultralytics'

        if masks.shape[0] == 0:
            return masks

        # 过滤较小的非连通区域和孔洞
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # 将发生变化的掩码分数设为 0，未变化的掩码分数设为 1，使 NMS 优先保留无需后处理的掩码
            scores.append(float(unchanged))

        # 重新计算边界框并移除新产生的重复项
        new_masks = torch.cat(new_masks, dim=0)
        # batched_mask_to_box 要求布尔掩码；如果使用 uint8，它会返回全零边界框，使下面的 NMS 去重失效
        boxes = batched_mask_to_box(new_masks.bool())
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        dst_shape=None,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """使用 SAM 模型对给定图像特征执行提示预处理和推理。

        参数：
            features (torch.Tensor | dict[str, Any]): Extracted image features from the SAM/SAM2 model image encoder.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            dst_shape (tuple[int, int] | None): The target shape (height, width) for the prompts. If None, defaults to
                (imgsz, imgsz).
            bboxes (np.ndarray | list[list[float]] | None): Bounding boxes in xyxy format with shape (N, 4).
            points (np.ndarray | list[list[float]] | None): Points indicating object locations with shape (N, 2), in
                pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N, ).
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        返回：
            pred_masks (torch.Tensor): The output masks in shape (C, H, W), where C is the number of generated masks.
            pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 6), where N is the number of boxes.
                Each box is in xyxy format with additional columns for score and class.

        注意：
            - 在 SAM 上执行时，输入特征是形状为 (B, C, H, W) 的 torch.Tensor；在 SAM2 上执行时，则是 dict[str, Any]。
        """
        dst_shape = dst_shape or (self.args.imgsz, self.args.imgsz)
        prompts = self._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        pred_masks, pred_scores = self._inference_features(features, *prompts, multimask_output)
        if pred_masks.shape[0] == 0:
            pred_masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = ops.scale_masks(pred_masks[None].float(), src_shape, padding=False)[0]
            pred_masks = pred_masks > self.model.mask_threshold  # 转换为 bool
            pred_bboxes = batched_mask_to_box(pred_masks)
            # 注意：SAM 模型不会返回 cls 信息，此处的 `cls` 仅作为接口一致性的占位符。
            cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
        return pred_masks, pred_bboxes


class SAM2Predictor(Predictor):
    """使用 Segment Anything Model 2 架构执行高级图像分割的 SAM2Predictor 类。

    此类继承基础 Predictor，为图像分割任务实现 SAM2 特有的功能。
    它提供模型初始化、特征提取和基于提示的推理方法。

    属性：
        _bb_feat_sizes (list[tuple]): Feature sizes for different backbone levels.
        model (torch.nn.Module): The loaded SAM2 model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        features (dict): Cached image features for efficient inference.
        segment_all (bool): Flag to indicate if all segments should be predicted.
        prompts (dict[str, Any]): Dictionary to store various types of prompts for inference.

    方法：
        get_model: Retrieve and initialize the SAM2 model.
        prompt_inference: Perform image segmentation inference based on various prompts.
        set_image: Preprocess and set a single image for inference.
        get_im_features: Extract and process image features using SAM2's image encoder.

    示例：
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> result = predictor(bboxes=bboxes)[0]
        >>> print(f"Predicted {len(result.masks)} masks with average score {result.boxes.conf.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]
    stride = 16

    def get_model(self):
        """获取并初始化用于图像分割任务的 Segment Anything Model 2（SAM2）。"""
        from .build import build_sam  # slow import

        return build_sam(self.args.model)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """根据目标尺寸准备并变换输入提示。

        参数：
            dst_shape (tuple[int, int]): 提示的目标尺寸（高度、宽度）。
            src_shape (tuple[int, int]): 输入图像的源尺寸（高度、宽度）。
            bboxes (np.ndarray | list | None): 形状为 (N, 4) 的 XYXY 格式边界框。
            points (np.ndarray | list | None): 表示目标位置的点，形状为 (N, 2) 或 (N, num_points, 2)，单位为像素。
            labels (np.ndarray | list | None): 形状为 (N,) 或 (N, num_points) 的点提示标签，1 表示前景，0 表示背景。
            masks (list | np.ndarray | None): 目标掩码，每个掩码均为二维数组。

        返回：
            points (torch.Tensor | None): 变换后的点。
            labels (torch.Tensor | None): 变换后的标签。
            masks (torch.Tensor | None): 变换后的掩码。

        异常：
            AssertionError: 提供 labels 时，点数与标签数不一致。
        """
        bboxes, points, labels, masks = super()._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(bboxes.shape[0], -1)
            # 注意：将 "boxes" 和 "points" 合并为单个 "points" 输入，
            # 并将边界框添加到开头，再传递给 model.sam_prompt_encoder
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        return points, labels, masks

    def setup_source(self, source):
        """为 SAM2 推理设置数据源和图像尺寸。"""
        super().setup_source(source)
        self._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]

    def get_im_features(self, im):
        """使用 SAM 图像编码器提取特征，供后续处理使用。"""
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats, self._bb_feat_sizes)
        ]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _inference_features(
        self,
        features,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """使用 SAM2 模型对图像特征执行推理。

        参数：
            features (torch.Tensor | dict[str, Any])：SAM2 图像编码器提取的图像特征，形状为 (B, C, H, W)。
                也可以是字典，其中 `image_embed` 为形状 (B, C, H, W) 的张量，`high_res_feats` 为骨干网络的高分辨率特征图列表。
            points (np.ndarray | list[list[float]] | None)：目标位置点，形状为 (N, 2)，单位为像素。
            labels (np.ndarray | list[int] | None)：点提示标签，形状为 (N)，1 表示前景，0 表示背景。
            masks (list[np.ndarray] | np.ndarray | None)：目标掩码，每个掩码均为二维数组。
            multimask_output (bool)：是否为有歧义的提示返回多个掩码。
            img_idx (int)：要处理的图像在批次中的索引。

        返回：
            pred_masks (torch.Tensor)：输出掩码，形状为 (C, H, W)，其中 C 为生成的掩码数量。
            pred_scores (torch.Tensor)：每个掩码的质量分数，长度为 C。
        """
        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # 预测掩码
        batched_mode = points is not None and points[0].shape[0] > 1  # 多对象预测
        high_res_features = None
        if isinstance(features, dict):
            high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            features = features["image_embed"][[img_idx]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可能为 1 或 3，取决于 `multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)


class SAM2VideoPredictor(SAM2Predictor):
    """处理用户与视频交互并管理推理状态的 SAM2VideoPredictor。

    此类扩展 SAM2Predictor 以支持视频处理，并维护推理操作的状态。它包含管理非重叠掩码以及清除非条件输入记忆的配置。

    属性：
        inference_state (dict): 保存当前推理操作状态的字典。
        non_overlap_masks (bool): 指示掩码是否应互不重叠的标志。
        clear_non_cond_mem_around_input (bool): 控制是否清除输入周围非条件记忆的标志。
        clear_non_cond_mem_for_multi_obj (bool): 控制多目标场景下是否清除非条件记忆的标志。

    方法：
        get_model: 获取并配置启用二值化的模型。
        inference: 根据给定输入提示执行图像分割推理。
        postprocess: 对预测结果执行后处理，必要时应用非重叠约束。
        add_new_prompts: 为指定目标 ID 在指定帧上添加新的点或掩码。
        propagate_in_video_preflight: 在跟踪前准备推理状态并整合临时输出。
        init_state: 为预测器初始化推理状态。
        get_im_features: 使用 SAM2 图像编码器提取图像特征，供后续分割任务使用。

    示例：
        >>> predictor = SAM2VideoPredictor(cfg=DEFAULT_CFG)
        >>> predictor.set_image("path/to/video_frame.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)

    注意：
        当前实现定义了 `fill_hole_area` 属性，但尚未使用。
    """

    # fill_hole_area = 8  # 当前未使用

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置和可选覆盖项初始化预测器。

        此构造函数使用给定配置初始化 SAM2VideoPredictor，应用指定覆盖项，并设置推理状态及控制预测器行为的标志。

        参数：
            cfg (dict): 包含默认设置的配置字典。
            overrides (dict | None): 用于覆盖默认配置的值字典。
            _callbacks (dict | None): 用于自定义行为的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.clear_non_cond_mem = True  # 是否定期清除非条件记忆

    def setup_source(self, source):
        """设置数据源，并在任何预测回调运行前构建视频推理状态。"""
        super().setup_source(source)
        if self.dataset is not None and self.dataset.mode == "video":
            self.init_state(self)

    def get_model(self):
        """获取并配置启用二值化的模型。

        注意：
            此方法覆盖基类实现，将 binarize 标志设置为 True。
        """
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        """使用当前加载的图像，根据给定输入提示执行图像分割推理。
        此方法利用 SAM（Segment Anything Model）的图像编码器、提示编码器和掩码解码器架构，
        支持实时且可提示的分割任务。

        参数：
            im (torch.Tensor): 预处理后的输入图像张量，形状为 (N, C, H, W)。
            bboxes (np.ndarray | list, optional): 形状为 (N, 4) 的 XYXY 格式边界框。
            points (np.ndarray | list, optional): 表示目标位置的点，形状为 (N, 2)，单位为像素。
            labels (np.ndarray | list, optional): 点提示标签，形状为 (N,)。1 表示前景，0 表示背景。
            masks (np.ndarray, optional): 之前预测得到的低分辨率掩码，形状为 (N,H,W)；对于 SAM，H=W=256。

        返回：
            pred_masks (torch.Tensor): 形状为 CxHxW 的输出掩码，其中 C 为生成的掩码数量。
            pred_scores (torch.Tensor): 长度为 C 的数组，包含每个掩码的预测质量分数。
        """
        # 如果 self.prompts 中保存了提示，则使用其覆盖传入提示
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        frame = self.dataset.frame
        self.inference_state["im"] = im
        output_dict = self.inference_state["output_dict"]
        if len(output_dict["cond_frame_outputs"]) == 0:  # 初始化提示
            points, labels, masks = self._prepare_prompts(
                im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks
            )
            if points is not None:
                for i in range(len(points)):
                    self.add_new_prompts(obj_id=i, points=points[[i]], labels=labels[[i]], frame_idx=frame)
            elif masks is not None:
                for i in range(len(masks)):
                    self.add_new_prompts(obj_id=i, masks=masks[[i]], frame_idx=frame)
        self.propagate_in_video_preflight()

        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # 清除相邻帧的非条件记忆
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame)
        # 创建每个目标输出的切片，供跟踪后与各目标进行后续交互
        self._add_output_per_object(frame, current_out, storage_key)
        self.inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        pred_masks = pred_masks[(pred_masks > self.model.mask_threshold).sum((1, 2)) > 0]  # 过滤空掩码

        return pred_masks, torch.ones(pred_masks.shape[0], dtype=pred_masks.dtype, device=pred_masks.device)

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果执行后处理，必要时应用非重叠约束。

        当 `non_overlap_masks` 标志为 True 时，此方法对预测掩码应用非重叠约束，确保掩码之间不重叠，
        这对某些应用很有用。

        参数：
            preds (tuple[torch.Tensor, torch.Tensor]): 模型输出的预测掩码和分数。
            img (torch.Tensor): 处理后的图像张量。
            orig_imgs (list[np.ndarray]): 处理前的原始图像。

        返回：
            (list): 后处理后的预测结果。

        注意：
            如果 `non_overlap_masks` 为 True，则应用约束以确保掩码互不重叠。
        """
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results

    @smart_inference_mode()
    def add_new_prompts(
        self,
        obj_id,
        points=None,
        labels=None,
        masks=None,
        frame_idx=0,
        inference_state: dict[str, Any] | None = None,
    ):
        """为指定目标 ID 在指定帧上添加新的点或掩码。

        此方法使用指定目标和帧索引的新提示（点或掩码）更新推理状态，确保每次调用只提供点或掩码之一，
        并相应更新内部状态。同时根据给定提示和已有状态生成新的分割结果。

        参数：
            obj_id (int): 提示所关联的目标 ID。
            points (torch.Tensor, optional): 感兴趣点的坐标。
            labels (torch.Tensor, optional): 点对应的标签。
            masks (torch.Tensor, optional): 目标的二值掩码。
            frame_idx (int, optional): 应用提示的帧索引。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            pred_masks (torch.Tensor): 展平后的预测掩码。
            pred_scores (torch.Tensor): 用于表示目标数量的全 1 张量。

        异常：
            AssertionError: 同时提供或同时未提供 `masks` 和 `points` 时抛出。

        注意：
            - 每次调用只能添加一种提示（点或掩码）。
            - 如果某帧首次被跟踪，则将其视为初始条件帧。
            - 此方法负责整合输出，并将掩码调整为原始视频分辨率。
        """
        inference_state = inference_state or self.inference_state
        assert (masks is None) ^ (points is None), "'masks' and 'points' prompts are not compatible with each other."
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        inference_state[pop_key][obj_idx].pop(frame_idx, None)
        # 如果该帧此前未被跟踪，则将其视为初始条件帧，即不使用其他帧记忆，
        # 仅使用输入点在当前帧生成分割结果（与 SAM 类似）。否则，输入点用于修正已跟踪掩码。
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # 如果是初始条件帧，或模型将所有接收点击/掩码的帧视为条件帧，则添加到条件输出。
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 获取该目标之前预测的掩码 logits，并与新点击一起输入 SAM 掩码解码器。
        prev_sam_mask_logits = None
        # 首先查找临时输出字典，其中包含最新输出；如果未找到，再查找条件帧和非条件帧输出。
        if point_inputs is not None:
            prev_out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )

            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(
                    device=self.device, non_blocking=self.device.type == "cuda"
                )
                # 限制 prev_sam_mask_logits 的数值范围，避免罕见的数值问题。
                prev_sam_mask_logits.clamp_(-32.0, 32.0)
        current_out = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # 在单个目标的切片上运行
            frame_idx=frame_idx,
            batch_size=1,  # 在单个目标的切片上运行
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=masks,
            reverse=False,
            # 添加点击或掩码时跳过记忆编码器。在 `propagate_in_video` 开始时（用户完成点击后）再执行记忆编码器，
            # 从而可以在将所有目标编码到记忆中之前应用非重叠约束。
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            inference_state=inference_state,
        )
        # 将输出添加到输出字典（供后续作为记忆使用）
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # 将输出掩码调整为原始视频分辨率
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            inference_state=inference_state,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self, inference_state: dict[str, Any] | None = None):
        """在跟踪前准备推理状态并整合临时输出。

        此方法标记跟踪开始，在会话重置前禁止添加新目标。它整合 `temp_output_dict_per_obj` 中的临时输出并合并到
        `output_dict`，同时清除输入帧周围的非条件记忆，并确保状态与给定输入一致。

        参数：
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        # 跟踪已开始，在会话重置前不允许添加新目标。
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])

        # 将每个目标的临时输出整合到“temp_output_dict_per_obj”中，
        # 并添加到“output_dict”。
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" 保存已添加整合临时输出的帧索引（可能来自本次调用或之前对
        # `propagate_in_video_preflight` 的调用）。
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in (False, True):
            # 分别整合条件帧和非条件帧的临时输出
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # 找出包含任意目标临时输出的所有帧（这些通常是刚刚通过 `add_new_points` 或 `add_new_mask`
            # 接收掩码点击输入的帧）
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # 整合该帧所有目标的临时输出
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True, inference_state=inference_state
                )
                # 将其合并到 "output_dict"，并创建每个目标对应的切片
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key, inference_state=inference_state)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    # 清除相邻帧的非条件记忆
                    self._clear_non_cond_mem_around_input(frame_idx)

            # 清除 `temp_output_dict_per_obj` 中的临时输出
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # 边界情况：如果向 "cond_frame_outputs" 添加了输出，则删除同一帧在
        # "non_cond_frame_outputs" 中已有的输出
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # 确保 "consolidated_frame_inds" 中的帧索引恰好对应包含点或掩码输入的帧，
        # 在正确的工作流中应始终满足这一条件。
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @staticmethod
    def init_state(predictor):
        """为预测器初始化推理状态。

        此函数设置视频推理所需的初始状态，包括初始化用于保存跟踪输入、输出及其他相关元数据的
        多个字典和有序字典。

        参数：
            predictor (SAM2VideoPredictor): 要为其初始化状态的预测器对象。
        """
        if len(predictor.inference_state) > 0:  # 表示已经初始化
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        predictor.inference_state = predictor._init_state(predictor.dataset.frames)

    @staticmethod
    def _init_state(num_frames):
        """初始化推理状态。

        此函数设置视频推理所需的初始状态，包括初始化用于保存跟踪输入、输出及其他相关元数据的
        多个字典和有序字典。

        参数：
            num_frames (int): 视频中的帧数。
        """
        inference_state = {
            "num_frames": num_frames,  # TODO：确认是否可以移除此字段
            "point_inputs_per_obj": {},  # 每个目标在各帧上的点输入
            "mask_inputs_per_obj": {},  # 每个目标在各帧上的掩码输入
            "constants": {},  # 跨帧不变的值（因此只需保存一份）
            # 客户端目标 ID 与模型目标索引之间的映射
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # 保存模型在各帧上的跟踪结果和状态
            "output_dict": {
                "cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
                "non_cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
            },
            # 每个目标跟踪结果的切片（视图），与 "output_dict" 共享内存
            "output_dict_per_obj": {},
            # 保存用户在帧上添加点击或掩码时产生的新输出的临时存储
            #（传播开始前会将其合并到 "output_dict"）
            "temp_output_dict_per_obj": {},
            # 已因点击或掩码输入而保存整合输出的帧
            #（跟踪过程中直接使用这些整合输出）
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),  # 保存帧索引的集合
                "non_cond_frame_outputs": set(),  # 保存帧索引的集合
            },
            # 每个跟踪帧的元数据（例如跟踪方向）
            "tracking_has_started": False,
            "frames_already_tracked": [],
        }
        return inference_state

    def get_im_features(self, im, batch=1):
        """使用 SAM2 图像编码器提取并处理图像特征，供后续分割任务使用。

        参数：
            im (torch.Tensor): 输入图像张量。
            batch (int, optional): 存在多个提示时用于扩展特征的批次大小。

        返回：
            vis_feats (torch.Tensor): 从图像中提取的视觉特征。
            vis_pos_embed (torch.Tensor): 视觉特征的位置嵌入。
            feat_sizes (list[tuple]): 包含提取特征尺寸的列表。

        注意：
            - 如果 `batch` 大于 1，则扩展特征以匹配批次大小。
            - 此方法调用模型的 `_prepare_backbone_features` 方法准备骨干网络特征。
        """
        # 检查是否存在预先计算的骨干网络输出
        backbone_out = getattr(self, "backbone_out", None)
        if backbone_out is None:
            backbone_out = self.model.forward_image(im)
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out, batch=batch)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id, inference_state: dict[str, Any] | None = None):
        """将客户端目标 ID 映射到模型目标索引。

        参数：
            obj_id (int): 客户端提供的目标唯一标识符。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            (int): 目标在模型侧的索引。

        异常：
            RuntimeError: 跟踪开始后尝试添加新目标时抛出。

        注意：
            - 更新或读取 `inference_state` 中保存的目标 ID 与索引映射。
            - 确保只能在跟踪开始前添加新目标。
            - 维护 ID 与索引之间的双向映射（`obj_id_to_idx` 和 `obj_idx_to_id`）。
            - 为新目标初始化用于保存输入和输出的附加数据结构。
        """
        inference_state = inference_state or self.inference_state
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # 这是此前未发送到服务端的新目标 ID。只允许在跟踪开始前添加新目标。
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # 获取下一个目标槽位
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # 为该目标设置输入和输出结构
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        inference_state: dict[str, Any] | None = None,
    ):
        """基于当前输入和之前的记忆信息，在单帧上执行跟踪。

        参数：
            output_dict (dict): 包含跟踪过程输出状态的字典。
            frame_idx (int): 当前帧索引。
            batch_size (int): 处理当前帧时的批次大小。
            is_init_cond_frame (bool): 当前帧是否为初始条件帧。
            point_inputs (dict | None): 输入点及其标签。
            mask_inputs (torch.Tensor | None): 输入二值掩码。
            reverse (bool): 是否按逆序执行跟踪。
            run_mem_encoder (bool): 是否执行记忆编码器。
            prev_sam_mask_logits (torch.Tensor | None): 当前目标之前的掩码 logits。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            (dict): 包含跟踪步骤输出的字典，包括更新后的特征和预测结果。

        异常：
            AssertionError: 同时提供 `point_inputs` 和 `mask_inputs` 时抛出。

        注意：
            - 此方法假设 `point_inputs` 和 `mask_inputs` 互斥。
            - 此方法通过 `get_im_features` 获取图像特征。
            - 假设 `maskmem_pos_enc` 跨帧不变，因此只保存一份副本。
            - `fill_holes_in_mask_scores` 函数当前因需要 CUDA 扩展而不受支持，相关代码已注释。
        """
        inference_state = inference_state or self.inference_state
        # 获取正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            inference_state["im"], batch_size
        )

        # 同一帧不能同时提供点输入和掩码输入
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.to(
                dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
            )
        # 注意：不支持 `fill_holes_in_mask_scores` 函数，因为它需要 CUDA 扩展来填补预测掩码中的孔洞
        # 如果 self.fill_hole_area > 0：
        #     pred_masks = current_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda")
        #     pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)

        # "maskmem_pos_enc" 跨帧相同，因此只需保存一份副本
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(current_out["maskmem_pos_enc"], inference_state)
        return current_out

    def _get_maskmem_pos_enc(self, out_maskmem_pos_enc, inference_state: dict[str, Any] | None = None):
        """缓存并管理跨帧、跨目标的掩码记忆位置编码。

        此方法缓存掩码记忆的位置编码 (`maskmem_pos_enc`) 以优化存储。该编码跨帧和目标保持不变，
        因而可以减少推理过程中保存的冗余信息。若位置编码尚未缓存，则缓存所提供编码的一份切片；
        当批次大小大于 1 时，将缓存的位置编码扩展到当前批次大小。

        参数：
            out_maskmem_pos_enc (list[torch.Tensor] | None): 掩码记忆的位置编码，应为张量列表或 None。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            (list[torch.Tensor]): 缓存或扩展后的掩码记忆位置编码。

        注意：
            - 假设 `out_maskmem_pos_enc` 为张量列表或 None。
            - 由于编码跨目标相同，只缓存单个目标的切片。
            - 检查会话常量中是否已经缓存位置编码。
            - 批次大小大于 1 时，将缓存编码扩展到对应批次大小。
        """
        inference_state = inference_state or self.inference_state
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" 应为张量列表或 None
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # 由于编码跨目标相同，只取一个目标的切片
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # 将缓存的 maskmem_pos_enc 扩展到实际批次大小
            batch_size = out_maskmem_pos_enc[0].shape[0]
            if batch_size > 1:
                out_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return out_maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        run_mem_encoder=False,
        inference_state: dict[str, Any] | None = None,
    ):
        """将每个目标的临时输出整合为包含所有目标的单个输出。

        此方法将指定帧上每个目标的临时输出合并为统一输出。缺失目标会从主输出字典中补齐；
        如果主输出中也不存在，则保留占位数据。可选地，在对目标分数应用非重叠约束后重新运行记忆编码器。

        参数：
            frame_idx (int): 要整合输出的帧索引。
            is_cond (bool, optional): 指示该帧是否为条件帧。
            run_mem_encoder (bool, optional): 指示整合输出后是否运行记忆编码器。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            (dict): 包含所有目标合并结果的整合输出字典。

        注意：
            - 使用占位值初始化缺失目标的整合输出。
            - 同时在临时输出字典和主输出字典中查找结果。
            - `run_mem_encoder` 为 True 时应用非重叠约束，并重新运行记忆编码器。
            - 仅当 `run_mem_encoder` 为 True 时才填充 `maskmem_features` 和 `maskmem_pos_enc`。
        """
        inference_state = inference_state or self.inference_state
        batch_size = len(inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 初始化 `consolidated_out`。对目标分数应用非重叠约束并重新运行记忆编码器时，
        # 才会补充 "maskmem_features" 和 "maskmem_pos_enc"。"pred_masks" 预先填充较大的
        # 负值（NO_OBJ_SCORE），用于表示缺失目标。
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                # size=(batch_size, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                size=(batch_size, 1, *self._bb_feat_sizes[0]),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # object_score_logits 默认设为 10.0，即假设目标存在且 sigmoid(10)=1，
                # 与 `MaskDecoder` 的 `predict_masks` 中一致
                fill_value=10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                # 如果该目标未出现在当前帧的 "temp_output_dict_per_obj" 中，
                # 则回退到 "output_dict_per_obj" 查找之前的输出。
                # 同时在 "output_dict_per_obj" 的 "cond_frame_outputs" 和 "non_cond_frame_outputs" 中查找该目标的输出。
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            # 如果目标在 "output_dict_per_obj" 中也不存在，则跳过它，保留默认掩码分数
            #（即上面的 NO_OBJ_SCORE 占位值），并将其目标指针设为虚拟指针。
            if out is None:
                # 为当前帧没有输入或跟踪结果的目标填充虚拟目标指针
                #（仅在 `run_mem_encoder=True`，即需要构建跟踪记忆时执行）。
                if run_mem_encoder:
                    # 使用基于空掩码的虚拟指针填充目标指针
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = self._get_empty_mask_ptr(frame_idx)
                continue
        # 将目标临时输出掩码添加到整合输出掩码中
            consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = out["pred_masks"]
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # 可选：对整合分数应用非重叠约束，并重新运行记忆编码器
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            consolidated_out["maskmem_features"], consolidated_out["maskmem_pos_enc"] = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # 这些帧是用户进行交互的帧
                object_score_logits=consolidated_out["object_score_logits"],
                inference_state=inference_state,
            )

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """根据当前帧的空掩码获取虚拟目标指针。

        参数：
            frame_idx (int): 要生成虚拟目标指针的当前帧索引。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            (torch.Tensor): 根据空掩码生成的虚拟目标指针张量。
        """
        inference_state = inference_state or self.inference_state
        # 获取正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(inference_state["im"])

        # 将空掩码和上述图像特征输入模型，获取虚拟目标指针
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            # 包含单个目标的虚拟（空）掩码
            mask_inputs=torch.zeros((1, 1, *self.imgsz), dtype=self.torch_dtype, device=self.device),
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(
        self,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        inference_state: dict[str, Any] | None = None,
    ):
        """对掩码运行记忆编码器。

        通常在对目标分数应用非重叠约束后调用此方法。由于分数发生了变化，还需要使用记忆编码器重新计算记忆信息。

        参数：
            batch_size (int): 处理当前帧时的批次大小。
            high_res_masks (torch.Tensor): 用于计算记忆信息的高分辨率掩码。
            object_score_logits (torch.Tensor): 表示目标分数的 logits。
            is_mask_from_pts (bool): 指示掩码是否来自点交互。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。

        返回：
            maskmem_features (torch.Tensor): 编码后的掩码特征。
            maskmem_pos_enc (torch.Tensor): 位置编码。
        """
        inference_state = inference_state or self.inference_state
        # 获取正确的图像特征
        current_vision_feats, _, feat_sizes = self.get_im_features(inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )

        # "maskmem_pos_enc" 跨帧相同，因此只需保存一份副本
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc, inference_state)
        return maskmem_features.to(
            dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
        ), maskmem_pos_enc

    def _add_output_per_object(
        self, frame_idx, current_out, storage_key, inference_state: dict[str, Any] | None = None
    ):
        """将多目标输出切分为每个目标的输出切片，并添加到 Output_Dict_Per_Obj。

        生成的切片与原输出共享相同的张量存储。

        参数：
            frame_idx (int): 当前帧索引。
            current_out (dict): 包含多目标输出的当前输出字典。
            storage_key (str): 在每目标输出字典中保存输出时使用的键。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """删除输入帧周围的非条件记忆。

        用户提供修正点击时，相邻帧的非条件记忆可能仍包含过时的目标外观信息，从而干扰模型。
        此方法清除交互帧周围的非条件记忆，避免同时向模型提供目标的新旧信息。

        参数：
            frame_idx (int): 发生用户交互的当前帧索引。
            inference_state (dict[str, Any], optional): 当前推理状态。为 None 时使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    @smart_inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False):
        """从跟踪状态中删除目标 ID。strict 为 True 时检查目标 ID 是否存在，不存在则抛出错误。"""
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        # 检查要删除的 object_id 是否存在，必要时抛出错误
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"]
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # 如果这是唯一剩余的目标 ID，直接重置状态
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"]

        # 删除该目标 ID 后仍有其他目标。此时需要从推理状态张量中删除该目标的存储。
        # 步骤 0：清除该目标 ID 在包含点或掩码输入的帧上的输入
        #（注意，此步骤可能将条件帧降级为非条件帧，因此不可省略）
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(inference_state["point_inputs_per_obj"][old_obj_idx_to_rm])
        obj_input_frames_inds.update(inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm])
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(inference_state, frame_idx, obj_id)

        # 步骤 1：更新目标 ID 映射（必须在步骤 0 后执行，因为步骤 0 仍需要推理状态中的旧映射）
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # 构建新映射
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # 步骤 2：对于每目标张量存储，调整字典键中的 obj_idx。
        #（注意，"consolidated_frame_inds" 已在步骤 0 中处理，此处无需更新）
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # 步骤 3：对于打包张量存储，根据剩余 ID 建立索引并重建每目标切片。
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]]
                # "maskmem_pos_enc" 跨帧相同，因此只需保存一份副本
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(out["maskmem_pos_enc"], inference_state)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][remain_old_obj_inds]
                # 同时更新每目标切片
                self._add_output_per_object(frame_idx, out, storage_key, inference_state=inference_state)

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        return inference_state["obj_ids"]

    @smart_inference_mode()
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id):
        """删除指定目标在指定帧上的所有点或掩码输入。"""
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        # 清除指定帧上的条件信息
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # 检查该帧是否仍有输入
        batch_size = len(inference_state["obj_idx_to_id"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # 如果该帧不再包含任何目标的输入，则进一步清除其条件帧状态
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # 删除该帧的条件输出（可能将其降级为非条件输出）
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # 该帧不再接收输入，因此不再是条件帧；将其输出（如果存在）“降级”为非条件帧输出。
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # 对每个目标的切片输出执行相同操作。
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # 如果所有条件帧都已删除，则同时清除跟踪输出
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

    @smart_inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """删除整个视频所有帧上的输入点或掩码。"""
        self._reset_tracking_results(inference_state)
        # 删除所有目标 ID
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    @staticmethod
    def _reset_tracking_results(inference_state):
        """重置整个视频中的所有跟踪输入和结果。"""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _prune_non_cond_memory(self, frame_idx, inference_state=None):
        """清理较早的非条件帧，以限制内存使用。"""
        if not self.clear_non_cond_mem:
            return
        inference_state = inference_state or self.inference_state

        # 确定窗口大小
        min_frame = frame_idx - self.model.num_maskmem * self.model.memory_temporal_stride_for_eval
        output_dict = inference_state["output_dict"]

        # 清理全局 non_cond_frame_outputs
        for f in [k for k in output_dict["non_cond_frame_outputs"] if k < min_frame]:
            output_dict["non_cond_frame_outputs"].pop(f, None)

        # 清理每个目标的 non_cond_frame_outputs
        for obj_output_dict in inference_state.get("output_dict_per_obj", {}).values():
            for f in [k for k in obj_output_dict["non_cond_frame_outputs"] if k < min_frame]:
                obj_output_dict["non_cond_frame_outputs"].pop(f, None)


class SAM2DynamicInteractivePredictor(SAM2Predictor):
    """SAM2DynamicInteractivePredictor 扩展 SAM2Predictor，支持与视频帧或图像序列进行动态交互。

    属性：
        memory_bank (list): OrderedDict: Stores the states of each image with prompts.
        obj_idx_set (set): A set to keep track of the object indices that have been added.
        obj_id_to_idx (OrderedDict): Maps object IDs to their corresponding indices.
        obj_idx_to_id (OrderedDict): Maps object indices to their corresponding IDs.

    方法：
        get_model: Retrieves and configures the model with binarization enabled.
        inference: Performs inference on a single image with optional prompts and object IDs.
        postprocess: Post-processes the predictions to apply non-overlapping constraints if required.
        update_memory: Append the imgState to the memory_bank and update the memory for the model.
        track_step: Tracking step for the current image state to predict masks.
        get_maskmem_enc: Get memory and positional encoding from the memory bank.

    示例：
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor(source=support_img1, bboxes=bboxes1, obj_ids=labels1, update_memory=True)
            >>> results1 = predictor(source=query_img1)
            >>> predictor(source=support_img2, bboxes=bboxes2, obj_ids=labels2, update_memory=True)
            >>> results2 = predictor(source=query_img2)
    """

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        max_obj_num: int = 3,
        _callbacks: dict | None = None,
    ) -> None:
        """使用配置和可选覆盖项初始化预测器。

        此构造函数使用给定配置初始化 SAM2DynamicInteractivePredictor，并应用指定的覆盖项。

        参数：
            cfg (Any): 包含默认设置的配置字典。
            overrides (dict[str, Any] | None): 用于覆盖默认配置的值字典。
            max_obj_num (int): 要跟踪的最大目标数，默认为 3，用于保持模型特征尺寸固定。
            _callbacks (dict | None): 用于自定义行为的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.non_overlap_masks = True

        # 初始化记忆库以保存图像状态
        # 注意：后续可能需要使用字典以便更好地查询
        self.memory_bank = []

        # 初始化目标索引集合和映射
        self.obj_idx_set = set()
        self.obj_id_to_idx = self.obj_idx_to_id = OrderedDict(enumerate(range(max_obj_num)))
        self._max_obj_num = max_obj_num

    @smart_inference_mode()
    def inference(
        self,
        im: torch.Tensor | np.ndarray,
        bboxes: list[list[float]] | None = None,
        masks: torch.Tensor | np.ndarray | None = None,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        obj_ids: list[int] | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在单张图像上执行推理，可选提供边界框、掩码、点和目标 ID。
        支持两种模式：一种是在不更新记忆的情况下对单张图像推理，另一种是使用给定提示和目标 ID 更新记忆。
        update_memory 为 True 时使用给定提示和 obj_ids 更新记忆；为 False 时仅对给定图像推理，不更新记忆。

        参数：
            im (torch.Tensor | np.ndarray): 输入图像张量或 numpy 数组。
            bboxes (list[list[float]] | None): 可选的边界框列表，用于更新记忆。
            masks (torch.Tensor | np.ndarray | None): 可选的掩码，用于更新记忆。
            points (list[list[float]] | None): 可选的点列表，用于更新记忆，每个点为 [x, y]。
            labels (list[int] | None): 点提示的可选标签（大于 0 为正点击，0 为负点击）。
            obj_ids (list[int] | None): 与提示对应的可选目标 ID 列表。
            update_memory (bool): 是否使用新目标更新记忆的标志。

        返回：
            res_masks (torch.Tensor): 形状为 (C, H, W) 的输出掩码。
            object_score_logits (torch.Tensor): 每个掩码的质量分数。
        """
        self.get_im_features(im)
        points, labels, masks = self._prepare_prompts(
            dst_shape=self.imgsz,
            src_shape=self.batch[1][0].shape[:2],
            points=points,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
        )

        if update_memory:
            if isinstance(obj_ids, int):
                obj_ids = [obj_ids]
            assert obj_ids is not None, "obj_ids must be provided when update_memory is True"
            assert masks is not None or points is not None, (
                "bboxes, masks, or points must be provided when update_memory is True"
            )
            if points is None:  # 占位输入
                points = torch.zeros((len(obj_ids), 0, 2), dtype=self.torch_dtype, device=self.device)
                labels = torch.zeros((len(obj_ids), 0), dtype=torch.int32, device=self.device)
            if masks is not None:
                assert len(masks) == len(obj_ids), "masks and obj_ids must have the same length."
            assert len(points) == len(obj_ids), "points and obj_ids must have the same length."
            self.update_memory(obj_ids, points, labels, masks)

        current_out = self.track_step()
        pred_masks, pred_scores = current_out["pred_masks"], current_out["object_score_logits"]
        # 根据目标索引筛选掩码和 logits
        if len(self.obj_idx_set) == 0:
            raise RuntimeError("No objects have been added to the state. Please add objects before inference.")
        idx = list(self.obj_idx_set)  # 类别 ID
        pred_masks, pred_scores = pred_masks[idx], pred_scores[idx]
        # 原始分数范围为 [-32, 32]，目标分数大于 0 表示目标存在。
        # 将分数映射到 [0, 1]，使目标分数 logits 非负并可作为掩码使用。
        pred_scores = torch.clamp_(pred_scores / 32, min=0)
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def get_im_features(self, img: torch.Tensor | np.ndarray) -> None:
        """处理输入图像并提取特征，以初始化图像状态。

        参数：
            img (torch.Tensor | np.ndarray): 输入图像张量或 numpy 数组。
        """
        vis_feats, vis_pos_embed, feat_sizes = SAM2VideoPredictor.get_im_features(self, img, batch=self._max_obj_num)
        self.high_res_features = [
            feat.permute(1, 2, 0).view(*feat.shape[1:], *feat_size)
            for feat, feat_size in zip(vis_feats[:-1], feat_sizes[:-1])
        ]

        self.vision_feats = vis_feats
        self.vision_pos_embeds = vis_pos_embed
        self.feat_sizes = feat_sizes

    @smart_inference_mode()
    def update_memory(
        self,
        obj_ids: list[int] | None = None,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        """将 imgState 添加到 memory_bank，并更新模型记忆。

        参数：
            obj_ids (list[int]): 与提示对应的目标 ID 列表。
            points (torch.Tensor | None): 形状为 (B, N, 2) 的张量，表示 N 个目标的输入点。
            labels (torch.Tensor | None): 形状为 (B, N) 的张量，表示输入点的标签。
            masks (torch.Tensor | None): 可选的形状为 (N, H, W) 的张量，表示 N 个目标的输入掩码。
        """
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(self._max_obj_num, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(self._max_obj_num, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(self._max_obj_num, 1),
                # object_score_logits 默认应为 10.0，即假设目标存在且 sigmoid(10)=1，
                # 与 `MaskDecoder` 的 `predict_masks` 中一致
                fill_value=-32,  # 10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }

        for i, obj_id in enumerate(obj_ids):
            assert obj_id < self._max_obj_num
            obj_idx = self._obj_id_to_idx(int(obj_id))
            self.obj_idx_set.add(obj_idx)
            point, label = points[[i]], labels[[i]]
            mask = masks[[i]][None] if masks is not None else None
            # 当前仅支持边界框提示或掩码提示，因此断言至少提供其中一种输入。
            assert point is not None or mask is not None, "Either bbox, points or mask is required"
            out = self.track_step(obj_idx, point, label, mask)
            if out is not None:
                obj_mask = out["pred_masks"]
                assert obj_mask.shape[-2:] == consolidated_out["pred_masks"].shape[-2:], (
                    f"Expected mask shape {consolidated_out['pred_masks'].shape[-2:]} but got {obj_mask.shape[-2:]} for object {obj_idx}."
                )
                consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = obj_mask
                consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

                if "object_score_logits" in out:
                    consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out["object_score_logits"]

        high_res_masks = F.interpolate(
            consolidated_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda"),
            size=self.imgsz,
            mode="bilinear",
            align_corners=False,
        )

        if self.model.non_overlap_masks_for_mem_enc:
            high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=self.vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=consolidated_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        consolidated_out["maskmem_features"] = maskmem_features
        consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        self.memory_bank.append(consolidated_out)

    def _prepare_memory_conditioned_features(self, obj_idx: int | None) -> torch.Tensor:
        """为当前图像状态准备记忆条件特征。

        如果提供 ``obj_idx``，则为图像中指定的提示目标准备特征；如果 ``obj_idx`` 为 None，则为所有目标准备特征。
        没有可用记忆时，将无记忆嵌入添加到当前视觉特征；否则通过 Transformer 注意力机制使用之前帧的记忆
        为当前视觉特征提供条件。

        参数：
            obj_idx (int | None): 要准备特征的目标索引。

        返回：
            pix_feat_with_mem (torch.Tensor): 带有记忆条件的像素特征。
        """
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            # 对初始条件帧不使用之前的记忆进行编码。
            # 直接添加无记忆嵌入（而不是使用 Transformer 编码器）。
            pix_feat_with_mem = self.vision_feats[-1] + self.model.no_mem_embed
        else:
            # 对推理帧使用之前帧的记忆特征
            memory, memory_pos_embed = self.get_maskmem_enc()
            pix_feat_with_mem = self.model.memory_attention(
                curr=self.vision_feats[-1:],
                curr_pos=self.vision_pos_embeds[-1:],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=0,  # 目标指针令牌数
            )
        # 将输出从 (HW)BC 重塑为 BCHW
        return pix_feat_with_mem.permute(1, 2, 0).view(
            self._max_obj_num,
            self.model.memory_attention.d_model,
            *self.feat_sizes[-1],
        )

    def get_maskmem_enc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """从记忆库获取记忆和位置编码，用于为当前图像特征提供条件。"""
        to_cat_memory, to_cat_memory_pos_embed = [], []
        for consolidated_out in self.memory_bank:
            to_cat_memory.append(consolidated_out["maskmem_features"].flatten(2).permute(2, 0, 1))  # (H*W, B, C)
            maskmem_enc = consolidated_out["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self.model.maskmem_tpos_enc[self.model.num_maskmem - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        return memory, memory_pos_embed

    def _obj_id_to_idx(self, obj_id: int) -> int | None:
        """将客户端目标 ID 映射到模型目标索引。

        参数：
            obj_id (int): 客户端目标 ID。

        返回：
            (int | None): 模型目标索引；未找到时返回 None。
        """
        return self.obj_id_to_idx.get(obj_id, None)

    def track_step(
        self,
        obj_idx: int | None = None,
        point: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """针对当前图像状态执行跟踪步骤并预测掩码。

        此方法处理图像特征并运行 SAM 头来预测掩码。提供 obj_idx 时处理图像中指定提示目标的特征；
        obj_idx 为 None 时处理所有目标的特征。此方法同时支持不使用 SAM 的基于掩码输出，以及使用记忆条件
        特征的完整 SAM 处理流程。

        参数：
            obj_idx (int | None): 要预测掩码的目标索引。为 None 时处理所有目标。
            point (torch.Tensor | None): 形状为 (N, 2) 的感兴趣点坐标。
            label (torch.Tensor | None): 点对应的标签，其中 1 表示正点击，0 表示负点击。
            mask (torch.Tensor | None): 形状为 (H, W) 的目标掩码输入。

        返回：
            current_out (dict[str, Any]): 包含当前掩码预测和目标指针的输出字典。键包括 'point_inputs'、'mask_inputs'、
                'pred_masks'、'pred_masks_high_res'、'obj_ptr' 和 'object_score_logits'。
        """
        if mask is not None and self.model.use_mask_input_as_output_without_sam:
            # 当 use_mask_input_as_output_without_sam=True 时，直接输出掩码输入
            #（将其视为真实标注掩码），不使用 SAM 提示编码器和掩码解码器。
            pix_feat = self.vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.model.memory_attention.d_model, *self.feat_sizes[-1])
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._use_mask_as_output(mask)
        else:
            # 将视觉特征与记忆库中的之前记忆特征融合。
            pix_feat_with_mem = self._prepare_memory_conditioned_features(obj_idx)
            # 如果提供 ``obj_idx``（即正在添加提示），仅保留第一个特征图。
            pix_feat_with_mem = pix_feat_with_mem[:1] if obj_idx is not None else pix_feat_with_mem
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs={"point_coords": point, "point_labels": label} if obj_idx is not None else None,
                mask_inputs=mask,
                multimask_output=False,
                high_res_features=[feat[: pix_feat_with_mem.shape[0]] for feat in self.high_res_features],
            )
        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }


class SAM3Predictor(SAM2Predictor):
    """用于图像分割任务的 Segment Anything Model 3（SAM3）交互式预测器。"""

    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def setup_model(self, model=None, verbose=True):
        """设置 SAM3 模型，并为预处理配置合适的均值和标准差。"""
        super().setup_model(model, verbose)
        # 更新均值和标准差
        self.mean = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)

    def get_model(self):
        """获取并初始化用于图像分割任务的 Segment Anything Model 3（SAM3）。"""
        from .build_sam3 import build_interactive_sam3  # slow import

        return build_interactive_sam3(self.args.model, compile=self.args.compile)


class SAM3SemanticPredictor(SAM3Predictor):
    """用于图像分割任务的 Segment Anything Model 3（SAM3）预测器。"""

    def get_model(self):
        """获取并初始化用于图像分割任务的 Segment Anything Model 3（SAM3）。"""
        from .build_sam3 import build_sam3_image_model  # slow import

        return build_sam3_image_model(self.args.model, compile=self.args.compile)

    @smart_inference_mode()
    def get_im_features(self, im):
        """使用模型骨干网络提取图像特征。"""
        return self.model.backbone.forward_image(im)

    def pre_transform(self, im):
        """对输入图像执行预处理所需的初始变换。

        此方法会应用缩放等变换，为后续预处理准备图像。目前不支持批量推理，因此列表长度应为 1。

        参数：
            im (list[np.ndarray])：包含一张 HWC 格式 NumPy 图像的列表。

        返回：
            (list[np.ndarray])：包含变换后图像的列表。

        异常：
            AssertionError：输入列表包含多张图像时引发。

        示例：
            >>> predictor = SAM3SemanticPredictor()
            >>> predictor.imgsz = [1024, 1024]  # 通常由 setup_source() 设置
            >>> image = np.random.rand(480, 640, 3)  # 单张 HWC 图像
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False, scale_fill=True)  # SAM3 在此处固定使用这些设置
        return [letterbox(image=x) for x in im]

    def _prepare_geometric_prompts(self, src_shape, bboxes=None, labels=None):
        """通过将边界框和点归一化到目标形状来准备提示。"""
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            # 输入需要使用 xywh 格式
            bboxes = ops.xyxy2xywh(bboxes)
            bboxes[:, 0::2] /= src_shape[1]
            bboxes[:, 1::2] /= src_shape[0]
            # 用户未传入标签时，默认所有标签均为正样本。
            if labels is None:
                labels = np.ones(bboxes.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert bboxes.shape[-2] == labels.shape[-1], (
                f"Number of points {bboxes.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            bboxes = bboxes.view(-1, 1, 4)  # (N, 1, 4)
            labels = labels.view(-1, 1)  # (N, 1)
        return bboxes, labels

    def _inference_features(self, features, bboxes=None, labels=None, text: list[str] | None = None):
        """使用提取的特征执行推理，可选提供边界框和标签。"""
        # 注意：优先级为 bboxes > text > 预设类别
        nc = 1 if bboxes is not None else len(text) if text is not None else len(self.model.names)
        geometric_prompt = None
        if bboxes is not None:
            geometric_prompt = self._get_dummy_prompt(nc)
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
            if text is None:
                text = ["visual"]  # 未传入文本时，边界框需要此 `visual` 文本提示
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        outputs = self.model.forward_grounding(
            backbone_out=features,
            text_ids=torch.arange(nc, device=self.device, dtype=torch.long),
            geometric_prompt=geometric_prompt,
        )
        return outputs

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果执行后处理，并在需要时应用不重叠约束。"""
        import torchvision

        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)  # 类别偏移量
        nms_boxes = pred_boxes[:, :4] + c  # 边界框（按类别偏移）
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)  # NMS
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = (
                    F.interpolate(masks.float()[None], orig_img.shape[:2], mode="bilinear")[0]
                    > self.model.mask_threshold
                )
                boxes[..., [0, 2]] *= orig_img.shape[1]
                boxes[..., [1, 3]] *= orig_img.shape[0]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """使用可选提示对单张图像执行推理。"""
        bboxes = self.prompts.pop("bboxes", bboxes)
        labels = self.prompts.pop("labels", labels)
        text = self.prompts.pop("text", text)
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        return self._inference_features(features, *prompts, text=text)

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
    ):
        """使用 SAM 模型对给定图像特征执行提示预处理和推理。

        参数：
            features (dict[str, Any])：SAM3 模型图像编码器提取的图像特征。
            src_shape (tuple[int, int])：输入图像的源形状（高度、宽度）。
            bboxes (np.ndarray | list[list[float]] | None)：xyxy 格式的边界框，形状为 (N, 4)，单位为像素。
            labels (np.ndarray | list[int] | None)：形状为 (N) 的点提示标签。
            text (list[str] | None)：与类别对应的文本提示列表。

        返回：
            pred_masks (torch.Tensor)：形状为 (C, H, W) 的输出掩码，其中 C 为生成的掩码数量。
            pred_bboxes (torch.Tensor)：每个掩码对应的边界框，形状为 (N, 6)，其中 N 为边界框数量。
                每个边界框使用 xyxy 格式，并额外包含分数和类别列。

        注意：
            - 在 SAM 上执行时，输入 features 是形状为 (B, C, H, W) 的 torch.Tensor；在 SAM2 上执行时则为 dict[str, Any]。
        """
        import torchvision

        prompts = self._prepare_geometric_prompts(src_shape[:2], bboxes, labels)
        preds = self._inference_features(features, *prompts, text=text)
        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)  # classes
        nms_boxes = pred_boxes[:, :4] + c  # boxes (offset by class)
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)  # NMS
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        if pred_masks.shape[0] == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = (
                F.interpolate(pred_masks.float()[None], src_shape[:2], mode="bilinear")[0] > self.model.mask_threshold
            )
            pred_boxes[..., 0] *= src_shape[1]
            pred_boxes[..., 1] *= src_shape[0]
            pred_boxes[..., 2] *= src_shape[1]
            pred_boxes[..., 3] *= src_shape[0]
        return pred_masks, pred_boxes

    def reset_prompts(self):
        """重置预测器的提示。"""
        self.prompts = {}
        self.model.text_embeddings = {}

    def _get_dummy_prompt(self, num_prompts=1):
        """获取不包含边界框的空几何提示。"""
        # 为提高 ultralytics 导入速度，SAM3 几何模块中的 torchvision 算子按需导入。
        from .sam3.geometry_encoders import Prompt

        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
            box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
        )
        return geometric_prompt


class SAM3VideoPredictor(SAM2VideoPredictor, SAM3Predictor):
    """用于视频分割任务的 Segment Anything Model 3（SAM3）视频预测器。"""

    def propagate_in_video(self, inference_state, frame_idx):
        """使用当前加载图像，根据给定输入提示执行图像分割推理。
        此方法利用由图像编码器、提示编码器和掩码解码器组成的 SAM 架构，支持实时且可提示的分割任务。

        参数：
            inference_state (dict): The current state of inference, including input cues and previous outputs.
            frame_idx (int): The index of the current frame in the video sequence.
        """
        frame = frame_idx
        output_dict = inference_state["output_dict"]
        obj_ids = inference_state["obj_ids"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # 清除相邻帧的非条件记忆
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                inference_state=inference_state,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame, inference_state=inference_state)
        # 创建每个目标输出的切片，供跟踪后与各目标进行后续交互。
        self._add_output_per_object(frame, current_out, storage_key, inference_state=inference_state)
        inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        obj_scores = current_out["object_score_logits"]

        return obj_ids, pred_masks, obj_scores


class SAM3VideoSemanticPredictor(SAM3SemanticPredictor):
    """Segment Anything Model 3（SAM3）视频语义预测器。"""

    HIGH_CONF_THRESH = 0.8
    HIGH_IOU_THRESH = 0.8
    NO_OBJ_LOGIT = -10.0
    NEVER_OCCLUDED = -1
    ALWAYS_OCCLUDED = 100000

    UNCONFIRMED = 1  # 新添加的 masklet，尚未被任何检测结果确认
    CONFIRMED = 2  # confirmed by at least one detection
    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks: dict | None = None,
        # 检测输出的概率阈值：仅保留高于此阈值、可进入 NMS 和检测-轨迹匹配的检测结果
        score_threshold_detection=0.5,
        # 检测 NMS 的 IoU 阈值
        det_nms_thresh=0.0,
        # 检测-轨迹匹配的 IoU 阈值：检测结果与轨迹重叠超过此阈值时视为“匹配”；通常使用 0.1 这样的宽松阈值
        assoc_iou_thresh=0.5,
        # 检测-轨迹匹配的 IoU 阈值，用于确定掩码轨迹是否未与任何检测结果匹配；通常使用 0.5 这样的严格阈值
        trk_assoc_iou_thresh=0.5,
        # 将检测结果添加为新目标的概率阈值
        new_det_thresh=0.0,
        # 热启动参数：延迟 `hotstart_delay` 帧输出，并执行以下处理：
        # 1）根据 `hotstart_unmatch_thresh` 删除未与任何检测结果匹配的轨迹；
        # 2）根据 `hotstart_dup_thresh` 删除彼此重叠的轨迹。
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        init_trk_keep_alive=10,
        max_trk_keep_alive=10,
        min_trk_keep_alive=-4,
        # 基于最近遮挡信息抑制重叠目标的阈值
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=True,
        o2o_matching_masklets_enable=False,  # 启用匈牙利匹配以匹配现有掩码轨迹
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        # 所有 GPU 总共跟踪的最大目标数（不限制时设为 -1）
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        # 掩码轨迹确认状态（用于抑制未确认的掩码轨迹）
        masklet_confirmation_enable=True,
        # 掩码轨迹连续被检测并匹配 `masklet_confirmation_consecutive_det_thresh` 次后视为已确认
        masklet_confirmation_consecutive_det_thresh=3,
        # 边界框启发式参数
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        """使用配置和可选覆盖项初始化 SAM3VideoSemanticPredictor。"""
        super().__init__(cfg, overrides, _callbacks)
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        # 热启动参数
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = decrease_trk_keep_alive_for_empty_masklets
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self._dist_pg_cpu = None  # CPU 进程组（首次使用时延迟初始化）

        max_num_objects = 10000  # 实际上不限制
        num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = masklet_confirmation_consecutive_det_thresh
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

        # 构建 SAM3 跟踪器
        self.tracker = SAM3VideoPredictor(overrides=overrides)

        self.inference_state = {}

    @smart_inference_mode(False)  # the tracker model is built after super() returns, outside its decorator
    def setup_model(self, model=None, verbose=True):
        """设置 SAM3VideoSemanticPredictor 模型。"""
        super().setup_model(model, verbose)
        from .build_sam3 import build_interactive_sam3

        # 初始化不包含骨干网络的 SAM3 跟踪器模型（骨干网络由检测器处理）
        model = build_interactive_sam3(self.args.model, with_backbone=False)
        self.tracker.setup_model(model=model, verbose=False)

    def setup_source(self, source):
        """为 SAM3VideoSemanticPredictor 模型设置数据源。"""
        super().setup_source(source)
        self.tracker.imgsz = self.imgsz
        self.tracker.model.set_imgsz(self.imgsz)
        self.tracker._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]
        self.interpol_size = self.tracker.model.memory_encoder.mask_downsampler.interpol_size
        if self.dataset is not None and self.dataset.mode == "video":
            self.init_state(self)

    @staticmethod
    def init_state(predictor):
        """为预测器初始化推理状态。

        此函数设置视频推理所需的初始状态，包括初始化用于保存输入、输出及其他跟踪元数据的字典和有序字典。

        参数：
            predictor (SAM3VideoSemanticPredictor): 要为其初始化状态的预测器对象。
        """
        if len(predictor.inference_state) > 0:  # 表示已经初始化
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        num_frames = predictor.dataset.frames
        inference_state = {
            "num_frames": num_frames,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * num_frames,
        }
        predictor.inference_state = inference_state

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """在视频序列上执行推理，可选提供提示。"""
        frame = self.dataset.frame - 1  # 将帧索引调整为从 0 开始
        self.inference_state["im"] = im  # 后续帧仅传递图像
        if "text_ids" not in self.inference_state:  # 处理第一帧
            self.add_prompt(frame_idx=frame, text=text, bboxes=bboxes, labels=labels)
        return self._run_single_frame_inference(frame, reverse=False)

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果执行后处理，必要时应用非重叠约束。"""
        obj_id_to_mask = preds["obj_id_to_mask"]  # 低分辨率掩码
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        names = self.model.names if self.model.names != "visual" else {}
        if len(curr_obj_ids) == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 7), device=self.device)
        else:
            pred_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)
            pred_masks = (
                F.interpolate(pred_masks.float()[None], orig_imgs[0].shape[:2], mode="bilinear")[0]
                > self.model.mask_threshold
            )
            pred_ids = torch.tensor(curr_obj_ids, dtype=torch.int32, device=pred_masks.device)
            pred_scores = torch.tensor(
                [preds["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            pred_cls = torch.tensor(
                [preds["obj_id_to_cls"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            keep = (pred_scores > self.args.conf) & pred_masks.any(dim=(1, 2))
            pred_masks = pred_masks[keep]
            pred_boxes = batched_mask_to_box(pred_masks)
            pred_boxes = torch.cat(
                [pred_boxes, pred_ids[keep][:, None], pred_scores[keep][..., None], pred_cls[keep][..., None]], dim=-1
            )
            if pred_boxes.shape[0]:
                names = names or dict(enumerate(str(i) for i in range(pred_boxes[:, 6].int().max() + 1)))
            if pred_masks.shape[0] > 1:
                tracker_scores = torch.tensor(
                    [(preds["obj_id_to_tracker_score"].get(obj_id, 0.0)) for obj_id in curr_obj_ids],
                    device=pred_masks.device,
                )[keep]
                pred_masks = (
                    self._apply_object_wise_non_overlapping_constraints(
                        pred_masks.unsqueeze(1),
                        tracker_scores.unsqueeze(1),
                        background_value=0,
                    ).squeeze(1)
                ) > 0

        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def _run_single_frame_inference(self, frame_idx, reverse=False, inference_state=None):
        """在单帧上执行推理并获取推理结果。"""
        inference_state = inference_state or self.inference_state
        # 准备输入
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        # 对当前帧执行推理
        (
            obj_id_to_mask,
            obj_id_to_score,
            obj_id_to_cls,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            im=inference_state["im"],
            text_ids=inference_state["text_ids"],
            geometric_prompt=(
                self._get_dummy_prompt(num_prompts=len(inference_state["text_ids"]))
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # 更新推理状态
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # 第一帧检测分数
            "obj_id_to_cls": obj_id_to_cls,  # 第一帧检测类别
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
        }
        # removed_obj_ids 仅在 rank 0 上用于处理热启动延迟缓冲区
        metadata = tracker_metadata_new["metadata"]
        removed_obj_ids = metadata["removed_obj_ids"]
        out["removed_obj_ids"] = removed_obj_ids
        out["frame_stats"] = frame_stats
        if self.masklet_confirmation_enable:
            status = metadata["masklet_confirmation"]["status"]
            is_unconfirmed = status == self.UNCONFIRMED
            out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids"][is_unconfirmed].tolist()
        else:
            out["unconfirmed_obj_ids"] = []
        return out

    @smart_inference_mode()
    def add_prompt(
        self,
        frame_idx,
        text=None,
        bboxes=None,
        labels=None,
        inference_state=None,
    ):
        """在单帧上添加文本、点或边界框提示。此方法仅返回提示帧上的推理输出。

        注意，文本提示不与特定帧关联（即对所有帧生效）；但此处仅对 `frame_idx` 指定的帧执行推理。
        """
        inference_state = inference_state or self.inference_state
        assert text is not None or bboxes is not None, "at least one type of prompt (text, boxes) must be provided"

        # 1）处理文本提示
        use_text = text is not None
        text = text if use_text else "visual"
        text_batch = [text] if isinstance(text, str) else text
        inference_state["text_prompt"] = text if use_text else None
        n = len(text_batch)
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)

        # 2）处理边界框提示
        bboxes, labels = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        assert (bboxes is not None) == (labels is not None)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        if bboxes is not None:
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """按目标应用非重叠约束（即重叠区域只能归属于一个目标）。"""
        # 用目标分数替换像素分数
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        # 根据掩码分数应用逐像素非重叠约束
        pixel_level_non_overlapping_masks = self.tracker.model._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        # 用像素分数替换目标分数。注意，此时重叠区域只能归属于一个目标
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    def _det_track_one_frame(
        self,
        im: torch.Tensor,
        text_ids: torch.Tensor,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        geometric_prompt: Prompt,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        allow_new_detections: bool = True,
    ):
        """以 SPMD 方式为 DenseTracking 模型执行单步推理。
        从整体上看，所有 GPU 像在单个 GPU 上运行一样执行相同函数调用；底层部分调用会基于分片后的 SAM2 状态
        执行分布式计算。

        - `input_batch` 包含整个视频的图像及其他输入，在所有 GPU 上应保持一致。
        - `tracker_states_local` 保存当前 GPU 分片中的本地掩码轨迹信息。
        - `tracker_metadata_prev` 管理 SAM2 目标的元数据，例如每个 GPU 保存哪些掩码轨迹，同时包含全局和本地掩码轨迹信息。
        """
        # 步骤 1：以分布式方式运行骨干网络和检测器，由 Sam3ImageOnVideoMultiGPU 完成；
        # 这是分配给 `self.detector` 的多 GPU 模型，以轮询方式对帧进行分片。
        det_out = self.run_backbone_and_detection(
            im=im,
            text_ids=text_ids,
            geometric_prompt=geometric_prompt,
            allow_new_detections=allow_new_detections,
        )

        # 步骤 2：每个 GPU 传播本地 SAM2 状态，获取 SAM2 预测掩码。
        # 返回的 `tracker_low_res_masks_global` 包含从所有 GPU 收集并拼接的掩码轨迹预测结果，
        # 效果等同于在单个 GPU 上传播。此步骤仅执行 SAM2 传播，不为预测掩码编码新记忆；
        # 所有启发式规则处理完毕后，再由 `run_tracker_update_execution_phase` 执行记忆编码。
        if tracker_metadata_prev == {}:
            # 如果掩码轨迹元数据尚未初始化（空字典），则进行初始化
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = self.run_tracker_propagation(
            frame_idx=frame_idx,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=tracker_metadata_prev,
        )

        # 步骤 3：根据检测输出和传播得到的 SAM2 预测掩码，制定 SAM2 掩码轨迹更新计划
        #（包括添加和删除哪些目标、如何进行负载均衡等）。此步骤还在全局运行 SAM2 记忆编码器，解决非重叠约束。
        # **所有更新所需的启发式规则都应在此步骤处理。**大部分更新计划在主进程（GPU 0）上完成，
        # 生成的 `tracker_update_plan` 广播到其他 GPU 执行。此步骤还根据旧元数据 `tracker_metadata_prev`
        # 生成新的掩码轨迹元数据 `tracker_metadata_new`。
        tracker_update_plan, tracker_metadata_new = self.run_tracker_update_planning_phase(
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_states_local=tracker_states_local,
        )

        # 从更新计划中获取重新条件化信息
        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())

        # 步骤 4：根据 `tracker_update_plan`，每个 GPU 针对本地 SAM2 推理状态执行更新
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
        )

        # 步骤 5：最后构建当前帧的输出（仅需在 GPU 0 上执行，因为只有 GPU 0 会向服务端发送输出）。
        obj_id_to_mask = self.build_outputs(
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        obj_id_to_cls = tracker_metadata_new["obj_id_to_cls"]
        # 将当前帧的部分统计信息作为输出的一部分
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        # 将跟踪器分数添加到元数据中；第一帧之外的帧才执行此操作
        if tracker_obj_scores_global.shape[0] > 0:
            # 更新前将 tracker_obj_scores_global 转换为 sigmoid 分数
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )
        return (
            obj_id_to_mask,  # 字典：obj_id --> 输出掩码
            obj_id_to_score,  # 字典：obj_id --> 输出分数（概率）
            obj_id_to_cls,  # 字典：obj_id --> 输出类别（整数）
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,  # a dict: obj_id --> tracker frame-level scores
        )

    @staticmethod
    def _suppress_detections_close_to_boundary(boxes, margin=0.025):
        """抑制距离图像边缘过近的检测结果（针对归一化边界框）。

        boxes: (N, 4) 的 xyxy 格式边界框，已归一化到 [0,1]。
        margin: 图像尺寸的比例。
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (x_c > margin) & (x_c < 1.0 - margin) & (y_c > margin) & (y_c < 1.0 - margin)

        return keep

    def run_backbone_and_detection(
        self, im: torch.Tensor, text_ids: torch.Tensor, geometric_prompt: Prompt, allow_new_detections: bool
    ):
        """对单帧运行骨干网络和检测器。"""
        features = self.get_im_features(im)
        sam3_image_out = self.model.forward_grounding(
            backbone_out=features, text_ids=text_ids, geometric_prompt=geometric_prompt
        )
        det_out = self._extract_detection_outputs(sam3_image_out, allow_new_detections)
        self._cache_backbone_features(sam3_image_out)
        return det_out

    def _extract_detection_outputs(self, sam3_image_out, allow_new_detections):
        """提取并筛选检测输出。"""
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8

        pred_cls = torch.tensor(
            list(range(pred_probs.shape[0])),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )[:, None].expand_as(pred_probs)

        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]

        keep = pred_probs > self.score_threshold_detection
        return {
            "bbox": pred_boxes_xyxy[keep],
            "mask": pred_masks[keep],
            "scores": pred_probs[keep],
            "cls": pred_cls[keep],
        }

    def _cache_backbone_features(self, sam3_image_out):
        """构建并缓存 SAM2 骨干网络特征。"""
        sam_mask_decoder = self.tracker.model.sam_mask_decoder
        feats = sam3_image_out["backbone_out"]["sam2_backbone_out"]
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(feats["backbone_fpn"][0]),
            sam_mask_decoder.conv_s1(feats["backbone_fpn"][1]),
            feats["backbone_fpn"][2],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": feats["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        # 在跟踪器中缓存 `frame_idx` 对应的 SAM2 骨干网络特征
        self.tracker.backbone_out = tracker_backbone_out

    def run_tracker_propagation(
        self, frame_idx: int, tracker_states_local: list[Any], tracker_metadata_prev: dict[str, np.ndarray]
    ):
        """以 SPMD 方式对单帧运行跟踪器传播阶段。"""
        # 步骤 1：传播本地 SAM2 状态，获取当前帧预测结果
        # 当前 GPU 上已有掩码轨迹的 `low_res_masks_local`
        # - obj_ids_local: list[int] —— 目标 ID 列表
        # - low_res_masks_local: Tensor —— (num_local_obj, H_mask, W_mask)
        obj_ids_local, low_res_masks_local, obj_scores_local = self._propogate_tracker_one_frame_local_gpu(
            tracker_states_local, frame_idx=frame_idx
        )

        assert np.all(obj_ids_local == tracker_metadata_prev["obj_ids"]), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids"]
        )

        # 步骤 2：将 `low_res_masks_local` 全量收集为 `low_res_masks_global`
        # - low_res_masks_global: Tensor —— (num_global_obj, H_mask, W_mask)
        low_res_masks_global = low_res_masks_local
        obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: dict[str, torch.Tensor],
        trk_id_to_max_iou_high_conf_det: list[int],
        tracker_states_local: list[Any],
        tracker_metadata: dict[str, np.ndarray],
        tracker_obj_scores_global: torch.Tensor,
    ):
        """根据新的高置信度检测结果对掩码轨迹重新进行条件化。"""
        # 根据新的检测结果重新进行掩码轨迹条件化
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            new_mask_binary = (
                F.interpolate(new_mask.unsqueeze(1), size=self.interpol_size, mode="bilinear", align_corners=False) > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids"] == trk_obj_id)[0].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    # 注意：此条件用于避免对被遮挡或质量较低的掩码重新进行条件化。
                    # 但由于批处理机制，这些掩码仍可能被重新条件化；后续应考虑移除这些启发式规则。
                    and obj_score > HIGH_CONF_THRESH
                ):
                    LOGGER.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_prompts(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        masks=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(tracker_states_local[idx])
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        reverse: bool,
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_obj_scores_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_states_local: list[Any],
    ):
        """以 SPMD 方式对单帧运行跟踪器更新规划阶段。"""
        # 根据旧元数据初始化新元数据（稍后会更新其中的值）
        tracker_metadata_new = {
            "obj_ids": deepcopy(tracker_metadata_prev["obj_ids"]),
            "num_obj": deepcopy(tracker_metadata_prev["num_obj"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_cls": deepcopy(tracker_metadata_prev["obj_id_to_cls"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        # 提前初始化 reconditioned_obj_ids，避免 UnboundLocalError
        reconditioned_obj_ids = set()

        # 步骤 1：在 GPU 0 上制定更新计划并处理启发式规则
        det_mask_preds: torch.Tensor = det_out["mask"]  # low-res mask logits
        det_scores_np: np.ndarray = det_out["scores"].float().cpu().numpy()
        det_cls_np: np.ndarray = det_out["cls"].float().cpu().numpy()
        det_bbox_xyxy: torch.Tensor = det_out["bbox"]
        # a）匹配检测器掩码和跟踪器掩码，并查找新目标
        (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        ) = self._associate_det_trk(
            det_masks=det_mask_preds,
            det_scores_np=det_scores_np,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=tracker_metadata_prev["obj_ids"],
        )
        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(det_bbox_xyxy[new_det_fa_inds])
            new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

        # 检查是否达到可跟踪目标数上限；如果达到，则丢弃部分检测结果
        prev_obj_num = np.sum(tracker_metadata_prev["num_obj"])
        new_det_num = len(new_det_fa_inds)
        num_obj_dropped_due_to_limit = 0
        if prev_obj_num + new_det_num > self.max_num_objects:
            LOGGER.warning(f"已达到 {self.max_num_objects=}，当前 {new_det_num=}，已有 {prev_obj_num=}")
            new_det_num_to_keep = self.max_num_objects - prev_obj_num
            num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
            new_det_fa_inds = self._drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, new_det_num_to_keep)
            assert len(new_det_fa_inds) == new_det_num_to_keep
            new_det_num = len(new_det_fa_inds)

        # 为新检测结果分配目标 ID，并确定其所在 GPU
        new_det_obj_ids = tracker_metadata_prev["max_obj_id"] + 1 + np.arange(new_det_num)

        # b）处理热启动启发式规则以删除目标
        # `metadata` 保存在 GPU 0 上且仅由 GPU 0 访问；假设其他 GPU 不需要这些数据，因此不广播以节省通信开销
        metadata_new = deepcopy(tracker_metadata_prev["metadata"])
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            obj_ids_newly_removed, metadata_new = self._process_hotstart(
                frame_idx=frame_idx,
                reverse=reverse,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
                empty_trk_obj_ids=empty_trk_obj_ids,
                unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                metadata=metadata_new,
            )
        else:
            # 预热未完成时不删除任何目标
            obj_ids_newly_removed = set()
        tracker_metadata_new["metadata"] = metadata_new

        # 广播后，所有 GPU 上的 `tracker_update_plan` 应完全一致
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # np.ndarray
            "new_det_obj_ids": new_det_obj_ids,  # np.ndarray
            # "new_det_gpu_ids": new_det_gpu_ids,  # np.ndarray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # np.ndarray
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,  # dict
            "reconditioned_obj_ids": reconditioned_obj_ids,  # set
        }

        # 步骤 3（可选）：在记忆编码前根据高置信度检测结果重新条件化掩码轨迹
        # 注意：在执行阶段（记忆编码后）运行此步骤可能导致结果不理想
        should_recondition_iou = False

        # 根据边界框 IoU 与检测结果的不匹配程度，评估需要重新条件化的轨迹
        if self.reconstruction_bbox_iou_thresh > 0 and len(trk_id_to_max_iou_high_conf_det) > 0:
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids"]).index(trk_obj_id)
                except ValueError:
                    continue  # 未找到轨迹时跳过

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue  # 掩码面积为零时跳过

                # 从 SAM2 掩码获取边界框，并转换为归一化坐标
                tracker_box_pixels = batched_mask_to_box(mask_binary.unsqueeze(0)).squeeze(0)
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                # 计算检测边界框与 SAM2 轨迹边界框之间的 IoU
                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = box_iou(det_box_batch, tracker_box_batch)[0]

                if iou < self.reconstruction_bbox_iou_thresh and det_score >= self.reconstruction_bbox_det_score:
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        # 满足周期条件或 IoU 条件时执行重新条件化
        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        # 步骤 4：对当前帧的预测掩码运行 SAM2 记忆编码器
        # 所有 GPU 都执行此操作
        batch_size = tracker_low_res_masks_global.size(0)
        if (
            batch_size > 0
            and (not hasattr(self, "_warm_up_complete") or self._warm_up_complete)
            and self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0
        ):
            # 注意：tracker_low_res_masks_global 会被原地更新并返回
            tracker_low_res_masks_global = self._suppress_overlapping_based_on_recent_occlusion(
                frame_idx,
                tracker_low_res_masks_global,
                tracker_metadata_prev,
                tracker_metadata_new,
                obj_ids_newly_removed,
                reverse,
            )

        if batch_size > 0:
            self._tracker_update_memories(tracker_states_local, frame_idx, low_res_masks=tracker_low_res_masks_global)

        # 步骤 4：根据更新计划更新 SAM2 元数据
        updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids"]
        if len(new_det_obj_ids) > 0:
            updated_obj_ids_this_gpu = np.concatenate([updated_obj_ids_this_gpu, new_det_obj_ids])
        if len(obj_ids_newly_removed) > 0:
            is_removed = np.isin(updated_obj_ids_this_gpu, list(obj_ids_newly_removed))
            updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
        tracker_metadata_new["obj_ids"] = updated_obj_ids_this_gpu
        tracker_metadata_new["num_obj"] = len(updated_obj_ids_this_gpu)
        # 更新目标分数和目前分配的最大目标 ID
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["obj_id_to_cls"].update(zip(new_det_obj_ids, det_cls_np[new_det_fa_inds]))
            # 新目标没有跟踪器分数，因此使用检测分数代替。
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["max_obj_id"] = max(tracker_metadata_new["max_obj_id"], np.max(new_det_obj_ids))
        # 对已删除目标，将分数设为很小的值（-1e4），但仍保留在 "obj_id_to_score" 中，便于统一处理输出
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # 检查仅在 GPU 0 上的 tracker_metadata_new 中包含 "metadata"
        assert "metadata" in tracker_metadata_new
        if self.masklet_confirmation_enable:
            metadata = self.update_masklet_confirmation_status(
                metadata=tracker_metadata_new["metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["metadata"] = metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, Any],
        tracker_metadata_new: dict[str, Any],
        obj_ids_newly_removed: set[int],
        reverse: bool = False,
    ):
        """根据最近的遮挡信息抑制重叠掩码。如果目标因热启动规则被删除，只要它与其他目标重叠，就始终抑制该目标。

        参数：
            frame_idx (int): 当前帧索引。
            tracker_low_res_masks_global (torch.Tensor): 当前帧的低分辨率掩码。
            tracker_metadata_prev (dict[str, Any]): 上一帧的元数据。
            tracker_metadata_new (dict[str, Any]): 当前帧的元数据。
            obj_ids_newly_removed (set[int]): 已删除的目标 ID。
            reverse (bool): 是否按逆序跟踪。

        返回：
            (torch.Tensor): 抑制部分目标后的更新低分辨率掩码。
        """
        obj_ids_global = tracker_metadata_prev["obj_ids"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                self.NEVER_OCCLUDED if obj_id not in obj_ids_newly_removed else self.ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            # 使用遮挡信息更新元数据
            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # 为每个目标提取最后一次遮挡帧
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1] for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # 在记忆编码前将被抑制的掩码置零
            tracker_low_res_masks_global[to_suppress] = self.NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        det_out: dict[str, torch.Tensor],
        tracker_states_local: list[Any],
        tracker_update_plan: dict[str, np.ndarray],
    ):
        """以 SPMD 方式执行单帧跟踪器更新计划。"""
        # 使用检测分数初始化跟踪分数
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        # new_det_gpu_ids: np.ndarray = tracker_update_plan["new_det_gpu_ids"]
        new_det_obj_ids_local: np.ndarray = new_det_obj_ids
        new_det_fa_inds_local: np.ndarray = new_det_fa_inds
        obj_ids_newly_removed: set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # 步骤 1：将检测器发现的新目标添加到 SAM2 推理状态
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: torch.Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # 使用新目标掩码初始化 SAM2
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
            )

        # 步骤 2：从 SAM2 推理状态中删除由启发式规则移除的目标
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    @staticmethod
    def build_outputs(
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_update_plan: dict[str, np.ndarray],
        reconditioned_obj_ids: set | None = None,
    ):
        """构建当前帧的输出掩码。"""
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> 输出掩码张量

        # 第 1 部分：来自之前 SAM2 传播的掩码
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids"]
        existing_masklet_logits = tracker_low_res_masks_global.unsqueeze(1)
        assert len(existing_masklet_obj_ids) == len(existing_masklet_logits)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_logits):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # 第 2 部分：来自新检测结果的掩码
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        assert len(new_det_obj_ids) == len(new_det_low_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # 第 3 部分：使用检测掩码覆盖重新条件化目标的掩码
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get("trk_id_to_max_iou_high_conf_det", {})

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    obj_id_to_mask[obj_id] = det_out["mask"][det_idx].unsqueeze(0)

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: torch.Tensor,
        last_occluded: list[int],
        obj_ids: list[int],
        frame_idx: int | None = None,
        reverse: bool = False,
    ):
        # 抑制最近被遮挡目标的重叠掩码
        assert binary_low_res_masks.dtype == torch.bool, f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks.flatten(1), binary_low_res_masks.flatten(1))  # [N,N]

        # 创建上三角矩阵（i < j）的掩码和 IoU 阈值掩码
        mask_iou_thresh = iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # 抑制最近被遮挡的目标
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)  # (last_occ_expanded_i > last_occ_expanded_j)
            & (last_occ_expanded_j > -1)  # 仅当 i 之前被遮挡时，j 才能抑制 i
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)  # 仅当 j 之前被遮挡时，i 才能抑制 j
        )
        # 应用抑制
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        # 记录调试日志
        if LOGGER.isEnabledFor(10) and frame_idx is not None:
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            # 不使用 torch.where 查找所有需要抑制的目标对
            batch_size = suppress_i_mask.shape[0]

            # 记录 i 被抑制、保留 j 的情况
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            # 记录 j 被抑制、保留 i 的情况
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(self, inference_states: list[Any], frame_idx: int):
        """Inference_states：推理状态列表，每个状态对应一组不同的目标。"""
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue  # 跳过空推理状态的传播

            out_obj_ids, out_low_res_masks, out_obj_scores = self.tracker.propagate_in_video(
                inference_state, frame_idx=frame_idx
            )
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        # 拼接所有本地推理状态输出的掩码轨迹
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, *self._bb_feat_sizes[0], device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: torch.Tensor,
        det_scores_np: np.ndarray,
        trk_masks: torch.Tensor,
        trk_obj_ids: np.ndarray,
    ):
        """将当前帧的检测结果与现有掩码轨迹匹配。

        参数：
            det_masks: 形状为 (N, H, W) 的预测掩码张量。
            det_scores_np: 形状为 (N,) 的检测分数数组。
            trk_masks: 形状为 (M, H, W) 的轨迹掩码张量。
            trk_obj_ids: 与 trk_masks 对应的形状为 (M,) 的目标 ID 数组。

        返回：
            new_det_fa_inds: 新目标索引数组。
            unmatched_trk_obj_ids: 当前帧未与任何检测结果匹配的现有掩码轨迹目标 ID 数组
               （仅统计面积大于 0 的轨迹）。
            det_to_matched_trk_obj_ids: dict[int, np.ndarray]：检测索引到匹配轨迹目标 ID 列表的映射。
            empty_trk_obj_ids: SAM2 预测中面积为零的现有掩码轨迹目标 ID 数组。
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(trk_obj_ids), (
            f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        )
        if trk_masks.size(0) == 0:
            # 所有检测结果都是新目标
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            # 如果之前的轨迹面积非零，则全部视为未匹配
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            # 调整到较小尺寸以节省 GPU 内存
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                # 将检测结果调整为轨迹尺寸
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary.flatten(1).float(), trk_masks_binary.flatten(1).float())  # (N, M)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from ultralytics.utils.ops import linear_sum_assignment

            # 对轨迹执行匈牙利匹配（一对一：每条轨迹最多匹配一个检测结果）
            cost_matrix = 1 - ious_np  # 匈牙利算法求解最小代价
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        # 未通过上述匈牙利分配匹配且面积非零的轨迹视为未匹配
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        # 同时记录 SAM2 预测中面积为零的掩码轨迹
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        # 对检测结果允许多条轨迹匹配同一个检测结果（多对一）。
        # 因此，未与任何轨迹超过阈值匹配的检测结果才视为“新目标”。
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        # 记录每个检测结果匹配到的轨迹（超过阈值）
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}  # 轨迹 ID -> 唯一的检测索引
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= self.HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= self.HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(np.nonzero(det_is_high_conf & det_is_high_iou)[0])
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        empty_trk_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        metadata: dict[str, Any],
    ):
        """处理热启动启发式规则，删除未匹配或重复目标。"""
        # obj_id -> 首次检测到该目标的帧索引
        obj_first_frame_idx = metadata["obj_first_frame_idx"]
        # obj_id -> [未匹配帧索引]
        unmatched_frame_inds = metadata["unmatched_frame_inds"]
        trk_keep_alive = metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) -> [重叠帧索引]
        overlap_pair_to_frame_inds = metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids：通过热启动规则抑制的目标 ID
        removed_obj_ids = metadata["removed_obj_ids"]

        obj_ids_newly_removed = set()  # 当前帧新删除的目标 ID
        hotstart_diff = frame_idx - self.hotstart_delay if not reverse else frame_idx + self.hotstart_delay

        # 步骤 1：记录每个目标 ID 首次出现的帧索引
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        # 使用 det-->tracks 列表检查匹配目标；否则需要计算面积来判断目标是否被遮挡
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            # 注意：为减少可配置参数数量，使用 hotstart_unmatch_thresh 设置 trk_keep_alive 的最大值
            trk_keep_alive[obj_id] = min(self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1)
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # 注意：为减少可配置参数数量，使用 hotstart_unmatch_thresh 设置 trk_keep_alive 的最小值。
            # 最大保持值是最小值的 2 倍，表示目标匹配时间足够长时，模型倾向于保留预测而不是抑制它。
            trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                # 注意：为减少可配置参数数量，使用 hotstart_unmatch_thresh 设置 trk_keep_alive 的最小值
                trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)

        # 步骤 2：在热启动期间，删除连续 `hotstart_unmatch_thresh` 帧未与检测结果匹配的轨迹。
        # a）为每个现有目标 ID 添加未匹配帧索引。
        # 注意，`unmatched_trk_obj_ids` 包含 SAM2 输出掩码未匹配任何检测结果的帧，
        # 不包括 SAM2 输出空掩码的帧。
        # b）如果掩码轨迹在 `hotstart_diff` 后首次出现，并且未匹配超过 `self.hotstart_unmatch_thresh` 帧，则删除它。
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # 目标已删除时跳过
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                    obj_first_frame_idx[obj_id] < hotstart_diff and reverse
                )
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0  # 目标长时间未匹配
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                LOGGER.debug(f"Removing object {obj_id} at frame {frame_idx}, due to being unmatched")
                # 直接删除目标，而不是抑制目标
                obj_ids_newly_removed.add(obj_id)

        # 步骤 3：删除连续 `hotstart_dup_thresh` 帧与其他轨迹重叠的轨迹。
        # a）查找重叠轨迹：如果多条轨迹匹配同一个检测结果，则视为重叠。
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            if len(matched_trk_obj_ids) < 2:
                continue  # 仅统计匹配多个（>=2）掩码轨迹的检测结果
            # 如果存在多个匹配轨迹 ID，需要找出最先出现的 ID；
            # 后出现的 ID 可能被视为重复目标并删除
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b）如果掩码轨迹在 `hotstart_diff` 后首次出现，并且与更早出现的掩码轨迹重叠超过
        # `self.hotstart_dup_thresh` 帧，则删除该掩码轨迹
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # 目标已删除时跳过
            if (
                (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse)
                or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
            ) and len(frame_indices) >= self.hotstart_dup_thresh:
                obj_ids_newly_removed.add(obj_id)
                LOGGER.debug(
                    f"Removing object {obj_id} at frame {frame_idx} "
                    f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, metadata

    def _tracker_update_memories(
        self, tracker_inference_states: list[Any], frame_idx: int, low_res_masks: torch.Tensor
    ):
        """运行 SAM2 记忆编码器，并在全局范围内执行非重叠约束。"""
        if len(tracker_inference_states) == 0:
            return
        # 注意：如果演示运行出现显存溢出，应检查此处
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        )
        # 记忆编码前先应用非重叠约束，其中可能包含一些抑制启发式规则。
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker.model._suppress_object_pw_area_shrinkage(high_res_masks)
        # 不收集预测目标分数，而是使用掩码面积作为代理指标。
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        # 在每个 GPU 的本地切片上运行记忆编码器
        start_idx_gpu = 0
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            # 获取此推理状态的本地高分辨率掩码和目标分数 logits
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[start_idx_state:end_idx_state]
            local_batch_size = local_high_res_masks.size(0)
            # 运行 SAM2 记忆编码器。注意，非重叠约束默认关闭，因此此处不重复执行。

            encoded_mem = self.tracker._run_memory_encoder(
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
                inference_state=tracker_state,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            # 将编码后的记忆保存到本地推理状态
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = local_maskmem_features
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [pos for pos in local_maskmem_pos_enc]
                # 对于批量推理状态，还需要添加每目标记忆切片以支持实例交互
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: list[int],
        new_obj_masks: torch.Tensor,
        tracker_states_local: list[Any],
    ):
        """向 SAM2 推理状态添加新目标。"""
        prev_tracker_state = tracker_states_local[0] if len(tracker_states_local) > 0 else None

        # 准备 inference_state
        # 将在同一帧首次出现的目标批量处理
        # 清除推理状态；如果存在缓存的图像特征则保留。
        new_tracker_state = self.tracker._init_state(num_frames=num_frames)
        # 注意：添加图像占位符
        new_tracker_state["im"] = None
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None) if prev_tracker_state is not None else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(0),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        new_obj_masks = new_obj_masks > 0

        # 逐个添加目标
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_prompts(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                masks=new_mask[None, None],  # 添加批次维度和通道维度
            )
        # 注意：添加新目标时跳过全局非重叠约束。
        self.tracker.propagate_in_video_preflight(new_tracker_state)
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_objects(self, tracker_states_local: list[Any], obj_ids: list[int]):
        """从 SAM2 推理状态中删除目标，同时删除该目标在视频所有帧中的信息。"""
        if not obj_ids:
            return
        # 过滤删除目标后变为空的状态
        active_states = []
        for state in tracker_states_local:
            for obj_id in obj_ids:
                # 尝试在每个推理状态中以 `strict=False` 删除 `obj_id`；
                # 如果状态不包含该 ID，则不执行任何操作
                self.tracker.remove_object(state, obj_id, strict=False)

            if len(state["obj_ids"]) > 0:
                active_states.append(state)

        # 原地更新列表
        tracker_states_local[:] = active_states

    def _initialize_metadata(self):
        """初始化掩码轨迹元数据。"""
        tracker_metadata = {
            "obj_ids": np.array([], np.int32),
            "num_obj": np.zeros(1, np.int32),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_cls": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        # "metadata" 包含仅保存在 GPU 0 上且仅由 GPU 0 访问的元数据
        # - obj_first_frame_idx：obj_id -> 首次检测到目标的帧索引
        # - unmatched_frame_inds：obj_id -> [未匹配帧索引]
        # - overlap_pair_to_frame_inds：(first_appear_obj_id, obj_id) -> [重叠帧索引]
        # - removed_obj_ids：通过热启动规则抑制的目标 ID
        metadata = {
            "obj_first_frame_idx": {},
            "unmatched_frame_inds": defaultdict(list),
            "trk_keep_alive": defaultdict(int),  # 仅用于目标抑制，不用于目标删除
            "overlap_pair_to_frame_inds": defaultdict(list),
            "removed_obj_ids": set(),
        }
        if self.masklet_confirmation_enable:
            # 以下数组与 `obj_ids_all_gpu` 具有相同形状
            metadata["masklet_confirmation"] = {
                # "status" 表示每个掩码轨迹的确认状态
                "status": np.array([], np.int64),
                # "consecutive_det_num" 表示掩码轨迹连续被检测器检测到（并匹配检测结果）的帧数
                "consecutive_det_num": np.array([], np.int64),
            }
        tracker_metadata["metadata"] = metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        metadata: dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
    ):
        """根据当前帧的检测结果更新掩码轨迹确认状态。"""
        confirmation_data = metadata["masklet_confirmation"]

        # a）首先扩展 "confirmation_data"，纳入当前帧新增的掩码轨迹
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert status_prev.shape == obj_ids_all_gpu_prev.shape, (
            f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"
        )

        obj_id_to_updated_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)}
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        # 新增掩码轨迹初始化为 "UNCONFIRMED" 状态
        unconfirmed_val = self.UNCONFIRMED
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[prev_elem_is_in_updated]

        # b）根据当前帧更新所有掩码轨迹的确认状态
        # b.1）更新 "consecutive_det_num"
        # "is_matched"：掩码轨迹是否在当前帧匹配到检测结果
        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        # b.2）更新 "status"
        change_to_confirmed = consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        status[change_to_confirmed] = self.CONFIRMED

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return metadata

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            LOGGER.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            LOGGER.info("Loaded ckpt successfully without missing or unexpected keys")

    def _encode_prompt(self, **kwargs):
        return self.model._encode_prompt(**kwargs)

    @staticmethod
    def _drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, num_to_keep):
        """根据最大目标数限制丢弃部分新检测结果。按照检测分数保留高分目标并丢弃低分目标。"""
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)  # 全部丢弃
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds  # 全部保留

        # 保留分数最高的检测结果
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
