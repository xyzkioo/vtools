# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import clip_boxes, clip_coords, scale_masks
from ultralytics.utils.torch_utils import TORCH_1_10

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """专用于快速 SAM（Segment Anything Model）分割预测任务的 FastSAMPredictor。

    此类继承 SegmentationPredictor，针对快速 SAM 定制预测流程。
    它调整后处理步骤以加入掩码预测和非极大值抑制，同时进行性能优化。
    针对单类别分割任务进行了优化。

    属性：
        prompts (dict): 包含分割提示信息的字典（边界框、点、标签和文本）。
        device (torch.device): 模型和张量执行处理的设备。
        clip (Any, 可选): 用于文本提示的 CLIP 模型，按需加载。

    方法：
        postprocess: 对 FastSAM 预测结果应用后处理并处理提示。
        prompt: 根据不同类型的提示执行图像分割推理。
        set_prompts: 设置推理期间使用的提示。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置和回调初始化 FastSAMPredictor。

        此方法初始化专用于 Fast SAM（Segment Anything Model）分割任务的预测器。
        预测器继承 SegmentationPredictor，并为掩码预测和非极大值抑制提供自定义后处理。
        针对单类别分割任务进行了优化。

        参数：
            cfg (dict): 预测器配置。
            overrides (dict, 可选): 配置覆盖项。
            _callbacks (dict, 可选): 回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """对 FastSAM 预测结果应用后处理，并处理提示。

        参数：
            preds (列表[torch.Tensor]): 模型输出的原始预测结果。
            img (torch.Tensor): 输入模型的图像张量。
            orig_imgs (列表[np.ndarray]): 预处理前的原始图像。

        返回：
            (列表[Results]): Processed 结果 with prompts applied.
        """
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=result.boxes.data.device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes)[0] > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """根据边界框、点和文本提示等线索执行图像分割推理。

        参数：
            results (Results | 列表[Results]): FastSAM 模型在未应用任何提示时生成的原始推理结果。
            bboxes (np.ndarray | 列表, 可选): XYXY 格式的边界框，形状为 (N, 4)。
            points (np.ndarray | 列表, 可选): 表示目标位置的点，像素坐标形状为 (N, 2)。
            labels (np.ndarray | list, 可选): 点提示的标签，形状为 (N,)。1 表示前景，0 表示背景。
            texts (str | 列表[str], 可选): 文本提示组成的字符串列表。

        返回：
            (列表[Results]): 根据给定提示筛选和确定的输出结果。
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = (scale_masks(masks[None].float(), result.orig_shape)[0] > 0.5).byte()
            # 边界框提示
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                boxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device).clone()
                boxes = boxes[None] if boxes.ndim == 1 else boxes
                boxes = clip_boxes(boxes, result.orig_shape)
                bbox_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in boxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                coords = torch.as_tensor(points, dtype=torch.int32, device=self.device).clone()
                coords = coords[None] if coords.ndim == 1 else coords
                coords = clip_coords(coords, tuple(x - 1 for x in result.orig_shape))
                if labels is None:
                    labels = torch.ones(coords.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(coords), "Labels and points must contain the same number of items."
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # 所有 negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(coords, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if (masks[i].sum() if TORCH_1_10 else masks[i].sum(0).sum()) <= 100:  # torch 1.9 bug workaround
                        filter_idx.append(i)
                        continue
                    crop = result.orig_img[y1:y2, x1:x2] * masks[i, y1:y2, x1:x2, None].cpu().numpy()
                    crop_ims.append(Image.fromarray(crop[:, :, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    # 过滤前将 text_idx 映射回原始索引（支持多个文本提示）
                    ori_idxs = torch.tensor([i for i in range(len(result)) if i not in filter_idx], device=self.device)
                    text_idx = ori_idxs[text_idx]
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """执行 CLIP 推理，计算图像与文本提示之间的相似度。

        参数：
            images (列表[PIL.Image]): 源图像列表，每个元素应为 RGB 通道顺序的 PIL.Image 对象。
            texts (列表[str]): 提示文本列表，每个元素应为字符串对象。

        返回：
            (torch.Tensor): 给定图像与文本之间的相似度矩阵，形状为 (M, N)。
        """
        from ultralytics.nn.text_model import CLIP

        if not hasattr(self, "clip"):
            self.clip = CLIP("ViT-B/32", device=self.device)
        images = torch.stack([self.clip.image_preprocess(image).to(self.device) for image in images])
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(self.clip.tokenize(texts))
        return text_features @ image_features.T  # (M, N)

    def set_prompts(self, prompts):
        """设置推理期间使用的提示。"""
        self.prompts = prompts
