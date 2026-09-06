# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch

from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor


class YOLOEVPDetectPredictor(DetectionPredictor):
    """继承 DetectionPredictor 的 YOLO-EVP（增强视觉提示）预测器。

    此类为使用视觉提示的 YOLO 模型提供通用功能，包括模型设置、提示处理和预处理变换。

    属性：
        model (torch.nn.Module): 用于推理的 YOLO 模型。
        device (torch.device): 运行模型的设备（CPU 或 CUDA）。
        prompts (dict): 包含类别索引以及边界框或掩码的视觉提示。
        visuals (torch.Tensor): 根据正在预处理的批次形状栅格化后的提示。

    方法：
        setup_model：初始化 YOLO 模型并将其设置为评估模式。
        set_prompts：为模型设置视觉提示。
        is_per_image：判断提示是否为每张图像分别提供一个数组。
        preprocess：预处理图像批次并栅格化其中的视觉提示。
        inference：使用视觉提示运行推理。
        get_vpe：处理源图像以获取视觉提示嵌入。
    """

    def setup_model(self, model, verbose: bool = True):
        """为预测设置模型。

        参数：
            model (torch.nn.Module): 要加载或使用的模型。
            verbose (bool, 可选): 为 True 时输出详细日志。
        """
        super().setup_model(model, verbose=verbose)
        self.done_warmup = True

    def set_prompts(self, prompts):
        """为模型设置视觉提示。

        参数：
            prompts (dict): 包含类别索引以及边界框或掩码的字典，必须包含带有类别索引的 'cls' 键。
        """
        self.prompts = prompts

    @staticmethod
    def is_per_image(prompts: dict) -> bool:
        """如果 'bboxes' 和 'cls' 为每张图像分别提供数组，则返回 True；否则表示所有图像共用一组提示。"""
        return all(
            isinstance(prompts.get(k), list) and all(isinstance(x, np.ndarray) for x in prompts[k])
            for k in ("bboxes", "cls")
        )

    def preprocess(self, im):
        """预处理批次并栅格化其中的视觉提示。"""
        imgs = super().preprocess(im)
        dst_shape = tuple(imgs.shape[2:])  # preprocess 会堆叠图像，因此每个批次使用同一个 letterbox 后尺寸
        # 张量源跳过 letterbox，因此其源尺寸和目标尺寸相同
        src_shapes = [dst_shape] * len(im) if isinstance(im, torch.Tensor) else [x.shape[:2] for x in im]
        self.visuals = self._prompts_to_tensor(dst_shape, src_shapes)
        return imgs

    def _prompts_to_tensor(self, dst_shape, src_shapes):
        """将提示栅格化为模型设备上的批量张量。"""
        bboxes, category = self.prompts.get("bboxes", None), self.prompts["cls"]
        if not self.is_per_image(self.prompts):  # 一组扁平提示，对批次中的每张图像进行栅格化
            masks = self.prompts.get("masks", None)
            visuals = [self._process_single_image(dst_shape, src, category, bboxes, masks) for src in src_shapes]
        else:
            assert len(src_shapes) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(src_shapes)}vs{len(category)}vs{len(bboxes)}!"
            )
            visuals = [
                self._process_single_image(dst_shape, src, category[i], bboxes[i]) for i, src in enumerate(src_shapes)
            ]
        prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)  # (B, N, H, W)
        return prompts.half() if self.model.fp16 else prompts.float()

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        """调整一张图像的提示大小并生成对应的视觉提示图。"""
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            # 计算缩放因子并调整边界框
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # gain = old / new
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - round(src_shape[1] * gain)) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - round(src_shape[0] * gain)) / 2 - 0.1)
        elif masks is not None:
            # 调整掩码大小并进行处理
            resized_masks = super().pre_transform(masks)
            masks = np.stack(resized_masks)  # (N, H, W)
            masks[masks == 114] = 0  # 将填充值重置为 0
        else:
            raise ValueError("Please provide valid bboxes or masks")

        # 使用视觉提示加载器生成视觉提示图
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)

    def inference(self, im, *args, **kwargs):
        """使用视觉提示运行推理。"""
        return super().inference(im, *args, vpe=self.visuals, **kwargs)

    def get_vpe(self, source):
        """从一张源图像中提取视觉提示嵌入。"""
        self.setup_source(source)
        assert len(self.dataset) == 1, "get_vpe 仅支持一张图像！"
        for _, im0s, _ in self.dataset:
            im = self.preprocess(im0s)
            return self.model(im, vpe=self.visuals, return_vpe=True)


class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor):
    """用于 YOLO-EVP 分割任务的预测器，结合检测和分割能力。"""
