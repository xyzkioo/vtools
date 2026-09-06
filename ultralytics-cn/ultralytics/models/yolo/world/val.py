# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import select_device


class WorldValidator(DetectionValidator):
    """用于 YOLO-World 模型的验证器，在验证前设置数据集类别名称。

    开放词汇 YOLO-World 模型默认使用 80 个 COCO 类别，因此在类别不同的数据集（例如 LVIS）上验证会失败或得到零指标。
    此验证器为数据集类别名称生成文本嵌入，使独立调用 `model.val()` 能够正常工作。
    训练期间则通过 `on_pretrain_routine_end` 回调设置类别。
    """

    def __call__(self, trainer=None, model=None):
        """为独立验证设置数据集类别，然后运行验证。"""
        if trainer is None:  # 独立验证；训练通过 on_pretrain_routine_end 回调设置类别
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model.eval().to(self.device)
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)  # 与 BaseValidator 的数据集处理保持一致
            names = [name.split("/", 1)[0] for name in check_det_dataset(self.args.data)["names"].values()]
            current = model.names.values() if isinstance(model.names, dict) else model.names  # names 也可能是列表
            if list(current) != names:  # 仅当类别顺序与数据集不同时重新生成提示词
                state = (model.names, model.txt_feats, model.model[-1].nc)  # 验证后恢复，避免影响调用方
                model.set_classes(names, cache_clip_model=False)
                model.names = dict(enumerate(names))  # set_classes 会更新嵌入和 nc，但不会更新 names
                try:
                    return super().__call__(trainer, model)
                finally:
                    model.names, model.txt_feats, model.model[-1].nc = state
        return super().__call__(trainer, model)
