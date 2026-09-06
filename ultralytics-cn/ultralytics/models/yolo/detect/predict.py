# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """继承 BasePredictor、用于根据检测模型生成预测结果的类。

    此预测器专用于对象检测任务，将模型输出处理为包含边界框和类别预测结果的检测结果。

    属性：
        args (namespace): 预测器的配置参数。
        model (nn.Module): 用于推理的检测模型。
        batch (列表): 待处理的图像及其元数据批次。

    方法：
        postprocess: 将模型原始预测结果处理为检测结果。
        construct_results: 根据处理后的预测结果构建 Results 对象。
        construct_result: 根据单个预测结果创建一个 Results 对象。
        get_obj_feats: 从特征图中提取对象特征。

    示例：
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """后处理预测结果，并返回 Results 对象列表。

        此方法对模型原始预测结果应用非极大值抑制，并为可视化和进一步分析准备数据。

        参数：
            preds (torch.Tensor): 模型输出的原始预测结果。
            img (torch.Tensor): 采用模型输入格式的处理后输入图像张量。
            orig_imgs (torch.Tensor | 列表): 预处理前的原始输入图像。
            **kwargs (Any): 其他关键字参数。

        返回：
            (列表): 包含后处理预测结果的 Results 对象列表。

        示例：
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            kwargs.pop("iou", self.args.iou),  # 允许调用方（例如 TrackTrack 宽松 NMS 恢复逻辑）覆盖 IoU
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # 将对象特征添加到结果

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """从特征图中提取对象特征。"""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # 查找最短向量长度
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # 对所有向量求均值，使其长度一致
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # 处理批次中的每张图像

    def construct_results(self, preds, img, orig_imgs):
        """根据模型预测结果构建 Results 对象列表。

        参数：
            preds (列表[torch.Tensor]): 每张图像的预测边界框和分数列表。
            img (torch.Tensor): 用于推理的预处理图像批次。
            orig_imgs (列表[np.ndarray]): 预处理前的原始图像列表。

        返回：
            (列表[Results]): 包含每张图像检测信息的 Results 对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """根据一张图像的预测结果构建单个 Results 对象。

        参数：
            pred (torch.Tensor): 预测边界框和分数，形状为 (N, 6)，其中 N 是检测数量。
            img (torch.Tensor): 用于推理的预处理图像张量。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像文件路径。

        返回：
            (Results): 包含原始图像、图像路径、类别名称和缩放后边界框的 Results 对象。
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
