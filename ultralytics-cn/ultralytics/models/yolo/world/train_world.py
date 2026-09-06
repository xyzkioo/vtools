# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import unwrap_model


class WorldTrainerFromScratch(WorldTrainer):
    """继承 WorldTrainer 的训练器，用于在开放集数据集上从头训练 World 模型。

    此训练器专门处理混合数据集，包括目标检测数据集和 grounding 数据集，支持训练具备视觉语言能力的 YOLO-World 模型。

    属性：
        cfg (dict): 包含模型训练默认参数的配置字典。
        overrides (dict): 用于自定义配置的参数覆盖字典。
        _callbacks (dict): 在训练不同阶段执行的回调函数字典。
        data (dict): 最终处理后的数据配置，包含 train/val 路径和元数据。
        training_data (dict): 训练数据集路径到其配置的映射字典。

    方法：
        build_dataset：构建支持混合数据集的 YOLO 训练或验证数据集。
        get_dataset：从数据字典获取训练和验证路径。
        plot_training_labels：跳过 YOLO-World 训练的标签绘图。
        final_eval：执行 YOLO-World 模型的最终评估和验证。

    示例：
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="flickr30k/images",
        ...                 json_file="flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="GQA/images",
        ...                 json_file="GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 WorldTrainerFromScratch 对象。

        此方法从头初始化 YOLO-World 模型训练器，支持包含目标检测数据集和 grounding 数据集的混合数据集，以实现视觉语言能力。

        参数：
            cfg (dict): 包含模型训练默认参数的配置字典。
            overrides (dict, 可选): 用于自定义配置的参数覆盖字典。
            _callbacks (dict, 可选): 在训练不同阶段运行的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建用于训练或验证的 YOLO 数据集。

        此方法根据模式和输入路径构建合适的数据集，同时处理标准 YOLO 数据集和不同格式的 grounding 数据集。

        参数：
            img_path (列表[str] | str): 包含图像的文件夹路径或路径列表。
            mode (str): 'train' 或 'val' 模式，可为每种模式自定义数据增强。
            batch (int, 可选): 批次大小，用于矩形训练或验证。

        返回：
            (YOLOConcatDataset | Dataset): 构建的训练或验证数据集。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(
                # 使用验证集的 `nc` 作为训练时文本样本数量上限，以保持一致性
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)  # 缓存文本嵌入以加速训练
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        """从 YAML 文件或字典检查并加载数据配置。

        参数：
            data (dict | str | Path): 数据配置，可以是字典或 YAML 文件路径。

        返回：
            (dict): 从 YAML 文件加载或直接传入的数据配置字典。
        """
        # 如果是字符串，则从 YAML 文件加载
        if not isinstance(data, dict):
            from ultralytics.utils import YAML

            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        """从数据字典获取训练和验证路径。

        处理数据配置，提取训练和验证数据集路径，同时处理 YOLO 检测数据集和 grounding 数据集。

        返回：
            (dict): 最终处理后的数据配置，包含 train/val 路径和元数据。

        异常：
            AssertionError: 未找到训练或验证数据集，或验证集包含多个数据集时抛出。
        """
        final_data = {}
        self.args.data = data_yaml = self.check_data_config(self.args.data)
        assert data_yaml.get("train", False), "未找到训练数据集"  # object365.yaml
        assert data_yaml.get("val", False), "未找到验证数据集"  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"当前仅支持在 1 个数据集上验证，但得到 {len(data['val'])} 个。"
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # 对于 LVIS 数据集
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in ("train", "val"):
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # 如果存在 grounding 数据，则保存
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding 数据必须以字典格式提供，但得到的是 {type(g)}"
                for k in ("img_path", "json_file"):
                    path = Path(g[k])
                    if not path.exists() and not path.is_absolute():
                        g[k] = str((DATASETS_DIR / g[k]).resolve())  # 路径相对于 DATASETS_DIR
            final_data[s] += grounding_data
        # 当前仅支持一个验证集，因此使用第一个验证数据集
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        # 注意：为确保训练正常运行，设置 `nc` 和 `names`
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        # 注意：添加包含 LVIS 路径的 path
        final_data["path"] = data["val"]["path"]
        final_data["channels"] = data["val"]["channels"]
        self.data = final_data
        if self.args.single_cls:  # 与基础训练器保持一致
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data

    def plot_training_labels(self):
        """跳过 YOLO-World 训练的标签绘图。"""

    def final_eval(self):
        """执行 YOLO-World 模型的最终评估和验证。

        在运行评估前，为验证器配置适当的数据集和数据划分信息。

        返回：
            (dict): 包含评估指标和结果的字典。
        """
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
