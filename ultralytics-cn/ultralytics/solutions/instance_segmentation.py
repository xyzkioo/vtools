# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class InstanceSegmentation(BaseSolution):
    """管理图像或视频流中实例分割任务的类。

    此类扩展 BaseSolution，提供实例分割功能，包括绘制分割掩码、边界框和标签。

    属性：
        model (YOLO): 用于推理的分割模型实例。
        line_width (int): 边界框和文本线宽。
        names (dict[int, str]): 类别索引到类别名称的映射。
        clss (列表[int]): 检测到的类别索引列表。
        track_ids (列表[int]): 检测实例的跟踪 ID 列表。
        masks (列表[np.ndarray]): 检测实例的分割掩码列表。
        show_conf (bool): 是否显示置信度分数。
        show_labels (bool): 是否显示类别标签。
        show_boxes (bool): 是否显示边界框。

    方法：
        process: 处理输入图像，执行实例分割并标注结果。
        extract_tracks: 从模型预测结果中提取边界框、类别和掩码等跟踪信息。

    示例：
        >>> segmenter = InstanceSegmentation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.process(frame)
        >>> print(f"Total segmented instances: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 InstanceSegmentation 类，用于检测和标注分割实例。

        参数：
            **kwargs (Any): 传递给 BaseSolution 父类的关键字参数，包括：
                - model (str): 模型名称或路径，默认为 "yolo26n-seg.pt"。
        """
        kwargs["model"] = kwargs.get("model", "yolo26n-seg.pt")
        super().__init__(**kwargs)

        self.show_conf = self.CFG.get("show_conf", True)
        self.show_labels = self.CFG.get("show_labels", True)
        self.show_boxes = self.CFG.get("show_boxes", True)

    def process(self, im0) -> SolutionResults:
        """在输入图像上执行实例分割并标注结果。

        参数：
            im0 (np.ndarray): 用于分割的输入图像。

        返回：
            (SolutionResults): 包含标注图像和跟踪实例总数的对象。

        示例：
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.process(frame)
            >>> print(summary)
        """
        self.extract_tracks(im0)  # 提取跟踪结果（边界框、类别和掩码）
        self.masks = getattr(self.tracks, "masks", None)

        # 遍历检测到的类别、跟踪 ID 和分割掩码
        if self.masks is None:
            self.LOGGER.warning("No masks detected! Ensure you're using a supported Ultralytics segmentation model.")
            plot_im = im0
        else:
            results = Results(im0, path=None, names=self.names, boxes=self.track_data.data, masks=self.masks.data)
            plot_im = results.plot(
                line_width=self.line_width,
                boxes=self.show_boxes,
                conf=self.show_conf,
                labels=self.show_labels,
                color_mode="instance",
            )

        self.display_output(plot_im)  # 使用基类函数显示标注后的输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
