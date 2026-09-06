# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class VisionEye(BaseSolution):
    """管理图像或视频流中对象检测和视觉映射的类。

    此类扩展 BaseSolution，提供检测对象、映射视觉点以及使用边界框和标签标注结果的功能。

    属性：
        vision_point (tuple[int, int]): 视觉观察对象并绘制跟踪线的坐标 `(x, y)`。

    方法：
        process: 处理输入图像，检测并标注对象，然后应用视觉映射。

    示例：
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = vision_eye.process(frame)
        >>> print(f"Total detected instances: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 VisionEye 类，用于检测对象并应用视觉映射。

        参数：
            **kwargs (Any): 传递给父类并用于配置 vision_point 的关键字参数。
        """
        super().__init__(**kwargs)
        # 设置系统观察对象并绘制跟踪线的视觉点
        self.vision_point = self.CFG["vision_point"]

    def process(self, im0) -> SolutionResults:
        """在输入图像上执行对象检测、视觉映射和标注。

        参数：
            im0 (np.ndarray): 用于检测和标注的输入图像。

        返回：
            (SolutionResults): 包含标注图像和跟踪统计信息的对象。
                - plot_im：包含边界框和视觉映射的标注输出图像
                - total_tracks：当前帧中的跟踪对象数量

        示例：
            >>> vision_eye = VisionEye()
            >>> frame = cv2.imread("image.jpg")
            >>> results = vision_eye.process(frame)
            >>> print(f"Detected {results.total_tracks} objects")
        """
        self.extract_tracks(im0)  # 提取跟踪结果（边界框、类别和掩码）
        annotator = SolutionAnnotator(im0, self.line_width)

        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            # 使用边界框、标签和视觉映射标注图像
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
            annotator.visioneye(self.get_enclosing_box(box), self.vision_point)

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示标注后的输出

        # 返回包含标注图像和跟踪统计信息的 SolutionResults 对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
