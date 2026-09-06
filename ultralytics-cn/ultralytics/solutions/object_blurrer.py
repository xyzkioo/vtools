# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    """管理实时视频流中检测对象模糊处理的类。

    此类扩展 BaseSolution，根据检测到的边界框对对象进行模糊处理。模糊区域会直接更新到输入图像中，
    可用于保护隐私或实现其他视觉效果。

    属性：
        blur_ratio (int): 应用于检测对象的模糊强度（值越大，模糊越明显）。
        iou (float): 对象检测的交并比阈值。
        conf (float): 对象检测的置信度阈值。

    方法：
        process: 对输入图像中的检测对象应用模糊效果。
        extract_tracks: 从检测对象中提取跟踪信息。
        display_output: 显示处理后的输出图像。

    示例：
        >>> blurrer = ObjectBlurrer()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = blurrer.process(frame)
        >>> print(f"Total blurred objects: {processed_results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 ObjectBlurrer 类，用于对视频流或图像中的检测对象应用模糊效果。

        参数：
            **kwargs (Any): 传递给父类并用于配置的关键字参数，包括：
                - blur_ratio (float): 模糊效果强度（0.1-1.0，默认为 0.5）。
        """
        super().__init__(**kwargs)
        blur_ratio = self.CFG["blur_ratio"]
        if blur_ratio < 0.1:
            LOGGER.warning("blur ratio cannot be less than 0.1, updating it to default value 0.5")
            blur_ratio = 0.5
        self.blur_ratio = int(blur_ratio * 100)

    def process(self, im0) -> SolutionResults:
        """对输入图像中的检测对象应用模糊效果。

        此方法提取跟踪信息，对应检测对象所在区域应用模糊，并使用边界框标注图像。

        参数：
            im0 (np.ndarray): 包含检测对象的输入图像。

        返回：
            (SolutionResults): 包含处理后图像和跟踪对象数量的对象。
                - plot_im (np.ndarray): 包含模糊对象标注的输出图像。
                - total_tracks (int): 当前帧中的跟踪对象总数。

        示例：
            >>> blurrer = ObjectBlurrer()
            >>> frame = cv2.imread("image.jpg")
            >>> results = blurrer.process(frame)
            >>> print(f"Blurred {results.total_tracks} objects")
        """
        self.extract_tracks(im0)  # 提取跟踪结果
        annotator = SolutionAnnotator(im0, self.line_width)

        # 遍历边界框和类别
        h, w = im0.shape[:2]
        for box, cls, conf in zip(self.boxes, self.clss, self.confs):
            x0, y0, x1, y1 = map(int, self.get_enclosing_box(box))
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)  # 将 OBB 角点裁剪到图像边界内
            if x0 >= x1 or y0 >= y1:  # 边界框完全位于画面外，裁剪后 ROI 为空，cv2.blur 会触发断言
                continue
            # 裁剪并模糊检测对象
            blur_obj = cv2.blur(im0[y0:y1, x0:x1], (self.blur_ratio, self.blur_ratio))
            # 更新原图像中的模糊区域
            im0[y0:y1, x0:x1] = blur_obj
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf), color=colors(cls, True)
            )  # 标注边界框

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
