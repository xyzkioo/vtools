# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """当边界框靠近图像边缘时，将其调整到图像边界。

    参数：
        boxes (torch.Tensor): 形状为 (N, 4) 的 xyxy 格式边界框。
        image_shape (tuple): 图像尺寸，格式为 (高度, 宽度)。
        threshold (int): 判断边界框是否靠近边缘时使用的像素阈值。

    返回：
        (torch.Tensor): 调整后的边界框，形状为 (N, 4)。
    """
    # 图像尺寸
    h, w = image_shape

    # 调整靠近图像边界的边界框
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes
