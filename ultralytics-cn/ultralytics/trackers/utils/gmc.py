# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """用于视频帧跟踪和目标检测的通用运动补偿（GMC）类。

    此类提供基于 ORB、SIFT、ECC 和稀疏光流等多种跟踪算法进行跟踪和目标检测的方法，
    同时支持缩小帧尺寸以提高计算效率。

    属性：
        method (str | None): 要使用的跟踪方法，可选 'orb'、'sift'、'ecc'、'sparseOptFlow' 或 None。
        downscale (int): 处理帧时的缩小倍数。
        prevFrame (np.ndarray | None): 用于跟踪的上一帧。
        prevKeyPoints (tuple | np.ndarray | None): 上一帧的关键点。
        prevDescriptors (np.ndarray | None): 上一帧的描述子。
        initializedFirstFrame (bool): 指示是否已处理第一帧的标志。

    方法：
        apply：将选定方法应用于原始帧，并可选择使用提供的检测结果。
        apply_ecc：将 ECC 算法应用于原始帧。
        apply_features：将 ORB 或 SIFT 等基于特征的方法应用于原始帧。
        apply_sparseoptflow：将稀疏光流方法应用于原始帧。
        reset_params：重置 GMC 对象的内部参数。

    示例：
        创建 GMC 对象并将其应用于一帧图像
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> warp = gmc.apply(frame)
        >>> print(warp.shape)
        (2, 3)
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        """使用跟踪方法和缩小倍数初始化通用运动补偿（GMC）对象。

        参数：
            method (str): 要使用的跟踪方法，可选 'orb'、'sift'、'ecc'、'sparseOptFlow' 或 'none'。
            downscale (int): 处理帧时的缩小倍数。
        """
        super().__init__()

        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == "ecc":
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == "sparseOptFlow":
            self.feature_params = {
                "maxCorners": 1000,
                "qualityLevel": 0.01,
                "minDistance": 1,
                "blockSize": 3,
                "useHarrisDetector": False,
                "k": 0.04,
            }

        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"Unknown GMC method: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        """估计一帧图像的 2×3 运动补偿变换矩阵。

        参数：
            raw_frame (np.ndarray): 要处理的原始帧，形状为 (H, W, C)。
            detections (列表, 可选): 处理时使用的检测结果列表。

        返回：
            (np.ndarray): 形状为 (2, 3) 的变换矩阵。

        示例：
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        if self.method in {"orb", "sift"}:
            return self.apply_features(raw_frame, detections)
        elif self.method == "ecc":
            return self.apply_ecc(raw_frame)
        elif self.method == "sparseOptFlow":
            return self.apply_sparseoptflow(raw_frame)
        else:
            return np.eye(2, 3)

    def apply_ecc(self, raw_frame: np.ndarray) -> np.ndarray:
        """使用 ECC（增强相关系数）算法对原始帧执行运动补偿。

        参数：
            raw_frame (np.ndarray): 要处理的原始帧，形状为 (H, W, C)。

        返回：
            (np.ndarray): 形状为 (2, 3) 的变换矩阵。

        示例：
            >>> gmc = GMC(method="ecc")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply_ecc(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3, dtype=np.float32)

        # 缩小图像以提高计算效率
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # 处理第一帧初始化
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        # 运行 ECC 算法以查找变换矩阵
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
            H[:, 2] *= (width / frame.shape[1], height / frame.shape[0])
        except Exception as e:
            LOGGER.warning(f"findTransformECC 失败，将使用单位变换。{e}")

        self.prevFrame = frame.copy()
        return H

    def apply_features(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        """将 ORB 或 SIFT 等基于特征的方法应用于原始帧。

        参数：
            raw_frame (np.ndarray): 要处理的原始帧，形状为 (H, W, C)。
            detections (列表, 可选): 处理时使用的检测结果列表。

        返回：
            (np.ndarray): 形状为 (2, 3) 的变换矩阵。

        示例：
            >>> gmc = GMC(method="orb")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply_features(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)

        # 缩小图像以提高计算效率
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # 创建用于关键点检测的掩码，排除边缘区域
        mask = np.zeros_like(frame)
        mask[int(0.02 * height) : int(0.98 * height), int(0.02 * width) : int(0.98 * width)] = 255

        # 从掩码中排除检测区域，避免跟踪已检测到的目标
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        # 查找关键点并计算描述子
        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # 处理第一帧初始化
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        # 匹配上一帧和当前帧之间的描述子
        knnMatches = (
            self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
            if self.prevDescriptors is not None and descriptors is not None
            else []
        )

        # 根据空间距离约束筛选匹配结果
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])

        # 应用 Lowe 比率测试和空间距离筛选
        prevPoints = []
        currPoints = []
        for matches in knnMatches:
            if len(matches) < 2:
                continue
            m, n = matches
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and (
                    np.abs(spatialDistance[1]) < maxSpatialDistance[1]
                ):
                    spatialDistances.append(spatialDistance)
                    prevPoints.append(prevKeyPointLocation)
                    currPoints.append(currKeyPointLocation)

        if not spatialDistances:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # 使用统计分析筛选异常值
        spatialDistances = np.asarray(spatialDistances).reshape(-1, 2)
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        # 保留位于边界上的匹配和方差为零的匹配。
        inliers = np.abs(spatialDistances - meanSpatialDistances) <= 2.5 * stdSpatialDistances

        # 保留通过异常值筛选的匹配点对
        good = inliers.all(axis=1)
        prevPoints = np.asarray(prevPoints).reshape(-1, 2)[good]
        currPoints = np.asarray(currPoints).reshape(-1, 2)[good]

        # 使用 RANSAC 估计变换矩阵
        if prevPoints.shape[0] > 4:
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # 将平移分量缩放回原始分辨率
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("匹配点数量不足")

        # 保存当前帧数据，供下一次迭代使用
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def apply_sparseoptflow(self, raw_frame: np.ndarray) -> np.ndarray:
        """将稀疏光流方法应用于原始帧。

        参数：
            raw_frame (np.ndarray): 要处理的原始帧，形状为 (H, W, C)。

        返回：
            (np.ndarray): 形状为 (2, 3) 的变换矩阵。

        示例：
            >>> gmc = GMC()
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply_sparseoptflow(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)

        # 缩小图像以提高计算效率
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # 查找适合跟踪的特征点
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # 处理第一帧初始化
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        # 使用 Lucas-Kanade 方法计算光流
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # 提取跟踪成功的点
        good = status.ravel().astype(bool)
        prevPoints = self.prevKeyPoints[good]
        currPoints = matchedKeypoints[good]

        # 使用 RANSAC 估计变换矩阵
        if prevPoints.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # 将平移分量缩放回原始分辨率
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("匹配点数量不足")

        # 保存当前帧数据，供下一次迭代使用
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def reset_params(self) -> None:
        """重置内部参数，包括上一帧、关键点和描述子。"""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
