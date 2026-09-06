# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    """使用卡尔曼滤波器在图像空间中跟踪边界框的 KalmanFilterXYAH 类。

    该类实现了一个用于图像空间边界框跟踪的简单卡尔曼滤波器。八维状态空间
    `(x, y, a, h, vx, vy, va, vh)` 包含边界框中心位置 `(x, y)`、宽高比 `a`、高度 `h` 及其对应速度。
    对象运动遵循匀速模型，边界框位置 `(x, y, a, h)` 作为状态空间的直接观测值（线性观测模型）。

    属性：
        _motion_mat (np.ndarray): 卡尔曼滤波器的运动矩阵。
        _update_mat (np.ndarray): 卡尔曼滤波器的更新矩阵。
        _std_weight_position (float): 位置标准差权重。
        _std_weight_velocity (float): 速度标准差权重。

    方法：
        initiate: 根据未关联的观测值创建跟踪对象。
        predict: 执行卡尔曼滤波预测步骤。
        project: 将状态分布投影到观测空间。
        multi_predict: 批量执行卡尔曼滤波预测步骤。
        update: 执行卡尔曼滤波校正步骤。
        gating_distance: 计算状态分布与观测值之间的门控距离。

    示例：
        初始化 the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
    """

    def __init__(self):
        """使用运动和观测不确定性权重初始化卡尔曼滤波器矩阵。

        卡尔曼滤波器使用八维状态空间 `(x, y, a, h, vx, vy, va, vh)`，其中 `(x, y)` 表示边界框中心位置，
        `a` 表示宽高比，`h` 表示高度，`(vx, vy, va, vh)` 表示对应速度。滤波器使用匀速模型描述对象运动，
        使用线性观测模型表示边界框位置。
        """
        ndim, dt = 4, 1.0

        # 创建卡尔曼滤波器矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 根据当前状态估计值设置运动和观测不确定性
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """根据未关联的观测值创建跟踪对象。

        参数：
            measurement (np.ndarray): 边界框坐标 `(x, y, a, h)`，其中 `(x, y)` 为中心位置，`a` 为宽高比，`h` 为高度。

        返回：
            mean (np.ndarray): Float64 类型的八维均值向量，速度部分初始化为零。
            covariance (np.ndarray): Float64 类型的 8x8 协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        measurement = np.asarray(measurement, dtype=np.float64)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """执行卡尔曼滤波预测步骤。

        参数：
            mean (np.ndarray): 对象在上一时间步的八维状态均值向量。
            covariance (np.ndarray): 对象在上一时间步的 8x8 状态协方差矩阵。

        返回：
            mean (np.ndarray): 预测状态的均值向量。
            covariance (np.ndarray): 预测状态的协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """将状态分布投影到观测空间。

        参数：
            mean (np.ndarray): 状态均值向量（八维数组）。
            covariance (np.ndarray): 状态协方差矩阵（8x8）。
            confidence (float, 可选): 检测置信度；设置后会按 `max(1 - confidence, 0.05)` 缩放观测噪声（NSA-Kalman）。

        返回：
            mean (np.ndarray): 给定状态估计的投影均值。
            covariance (np.ndarray): 给定状态估计的投影协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:  # NSA-Kalman：根据检测置信度缩放观测噪声（StrongSORT）
            innovation_cov *= max(1.0 - float(confidence), 0.05)

        mean = mean[:4].copy()
        covariance = covariance[:4, :4]
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """批量执行多个对象状态的卡尔曼滤波预测步骤。

        参数：
            mean (np.ndarray): 对象状态在上一时间步的 N×8 均值矩阵。
            covariance (np.ndarray): 对象状态在上一时间步的 N×8×8 协方差矩阵。

        返回：
            mean (np.ndarray): 形状为 `(N, 8)` 的预测状态均值矩阵。
            covariance (np.ndarray): 形状为 `(N, 8, 8)` 的预测状态协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.random.rand(10, 8)  # 10 object states
            >>> covariance = np.random.rand(10, 8, 8)  # 10 个对象状态的协方差矩阵
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.zeros((sqr.shape[0], 8, 8))
        motion_cov[:, range(8), range(8)] = sqr

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float | None = None
    ):
        """执行卡尔曼滤波校正步骤。

        参数：
            mean (np.ndarray): 预测状态的八维均值向量。
            covariance (np.ndarray): 状态的 8x8 协方差矩阵。
            measurement (np.ndarray): 四维观测向量 `(x, y, a, h)`，其中 `(x, y)` 为中心位置，`a` 为宽高比，`h` 为边界框高度。
            confidence (float, 可选): 检测置信度；设置后会按 `max(1 - confidence, 0.05)` 缩放观测噪声（NSA-Kalman）。

        返回：
            new_mean (np.ndarray): 经观测值校正后的状态均值。
            new_covariance (np.ndarray): 经观测值校正后的状态协方差。

        示例：
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        kalman_gain = np.linalg.solve(projected_cov, covariance[:, :4].T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """计算状态分布与观测值之间的门控距离。

        返回的平方马氏距离可使用卡方分布的第 95 百分位数作为合适阈值：当 `only_position` 为 False 时，
        四个自由度对应 9.4877；否则两个自由度对应 5.9915。`"gaussian"` 度量返回平方欧氏距离，不适用上述阈值。

        参数：
            mean (np.ndarray): 状态分布的均值向量（八维）。
            covariance (np.ndarray): 状态分布的协方差矩阵（8x8）。
            measurements (np.ndarray): N 个观测值组成的 `(N, 4)` 矩阵，每行格式为 `(x, y, a, h)`，其中 `(x, y)` 为边界框中心位置，
                `a` 为宽高比，`h` 为高度。
            only_position (bool, 可选): 为 True 时，仅根据边界框中心位置计算距离。
            metric (str, 可选): 距离度量方式。`'gaussian'` 表示平方欧氏距离，`'maha'` 表示平方马氏距离。

        返回：
            (np.ndarray): 长度为 N 的数组，第 i 个元素包含 `(mean, covariance)` 与 `measurements[i]` 之间的平方距离。

        示例：
            计算 gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T)
            return np.sum(z * z, axis=0)  # 平方马氏距离
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """使用卡尔曼滤波器在图像空间中跟踪边界框的 KalmanFilterXYWH 类。

    该类实现了一个用于跟踪边界框的卡尔曼滤波器，状态空间为 `(x, y, w, h, vx, vy, vw, vh)`。
    其中 `(x, y)` 为中心位置，`w` 为宽度，`h` 为高度，`vx`、`vy`、`vw`、`vh` 为对应速度。
    对象运动遵循匀速模型，边界框位置 `(x, y, w, h)` 作为状态空间的直接观测值（线性观测模型）。

    属性：
        _motion_mat (np.ndarray): 卡尔曼滤波器的运动矩阵。
        _update_mat (np.ndarray): 卡尔曼滤波器的更新矩阵。
        _std_weight_position (float): 位置标准差权重。
        _std_weight_velocity (float): 速度标准差权重。

    方法：
        initiate: 根据未关联的观测值创建跟踪对象。
        predict: 执行卡尔曼滤波预测步骤。
        project: 将状态分布投影到观测空间。
        multi_predict: 以向量化方式批量执行卡尔曼滤波预测步骤。
        update: 执行卡尔曼滤波校正步骤。

    示例：
        创建卡尔曼滤波器并初始化跟踪对象
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
    """

    def initiate(self, measurement: np.ndarray):
        """根据未关联的观测值创建跟踪对象。

        参数：
            measurement (np.ndarray): 边界框坐标 `(x, y, w, h)`，其中 `(x, y)` 为中心位置，`w` 为宽度，`h` 为高度。

        返回：
            mean (np.ndarray): Float64 类型的八维均值向量，速度部分初始化为零。
            covariance (np.ndarray): Float64 类型的 8x8 协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
        """
        measurement = np.asarray(measurement, dtype=np.float64)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """执行卡尔曼滤波预测步骤。

        参数：
            mean (np.ndarray): 对象在上一时间步的八维状态均值向量。
            covariance (np.ndarray): 对象在上一时间步的 8x8 状态协方差矩阵。

        返回：
            mean (np.ndarray): 预测状态的均值向量。
            covariance (np.ndarray): 预测状态的协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """将状态分布投影到观测空间。

        参数：
            mean (np.ndarray): 状态均值向量（八维数组）。
            covariance (np.ndarray): 状态协方差矩阵（8x8）。
            confidence (float, 可选): 检测置信度；设置后会按 `max(1 - confidence, 0.05)` 缩放观测噪声（NSA-Kalman）。

        返回：
            mean (np.ndarray): 给定状态估计的投影均值。
            covariance (np.ndarray): 给定状态估计的投影协方差矩阵。

        示例：
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:  # NSA-Kalman：根据检测置信度缩放观测噪声（StrongSORT）
            innovation_cov *= max(1.0 - float(confidence), 0.05)

        mean = mean[:4].copy()
        covariance = covariance[:4, :4]
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """批量执行卡尔曼滤波预测步骤。

        参数：
            mean (np.ndarray): 对象状态在上一时间步的 N×8 均值矩阵。
            covariance (np.ndarray): 对象状态在上一时间步的 N×8×8 协方差矩阵。

        返回：
            mean (np.ndarray): 形状为 `(N, 8)` 的预测状态均值矩阵。
            covariance (np.ndarray): 形状为 `(N, 8, 8)` 的预测状态协方差矩阵。

        示例：
            >>> mean = np.random.rand(5, 8)  # 5 个对象，每个对象具有 8 维状态向量
            >>> covariance = np.random.rand(5, 8, 8)  # 5 个对象的 8x8 协方差矩阵
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.zeros((sqr.shape[0], 8, 8))
        motion_cov[:, range(8), range(8)] = sqr

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float | None = None
    ):
        """执行卡尔曼滤波校正步骤。

        参数：
            mean (np.ndarray): 预测状态的八维均值向量。
            covariance (np.ndarray): 状态的 8x8 协方差矩阵。
            measurement (np.ndarray): 四维观测向量 `(x, y, w, h)`，其中 `(x, y)` 为中心位置，`w` 为边界框宽度，`h` 为高度。
            confidence (float, 可选): 检测置信度；设置后会按 `max(1 - confidence, 0.05)` 缩放观测噪声（NSA-Kalman）。

        返回：
            new_mean (np.ndarray): 经观测值校正后的状态均值。
            new_covariance (np.ndarray): 经观测值校正后的状态协方差。

        示例：
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement, confidence=confidence)
