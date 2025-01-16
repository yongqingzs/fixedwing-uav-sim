import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        self.A = A  # 状态转移矩阵
        self.B = B  # 控制矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.P = P  # 估计误差协方差
        self.x = x  # 状态估计

    def predict(self, u=0):
        # 预测步骤
        self.x = self.A @ self.x + self.B @ np.array([[u]])
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        # 更新步骤
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (np.array([[z]]) - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x

def simulate_kalman_filter():
    # 定义卡尔曼滤波器的参数
    dt = 0.1  # 时间步长
    A = np.array([[1]])  # 状态转移矩阵
    B = np.array([[0]])  # 控制矩阵
    H = np.array([[1]])  # 观测矩阵
    Q = np.array([[1e-5]])  # 过程噪声协方差
    R = np.array([[1e-2]])  # 观测噪声协方差
    P = np.array([[1]])  # 估计误差协方差
    x = np.array([[0]])  # 初始状态估计

    # 创建卡尔曼滤波器实例
    kf = KalmanFilter(A, B, H, Q, R, P, x)

    # 生成模拟数据
    true_values = []
    measurements = []
    estimates = []
    sim_time = 0.0
    while sim_time < 10.0:
        true_value = np.sin(2 * np.pi * 0.1 * sim_time)  # 真值
        measurement = true_value + np.random.normal(0, np.sqrt(R[0, 0]))  # 观测值
        kf.predict()  # 预测
        estimate = kf.update(measurement)  # 更新

        true_values.append(true_value)
        measurements.append(measurement)
        estimates.append(estimate[0, 0])

        sim_time += dt

    # 绘制结果
    time = np.arange(0, 20, dt)
    time = time[0:len(true_values)]
    plt.plot(time, true_values, label='True Value')
    plt.plot(time, measurements, label='Measurements', linestyle='--')
    plt.plot(time, estimates, label='Estimates')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Kalman Filter Simulation')
    plt.legend()
    plt.show()

# 运行示例
simulate_kalman_filter()