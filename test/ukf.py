import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, state_dim, meas_dim, process_noise, meas_noise, dt):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.process_noise = process_noise
        self.meas_noise = meas_noise
        self.dt = dt

        self.x = np.zeros((state_dim, 1))  # 状态向量
        self.P = np.eye(state_dim)  # 状态协方差矩阵

        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lmbda = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim

        self.gamma = np.sqrt(self.state_dim + self.lmbda)
        self.Wm = np.full(2 * self.state_dim + 1, 0.5 / (self.state_dim + self.lmbda))
        self.Wm[0] = self.lmbda / (self.state_dim + self.lmbda)
        self.Wc = np.copy(self.Wm)
        self.Wc[0] += 1 - self.alpha**2 + self.beta

    def predict(self, f):
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_points_pred = np.array([f(sigma_points[:, i], self.dt) for i in range(2 * self.state_dim + 1)]).T

        self.x = np.dot(self.Wm, sigma_points_pred.T).reshape(-1, 1)
        self.P = self.process_noise + np.dot(self.Wc * (sigma_points_pred - self.x), (sigma_points_pred - self.x).T)

    def update(self, z, h):
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_points_meas = np.array([h(sigma_points[:, i]) for i in range(2 * self.state_dim + 1)]).T

        z_pred = np.dot(self.Wm, sigma_points_meas.T).reshape(-1, 1)
        Pz = self.meas_noise + np.dot(self.Wc * (sigma_points_meas - z_pred), (sigma_points_meas - z_pred).T)
        Pxz = np.dot(self.Wc * (sigma_points - self.x), (sigma_points_meas - z_pred).T)

        K = np.dot(Pxz, np.linalg.inv(Pz))
        self.x += np.dot(K, (z - z_pred))
        self.P -= np.dot(K, np.dot(Pz, K.T))

    def _generate_sigma_points(self, x, P):
        sigma_points = np.zeros((self.state_dim, 2 * self.state_dim + 1))
        sigma_points[:, 0] = x[:, 0]
        sqrt_P = np.linalg.cholesky(P)

        for i in range(self.state_dim):
            sigma_points[:, i + 1] = x[:, 0] + self.gamma * sqrt_P[:, i]
            sigma_points[:, self.state_dim + i + 1] = x[:, 0] - self.gamma * sqrt_P[:, i]

        return sigma_points

# 示例使用
def f(x, dt):
    # 状态转移函数
    return np.array([x[0] + dt * x[1], x[1]])

def h(x):
    # 观测函数
    return np.array([x[0]])

state_dim = 2
meas_dim = 1
process_noise = np.diag([1e-5, 1e-5])
meas_noise = np.diag([1e-2])
dt = 0.1

ukf = UnscentedKalmanFilter(state_dim, meas_dim, process_noise, meas_noise, dt)

# 生成模拟数据
true_values = []
measurements = []
estimates = []
sim_time = 0.0
while sim_time < 10.0:
    true_value = np.array([np.sin(2 * np.pi * 0.1 * sim_time), 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * sim_time)])
    measurement = true_value[0] + np.random.normal(0, np.sqrt(meas_noise[0, 0]))
    ukf.predict(f)
    ukf.update(np.array([measurement]), h)

    true_values.append(true_value[0])
    measurements.append(measurement)
    estimates.append(ukf.x[0, 0])

    sim_time += dt

# 绘制结果
import matplotlib.pyplot as plt

time = np.arange(0, 20, dt)
time = time[0:len(true_values)]
plt.plot(time, true_values, label='True Value')
plt.plot(time, measurements, label='Measurements', linestyle='--')
plt.plot(time, estimates, label='Estimates')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Unscented Kalman Filter Simulation')
plt.legend()
plt.show()