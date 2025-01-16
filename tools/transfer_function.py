"""
transfer function block (SISO)
"""
import numpy as np
import matplotlib.pyplot as plt

class transfer_function:
    def __init__(self, num, den, Ts):
        # expects num and den to be numpy arrays of shape (1,m) and (1,n)
        m = num.shape[1]
        n = den.shape[1]
        # set initial conditions
        self._state = np.zeros((n-1, 1))
        # make the leading coef of den == 1
        if den[0][0] != 1:
            den = den / den[0][0]
            num = num / den[0][0]
        self.num = num
        self.den = den
        # set up state space equations in control canonical form
        self._A = np.eye(n-1)
        self._B = np.zeros((n-1, 1))
        self._C = np.zeros((1, n-1))
        self._B[0][0] = Ts
        if m == n:
            self._D = num[0][0]
            for i in range(0, m):
                self._C[0][n-i-2] = num[0][m-i-1] - num[0][0]*den[0][n-i-2]
            for i in range(0, n-1):
                self._A[0][i] += - Ts * den[0][i+1]
            for i in range(1, n-1):
                self._A[i][i-1] += Ts
        else:
            self._D = 0.0
            for i in range(0, m):
                self._C[0][n-i-2] = num[0][m-i-1]
            for i in range(0, n-1):
                self._A[0][i] += - Ts * den[0][i]
            for i in range(1, n-1):
                self._A[i][i-1] += Ts

    def update(self, u):
        '''Update state space model'''
        self._state = self._A @ self._state + self._B * u
        y = self._C @ self._state + self._D * u
        return y[0][0]


def simulate_low_pass_filter():
    # 定义低通滤波器的参数
    Ts = 0.01  # 采样时间

    # 连续时间传递函数的分子和分母多项式系数
    num = np.array([[1]])
    den = np.array([[1 / (2 * np.pi), 1]])

    # 创建离散时间传递函数模型
    system = transfer_function(num, den, Ts)

    # 主仿真循环
    sim_time = 0.0
    time = [sim_time]
    output = [system.update(0.)]
    input_signal = []

    while sim_time < 20.0:
        u = np.sin(2 * np.pi * 0.5 * sim_time)  # 正弦函数输入，频率为0.5Hz
        y = system.update(u)  # 基于当前输入更新系统
        sim_time += Ts  # 增量仿真时间

        # 更新绘图数据
        time.append(sim_time)
        output.append(y)
        input_signal.append(u)

    # 绘制输入和输出随时间变化的关系
    plt.plot(time, output, label='Output')
    plt.plot(time[:-1], input_signal, label='Input', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Low Pass Filter Response to Sine Wave Input')
    plt.legend()
    plt.show() 


if __name__ == "__main__":
    def Demo():
        # instantiate the system
        Ts = 0.01  # simulation step size
        num = np.array([[1, 2]])  # numerator polynomial
        den = np.array([[1, 4, 5, 6]])  # denominator polynomial (no leading 1: s^3+4s^2+5s+6)
        system = transfer_function(num, den, Ts)

        # main simulation loop
        sim_time = 0.0
        time = [sim_time]
        output = [system.update(0.)]
        while sim_time < 10.0:
            u = np.random.randn()  # white noise
            y = system.update(u)  # update based on current input
            sim_time += Ts   # increment the simulation time

            # update date for plotting
            time.append(sim_time)
            output.append(y)

        # plot output vs time
        plt.plot(time, output)
        plt.show()

    def Demo1():
        # instantiate the system
        omega_n = 1.0  # 自然频率
        zeta = 0.5  # 阻尼比
        Ts = 0.01  # simulation step size
        num = np.array([[omega_n**2]])
        den = np.array([[1, 2*zeta*omega_n, omega_n**2]])
        system = transfer_function(num, den, Ts)

        # main simulation loop
        sim_time = 0.0
        time = [sim_time]
        output = [system.update(0.)]
        u = 1
        while sim_time < 100.0:
            # u = np.random.randn()  # white noise
            y = system.update(u)  # update based on current input
            sim_time += Ts   # increment the simulation time

            # update date for plotting
            time.append(sim_time)
            output.append(y)

        # plot output vs time
        plt.plot(time, output)
        plt.plot(time, u*np.ones_like(time))
        # plt曲线标识
        plt.legend(['output', 'input'])
        plt.show()

    def Demo2():
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import parameters.control_parameters as AP
        Ts = 0.01
        system = transfer_function(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1 / AP.yaw_damper_tau_r]]),
                        Ts=Ts)
        
        # main simulation loop
        sim_time = 0.0
        time = [sim_time]
        output = [system.update(0.)]
        u = 20
        while sim_time < 100.0:
            # u = np.random.randn()  # white noise
            y = system.update(u)  # update based on current input
            sim_time += Ts   # increment the simulation time

            # update date for plotting
            time.append(sim_time)
            output.append(y)

        # plot output vs time
        plt.plot(time, output)
        plt.plot(time, u*np.ones_like(time))
        # plt曲线标识
        plt.legend(['output', 'input'])
        plt.show()

    # Demo()
    # Demo1()
    # Demo2()
    simulate_low_pass_filter()
