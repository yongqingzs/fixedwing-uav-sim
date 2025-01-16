import numpy as np
import matplotlib.pyplot as plt

class LowPassFilter:
    def __init__(self, cutoff_frequency, Ts):
        self.cutoff_frequency = cutoff_frequency
        self.Ts = Ts
        self.alpha = self.cutoff_frequency / (self.cutoff_frequency + 1/(2*np.pi*Ts))
        self.prev_output = 0.0

    def update(self, input_signal):
        output = self.alpha * input_signal + (1 - self.alpha) * self.prev_output
        self.prev_output = output
        return output

def simulate_low_pass_filter():
    # 定义低通滤波器的参数
    Ts = 0.10  # 采样时间
    cutoff_frequency = 5  # 截止频率 (Hz)

    # 创建低通滤波器实例
    lpf = LowPassFilter(cutoff_frequency, Ts)

    # 主仿真循环
    sim_time = 0.0
    time = [sim_time]
    output = [lpf.update(0.)]
    input_signal = []

    while sim_time < 20:
        u = np.sin(2 * np.pi * 1 * sim_time)  # 正弦函数输入，频率为10Hz
        y = lpf.update(u)  # 基于当前输入更新系统
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
    plt.title('Low Pass Filter Response to 10 Hz Sine Wave Input')
    plt.legend()
    plt.show()

# 运行示例
simulate_low_pass_filter()