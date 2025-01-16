import numpy as np
import matplotlib.pyplot as plt

def kalman_variable():
    # -*- coding: utf-8 -*-
    """
    @对理想的一维匀加速直线运动模型，配有不精确的imu和不精确的gps，进行位置观测
    """
    t = np.linspace(1,100,100) # 在1~100s内采样100次
    a = 0.6 # 加速度值，匀加速直线运动模型
    v0 = 0 # 初始速度
    s0 = 0 # 初始位置
    m_var = 120**2 #这是我们自己设定的位置测量仪器的方差，越大则测量值占比越低，Q~N(0,m_var)
    v_var = 50**2 # 速度测量仪器的方差（这个方差在现实生活中是需要我们进行传感器标定才能算出来的，可搜Allan方差标定），R~N(0,v_var)
    nums = t.shape[0]

    # 根据理想模型推导出来的真实位置值，实际生活中不会存在如此简单的运动模型，真实位置也不可知，本程序中使用真值的目的是模拟观测噪声数据和测量噪声数据
    # 对于实际应用的卡尔曼滤波而言，并不需要知道真实值，而是通过预测值和观测值，来求解最优估计值，从而不断逼近该真值
    real_positions = [0] * nums
    real_positions[0] = s0
    # 实际观测值，通过理论值加上观测噪声模拟获得，初值即理论初始点加上观测噪声
    measure_positions = [0] * nums
    measure_positions[0] = real_positions[0] + np.random.normal(0, m_var**0.5)
    # 不使用卡尔曼滤波，也不使用实际观测值修正，单纯依靠运动模型来预估的预测值，仅初值由观测值决定
    predict_positions = [0] * nums
    predict_positions[0] = measure_positions[0]
    # 最优估计值，也就是卡尔曼滤波输出的真实值的近似逼近，同样地，初始值由观测值决定
    optim_positions = [0] * nums
    optim_positions[0] = measure_positions[0]
    # 卡尔曼滤波算法的中间变量
    pos_k_1 = optim_positions[0]

    predict_var = 0
    for i in range(1,t.shape[0]):
        # 根据理想模型获得当前的速度、位置真实值（实际应用中不需要）
        real_v = v0 + a * i;
        real_pos = s0 + (v0 + real_v) * i / 2
        real_positions[i] = real_pos
        # 模拟输入数据，实际应用中从传感器测量获得
        v = real_v + np.random.normal(0,v_var**0.5)
        measure_positions[i] = real_pos + np.random.normal(0,m_var**0.5)
        # 如果仅使用运动模型来预测整个轨迹，而不使用观测值，则得到的位置如下
        predict_positions[i] = predict_positions[i-1] + (v + v + a) * (i - (i - 1))/2

        # 以下是卡尔曼滤波的整个过程
        # 根据实际模型预测，利用上个时刻的位置（上一时刻的最优估计值）和速度预测当前位置
        pos_k_pred = pos_k_1 + v + a/2
        # 更新预测数据的方差
        predict_var += v_var 
        # 求得最优估计值
        pos_k = pos_k_pred * m_var/(predict_var + m_var) + measure_positions[i] * predict_var/(predict_var + m_var)
        # 更新
        predict_var = (predict_var * m_var)/(predict_var + m_var)
        pos_k_1 = pos_k
        optim_positions[i] = pos_k

    plt.plot(t,real_positions,label='real positions')
    plt.plot(t,measure_positions,label='measured positions')    
    plt.plot(t,optim_positions,label='kalman filtered positions')
    # 预测噪声比测量噪声低，但是运动模型预测值比观测值差很多，原因是在于运动模型是基于前一刻预测结果进行下一次的预测，而测量噪声是基于当前位置给出的测量结果
    # 意思就是，运动模型会积累噪声，而观测结果只是单次噪声
    plt.plot(t,predict_positions,label='predicted positions')

    plt.legend()
    plt.show()


def kalman_mat():
    # -*- coding: utf-8 -*-
    """
    @对理想的一维匀加速直线运动模型，配有不精确的imu和不精确的gps，进行位置观测，协方差均使用矩阵的方式表示，以适配多维特征
    """
    t = np.linspace(1,100,100) # 在1~100s内采样100次
    u = 0.6 # 加速度值，匀加速直线运动模型
    v0 = 5 # 初始速度
    s0 = 0 # 初始位置
    X_true = np.array([[s0], [v0]])
    size = t.shape[0] + 1
    dims = 2 # x, v, [位置, 速度]

    Q = np.array([[1e1,0], [0,1e1]]) # 过程噪声的协方差矩阵，这是一个超参数
    R = np.array([[1e4,0], [0,1e4]]) # 观测噪声的协方差矩阵，也是一个超参数。
    # R_var = R.trace()
    # 初始化
    X = np.array([[0], [0]]) # 估计的初始状态，[位置, 速度]，就是我们要估计的内容，可以用v0，s0填入，也可以默认为0，相差越大，收敛时间越长
    P = np.array([[0.1, 0], [0, 0.1]]) # 先验误差协方差矩阵的初始值，根据经验给出
    # 已知的线性变换矩阵
    F = np.array([[1, 1], [0, 1]]) # 状态转移矩阵
    B = np.array([[1/2], [1]]) # 控制矩阵
    H = np.array([[1,0],[0,1]]) # 观测矩阵

    # 根据理想模型推导出来的真实位置值，实际生活中不会存在如此简单的运动模型，真实位置也不可知，本程序中使用真值的目的是模拟观测噪声数据和测量噪声数据
    # 对于实际应用的卡尔曼滤波而言，并不需要知道真实值，而是通过预测值和观测值，来求解最优估计值，从而不断逼近该真值
    real_positions = np.array([0] * size)
    real_speeds = np.array([0] * size)
    real_positions[0] = s0
    # 实际观测值，通过理论值加上观测噪声模拟获得，初值即理论初始点加上观测噪声
    measure_positions = np.array([0] * size)
    measure_speeds = np.array([0] * size)
    measure_positions[0] = real_positions[0] + np.random.normal(0, R[0][0]**0.5)
    # 最优估计值，也就是卡尔曼滤波输出的真实值的近似逼近，同样地，初始值由观测值决定
    optim_positions = np.array([0] * size)
    optim_positions[0] = measure_positions[0]
    optim_speeds = np.array([0] * size)

    for i in range(1,t.shape[0]+1):
        # 根据理想模型获得当前的速度、位置真实值（实际应用中不需要），程序中只是为了模拟测试值和比较
        w = np.array([[np.random.normal(0, Q[0][0]**0.5)], [np.random.normal(0, Q[1][1]**0.5)]])
        X_true = F @ X_true + B * u + w
        real_positions[i] = X_true[0]
        real_speeds[i] = X_true[1]
        v = np.array([[np.random.normal(0, R[0][0]**0.5)], [np.random.normal(0, R[1][1]**0.5)]])
        # 观测矩阵用于产生真实的观测数据，注意各量之间的关联
        Z = H @ X_true + v  # 观测值
        # 以下是卡尔曼滤波的整个过程
        X_ = F @ X + B * u  # 预测状态
        P_ = F @ P @ F.T + Q
        # 注意矩阵运算的顺序
        K = P_@ H.T @ np.linalg.inv(H @ P_@ H.T + R)
        X = X_ + K @ (Z - H @ X_)  # 最优估计值
        P = (np.eye(2) - K @ H ) @ P_
        # 记录结果
        optim_positions[i] = X[0][0]
        optim_speeds[i] = X[1][0]
        measure_positions[i] = Z[0]
        measure_speeds[i] = Z[1]
        
    t = np.concatenate((np.array([0]), t))
    plt.plot(t,real_positions,label='real positions')
    plt.plot(t,measure_positions,label='measured positions')    
    plt.plot(t,optim_positions,label='kalman filtered positions')

    plt.legend()
    plt.show()

    plt.plot(t,real_speeds,label='real speeds')
    plt.plot(t,measure_speeds,label='measured speeds')    
    plt.plot(t,optim_speeds,label='kalman filtered speeds')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # kalman_variable()
    kalman_mat()
