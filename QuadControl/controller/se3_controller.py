from QuadControl.mixer.linear_mixer import *
import scipy.spatial.transform


class SE3Controller:
    def __init__(self, m, J, g=9.81):
        self.m = m
        self.J = J
        self.g = g
        self.mixer = LinearMixer()  # 4x4 混控矩阵
        self.max_total_thrust = 0.6292
        # 控制增益（针对 33g 无人机调整）
        self.Kp = np.array([0.2, 0.2, 0.5])  
        self.Kv = np.array([0.5, 0.5, 1.0])  
        self.KR = np.array([2.0, 2.0, 2.0])  
        self.Kw = np.array([0.5, 0.5, 0.5])
    
    def update(self, state, desired_state):
        pos = state['pos']
        vel = state['vel']
        quat = state['quat']  # [w, x, y, z]
        omega = state['omega']
        
        # 归一化四元数并转换为旋转矩阵
        quat = quat / np.linalg.norm(quat)
        # quat_scipy = np.roll(quat, -1)  # [w,x,y,z] -> [x,y,z,w]
        R = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
        
        pos_des = desired_state['pos_des']
        vel_des = desired_state['vel_des']
        acc_des = desired_state['acc_des']
        yaw_des = desired_state['yaw_des']
        
        # 位置控制器：期望推力向量（世界系）
        e_p = pos - pos_des
        e_v = vel - vel_des
        F_des = -self.Kp * e_p - self.Kv * e_v + self.m * (np.array([0, 0, self.g]) + acc_des)
        # print(f"F_des:{F_des}")
        # 推力大小限制（饱和）
        thrust_norm = np.linalg.norm(F_des)
        if thrust_norm > self.max_total_thrust:
            F_des = F_des / thrust_norm * self.max_total_thrust
            thrust_norm = self.max_total_thrust
        # print(f"thrust_norm:{thrust_norm}")
        # 期望推力方向（归一化）
        if thrust_norm < 1e-6:
            z_b_des = np.array([0, 0, 1])  # 默认向上
        else:
            z_b_des = F_des / thrust_norm
        
        # 根据期望偏航构建期望姿态
        x_c_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        y_b_des = np.cross(z_b_des, x_c_des)
        y_b_des_norm = np.linalg.norm(y_b_des)
        if y_b_des_norm < 1e-6:
            y_b_des = np.array([0, 1, 0])
        else:
            y_b_des /= y_b_des_norm
        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        
        # 姿态误差
        e_R = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R_vec = np.array([e_R[2, 1], e_R[0, 2], e_R[1, 0]])
        
        # 角速度误差（期望角速度假设为零）
        e_omega = omega
        tau_des = -self.KR * e_R_vec - self.Kw * e_omega
        # print(f"限幅前tau_des:{tau_des}")
        tau_max = 0.005  # Nm (根据 max_thrust * L * 2 估算)
        tau_des = np.clip(tau_des, -tau_max, tau_max)
        # print(f"限幅后tau_des:{tau_des}")
        # 解算电机推力（N）
        ctrl, scale = self.mixer.calculate(thrust_norm, tau_des[0], tau_des[1], tau_des[2])
        # print(f"混控器ctrl:{ctrl}, scale:{scale}")
        return ctrl, thrust_norm, scale, max(tau_des)
