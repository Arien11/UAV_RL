import mujoco.viewer
import scipy.spatial.transform
import matplotlib.pyplot as plt
from QuadControl.mixer.linear_mixer import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ================== 几何控制器类 ==================
class GeometricController:
    def __init__(self, m, J, g=9.81):
        self.m = m
        self.J = J
        self.g = g
        self.mixer = LinearMixer()  # 4x4 混控矩阵
        self.max_total_thrust = 0.6292
        # 控制增益（针对 33g 无人机调整）
        self.Kp = np.array([0.2, 0.2, 0.5])  # 位置增益
        self.Kv = np.array([0.5, 0.5, 1.0])  # 速度增益
        self.KR = np.array([2.0, 2.0, 2.0])  # 再减半
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


def quat_to_euler(q):
    """四元数 (x,y,z,w) 转欧拉角 (roll, pitch, yaw)"""
    x, y, z, w = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def Circle_traj(t):
    radius = 0.5
    height = 0.7
    period = 8.0
    # 圆周轨迹
    
    if t < 2.0:
        pos_des = np.array([0.0, 0.0, 1.0])
        vel_des = np.zeros(3)
        acc_des = np.zeros(3)
    else:
        w = 2 * np.pi / period
        pos_des = np.array([radius * np.cos(w * t),
                            radius * np.sin(w * t),
                            height])
        vel_des = np.array([-radius * w * np.sin(w * t),
                            radius * w * np.cos(w * t),
                            0.0])
        acc_des = np.array([-radius * w ** 2 * np.cos(w * t),
                            -radius * w ** 2 * np.sin(w * t),
                            0.0])
    return pos_des, vel_des, acc_des


def Eight_traj(t):
    # 8字形轨迹
    # 轨迹参数
    if t < 2.0:
        pos_des = np.array([0.0, 0.0, 1.0])
        vel_des = np.zeros(3)
        acc_des = np.zeros(3)
    else:
        A = 0.5
        H = 1.5
        T = 15.0
        omega = 2 * np.pi / T
        
        # 生成期望状态
        pos_des = np.array([A * np.sin(omega * t),
                            A * np.sin(2 * omega * t),
                            H])
        vel_des = np.array([A * omega * np.cos(omega * t),
                            2 * A * omega * np.cos(2 * omega * t),
                            0.0])
        acc_des = np.array([-A * omega ** 2 * np.sin(omega * t),
                            -4 * A * omega ** 2 * np.sin(2 * omega * t),
                            0.0])
    return pos_des, vel_des, acc_des


def offset_generator(t, start_time=2.0, period=5.0, amplitude=0.05):
    """
    按时间产生偏移量：在 start_time 之后，每 period 秒切换一次符号
    偏移方向：x, y, z 分别独立方波，也可同相。
    这里以同相方波为例（所有轴同时正或同时负）。
    """
    if t < start_time:
        return np.zeros(3)
    cycle = int((t - start_time) / period) % 2
    sign = 1 if cycle == 0 else -1
    return amplitude * sign * np.array([1, 1, 1])


# ================== 主仿真程序 ==================
def main():
    # # 加载模型
    # model = mujoco.MjModel.from_xml_path("E:\\UAV_RL\envs\crazyfile\scene.xml")
    # data = mujoco.MjData(model)
    import yaml
    import mujoco
    from envs.config_builder import Configuration
    from envs.QuadEnv import QuadEnv
    # ========== 加载环境 ==========
    with open(r"E:\UAV_RL\config\Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    env = QuadEnv(r"E:\UAV_RL\config\env_config.yaml", cfg)
    obs = env.reset()
    
    # 控制参数（应与环境匹配）
    control_dt = 0.02  # 控制周期 (s)
    sim_time = 10.0  # 仿真总时长 (s)
    steps = int(sim_time / control_dt)
    
    # 计算 frame_skip（物理步长从 interface 获取）
    phys_dt = env.interface.sim_dt()
    frame_skip = int(control_dt / phys_dt)
    print(f"物理步长: {phys_dt * 1000:.2f} ms, 控制周期: {control_dt * 1000:.2f} ms, frame_skip: {frame_skip}")
    
    # 启动 viewer
    try:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        use_viewer = True
        print("可视化窗口已启动。使用鼠标拖动旋转视角，滚轮缩放。")
    except Exception as e:
        print(f"无法启动 viewer：{e}")
        use_viewer = False
    
    # 从模型提取参数
    m = 0.033  # kg
    J = np.diag([1.395e-5, 1.395e-5, 2.173e-5])  # 惯量矩阵
    
    controller = GeometricController(m, J)
    
    # 初始位置（模型初始 z=0.1）
    env.data.qpos[2] = 0.2  # 确保初始高度
    mujoco.mj_forward(env.model, env.data)  # 更新传感器和渲染状态
    print(f"初始高度设置后: data.qpos[2] = {env.data.qpos[2]:.3f}")
    
    sim_steps = 0
    last_yaw = 0
    pos_lst = []
    pos_des_lst = []
    time_lst = []
    rpy_lst = []
    thrust_norm_lst = []
    scale_lst = []
    tau_des_max_lst = []
    while True:
        # 获取当前状态
        pos = env.interface.get_pos()
        quat = env.interface.get_quat()
        vel = env.interface.get_vel()
        omega = env.interface.get_omega()
        # pos = env.data.qpos[0:3].copy()
        #         # quat = env.data.qpos[3:7].copy()  # w,x,y,z
        #         # vel = env.data.qvel[0:3].copy()
        #         # omega = env.data.qvel[3:6].copy()
        
        t = env.data.time
        # pos_des, vel_des, acc_des = Circle_traj(t)
        # pos_des, vel_des, acc_des = Eight_traj(t)
        offset = offset_generator(t)
        # 悬停
        pos_des = [0, 0, 0.5] + offset
        vel_des = [0, 0, 0]
        acc_des = [0, 0, 0]
        
        dx = 0 - pos_des[0]
        dy = 0 - pos_des[1]
        # 在每个时间步
        raw_yaw = np.arctan2(dy, dx)  # 原始偏航角（可能跳变）
        delta = raw_yaw - last_yaw
        # 将差值调整到 (-π, π] 区间
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        yaw_des = last_yaw + delta
        last_yaw = yaw_des
        
        # yaw_des = 0
        desired_state = {'pos_des': pos_des, 'vel_des': vel_des,
                         'acc_des': acc_des, 'yaw_des': yaw_des}
        state = {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}
        
        # 计算电机推力
        ctrl, thrust_norm, scale, tau_des_max = controller.update(state, desired_state)
        
        # 转换为控制量 (0~1)
        env.data.ctrl[:] = ctrl
        
        # 仿真步进
        mujoco.mj_step(env.model, env.data)
        
        # 更新可视化
        if use_viewer:
            viewer.sync()
            if not viewer.is_running():
                break
        
        # 打印误差
        sim_steps += 1
        if sim_steps % 50 == 0:
            error = np.linalg.norm(pos - pos_des)
            print(f"\n--- t={t:.2f} ---")
            print(f"位置误差:{error}")
            # print(f"期望: x={pos_des[0]:.3f}, y={pos_des[1]:.3f}, z={pos_des[2]:.3f}")
            # print(f"位置误差: x={e_p[0]:.3f}, y={e_p[1]:.3f}, z={e_p[2]:.3f}")
            # print(f"期望推力向量 F_des: {F_des}")
            # print(f"期望总推力 norm: {thrust_norm:.4f} N")
            # print(f"实际总推力: {actual_total_thrust:.4f} N")
            # print(f"电机推力: {motor_thrusts}")
            # print(f"控制量 ctrl: {ctrl}")
            
            rpy = quat_to_euler(np.roll(quat, -1))
            rpy_lst.append(rpy)
            pos_lst.append(pos)
            pos_des_lst.append(pos_des)
            time_lst.append(t)
            thrust_norm_lst.append(thrust_norm)
            scale_lst.append(scale)
            tau_des_max_lst.append(tau_des_max)
    
    # ========== 绘图 ==========
    plt.figure(figsize=(14, 10))
    pos_lst = np.array(pos_lst)
    rpy_lst = np.array(rpy_lst)
    pos_des_lst = np.array(pos_des_lst)
    time_lst = np.array(time_lst)
    # 位置
    plt.subplot(2, 3, 1)
    plt.plot(time_lst, pos_lst[:, 0], label='x')
    plt.plot(time_lst, pos_des_lst[:, 0], linestyle='--', label='目标 x')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('位置 (m)')
    plt.legend()
    plt.grid(True)
    plt.title('x 位置变化')
    
    plt.subplot(2, 3, 2)
    plt.plot(time_lst, pos_lst[:, 1], label='y')
    plt.plot(time_lst, pos_des_lst[:, 1], linestyle='--', label='目标 y')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置 (m)')
    plt.legend()
    plt.grid(True)
    plt.title('y 位置变化')
    
    plt.subplot(2, 3, 3)
    plt.plot(time_lst, pos_lst[:, 2], label='z')
    plt.plot(time_lst, pos_des_lst[:, 2], linestyle='--', label='目标 z')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置 (m)')
    plt.legend()
    plt.grid(True)
    plt.title('z 位置变化')
    
    # 姿态角
    plt.subplot(2, 3, 4)
    plt.plot(time_lst, np.degrees(rpy_lst[:, 0]), label='roll')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (deg)')
    plt.legend()
    plt.grid(True)
    plt.title('roll')
    
    plt.subplot(2, 3, 5)
    plt.plot(time_lst, np.degrees(rpy_lst[:, 1]), label='pitch')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (deg)')
    plt.legend()
    plt.grid(True)
    plt.title('pitch')
    
    plt.subplot(2, 3, 6)
    plt.plot(time_lst, np.degrees(rpy_lst[:, 2]), label='yaw')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (deg)')
    plt.legend()
    plt.grid(True)
    plt.title('yaw')
    
    # 混控器
    # plt.subplot(2, 3, 4)
    # plt.plot(time_lst, thrust_norm_lst, label='thrust')
    # plt.xlabel('时间 (s)')
    # plt.ylabel('thrust N')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 5)
    # plt.plot(time_lst, scale_lst, label='scale')
    # plt.xlabel('时间 (s)')
    # plt.ylabel('scale')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 6)
    # plt.plot(time_lst, tau_des_max_lst, label='tau_des')
    # plt.xlabel('时间 (s)')
    # plt.ylabel('tau')
    # plt.legend()
    # plt.grid(True)
    # thrust_norm_lst.append(thrust_norm)
    # scale_lst.append(scale)
    # tau_des_max_lst.append(tau_des_max)
    
    plt.tight_layout()
    plt.show()
    if use_viewer:
        viewer.close()


if __name__ == "__main__":
    main()
