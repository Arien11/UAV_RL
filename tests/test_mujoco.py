import sys
import os
from tkinter.constants import FALSE
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
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
    import yaml
    import mujoco
    from envs.config_builder import Configuration
    from envs.QuadEnv import QuadEnv
    
    # ========== 加载环境 ==========
    with open(r"E:\UAV_RL\config\QuadEnv_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    env = QuadEnv(config_data)
    obs = env.reset()
    
    # 控制参数（应与环境匹配）
    control_dt = 0.005  # 控制周期 (s)
    
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
    print(f"初始高度设置后: z = {env.data.qpos[2]:.3f}")
    
    # ========== 初始化相机显示窗口（可选）==========
    show_camera = True  # 设置为False可以大幅提升性能
    show_plt = False
    fig_rgb, fig_depth = None, None
    im_rgb, im_depth = None, None
    
    if show_camera:
        # RGB图像窗口
        # fig_rgb = plt.figure('RGB Image', figsize=(3, 2))
        # ax_rgb = fig_rgb.add_subplot(111)
        # im_rgb = ax_rgb.imshow(np.zeros((120, 160, 3), dtype=np.uint8))
        # ax_rgb.set_title('RGB Image')
        # ax_rgb.axis('off')
        # plt.tight_layout()
        
        # Depth图像窗口
        fig_depth = plt.figure('Depth Image', figsize=(3, 2))
        ax_depth = fig_depth.add_subplot(111)
        im_depth = ax_depth.imshow(np.zeros((120, 160)), cmap='jet', vmin=0, vmax=5)
        ax_depth.set_title('Depth Image')
        ax_depth.axis('off')
        cbar = plt.colorbar(im_depth, ax=ax_depth, label='Depth (m)')
        plt.tight_layout()
        
        plt.ion()
        plt.show()
    
    sim_steps = 0
    last_yaw = 0
    pos_lst = []
    pos_des_lst = []
    time_lst = []
    rpy_lst = []
    thrust_norm_lst = []
    scale_lst = []
    tau_des_max_lst = []
    
    # 时间同步变量
    last_wall_time = time.time()
    
    try:
        while True:
            # 获取当前状态
            pos = env.interface.get_pos()
            quat = env.interface.get_quat()
            vel = env.interface.get_vel()
            omega = env.interface.get_omega()
            t = env.data.time
            obs = env.get_obs()
            pos_des, vel_des, acc_des = Eight_traj(t)
            #pos_des, vel_des, acc_des = Circle_traj(t)
            # offset = offset_generator(t)
            # 悬停
            # pos_des = [0, 0, 0.5] + offset
            # vel_des = [0, 0, 0]
            # acc_des = [0, 0, 0]
            
            # dx = 0 - pos_des[0]
            # dy = 0 - pos_des[1]
            
            yaw_des = 0
            desired_state = {'pos_des': pos_des, 'vel_des': vel_des,
                             'acc_des': acc_des, 'yaw_des': yaw_des}
            state = {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}
            
            # 计算电机推力
            ctrl, thrust_norm, scale, tau_des_max = controller.update(state, desired_state)
            
            # 转换为控制量 (0~1)
            env.data.ctrl[:] = ctrl
            
            # ========== 多步物理（正确做法）==========
            for _ in range(frame_skip):
                mujoco.mj_step(env.model, env.data)
            
            # ========== 时间同步：按控制周期同步（关键！）==========
            # current_wall_time = time.time()
            # elapsed_wall = current_wall_time - last_wall_time
            # if elapsed_wall < control_dt:
            #     time.sleep(control_dt - elapsed_wall)
            # last_wall_time = time.time()
            
            # 更新可视化
            if use_viewer:
                viewer.sync()
                if not viewer.is_running():
                    break
            
            # 更新相机图像（降低频率以提升性能）
            if show_camera and sim_steps % 50 == 0:
                try:
                    # rgb, depth = env.get_camera_data()
                    depth = obs['sensors']['depth'][0]
                    # im_rgb.set_data(rgb)
                    im_depth.set_data(depth)
                    fig_depth.canvas.draw_idle()  # 使用draw_idle代替draw更高效
                    fig_depth.canvas.flush_events()
                except Exception as e:
                    print(f"获取相机数据失败: {e}")
            
            # 打印误差
            sim_steps += 1
            if sim_steps % 50 == 0:
                error = np.linalg.norm(pos - pos_des)
                # print(f"\n--- t={t:.2f} ---")
                # print(f"位置误差:{error}")
                
                rpy = quat_to_euler(np.roll(quat, -1))
                rpy_lst.append(rpy)
                pos_lst.append(pos)
                pos_des_lst.append(pos_des)
                time_lst.append(t)
                thrust_norm_lst.append(thrust_norm)
                scale_lst.append(scale)
                tau_des_max_lst.append(tau_des_max)
            
            # if t > sim_time:
            #     break
                
    except KeyboardInterrupt:
        print("仿真被用户中断")
    finally:
        if show_camera:
            plt.ioff()
            plt.close(fig_rgb)
            plt.close(fig_depth)
        if use_viewer:
            viewer.close()
        env.close()
    
    # ========== 绘图 ==========
    if show_plt:
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
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

