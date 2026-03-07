import sys
import os
from tkinter.constants import FALSE
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import torch
import mujoco.viewer
import scipy.spatial.transform
import matplotlib.pyplot as plt
from QuadControl.mixer.linear_mixer import *
from utils.logger import VisLogger
from envs.maps.esdf_generate import ESDFGenerator, DifferentiableESDF

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ========== 轨迹生成 ==========
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


def grad_check(generator, pos, grad_autograd, eps = 1e-4):
    grad_numerical = np.zeros_like(pos)
        
    for dim in range(3):
        # 正向扰动
        point_plus = pos.copy()
        point_plus[dim] += eps
        dist_plus, _ = generator.distance_kdtree(point_plus)
        
        # 负向扰动
        point_minus = pos.copy()
        point_minus[dim] -= eps
        dist_minus, _ = generator.distance_kdtree(point_minus)
        
        # 中心差分
        grad_numerical[dim] = (dist_plus - dist_minus) / (2 * eps)
    
    # ========== 验证梯度正确性 ==========
    grad_error = np.linalg.norm(grad_autograd - grad_numerical)
    
    print(f"  自动微分梯度: {grad_autograd}")
    print(f"  数值梯度:     {grad_numerical}")
    print(f"  梯度误差:     {grad_error:.2e}")
    if grad_error < 1e-2:
        print("  ✅ 梯度验证通过！")
    else:
        print("  ⚠️  梯度误差较大")

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
    print(f"\n初始状态: {env.nominal_pose}")

    # ========== 创建ESDF生成器（离线预计算）==========
    print("\n创建ESDF数据...")
    generator = ESDFGenerator(env.model, env.data)
    generator.sample_obstacle_points(num_points_per_geom=100)
    esdf_module = DifferentiableESDF(generator)

    # 控制参数（应与环境匹配）
    control_dt = 0.005  # 控制周期 (s)
    
    # 计算 frame_skip（物理步长从 interface 获取）
    phys_dt = env.interface.sim_dt()
    frame_skip = int(control_dt / phys_dt)
    print(f"\n物理步长: {phys_dt * 1000:.2f} ms, 控制周期: {control_dt * 1000:.2f} ms, frame_skip: {frame_skip}")

    # 启动 viewer
    try:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        use_viewer = True                                                                           
    except Exception as e:
        print(f"无法启动 viewer：{e}")
        use_viewer = False
    
    # ========== 初始化日志记录器 ==========
    print("\n初始化日志记录器...")
    logger = VisLogger(
        logging_freq_hz=100,
        output_folder="my_results",
        duration_sec=10
    )


    # ========== 初始化相机显示窗口（可选）==========
    show_camera = False  # 设置为False可以大幅提升性能
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
    pos_lst = []
    pos_des_lst = []
    dist_lst = []
    time_lst = []
    rpy_lst = []
    thrust_norm_lst = []
    scale_lst = []
    tau_des_max_lst = []
    
    # 时间同步变量
    last_wall_time = time.time()
    
    print("\n开始仿真...")
    try:
        while True:
            # ========== 获取当前时间 ==========
            t = env.data.time

            # ========== 获取当前状态 ==========    
            pos = env.interface.get_pos()    
            quat = env.interface.get_quat()
            rpy = env.interface.get_euler()
            vel = env.interface.get_vel()
            omega = env.interface.get_omega()
            obs = env.get_obs()

            esdf_pos = torch.tensor(pos, dtype=torch.float32, requires_grad=True) 
            distance = esdf_module(esdf_pos)                # 计算到最近障碍物的距离
            distance.backward()                             # 反向传播计算梯度
            grad_autograd = esdf_pos.grad.numpy().copy()    # 计算得到esdf梯度
            
            grad_check(generator, pos, grad_autograd)
            
            # ========== 跟踪任务 ==========
            pos_des, vel_des, acc_des = Eight_traj(t)   # 8字轨迹
            #pos_des, vel_des, acc_des = Circle_traj(t)  # 圆周轨迹
            # offset = offset_generator(t)
            # 悬停
            # pos_des = [0, 0, 0.5]
            # vel_des = [0, 0, 0]
            # acc_des = [0, 0, 0]     
            
            yaw_des = 0
            desired_state = {'pos_des': pos_des, 'vel_des': vel_des,
                             'acc_des': acc_des, 'yaw_des': yaw_des}
            state = {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}
   
            # ========== 控制器 ==========    
            ctrl, thrust_norm, scale, tau_des_max = env.robot.ctrl.update(state, desired_state) # 计算控制量：推力与力矩
            log_state = np.array([
                    pos[0], pos[1], pos[2],                 # 位置 (0-2)
                    vel[0], vel[1], vel[2],                 # 速度 (3-5)
                    rpy[0], rpy[1], rpy[2],                 # 欧拉角 (6-8)
                    omega[0], omega[1], omega[2],           # 角速度 (9-11)
                    quat[0], quat[1], quat[2], quat[3],     # 目标 (16-18)
                    pos_des[0], pos_des[1], pos_des[2],     # 目标位置 (19-21)
                    thrust_norm                             # 推力 (22)
            ])    
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
            
            # ========== 记录数据 ==========        
            logger.log(t, log_state, None)
            dist_lst.append(distance.detach().numpy())
            rpy_lst.append(rpy)
            pos_lst.append(pos)
            pos_des_lst.append(pos_des)
            time_lst.append(t)
            thrust_norm_lst.append(thrust_norm)
            scale_lst.append(scale)
            tau_des_max_lst.append(tau_des_max)

            sim_steps += 1

                
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
    

    plt.plot(time_lst, dist_lst, label='x')
    plt.show()
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

