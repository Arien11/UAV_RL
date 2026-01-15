import time

import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import yaml
from envs.Envs import QuadEnv
import matplotlib.pyplot as plt
from envs.MujocoSim import *
from QuadControl.Quad import Quadrotor
import mujoco.viewer as viewer


# 可视化（可选）
def plot_results(data_log):
    plt.figure(figsize=(12, 8))
    
    # 位置跟踪图
    plt.subplot(2, 2, 1)
    plt.plot(data_log['time'], data_log['pos_z'], 'b-', label='实际高度')
    plt.plot(data_log['time'], data_log['target_z'], 'r--', label='目标高度')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('高度跟踪')
    plt.legend()
    plt.grid(True)
    
    # 推力变化图
    plt.subplot(2, 2, 2)
    plt.plot(data_log['time'], data_log['thrust'], 'g-')
    plt.xlabel('时间 (s)')
    plt.ylabel('推力 (mg)')
    plt.title('推力变化')
    plt.grid(True)
    
    # 电机输入图
    plt.subplot(2, 2, 3)
    for i in range(4):
        plt.plot(data_log['time'], [m[i] for m in data_log['motor_inputs']],
                 label=f'电机{i + 1}')
    plt.xlabel('时间 (s)')
    plt.ylabel('归一化输入')
    plt.title('电机输入')
    plt.legend()
    plt.grid(True)
    
    # 误差图
    plt.subplot(2, 2, 4)
    errors = [p - t for p, t in zip(data_log['pos_z'], data_log['target_z'])]
    plt.plot(data_log['time'], errors, 'm-')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度误差 (m)')
    plt.title('跟踪误差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quadrotor_test_results.png', dpi=150)
    print("\n结果图表已保存到: quadrotor_test_results.png")


if __name__ == '__main__':
    config_path = "E:\\UAV_RL\config\env_config.yaml"
    
    # =========================== 仿真器 =========================== #
    XML_loader = XMLModelLoader()
    env_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
    model = XML_loader.load(env_config.get('model', {}))
    Mujoco_simulator = MuJoCoSimulator(model)
    
    # =========================== 环境 =========================== #
    env = QuadEnv(
        # observation_builder=observation_builder,
        simulator=Mujoco_simulator,
        config=env_config,
    )
    
    # =========================== 控制、决策 =========================== #
    # 控制器配置
    QuadController = Quadrotor()
    obs, info = env.reset()
    print(f"Init Height: {info['qpos'][2]:.3f}m")
    
    # =========================== 数据可视化 =========================== #
    data_log = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'target_z': [],
        'reward': [],
        'actions': []
    }
    
    # =========================== 训练 =========================== #
    max_steps = int(1000 / QuadController._get_dt())
    render_fps = 100
    frame_interval = 1.0 / render_fps if render_fps > 0 else 0
    with mujoco.viewer.launch_passive(Mujoco_simulator.model, Mujoco_simulator.data) as viewer:
        for i in range(30):  # 等待3秒
            viewer.sync()
            time.sleep(0.1)
        while viewer.is_running():
            # 轨迹采集重置 ==============
            total_reward = 0
            obs, info = env.reset()
            done = False
            for step in range(200000):
                frame_start = time.time()
                
                # 动作选择(控制器 or 强化学习) ==============
                # action = np.random.uniform(-1, 1, size=env.model.nu)
                action, _ = QuadController._cal_control(obs)
                obs, reward, done, _, info = env.step(action)
                
                # 可视化同步 =============
                if viewer is not None:
                    viewer.sync()
                    # 控制帧率
                    elapsed = time.time() - frame_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                        
                # 打印数据 =============
                if step % 10 == 0:
                    data_log['time'].append(info['time'])
                    data_log['pos_z'].append(info['qpos'][2])
                    data_log['reward'].append(total_reward)
                    data_log['actions'].append(action.copy())
                    print(f"pos_z:{info['qpos'][2]:.2f}")
                    print(f"action:{action.copy()}")
                    print()
                total_reward += reward
                

# print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}\n")
# 记录数据
# if step % 10 == 0:
# data_log['time'].append(info['time'])
# # data_log['height'].append(info['qpos'][2])
# # data_log['target_height'].append(0.3)
# data_log['reward'].append(reward)
# data_log['actions'].append(action.copy())
# print(f"step: {step}")
# print(f"pos_z:{info['qpos'][2]:.2f}")
# print(f"action:{action.copy()}")
# # # print(f"reward:{reward:.2f}")
# # print(f"time:{info['time']:.2f}s")
# print()
# parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
# parser.add_argument("--train_config", type=str, default=os.path.join("config", "train_config.yaml"))
# args = parser.parse_args()
# print(args.train_config)
# if torch.cuda.is_available():
#     print("choose to use gpu...")
#     args.device = torch.device("cuda:0")
# else:
#     print("choose to use cpu...")
#     args.device = torch.device("cpu")
