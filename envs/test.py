import yaml
import numpy as np
from envs.QuadEnv import QuadEnv
from envs.config_builder import Configuration
from Tasks.Hover_Task import HoverTask

# 加载配置
with open("E:\\UAV_RL\\config\\Quad_config.yaml", 'r') as f:
    config_data = yaml.safe_load(f)
cfg = Configuration(**config_data)

# 创建任务
task = HoverTask(target_pos=np.array([0, 0, 0.5]))
# 创建环境
env = QuadEnv("E:\\UAV_RL\\config\\env_config.yaml", cfg)

# 固定推力值
hover_cmd = 0.5145
action = np.array([hover_cmd, hover_cmd, hover_cmd, hover_cmd])

obs = env.reset()
step = 0
while step < 1000:
    obs, reward, done, info = env.step(action)  # 传入固定动作
    pos = env.interface.get_pos()
    print(f"step={step}, height={pos[2]:.3f}, reward={reward:.2f}")
    step += 1
    if done:
        print(f"Done at step {step}, reason: check task.done()")
        break