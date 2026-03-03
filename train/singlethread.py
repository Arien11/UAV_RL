import mujoco.viewer
import scipy.spatial.transform
import matplotlib.pyplot as plt
from QuadControl.mixer.linear_mixer import *
import yaml
import mujoco
from envs.config_builder import Configuration
from envs.QuadEnv import QuadEnv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    # ========== 加载环境 ==========
    with open(r"E:\UAV_RL\config\Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    env = QuadEnv(r"E:\UAV_RL\config\env_config.yaml", cfg)
    obs = env.reset()
    # 启动 viewer
    action = np.zeros(3)
    try:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        use_viewer = True
        print("可视化窗口已启动。使用鼠标拖动旋转视角，滚轮缩放。")
    except Exception as e:
        print(f"无法启动 viewer：{e}")
        use_viewer = False
    while True:
        next_state, reward, done, _ = env.step(action)
        # print(next_state[2])
        # 更新可视化
        if use_viewer:
            viewer.sync()
            if not viewer.is_running():
                break
