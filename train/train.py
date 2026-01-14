import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import yaml
from envs.Envs import EnvTest
from envs.MujocoSim import *

if __name__ == '__main__':
    config_path = "E:\\UAV_RL\config\env_config.yaml"
    XML_loader = XMLModelLoader()
    # 加载配置
    env_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
    
    env = EnvTest(
        model_loader=XML_loader,
        # observation_builder=observation_builder,
        config=env_config,
        VisEnv=True
    )
    
    obs = env.reset()
    
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        for _ in range(10):
            action = np.random.uniform(-1, 1, size=env.model.nu)
            obs, reward, done, _, info = env.step(action)
            print("info:", info)
            total_reward += reward

        print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}\n")
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
