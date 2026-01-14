import gym

from .StateObs import *
from .MujocoSim import *
from typing import Dict, Any, Callable
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import yaml


class EnvTest(gym.Env):
    
    def __init__(
            self,
            model_loader: Optional[XMLModelLoader] = None,
            observation_builder: Optional[ObservationSpace] = None,
            config: Optional[Dict] = None,
            VisEnv: bool = False,
    ):
        self.model_loader = model_loader
        self.observation_builder = observation_builder
        self.config = config or {}
        self.VisEnv = VisEnv
        
        self.model = None
        self.simulator = None
        self.renderer = None
        self._initEnvs()
    
    def _initEnvs(self):
        """环境初始化"""
        if self.model is None:
            # 加载模型
            self.model = self.model_loader.load(self.config.get('model', {}))
        
        # 创建仿真器
        self.simulator = MuJoCoSimulator(self.model)
        
        # 初始化观测构建器(外部若未导入，则自己生成)
        if self.observation_builder is None:
            self.observation_builder = ObservationSpace()
        
        # 初始化动作空间
        self._setup_action_space()
        
        # 初始化观测空间
        self._setup_observation_space()
    
    def _setup_action_space(self):
        """设置动作空间"""
        n_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_actuators,), dtype=np.float32
        )
    
    def _setup_observation_space(self):
        init_state = self.simulator._get_state()
        obs_space_info = self.observation_builder.build_space(init_state)
        self.observation_space = gym.spaces.Box(
            low=obs_space_info['low'],
            high=obs_space_info['high'],
            shape=obs_space_info['shape'],
            dtype=obs_space_info['dtype']
        )
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self._initEnvs()
        
        # 重置仿真器
        sim_state = self.simulator.reset()
        
        # 获取观测
        if self.observation_builder:
            obs = self.observation_builder.get_observation(sim_state)
        else:
            ...
            # obs = self.task.get_observation(sim_state)
        
        # 获取info
        info = self._get_info(sim_state)
        
        return obs, info
    
    def step(self, action):
        """"""
        state = self.simulator.step(action)
        
        # ============= calculate reward ============= #
        reward = 0
        # if self.reward_calculator:
        #     reward, reward_info = self.reward_calculator.calculate(
        #         sim_state, action
        #     )
        # else:
        #     reward = self.task.compute_reward(sim_state, action)
        #     reward_info = {}
        
        # ============= get observation ============= #
        if self.observation_builder:
            obs = self.observation_builder.get_observation(state)
        
        # ============= check end subjects ============= #
        terminated = 0
        truncated = 0
        # terminated, termination_reason = self.task.check_termination(sim_state)
        # truncated, truncation_reason = self.task.check_truncation(sim_state)
        
        # ============= create traj_info ============= #
        ### used for data visualization
        info = self._get_info(state)
        info.update({
            # 'termination_reason': termination_reason,
            # 'truncation_reason': truncation_reason,
            'sim_time': state['time']
        })
        # info.update(reward_info)
        
        return obs, reward, terminated, truncated, info
    
    @staticmethod
    def _get_info(sim_state):
        """获取环境信息"""
        return {
            'qpos': sim_state['qpos'].copy(),
            'qvel': sim_state['qvel'].copy()
        }
    
    def Vis_Env(self):
        """渲染环境"""
        self.simulator.render()


if __name__ == '__main__':
    config_path = "../config/env_config.yaml"
    XML_loader = XMLModelLoader()
    # 加载配置
    env_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
    
    Env = EnvTest(
        model_loader=XML_loader,
        # observation_builder=observation_builder,
        config=env_config,
        VisEnv=True
    )
    Env.Vis_Env()
    # print(Env.model)
