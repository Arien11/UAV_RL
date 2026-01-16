import gym

from typing import Dict, Any, Callable
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import yaml

from .Observe import *
from .RewardTask import *
from .MujocoSim import *


class QuadEnv(gym.Env):
    
    def __init__(
            self,
            observation_builder: Optional[ObservationSpace] = None,
            simulator: Optional[BaseSimulator] = None,
            task: Optional[BaseTask] = None,
            config=None
    ):
        self.observation_builder = observation_builder
        self.simulator = simulator or {}
        self.task = task
        self.config = config or {}
        self.last_state = None
        self._initEnvs()
    
    def _initEnvs(self):
        # 初始化观测构建器(外部若未导入，则自己生成)
        if self.observation_builder is None:
            self.observation_builder = ObservationSpace()
        
        # 初始化相关任务
        self.task.setup(self.simulator)
        
        # 初始化动作空间
        # self._setup_action_space()
        
        # 初始化观测空间
        self._setup_observation_space()
    
    def _setup_action_space(self):
        """设置动作空间"""
        # n_actuators = self.model.nu
        # self.action_space = gym.spaces.Box(
        #     low=-1.0, high=1.0,
        #     shape=(n_actuators,), dtype=np.float32
        # )
        ...
    
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
        self.last_state = sim_state
        
        # 获取观测
        if self.observation_builder:
            obs = self.observation_builder.get_observation(sim_state)
        else:
            ...
            # obs = self.task.get_observation(sim_state)
        
        # 获取info
        info = self._get_info(sim_state)
        
        return obs, info
    
    def step(self, action=None):
        # action = np.clip(action, 0.0, 1.0) # 确保动作在有效范围内
        state = self.simulator.step(action)
        self.last_state = state
        # ============= calculate reward ============= #
        reward = 0
        if self.task:
            reward = self.task.compute_reward(state, action)
        
        # ============= get observation ============= #
        if self.observation_builder:
            obs = self.observation_builder.get_observation(state)
        
        # ============= check end subjects ============= #
        terminated = 0
        truncated = 0
        # terminated, termination_reason = self.task.check_termination(sim_state)
        # truncated, truncation_reason = self.task.check_truncation(sim_state)
        
        # ============= create traj_info ============= #
        info = self._get_info(state)
        info.update({
            # 'termination_reason': termination_reason,
            # 'truncation_reason': truncation_reason,
            'time': state['time']
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


if __name__ == '__main__':
    ...
