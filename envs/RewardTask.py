import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


# =========================== 任务抽象基类 =========================== #
class BaseTask(ABC):
    
    def __init__(self, config=None):
        self.simulator = None
        self.goal = None
        self.Task_Settings = config or {}
        self.Rewards = None
        self._termination_conditions = []
        self._truncation_conditions = []
    
    @abstractmethod
    def setup(self, simulator):
        pass
    
    @abstractmethod
    def compute_reward(self, state, action, info=None):
        """计算任务奖励"""
        pass
    
    @abstractmethod
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def check_termination(self, sim_state, info=None):
        """检查终止条件"""
        for condition in self._termination_conditions:
            if condition(sim_state, info):
                return True, condition.__name__
        return False, None
    
    def check_truncation(self, sim_state, info=None):
        """检查截断条件"""
        for condition in self._truncation_conditions:
            if condition(sim_state, info):
                return True, condition.__name__
        return False, None
    
    def add_termination_condition(self, condition):
        self._termination_conditions.append(condition)
    
    def add_truncation_condition(self, condition):
        self._truncation_conditions.append(condition)


# =========================== 具体任务 =========================== #
class Hover(BaseTask):
    def __init__(self, config=None):
        super().__init__(config)
        # 添加终止条件
        self.add_termination_condition(self._check_fall)
        self.goal = 1
        self.Task_name = "Hover"
    
    def setup(self, simulator):
        self.simulator = simulator
        self.Rewards = self.Task_Settings[self.Task_name]
    
    def compute_reward(self, state, action, info=None):
        position = state['qpos'][0]
        velocity = state['qvel'][0]
        
        # 平衡奖励
        Balance_Param = self.Rewards["Balance"]
        pos_reward = abs(position - self.goal) * Balance_Param["lambda_pos"]
        vel_reward = abs(velocity) * Balance_Param["lambda_vel"]
        
        # 动力损耗奖励
        Control_Param = self.Rewards["effort"]
        control_reward = np.sum(np.square(action)) * Control_Param["lambda_effort"]
        
        return pos_reward + vel_reward + control_reward
    
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def _check_fall(self, sim_state, info=None):
        # 检查是否摔倒
        angle = sim_state['qpos'][1]
        return abs(angle) > 0.5


if __name__ == '__main__':
    
    import yaml
    
    config_path = "../config/Task_config.yaml"
    # 加载配置
    task_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            task_config = yaml.safe_load(f)
    # 处理配置
    rewards = task_config
    
    print(rewards["Hover"]["Balance"]["lambda_pos"])
    # 输出: {'lambda_1': 2, 'lambda_2': 1, 'lambda_3': 1}
    
    print(rewards["Hover"]["effort"])
