import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RewardComponent:
    """奖励组件"""
    name: str
    weight: float
    value: float
    calculator: callable


class RewardCalculate:
    """组合奖励计算器"""
    
    def __init__(self):
        self.Rewards: List[RewardComponent] = []
        self.reward_registry = {}
    
    def add_reward(self, reward: RewardComponent):
        """注册奖励组件"""
        self.Rewards.append(reward)
    
    def calculate(self, state, action, info=None) -> Dict:
        """计算总奖励"""
        total_reward = 0.0
        reward_info = {}
        
        for reward in self.Rewards:
            reward.value = reward.calculator(state, action, info)
            weighted_reward = reward.value * reward.weight
            total_reward += weighted_reward
            reward_info[f"reward_{reward.name}"] = reward.value
            reward_info[f"reward_{reward.name}_weighted"] = weighted_reward
        
        reward_info["reward_total"] = total_reward
        return total_reward, reward_info


# =========================== 任务抽象基类 =========================== #
class BaseTask(ABC):
    
    def __init__(self, config=None):
        self.config = config or {}
        self.simulator = None
        self.goal = None
        self._termination_conditions = []
        self._truncation_conditions = []
    
    @abstractmethod
    def setup(self, simulator, **kwargs):
        """
        任务初始化
        :param simulator: 使用的仿真器
        :param kwargs: 目标任务
        """
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
class BalanceTask(BaseTask):
    def __init__(self, config=None):
        super().__init__(config)
        # 添加终止条件
        self.add_termination_condition(self._check_fall)
    
    def setup(self, simulator, **kwargs):
        self.simulator = simulator
        self.goal = self.config.get('target_position', 0.0)
    
    def compute_reward(self, state, action, info=None):
        # 平衡任务奖励
        position = state['qpos'][0]
        velocity = state['qvel'][0]
        
        # 位置奖励
        position_cost = -abs(position - self.goal)
        # 速度惩罚
        velocity_cost = -0.1 * abs(velocity)
        # 控制代价
        control_cost = -0.001 * np.sum(np.square(action))
        
        return position_cost + velocity_cost + control_cost
    
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def _check_fall(self, sim_state, info=None):
        # 检查是否摔倒
        angle = sim_state['qpos'][1]
        return abs(angle) > 0.5


class AvoidTask(BaseTask):
    def __init__(self, config=None):
        super().__init__(config)
        # 添加终止条件
        self.add_termination_condition(self._check_collide)
    
    def setup(self, simulator, **kwargs):
        self.simulator = simulator
        self.goal = self.config.get('target_position', 0.0)
    
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def compute_reward(self, state, action, info=None):
        # 平衡任务奖励
        position = state['qpos'][0]
        velocity = state['qvel'][0]
        
        # 位置奖励
        position_cost = -abs(position - self.goal)
        # 速度惩罚
        velocity_cost = -0.1 * abs(velocity)
        # 控制代价
        control_cost = -0.001 * np.sum(np.square(action))
        
        return position_cost + velocity_cost + control_cost
    
    def _check_collide(self, sim_state, info=None):
        # 检查是否摔倒
        angle = sim_state['qpos'][1]
        return abs(angle) > 0.5
