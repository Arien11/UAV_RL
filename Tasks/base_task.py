from abc import ABC, abstractmethod
import numpy as np


class ActionSpace:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
    
    def sample(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)
    
    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)


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
    def step(self):
        """任务状态的更新"""
        pass
    
    @abstractmethod
    def setup(self):
        """任务的设置"""
        pass
    
    @abstractmethod
    def done(self):
        """检查任务是否结束"""
        pass
    
    @abstractmethod
    def reset(self, iter_count):
        pass
    
    @abstractmethod
    def calc_reward(self, action):
        """计算任务奖励"""
        pass
    
    @abstractmethod
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def interpret_action_e2e(self, action):
        """端到端的动作解析"""
        pass
    
    def interpret_action_controller(self, action):
        """动作解析到控制信号"""
        pass
