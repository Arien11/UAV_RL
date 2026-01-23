from abc import ABC, abstractmethod


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
    def calc_reward(self, action):
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
