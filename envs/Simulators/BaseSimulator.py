"""
仿真器抽象接口 - 定义所有仿真器必须实现的方法
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSimulator(ABC):
    """仿真器抽象基类"""
    
    @abstractmethod
    def reset(self):
        """
        重置仿真器到初始状态
        
        Returns:
            初始观测
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        执行一步仿真
        
        Args:
            action: 控制动作
            
        Returns:
            (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            状态字典
        """
        pass
    
    @abstractmethod
    def set_state(self, qpos, qvel):
        """
        设置状态
        
        Args:
            qpos: 位置
            qvel: 速度
        """
        pass
    
    @abstractmethod
    def render(self):
        """
        渲染当前状态
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        关闭仿真器，释放资源
        """
        pass
    
    @abstractmethod
    def get_time(self):
        """
        获取当前仿真时间
        
        Returns:
            时间（秒）
        """
        pass
    
    @abstractmethod
    def get_dt(self):
        """
        获取仿真时间步长
        
        Returns:
            时间步长（秒）
        """
        pass

