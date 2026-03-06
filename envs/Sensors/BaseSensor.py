"""
传感器抽象基类 - 定义所有传感器的统一接口
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSensor(ABC):
    """传感器抽象基类"""
    
    def __init__(self, config):
        """
        初始化传感器
        
        Args:
            config: 传感器配置字典
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self._sim = None  # 仿真器引用，需要在外部设置
    
    def set_simulator(self, sim):
        """
        设置仿真器引用
        
        Args:
            sim: 仿真器实例
        """
        self._sim = sim
    
    @abstractmethod
    def get_observation(self):
        """
        获取传感器原始数据
        
        Returns:
            原始传感器数据
        """
        pass
    
    @abstractmethod
    def preprocess_observation(self, raw_data):
        """
        预处理传感器数据（如归一化、压缩）
        
        Args:
            raw_data: 原始传感器数据
            
        Returns:
            预处理后的传感器数据
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置传感器状态"""
        pass
    
    def get_processed_observation(self):
        """
        获取预处理后的观测数据
        
        Returns:
            预处理后的观测数据
        """
        if not self.enabled:
            return np.zeros(self.get_observation_shape())
        
        raw_data = self.get_observation()
        return self.preprocess_observation(raw_data)
    
    @abstractmethod
    def get_observation_shape(self):
        """
        获取观测形状（用于空间定义）
        
        Returns:
            观测形状元组
        """
        pass