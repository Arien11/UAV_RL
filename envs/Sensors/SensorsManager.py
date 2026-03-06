"""
传感器管理器 - 统一管理所有传感器
"""
import numpy as np
from .BaseSensor import BaseSensor
from .MujocoCamera import MujocoDepthCamera, MujocoRGBCamera


class SensorManager:
    """传感器管理器"""
    
    def __init__(self, config, sim=None):
        """
        初始化传感器管理器
        
        Args:
            config: 传感器配置字典
            sim: 仿真器实例（可选）
        """
        self.sensors = []
        self.config = config
        self._sim = sim
        
        # 自动创建传感器实例
        for sensor_config in config.get('sensors', []):
            try:
                sensor_class = self._get_sensor_class(sensor_config['type'])
                sensor = sensor_class(sensor_config)
                
                # 如果提供了仿真器，设置到传感器
                if sim is not None:
                    sensor.set_simulator(sim)
                
                self.sensors.append(sensor)
                print(f"传感器已初始化: {sensor_config['type']}")
            except Exception as e:
                print(f"初始化传感器失败: {sensor_config.get('type', 'unknown')}, 错误: {e}")
    
    def set_simulator(self, sim):
        """
        设置仿真器引用到所有传感器
        
        Args:
            sim: 仿真器实例
        """
        self._sim = sim
        for sensor in self.sensors:
            sensor.set_simulator(sim)
    
    def _get_sensor_class(self, sensor_type):
        """
        根据传感器类型获取对应实现类
        
        Args:
            sensor_type: 传感器类型字符串
            
        Returns:
            传感器类
        """
        sensor_classes = {
            'mujoco_depth_camera': MujocoDepthCamera,
            'mujoco_rgb_camera': MujocoRGBCamera,
            # 未来可以添加实机传感器
        }
        
        if sensor_type not in sensor_classes:
            raise ValueError(f"未知的传感器类型: {sensor_type}. 可用类型: {list(sensor_classes.keys())}")
        
        return sensor_classes[sensor_type]
    
    def get_all_observations(self):
        """
        获取所有传感器的预处理观测数据
        
        Returns:
            所有传感器观测数据的合并数组
        """
        observations = []
        for sensor in self.sensors:
            try:
                obs = sensor.get_processed_observation()
                obs_type = sensor.sensor_type
                
                observations.append({obs_type: obs})

            except Exception as e:
                print(f"获取传感器观测失败: {e}")
                # 添加零观测作为默认值
                obs_shape = sensor.get_observation_shape()
                zero_obs = np.zeros(np.prod(obs_shape))
                observations.append(zero_obs)
        
        # 合并所有观测
        # if len(observations) > 0:
        #     return np.concatenate(observations)
        return np.array([])
    
    def reset(self):
        """
        重置所有传感器
        """
        for sensor in self.sensors:
            try:
                sensor.reset()
            except Exception as e:
                print(f"重置传感器失败: {e}")
    
    def get_observation_shape(self):
        """
        获取总观测形状
        
        Returns:
            总观测形状元组
        """
        total_obs_size = {}
        for sensor in self.sensors:
            try:
                cur_shape = sensor.get_observation_shape()
                cur_type = sensor.type
                cur_obs = {cur_type: cur_shape}
                total_obs_size.update(cur_obs)
            except Exception as e:
                print(f"获取传感器观测形状失败: {e}")
        
        return total_obs_size
    
    def get_sensor_by_type(self, sensor_type):
        """
        根据类型获取特定传感器
        
        Args:
            sensor_type: 传感器类型
            
        Returns:
            传感器实例或None
        """
        for sensor in self.sensors:
            if sensor.__class__.__name__.lower() == sensor_type.lower():
                return sensor
        return None