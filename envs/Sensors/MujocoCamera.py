"""
MuJoCo深度相机传感器实现
"""
import numpy as np
from .BaseSensor import BaseSensor


class MujocoDepthCamera(BaseSensor):
    """MuJoCo深度相机传感器"""
    
    def __init__(self, config):
        """
        初始化MuJoCo深度相机
        
        Args:
            config: 传感器配置字典
        """
        super().__init__(config)
        self.camera_name = config.get('camera_name', 'depth_camera')
        self.type = config.get('sensor_type', 'depth')
        self.width = config.get('width', 160)
        self.height = config.get('height', 120)
        self.fov = config.get('fov', 90)
        self.max_depth = config.get('max_depth', 5.0)
        self.min_depth = config.get('min_depth', 0.1)
        
        # 初始化相机参数
        self._cached_depth = None
        self._last_frame = -1
    
    def get_observation(self):
        """
        获取原始深度图像
        
        Returns:
            原始深度图像数组
        """
        if self._sim is None:
            raise RuntimeError("Simulator not set. Call set_simulator() first.")
        
        # 从仿真器获取深度数据
        try:
            depth = self._sim.get_camera_depth()
            self._cached_depth = depth
            return depth
        except Exception as e:
            print(f"获取深度图像失败: {e}")
            # 返回黑色图像作为默认值
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def preprocess_observation(self, raw_depth):
        """
        深度图像预处理：归一化和裁剪
        
        Args:
            raw_depth: 原始深度图像
            
        Returns:
            预处理后的深度图像
        """
        # 深度裁剪到有效范围
        depth_clipped = np.clip(raw_depth, self.min_depth, self.max_depth)
        
        # 归一化到 [0, 1] 范围
        normalized_depth = (depth_clipped - self.min_depth) / (self.max_depth - self.min_depth)
        normalized_depth = np.clip(normalized_depth, 0, 1)
        
        return normalized_depth
    
    def get_observation_shape(self):
        """
        获取观测形状
        
        Returns:
            观测形状元组 (height, width)
        """
        return (self.height, self.width)
    
    def reset(self):
        """
        重置相机状态
        """
        self._cached_depth = None
        self._last_frame = -1


class MujocoRGBCamera(BaseSensor):
    """MuJoCo RGB相机传感器"""
    
    def __init__(self, config):
        """
        初始化MuJoCo RGB相机
        
        Args:
            config: 传感器配置字典
        """
        super().__init__(config)
        self.camera_name = config.get('camera_name', 'drone_rgb_camera')
        self.width = config.get('width', 160)
        self.height = config.get('height', 120)
        self.fov = config.get('fov', 90)
        
        # 初始化相机参数
        self._cached_rgb = None
    
    def get_observation(self):
        """
        获取原始RGB图像
        
        Returns:
            原始RGB图像数组 (H, W, 3)
        """
        if self._sim is None:
            raise RuntimeError("Simulator not set. Call set_simulator() first.")
        
        # 从仿真器获取RGB数据
        try:
            rgb = self._sim.get_camera_rgb(self.camera_name)
            self._cached_rgb = rgb
            return rgb
        except Exception as e:
            print(f"获取RGB图像失败: {e}")
            # 返回黑色图像作为默认值
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def preprocess_observation(self, raw_rgb):
        """
        RGB图像预处理：归一化到 [0, 1]
        
        Args:
            raw_rgb: 原始RGB图像
            
        Returns:
            预处理后的RGB图像
        """
        # 归一化到 [0, 1] 范围
        normalized_rgb = raw_rgb.astype(np.float32) / 255.0
        return normalized_rgb
    
    def get_observation_shape(self):
        """
        获取观测形状
        
        Returns:
            观测形状元组 (height, width, 3)
        """
        return (self.height, self.width, 3)
    
    def reset(self):
        """
        重置相机状态
        """
        self._cached_rgb = None