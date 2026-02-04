import numpy as np
from .base_task import BaseTask
from utils.quadtools import *


def simple_trajectory(time):
    # time为实时时间
    """简易轨迹生成器"""
    wait_time = 1.5
    height = 0.3
    radius = 0.5
    speed = 0.3
    # 构建机头朝向
    _cos = np.cos(2 * np.pi * speed * (time - wait_time))
    _sin = np.sin(2 * np.pi * speed * (time - wait_time))
    # 首先等待起飞到目标开始点位
    if time < wait_time:
        return np.array([radius, 0, height]), np.array([0.0, 1.0, 0.0])
    
    # 随后开始绕圈(逆时针旋转)
    _heading = np.array([-_sin, _cos, 0])
    _pos = np.array([radius * _cos, radius * _sin, height])
    
    return _pos, _heading


class TraceTask(BaseTask):
    """任务目标是追踪轨迹，做误差补偿，输出的是对参考轨迹的小修正量"""
    
    def __init__(self, client=None, dt=None):
        super().__init__()
        self.client = client
        self.ref_pos = 0
        self.dt = dt
        
        self.height = 0.3
        
        # 终止阈值
        self.max_pos_error = 0.6
        self.min_height = 0.1
        self.max_tilt = np.deg2rad(60)
    
    def step(self):
        pass
    
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def setup(self):
        pass
    
    def get_reference(self, time):
        ref_pos, ref_heading = simple_trajectory(time)
        return ref_pos, ref_heading
    
    def calc_reward(self, action):
        current_pos = self.client.get_pos()
        current_vel = self.client.get_vel()
        current_quat = self.client.get_quat()
        current_time = self.client.get_time()
        ref_pos, ref_heading = self.get_reference(current_time)
        
        # 位置误差
        pos_error = np.linalg.norm(current_pos - ref_pos)
        
        # yaw 误差
        yaw = quat_to_yaw(current_quat)
        ref_yaw = np.arctan2(ref_heading[1], ref_heading[0])
        yaw_error = angle_wrap(yaw - ref_yaw)
        
        reward = 0.0
        reward -= 2.0 * pos_error ** 2
        reward -= 0.5 * np.linalg.norm(current_vel) ** 2
        reward -= 0.3 * yaw_error ** 2
        reward -= 0.01 * np.linalg.norm(action) ** 2
        
        # 稳定飞行奖励
        if pos_error < 0.1:
            reward += 0.5
        return reward
    
    def done(self):
        current_pos = self.client.get_pos()
        current_quat = self.client.get_quat()
        current_time = self.client.get_time()
        
        # 掉高度
        if current_pos[2] < self.min_height:
            return True
        
        # 离轨太远
        ref_pos, _ = self.get_reference(current_time)
        if np.linalg.norm(current_pos - ref_pos) > self.max_pos_error:
            return True
        
        # 翻机
        roll, pitch, _ = quat_to_euler(current_quat)
        if abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            return True
        
        return False
