import numpy as np
from base_task import BaseTask


def simple_trajectory(time):
    """简易轨迹生成器"""
    wait_time = 1.5
    height = 0.3
    radius = 0.5
    speed = 0.3
    
    if time < wait_time:
        return np.array([radius, 0, height]), np.array([0.0, 1.0, 0.0])
    
    _cos = np.cos(2 * np.pi * speed * (time - wait_time))
    _sin = np.sin(2 * np.pi * speed * (time - wait_time))
    _heading = np.array([-_sin, _cos, 0])
    _pos = np.array([radius * _cos, radius * _sin, height])
    
    return _pos, _heading


class TraceTask(BaseTask):
    """任务目标是追踪轨迹，做误差补偿，输出的是对参考轨迹的小修正量"""
    
    def __init__(self, client=None):
        ref_pos, ref_heading = simple_trajectory(time)
        self.ref_pos = ref_pos
        self.ref_heading = ref_heading
        self._client = client
    
    def step(self):
        pass
    
    def setup(self):
        pass
    
    def done(self):
        pass
    
    def calc_reward(self, action):
        current_pos = self._client.get_pos()
        current_vel = self._client.get_vel()
        current_orientation = self._client.get_orientation()

        Params = self.Rewards["Balance"]
        pos_reward = np.linalg.norm(current_pos - self.ref_pos) * Params["lambda_pos"]
        vel_reward = abs(current_vel - 0) * Params["lambda_vel"]

        action_reward = (np.linalg.norm(action) ** 2) * Params["lambda_act"]
        
        total_rewards = (
            -pos_reward
            -vel_reward
            -action_reward
        )
        return total_rewards
