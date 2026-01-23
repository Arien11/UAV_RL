import collections
from typing import Optional

import Observe
from Tasks.Hover_Task import *
from envs.Simulators.MujocoSim import *
from QuadBaseEnv import QuadBaseEnv
from interface import RobotInterface
from QuadControl.Quad import *
from envs.config_builder import Configuration


# 指定无人机的基类配置
class QuadEnv(QuadBaseEnv):
    @abstractmethod
    def _setup_task(self, control_dt: float) -> None:
        """Setup the task instance. Must set self.task."""
        pass
    
    def _setup_robot(self):
        """设置机器人组件"""
        
        control_dt = self.cfg.contorl_dt
        # 设置交互接口interface
        self.interface = RobotInterface(self.model, self.data)
        
        # 设置任务task
        self._setup_task(control_dt)
        
        # 设置Robot
        self.robot = Quadrotor()
    
    def _setup_spaces(self):
        """设置动作空间与观测空间"""
        # 动作空间
        action_space_size = 4
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size)
        
        # 观测空间
        self.base_obs_len = self._get_robot_state_len() + self._get_num_external_obs()  # 机器人状态 + 外部观测
        self.observation_space = np.zeros(self.base_obs_len * self.history_len)  # 历史状态堆叠
        
        # 观测量标准化
        # self._setup_obs_normalization()
    
    def _get_robot_state(self):
        """获得机器人状态观测量"""
        pos = Observe.get_pos(self.interface)
        quat = Observe.get_quat(self.interface)
        vel = Observe.get_vel(self.interface)
        omega = Observe.get_angular_vel(self.interface)
        # 有需要可以添加观测噪声
        
        return np.concatenate([pos, quat, vel, omega])
    
    def _get_external_state(self):
        """获得机器人外部观测量"""
        return None
    
    def _setup_domain_randomization(self):
        pass
    
    def _setup_obs_normalization(self):
        self.obs_mean = np.concatenate(
            ()
        )
        self.obs_std = np.concatenate(
            (
                [0.2, 0.2, 1, 1, 1],
                0.5 * np.ones(10),
                4 * np.ones(10),
                100 * np.ones(10),
            )
        )
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)
    
    @staticmethod
    def _get_robot_state_len():
        """Return length of UAV state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, q1, q2, q3, q4,
        """
        return 13
    
    @staticmethod
    def _get_num_external_obs():
        """Return length of UAV  external state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, q1, q2, q3, q4,
        """
        return 0


if __name__ == '__main__':
    with open("../config/Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    temp = QuadEnv("../config/env_config.yaml", cfg)
    print()
