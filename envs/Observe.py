from abc import ABC, abstractmethod
import numpy as np
import interface


# class ObservationSpaceBuilder(ABC):
#     """观测空间构建器"""
#
#     @abstractmethod
#     def build_space(self, sim_state):
#         """构建观测空间"""
#         pass
#
#     @abstractmethod
#     def get_observation(self, sim_state):
#         """获取观测值"""
#         pass


def get_pos(interface):
    return interface.get_pos()


def get_quat(interface):
    return interface.get_quat()


def get_vel(interface):
    return interface.get_vel()


def get_angular_vel(interface):
    return interface.get_angular_vel()


def get_acc(interface):
    return interface.get_acc()


def get_cvel(interface):
    return interface.get_cvel()


class ObservationSpace:
    def __init__(self, include_qpos=True, include_qvel=True, normalize=True):
        self.include_qpos = include_qpos
        self.include_qvel = include_qvel
        self.normalize = normalize
    
    def build_space(self, state):
        """构建观测空间，用于描述观测空间信息"""
        dim = 0
        if self.include_qpos:
            dim += len(state['qpos'])
        if self.include_qvel:
            dim += len(state['qvel'])
        return {
            'shape': (dim,),
            # 'data_type': [key for key in state.keys()]
            'dtype': np.float32,
            'low': -np.inf,
            'high': np.inf
        }
    
    def get_quat(self):
        pass
    
    def get_orientation(self):
        pass
    
    def get_velocity(self):
        pass
    
    def get_positions(self):
        pass
    
    def get_angel_vel(self):
        pass
    
    def get_observation(self, state):
        if self.include_qpos:
            position = state['qpos'][:3]
            qw, qx, qy, qz = state['qpos'][3:]
            quaternion = np.array([qx, qy, qz, qw])
        if self.include_qvel:
            linear_vel = state['qvel'][:3]
            angular_vel = state['qvel'][3:]
        obs = np.concatenate([
            position,  # 3
            linear_vel,  # 3
            quaternion,  # 4
            angular_vel,  # 3
        ])
        
        if self.normalize:
            obs = self._normalize(obs)
        
        return obs
    
    def _get_rl_state(self):
        ...
        # state = self._get_state()
        # # sensor_state = self._get_sensor_state()
        #
        # # 位置和姿态
        # position = state['qpos'][:3]
        # quaternion = state['qpos'][3:]
        #
        # # 线速度和角速度
        # linear_vel = state['qvel'][:3]
        # angular_vel = state['qvel'][3:]
        #
        # # 组合成RL状态
        # rl_state = np.concatenate([
        #     position,  # 3
        #     quaternion,  # 4
        #     linear_vel,  # 3
        #     angular_vel,  # 3
        #     state['ctrl']  # 4 (当前控制输入)
        # ])  # 总共 17 维
        #
        # return rl_state
    
    def _normalize(self, obs):
        # 标准化逻辑
        return obs


if __name__ == '__main__':
    state = {
        'qpos': [0, 1, 1],
        'qvel': [0, 2, 2]
    }
    
    obs = ObservationSpace()
    temp = obs.build_space(state)
    print(temp)
