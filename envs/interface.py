# 与环境交互的各种接口
import collections
import os
import mujoco


class RobotInterface:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.stepCounter = 0
    
    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)
    
    def get_pos(self):
        return self.data.qpos.copy()[:3]
    
    def get_quat(self):
        return self.data.qpos.copy()[3:]
    
    def get_vel(self):
        return self.data.qvel.copy()[:3]
    
    def get_angular_vel(self):
        return self.data.qvel.copy()[3:]
    
    def get_acc(self):
        return self.data.qacc.copy()
    
    def get_cvel(self):
        return self.data.cvel.copy()
    
    def get_orientation(self):
        return 0
    
    def step(self):
        ...
