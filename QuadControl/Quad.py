# QuadrotorController.py
from QuadControl.controller.se3_controller import *


class Quadrotor:
    def __init__(self, task=None, client=None):
        self.task = task
        self.iteration_count = 0
        self.client = client
        # 物理参数
        self.gravity = 9.8066  # 重力加速度 单位m/s^2
        self.mass = 0.033  # 飞行器质量 单位kg
        self.J = np.diag([1.395e-5, 1.395e-5, 2.173e-5])  # 惯量矩阵
        
        self.Ct = 3.25e-4  # 电机推力系数 (N/krpm^2)
        self.Cd = 7.9379e-6  # 电机反扭系数 (Nm/krpm^2)
        
        # 电机参数
        self.arm_length = 0.065 / 2.0  # 电机力臂长度 单位m
        self.max_thrust = 0.1573  # 单个电机最大推力 单位N
        self.max_torque = 3.842e-03  # 单个电机最大扭矩 单位Nm
        
        self.motor_names = ['motor1', 'motor2', 'motor3', 'motor4']
        # ======================== controller ======================== #
        # 1.SE3控制器
        self.ctrl = SE3Controller(self.mass, self.J)
        # 2.PID控制器
        
        # 仿真周期 1000Hz 1ms 0.001s
        self.dt = 0.001
        
        # 日志
        self.log_count = 0
    
    def _get_dt(self):
        return self.dt
