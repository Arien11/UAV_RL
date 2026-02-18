# 与环境交互的各种接口
import collections
import os
import mujoco
import numpy as np
from utils.quadtools import *


class RobotInterface:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.stepCounter = 0
    
    def nq(self):
        return self.model.nq
    
    def nu(self):
        return self.model.nu
    
    def nv(self):
        return self.model.nv
    
    def sim_dt(self):
        return self.model.opt.timestep
    
    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)
    
    def get_pos(self):
        return self.data.qpos.copy()[:3]
    
    def get_quat(self):
        qw, qx, qy, qz = self.data.qpos.copy()[3:]
        quat = np.array([qx, qy, qz, qw])
        return quat
    
    def get_euler(self):
        return quat_to_euler(self.get_quat())
    
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
    
    def get_obs(self):
        import numpy as np
        pos = self.get_pos()
        quat = self.get_quat()
        vel = self.get_vel()
        omega = self.get_angular_vel()
        return np.concatenate([pos, quat, vel, omega])
    
    def get_time(self):
        return self.data.time
    
    def step(self, mj_step=True, nstep=1):
        """
        (Adapted from dm_control/mujoco/engine.py)

        Advances physics with up-to-date position and velocity dependent fields.
        Args:
          nstep: Optional integer, number of steps to take.
        """
        if mj_step:
            mujoco.mj_step(self.model, self.data, nstep)
            self.stepCounter += nstep
            return
        
        # In the case of Euler integration we assume mj_step1 has already been
        # called for this state, finish the step with mj_step2 and then update all
        # position and velocity related fields with mj_step1. This ensures that
        # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
        # integrators (e.g. RK4) an additional mj_step1 must be called after the
        # last mj_step to ensure mjData syncing.
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
            mujoco.mj_step2(self.model, self.data)
            if nstep > 1:
                mujoco.mj_step(self.model, self.data, nstep - 1)
        else:
            mujoco.mj_step(self.model, self.data, nstep)
        
        mujoco.mj_step1(self.model, self.data)
        
        self.stepCounter += nstep
