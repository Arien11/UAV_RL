# QuadrotorController.py
from QuadControl.controller.se3_controller import *
from QuadControl.utils.motor_mixer import *


class Quadrotor:
    def __init__(self, task=None, client=None):
        self.task = task
        self.iteration_count = 0
        self.client = client
        # 物理参数
        self.gravity = 9.8066  # 重力加速度 单位m/s^2
        self.mass = 0.033  # 飞行器质量 单位kg
        self.Ct = 3.25e-4  # 电机推力系数 (N/krpm^2)
        self.Cd = 7.9379e-6  # 电机反扭系数 (Nm/krpm^2)
        
        # 电机参数
        self.arm_length = 0.065 / 2.0  # 电机力臂长度 单位m
        self.max_thrust = 0.1573  # 单个电机最大推力 单位N
        self.max_torque = 3.842e-03  # 单个电机最大扭矩 单位Nm
        
        # ======================== controller ======================== #
        # 1.SE3控制器
        self.ctrl = SE3Controller()
        self.ctrl.kx = 0.6
        self.ctrl.kv = 0.4
        self.ctrl.kR = 6.0
        self.ctrl.kw = 1.0
        # 2.PID控制器
        
        # 混控器(电机动力分配器)
        self.mixer = Mixer()
        self.torque_scale = 0.001  # 控制器控制量到实际扭矩(Nm)的缩放系数(因为是无模型控制所以需要此系数)
        
        # 仿真周期 1000Hz 1ms 0.001s
        self.dt = 0.001
        
        # 日志
        self.log_count = 0
    
    def calc_motor_force(self, krpm):
        """根据电机转速计算电机推力"""
        return self.Ct * krpm ** 2
    
    def calc_motor_speed_by_force(self, force):
        """根据推力计算电机转速"""
        if force > self.max_thrust:
            force = self.max_thrust
        elif force < 0:
            force = 0
        return np.sqrt(force / self.Ct)
    
    def calc_motor_speed_by_torque(self, torque):
        """根据扭矩计算电机转速"""
        # 注意返回数值为转速绝对值 根据实际情况决定转速是增加还是减少
        if torque > self.max_torque:  # 扭矩绝对值限制
            torque = self.max_torque
        return np.sqrt(torque / self.Cd)
    
    def calc_motor_speed(self, force):
        """根据推力计算电机转速"""
        if force > 0:
            return self.calc_motor_speed_by_force(force)
    
    def calc_motor_torque(self, krpm):
        """根据电机转速计算电机扭矩"""
        return self.Cd * krpm ** 2
    
    def calc_motor_input(self, krpm):
        """根据电机转速计算电机归一化输入"""
        if krpm > 22:
            krpm = 22
        elif krpm < 0:
            krpm = 0
        _force = self.calc_motor_force(krpm)
        _input = _force / self.max_thrust
        if _input > 1:
            _input = 1
        elif _input < 0:
            _input = 0
        return _input
    
    def _cal_control(self, obs, goal=None, time=None):
        """
        
        :param obs:
            position       # 3
            linear_vel     # 3
            quaternion     # 4
            angular_vel    # 3
        :param goal:
        :param time:
        :return: motor_inputs, control_info
        """
        # ================== 构建当前状态 ================== #
        curr_pos = obs[0:3]
        curr_vel = obs[3:6]
        curr_quat = obs[6:10]
        curr_omega = obs[10:13]
        curr_state = State(curr_pos, curr_vel, curr_quat, curr_omega)
        
        # ================== 关键：构建目标状态(轨迹输入部分) ================== #
        if goal is None:
            # 使用默认目标
            goal_pos = np.array([0.0, 0.0, 0.3])
            goal_heading = np.array([1.0, 0.0, 0.0])
        goal_pos = goal[0]
        goal_heading = goal[1]
        goal_vel = np.array([0, 0, 0])
        goal_quat = np.array([0.0, 0.0, 0.0, 1.0])
        goal_omega = np.array([0, 0, 0])
        goal_state = State(goal_pos, goal_vel, goal_quat, goal_omega)
        
        # ================== 更新控制器 ================== #
        control_command = self.ctrl.control_update(
            curr_state, goal_state, self.dt, goal_heading
        )
        ctrl_thrust = control_command.thrust  # 总推力控制量(mg为单位)
        ctrl_torque = control_command.angular  # 三轴扭矩控制量
        # 混控
        mixer_thrust = ctrl_thrust * self.gravity * self.mass  # 机体总推力(N)
        mixer_torque = ctrl_torque * self.torque_scale  # 机体扭矩(Nm)
        
        # 动力分配,
        # input: thrust(机体总推力 单位N）, mx, my, mz(三轴扭矩 单位Nm)
        motor_speed = self.mixer.calculate(
            mixer_thrust, mixer_torque[0], mixer_torque[1], mixer_torque[2]
        )
        
        # 计算电机输入
        motor_inputs = np.array([
            self.calc_motor_input(motor_speed[0]),
            self.calc_motor_input(motor_speed[1]),
            self.calc_motor_input(motor_speed[2]),
            self.calc_motor_input(motor_speed[3])
        ])
        
        # 返回控制信息
        control_info = {
            'thrust': control_command.thrust,
            'torque': control_command.angular,
            'goal_pos': goal_pos,
            'motor_speed': motor_speed
        }
        
        return motor_inputs, control_info
    
    def _get_dt(self):
        return self.dt
    
    def _do_simulation(self, target, n_frames=5):
        """机器人与仿真器环境的交互"""
        ...
    
    def step(self, action, offset=None):
        if not isinstance(action, np.ndarray):
            raise TypeError("Expected action to be a numpy array")
        
        if not isinstance(action, np.ndarray):
            raise TypeError("Expected action to be a numpy array")
        
        action = np.copy(action)
        
        # === 解算 RL action ===
        delta_pos = action[:3]  # Δx, Δy, Δz
        delta_yaw = action[3]  # Δψ
        
        # === 参考轨迹计算 ===
        ref_pos, ref_heading = self.task.get_reference(self.client.data.time)
        
        # === 期望量计算 ===
        goal_pos = ref_pos + delta_pos
        
        ref_yaw = np.arctan2(ref_heading[1], ref_heading[0])
        goal_yaw = ref_yaw + delta_yaw
        goal_heading = np.array([
            np.cos(goal_yaw),
            np.sin(goal_yaw),
            0.0
        ])
        
        # === 期望输入给到robot控制器 ===
        obs = self.client.get_obs()
        motor_inputs, _ = self._cal_control(
            obs=obs,
            goal=(goal_pos, goal_heading),
        )
        
        # === 与仿真器交互 ===
        # self._do_simulation(motor_inputs, self.frame_skip)
        
        # === task计算reward ===
        self.task.step()
        rewards = self.task.calc_reward(action)
        done = self.task.done()
        return rewards, done


if __name__ == '__main__':
    ...
    
