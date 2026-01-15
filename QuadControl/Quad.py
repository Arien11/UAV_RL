# QuadrotorController.py
from QuadControl.controller.se3_controller import *
from QuadControl.utils.motor_mixer import *


class Quadrotor:
    def __init__(self):
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
            goal_pos = np.array([0.0, 0.0, 0.6])
            goal_heading = np.array([1.0, 0.0, 0.0])
        
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
    
    # def _log_control(self, curr_state, goal_state, control_command):
    #     """记录控制信息"""
    #     print(f"Control Thrust: {control_command.thrust:.4f}")
    #     print(f"Position: X:{curr_state.pos[0]:.2f} Y:{curr_state.pos[1]:.2f} Z:{curr_state.pos[2]:.2f}")
    #     print(f"Goal Pos: X:{goal_state.pos[0]:.2f} Y:{goal_state.pos[1]:.2f} Z:{goal_state.pos[2]:.2f}")
    #     print("-" * 40)
    
    def _get_dt(self):
        return self.dt


if __name__ == '__main__':
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
    
    
    def test_quadrotor_basic():
        """测试Quadrotor类的基础功能"""
        print("=== Quadrotor类基础功能测试 ===")
        
        # 1. 创建四旋翼对象
        quad = Quadrotor()
        print(f"1. 成功创建Quadrotor对象")
        print(f"   质量: {quad.mass} kg")
        print(f"   重力: {quad.gravity} m/s²")
        print(f"   最大推力: {quad.max_thrust} N")
        
        # 模拟当前状态
        curr_pos = np.array([0.0, 0.0, 0.1])  # 当前位置
        curr_vel = np.array([0.0, 0.0, 0.0])  # 当前速度
        curr_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 当前姿态（无旋转）
        curr_omega = np.array([0.0, 0.0, 0.0])  # 当前角速度
        
        # 目标状态
        goal_pos = np.array([0.0, 0.0, 0.3])  # 目标高度0.3m
        goal_heading = np.array([1.0, 0.0, 0.0])  # 朝向X轴
        # obs =
        # try:
        #     # 计算控制信号
        #     motor_inputs, control_info = quad._cal_control(
        #         curr_pos=curr_pos,
        #         curr_vel=curr_vel,
        #         curr_quat=curr_quat,
        #         curr_omega=curr_omega,
        #         goal_pos=goal_pos,
        #         goal_heading=goal_heading
        #     )
        #
        #     print(f"   电机输入: {motor_inputs}")
        #     print(f"   推力: {control_info['thrust']:.4f}")
        #     print(f"   扭矩: {control_info['torque']}")
        #     print(f"   电机转速: {control_info['motor_speed']}")
        #     print(f"   目标位置: {control_info['goal_pos']}")
        #
        # except Exception as e:
        #     print(f"   控制计算失败: {e}")
    
    
    test_quadrotor_basic()
