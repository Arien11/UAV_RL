import collections
import copy
import yaml
from typing import Optional, Dict, Any, Union
import envs.Observe as Observe
from envs.Simulators.MujocoSim import *
from envs.interface import RobotInterface
from envs.config_builder import Configuration
from envs.Sensors.SensorsManager import SensorManager
from Tasks.trace_task import TraceTask
from QuadControl.Quad import *
from Tasks.Hover_Task import *


class QuadEnv(MuJoCoSimulator):
    def __init__(self, config):
        MuJoCoSimulator.__init__(self, config)
        self.cfg = Configuration(**config)
        
        # 历史记录长度
        self.motion_history_len = getattr(self.cfg, 'motion_history_len', 1)
        self.ext_history_len = getattr(self.cfg, 'ext_history_len', 1)
        
        # 历史记录 - 分别存储运动状态和传感器数据
        self.motion_history = collections.deque(maxlen=self.motion_history_len)
        self.ext_history = {}  # {sensor_name: deque}
        
        # 智能体设置
        self.interface = None
        self.nominal_pose = None
        self.task = None
        self.robot = None
        self._setup_robot()
        
        self.default_model = copy.deepcopy(self.model)
        self.action_space = None
        self._setup_spaces()
    
    def _setup_robot(self):
        """设置机器人组件"""
        control_dt = 0.02

        # 设置数据交互接口interface
        self.interface = RobotInterface(self.model, self.data)
        self.frame_skip = int(control_dt / self.interface.sim_dt())

        # 设置任务task
        self._setup_task()
        
        # 设置Robot
        self.robot = Quadrotor(self.task, self.interface)
        
        # 设置传感器管理器
        sensor_config = self._get_sensor_config()
        self.sensor_manager = SensorManager(sensor_config, sim=self)

        # 设置初始状态init state
        self.nominal_pose = [0, 0, 0.5, 1.0, 0.0, 0.0, 0.0]
    
    def _get_sensor_config(self):
        """
        从配置中获取传感器配置
        
        Returns:
            传感器配置字典
        """
        if hasattr(self.cfg, 'sensors') and self.cfg.sensors:
            sensors_list = []
            for sensor in self.cfg.sensors:
                if hasattr(sensor, 'to_dict'):
                    sensors_list.append(sensor.to_dict())
                else:
                    sensors_list.append(sensor)
            return {
                'sensors': sensors_list
            }
        # 如果配置中没有传感器，返回默认配置
        return {}
    
    def _setup_task(self) -> None:
        """Setup the task instance. Must set self.task."""
        # self.task = TraceTask(self.interface)
        self.task = HoverTask(self.interface)
    
    def _setup_spaces(self):
        """
        设置动作空间与观测空间
        
        观测空间结构：
        {
            'motion': np.ndarray,  # 运动状态 (位置、速度、姿态)
            'sensors': Dict[str, np.ndarray]  # 传感器数据 (按传感器名称)
        }
        """
        self.action_space = self.task.action_space
        self.prev_prediction = self.task.action_space
        
        base_motion_len = self._get_robot_state_len()
        
        self.observation_space = {
            'motion_shape': (base_motion_len,),
            'sensors_shapes': self._get_sensors_shapes()
        }
    
    def get_obs(self) -> Dict[str, Any]:
        """
        获取完整的观测量 - 分离运动状态和传感器数据
        Returns:
            {
                'motion': np.ndarray,             # 运动状态 (motion_history_len, motion_len)
                'sensors': Dict[str, np.ndarray]  # 传感器数据 (按传感器名称)
            }
        """
        motion_state = self._get_robot_state()
        sensors_data = self._get_sensors_data()
        
        # 存储运动状态到历史记录
        self.motion_history.appendleft(motion_state)
        
        # 存储传感器数据到历史记录
        for sensor_name, sensor_data in sensors_data.items():
            if sensor_name not in self.ext_history:
                self.ext_history[sensor_name] = collections.deque(maxlen=self.ext_history_len)
            self.ext_history[sensor_name].appendleft(sensor_data)
        
        # 填充历史记录，长度为motion_history_len和ext_history_len
        if len(self.motion_history) == 0:
            for _ in range(self.motion_history_len):
                self.motion_history.appendleft(np.zeros_like(motion_state))
        
        for sensor_name, sensor_data in sensors_data.items():
            if len(self.ext_history.get(sensor_name, [])) == 0:
                for _ in range(self.ext_history_len):
                    self.ext_history[sensor_name].appendleft(np.zeros_like(sensor_data))
        
        return {
            'motion': np.array(self.motion_history),
            'sensors': {
                sensor_name: np.array(history)
                for sensor_name, history in self.ext_history.items()
            }
        }
    
    def _get_robot_state(self):
        """获得机器人运动状态观测量"""
        pos = Observe.get_pos(self.interface)
        quat = Observe.get_quat(self.interface)
        vel = Observe.get_vel(self.interface)
        omega = Observe.get_omega(self.interface)
        
        return np.concatenate([pos, quat, vel, omega])
    
    def _get_sensors_data(self) -> Dict[str, np.ndarray]:
        """
        获得所有传感器的数据
        
        Returns:
            {sensor_name: sensor_data}
        """
        sensors_data = {}
        # 遍历所有传感器，获取数据
        if hasattr(self, 'sensor_manager'):
            for i, sensor in enumerate(self.sensor_manager.sensors):
                sensor_name = sensor.type     # 获取传感器名称，或使用默认值
                try:
                    data = sensor.get_observation()
                    sensors_data[sensor_name] = data
                except Exception as e:
                    print(f"获取传感器 {sensor_name} 数据失败: {e}")
        return sensors_data
    
    def _get_sensors_shapes(self) -> Dict[str, tuple]:
        """
        获取所有传感器的形状
        
        Returns:
            {sensor_name: shape_tuple}
        """
        shapes = {}
        if hasattr(self, 'sensor_manager'):
            for i, sensor in enumerate(self.sensor_manager.sensors):
                sensor_name = sensor.type
                try:
                    shape = sensor.get_observation_shape()
                    shapes[sensor_name] = shape
                except Exception as e:
                    print(f"获取传感器 {sensor_name} 形状失败: {e}")
        return shapes

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
    
    def _do_simulation(self, traj_info, n_frames):
        for _ in range(n_frames):
            # 轨迹解算
            if traj_info["traj_active"]:
                t = self.interface.data.time - traj_info["traj_start_time"]
                if t <= traj_info["traj_duration"]:
                    # 从求解器获取期望位置（可选速度、加速度）
                    des_pos = np.array([
                        traj_info["solver_x"].get_position(t),
                        traj_info["solver_y"].get_position(t),
                        traj_info["solver_z"].get_position(t)
                    ])
                    des_vel = np.array([
                        traj_info["solver_x"].get_velocity(t),
                        traj_info["solver_y"].get_velocity(t),
                        traj_info["solver_z"].get_velocity(t)
                    ])
                    des_acc = np.array([
                        traj_info["solver_x"].get_acceleration(t),
                        traj_info["solver_y"].get_acceleration(t),
                        traj_info["solver_z"].get_acceleration(t)
                    ])
                else:
                    # 轨迹结束，悬停在最后一点（或等待新轨迹）
                    traj_info["traj_active"] = False
                    # 可以选择保持最后一点作为目标，或让控制器悬停
            else:
                # 无轨迹时，使用任务默认参考点（例如悬停）
                ref_pos, ref_yaw = self.task.get_reference(self.interface.data.time)
                des_pos = ref_pos
            pos = self.interface.get_pos()
            quat = self.interface.get_quat()
            vel = self.interface.get_vel()
            omega = self.interface.get_omega()

            # 构建期望状态
            desired_state = {'pos_des': des_pos, 'vel_des': des_vel,
                             'acc_des': des_acc, 'yaw_des': 0.0}
            # print(f"t={t:.3f}, pos0={traj_info['pos0']}, pos1={traj_info['pos1']}, des_pos={des_pos}")
            state = {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}
            
            # 计算控制输入
            ctrl, thrust_norm, scale, tau_des_max = self.robot.ctrl.update(state, desired_state)
            self.data.ctrl[:] = ctrl

            # 执行物理模拟
            mujoco.mj_step(self.model, self.data)
            # print(ctrl)
    
    def step(self, action):
        # Get offsets from nominal pose
        # offsets = self._get_action_offsets()
        
        # 解析动作，生成轨迹信息
        traj_info = self.task.interpret_action_e2e(action)
        
        self._do_simulation(traj_info, self.frame_skip)
        
        obs = self.get_obs()
        
        rewards = self.task.calc_reward(action)
        total_reward = sum(rewards.values())
        done = self.task.done()
        
        self.prev_prediction = action
        
        return obs, total_reward, done, rewards
    
    def _get_robot_state_len(self):
        """Return length of UAV state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, qx, qy, qz, qw,
        """
        return 13
    
    def reset_model(self):
        """
        重置环境，包含传感器重置
        """
        # 重置初始状态
        init_qpos = self.nominal_pose.copy()
        init_qvel = [0] * self.interface.nv()
        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        
        # 重置任务状态
        self.task.reset(iter_count=self.robot.iteration_count)

        # 重置历史记录
        self.motion_history.clear()
        self.ext_history.clear()
        
        # 重置传感器状态
        if hasattr(self, 'sensor_manager'):
            try:
                self.sensor_manager.reset()
            except Exception as e:
                print(f"重置传感器失败: {e}")
        
        return self.get_obs()
    
    def _apply_init_noise(self):
        pass
    
    def _apply_observation_noise(self):
        pass
