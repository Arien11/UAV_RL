import collections
from typing import Optional

import envs.Observe as Observe
from Tasks.Hover_Task import *
from envs.Simulators.MujocoSim import *
from envs.QuadBaseEnv import QuadBaseEnv
from envs.interface import RobotInterface
from QuadControl.Quad import *
from envs.config_builder import Configuration
from Tasks.trace_task import TraceTask


# 指定无人机的基类配置
class QuadEnv(QuadBaseEnv):
    def _setup_robot(self):
        """设置机器人组件"""
        control_dt = 0.02
        # 设置交互接口interface
        self.interface = RobotInterface(self.model, self.data)
        self.frame_skip = int(control_dt / self.interface.sim_dt())
        # 设置任务task
        self._setup_task(control_dt)
        
        # 设置Robot
        self.robot = Quadrotor(self.task, self.interface)
        
        # 设置初始状态init state
        self.nominal_pose = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    
    def _setup_task(self, control_dt: float) -> None:
        """Setup the task instance. Must set self.task."""
        # self.task = TraceTask(self.interface)
        self.task = HoverTask(self.interface)
        
        # self.task.setup()
        pass
    
    def _setup_spaces(self):
        """设置动作空间与观测空间"""
        # 动作空间
        self.action_space = self.task.action_space
        self.prev_prediction = self.task.action_space
        
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
    
    def _do_simulation(self, ctrl_cmds, n_frames):
        for _ in range(n_frames):
            for i, motor_name in enumerate(self.robot.motor_names):
                self.data.actuator(motor_name).ctrl[0] = ctrl_cmds[i]
            mujoco.mj_step(self.model, self.data, 1)
        
        # 每 100 步打印高度（需能从环境访问 task）
        # if hasattr(self, 'task'):
        #     h = self.task.client.get_pos()[2]
        #     print(f"[CALIB] thrust={ctrl_cmds[0]:.3f}, height={h:.3f}")
    
    def step(self, action):
        # Get offsets from nominal pose
        # offsets = self._get_action_offsets()
        ctrl_cmds = self.task.interpret_action_e2e(action)
        # === task计算reward ===
        self.task.step()
        rewards = self.task.calc_reward(action)
        done = self.task.done()
        total_reward = sum(rewards.values())

        # === robot step ===
        # sim_input = self.robot.step(action)
        obs = self.get_obs()
        self.prev_prediction = action
        
        # === 与仿真器交互(端到端就直接给仿真器) ===
        # self._do_simulation(sim_input, self.frame_skip)
        self._do_simulation(ctrl_cmds, self.frame_skip)
        # print(f"[DEBUG] raw action: {action}, motor_cmds: {ctrl_cmds}")
        # 域随机化
        # if self.dynrand_interval > 0 and np.random.randint(self.dynrand_interval) == 0:
        #     self._randomize_dynamics()
        #
        # if self.perturb_interval > 0 and np.random.randint(self.perturb_interval) == 0:
        #     self._apply_perturbation()
        
        return obs, total_reward, done, rewards
    
    @staticmethod
    def _get_robot_state_len():
        """Return length of UAV state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, qx, qy, qz, qw,
        """
        return 13
    
    @staticmethod
    def _get_num_external_obs():
        """Return length of UAV  external state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, q1, q2, q3, q4,
        """
        return 0
