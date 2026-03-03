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
        omega = Observe.get_omega(self.interface)
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
    
    def _do_simulation(self, traj_info, n_frames):
        for _ in range(n_frames):
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
            desired_state = {'pos_des': des_pos, 'vel_des': des_vel,
                             'acc_des': des_acc, 'yaw_des': 0.0}
            # print(f"t={t:.3f}, pos0={traj_info['pos0']}, pos1={traj_info['pos1']}, des_pos={des_pos}")
            state = {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}
            ctrl, thrust_norm, scale, tau_des_max = self.robot.ctrl.update(state, desired_state)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            # print(ctrl)
    
    def step(self, action):
        # Get offsets from nominal pose
        # offsets = self._get_action_offsets()
        
        # 解析动作，生成轨迹信息
        traj_info = self.task.interpret_action_e2e(action)
        
        # 执行仿真
        self._do_simulation(traj_info, self.frame_skip)
        
        # 仿真后获取观测并计算奖励
        obs = self.get_obs()
        
        rewards = self.task.calc_reward(action)  # 现在 action 是网络输出
        total_reward = sum(rewards.values())
        done = self.task.done()
        
        self.prev_prediction = action
        
        # 域随机化
        # if self.dynrand_interval > 0 and np.random.randint(self.dynrand_interval) == 0:
        #     self._randomize_dynamics()
        #
        # if self.perturb_interval > 0 and np.random.randint(self.perturb_interval) == 0:
        #     self._apply_perturbation()
        
        return obs, total_reward, done, rewards
    
    def _get_robot_state_len(self):
        """Return length of UAV state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, qx, qy, qz, qw,
        """
        return 13 * self.history_len
    
    def _get_num_external_obs(self):
        """Return length of UAV  external state vector
        Px, Py, Pz, Vx, Vy, Vz, Wx, Wy, Wz, q1, q2, q3, q4,
        """
        return 0
