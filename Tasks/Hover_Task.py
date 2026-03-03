import numpy as np
from .base_task import *
from utils.quadtools import *  # 如果不需要可删除
from QuadControl.poly_solver import *


class HoverTask(BaseTask):
    def __init__(self, client=None, dt=None, target_pos=None):
        super().__init__()
        self.client = client
        self.dt = dt
        self.target_pos = np.array([0.0, 0.0, 0.5])  # 悬停目标位置
        self.step_count = 0
        self._termination_printed = False
        
        # ========== 轨迹部分定义 ==========
        self.traj_start_time = 0.0  # 轨迹开始时的仿真时间
        self.traj_duration = 1.0  # 轨迹总时长（秒），可调参数
        self.traj_active = False  # 是否有有效轨迹
        
        # ========== 动作空间定义 ==========
        # 动作 = [Δpos (3), v_end (3), a_end (3)]
        # 物理意义：对参考位置/航向的修正量
        self.base_thrust = 0.5145  # 校准后的悬停推力
        self.action_dim = 3
        self.action_low = np.array([-0.05, -0.05, -0.05], dtype=np.float32)
        self.action_high = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.action_space = ActionSpace(
            shape=(self.action_dim,),
            low=self.action_low,
            high=self.action_high
        )
        
        # ========== 动力学约束 ==========
        self.max_thrust_rate = 0.05  # 最大推力变化率（每步），防止急动
        self.max_single_thrust = 0.8  # 单个电机最大推力
        self.min_single_thrust = 0.2  # 单个电机最小推力
        
        # ========== 奖励权重 ==========
        self.w_pos = 2.0  # 提高位置惩罚
        self.w_vel = 0.2  # 提高速度惩罚
        self.w_att = 0.01  # 姿态惩罚降至可忽略
        self.alive_bonus = 1.0  # 提高存活奖励
        self.smooth_penalty_coef = 0.05  # 降低动作平滑惩罚
        self.integral_pos_error = np.zeros(3)
        self.integral_decay = 0.99  # 衰减因子，防止无限累积
        self.integral_penalty_coef = 0.05  # 降低积分惩罚系数
        self.close_to_target_bonus = 5.0  # 接近目标的额外奖励
        self.target_threshold = 0.2  # 目标区域阈值

        # ========== 终止阈值（可选放宽） ==========
        self.max_pos_error = 2
        self.min_height = 0.15  # 保持不变（安全考虑）
        self.max_tilt = 1.2  # 原1.047，放宽到约69°，减少因轻微倾斜终止
        
        # 用于动作平滑惩罚的上一时刻动作
        self.prev_action = None
    
    def step(self):
        pass
    
    def get_observation(self, state):
        """获取观测"""
        pass
    
    def setup(self):
        pass
    
    def get_reference(self, time):
        """
        悬停任务：参考位置始终为目标点，无朝向要求（可忽略）。
        返回 (target_pos, 任意朝向向量)
        """
        # 偏航自由，因此返回任意单位向量即可
        return self.target_pos, np.array([0.0, 0.0, 0.0])
    
    def interpret_action_e2e(self, action):
        """
        
        :param action: [Δx, Δy, Δz]，位置偏移修正量
        :return:
        """
        delta_pos = action  # 终点位置偏移
        # v_end = action[3:6]  # 终点速度
        # a_end = action[6:9]  # 终点加速度
        
        # 2. 获取当前无人机状态（起点）
        pos0 = self.client.get_pos()
        vel0 = self.client.get_vel()
        # 加速度可通过差分获得，或暂时设为0（如果无法估计）
        acc0 = np.zeros(3)  # 或使用上一时刻的速度差分
        # if self.client.data.time % 1.0 < 0.01:  # 每秒打印一次
        #     print(f"Action delta: {action}, Actual pos: {pos0}")
        # 3. 终点绝对位置 = 任务参考点 + 偏移（根据你的任务逻辑调整）
        ref_pos, ref_yaw = self.get_reference(self.client.data.time)
        pos1 = ref_pos + delta_pos  # 或直接使用绝对位置
        
        # print(f"ref_pos = {pos1}")
        # 4. 创建 Hermite 求解器（每个轴独立）
        T = self.traj_duration
        solver_x = Poly5Solver(pos0[0], vel0[0], acc0[0], pos1[0], 0, 0, T)
        solver_y = Poly5Solver(pos0[1], vel0[1], acc0[1], pos1[1], 0, 0, T)
        solver_z = Poly5Solver(pos0[2], vel0[2], acc0[2], pos1[2], 0, 0, T)
        
        # 5. 记录轨迹开始时间
        self.traj_start_time = self.client.data.time
        self.traj_active = True
        
        return {
            "pos0": pos0,
            "pos1": pos1,
            "solver_x": solver_x,
            "solver_y": solver_y,
            "solver_z": solver_z,
            "traj_active": self.traj_active,
            "traj_start_time": self.traj_start_time,
            "traj_duration": self.traj_duration,
            "ref_yaw": ref_yaw
        }
    
    def calc_reward(self, action):
        # 获取当前状态
        pos = self.client.get_pos()
        vel = self.client.get_vel()
        target = self.target_pos
        dt = 0.02
        
        # 核心项 1: 位置误差惩罚 (L2范数) - 引导无人机到达目标
        pos_error = np.linalg.norm(pos - target)
        reward_pos = -self.w_pos * pos_error
        
        # 核心项 2: 速度惩罚 (L2范数) - 抑制运动，使无人机趋于静止
        speed = np.linalg.norm(vel)
        reward_vel = -self.w_vel * speed
        
        # 核心项 3: 存活奖励 - 鼓励长时间存活，避免过早终止
        reward_alive = self.alive_bonus
        
        # 核心项 4: 积分惩罚 - 消除累积漂移 (防止缓慢漂移出界)
        self.integral_pos_error = self.integral_decay * self.integral_pos_error + (pos - target) * dt
        integral_penalty = -self.integral_penalty_coef * np.linalg.norm(self.integral_pos_error)
        
        # 核心项 5 (可选): 动作平滑惩罚 - 抑制动作突变，提高稳定性
        if self.prev_action is not None:
            delta_action = np.linalg.norm(action - self.prev_action)
            action_smooth_penalty = -self.smooth_penalty_coef * delta_action
        else:
            action_smooth_penalty = 0.0
        self.prev_action = action.copy()
        
        # 核心项 6: 接近目标的额外奖励 - 提供正反馈
        close_to_target = 0.0
        if pos_error < self.target_threshold:
            close_to_target = self.close_to_target_bonus
        
        # 汇总奖励
        total_reward = (reward_pos + reward_vel + reward_alive +
                        integral_penalty + action_smooth_penalty + close_to_target)
        
        return {
            "pos_err": reward_pos,
            "vel_err": reward_vel,
            "alive": reward_alive,
            "integral_penalty": integral_penalty,
            "action_smooth_penalty": action_smooth_penalty,
            "close_to_target": close_to_target,
        }
    
    def done(self):
        self.step_count += 1
        if self.step_count <= 10:
            return False
        
        pos = self.client.get_pos()
        roll, pitch, _ = self.client.get_euler()
        pos_error = np.linalg.norm(pos - self.target_pos)
        
        # 打印当前状态
        # print(
        #     f"[DONE DEBUG] step={self.step_count}, height={pos[2]:.3f}, min_height={self.min_height}, pos_error={pos_error:.3f}, max_pos_error={self.max_pos_error}, tilt=({abs(roll):.3f},{abs(pitch):.3f}), max_tilt={self.max_tilt}")
        #
        if pos[2] < self.min_height:
            print("  -> TERMINATE: low height")
            return True
        if pos_error > self.max_pos_error:
            print("  -> TERMINATE: pos error")
            return True
        if abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            # print("  -> TERMINATE: tilt")
            return True
        if self.step_count >= 1000:
            # print("  -> TERMINATE: max steps")
            return True
        return False
    
    def reset(self, iter_count):
        self.prev_action = None
        self.step_count = 0
        self._termination_printed = False  # 重置打印标志
        self.integral_pos_error = np.zeros(3)
