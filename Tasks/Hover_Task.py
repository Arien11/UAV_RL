import numpy as np
from .base_task import *
from utils.quadtools import *  # 如果不需要可删除


class HoverTask(BaseTask):
    """
    悬停任务：无人机在指定目标点保持静止且水平。
    奖励设计：
        - 位置误差惩罚
        - 速度惩罚
        - 姿态（滚转/俯仰）惩罚
        - 动作平滑惩罚
        - 存活奖励
        - 成功奖励（在目标附近且稳定时给予）
    """
    
    def __init__(self, client=None, dt=None, target_pos=None):
        super().__init__()
        self.client = client
        self.dt = dt
        self.target_pos = np.array([0.0, 0.0, 0.5])  # 悬停目标位置
        self.step_count = 0
        self._termination_printed = False
        # ========== 动作空间定义 ==========
        # 动作 = [Δx, Δy, Δz, Δyaw]
        # 物理意义：对参考位置/航向的修正量
        self.base_thrust = 0.5145  # 校准后的悬停推力
        self.action_dim = 4
        self.action_low = np.array([-0.01, -0.01, -0.01, -0.01], dtype=np.float32)
        self.action_high = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        self.action_space = ActionSpace(
            shape=(self.action_dim,),
            low=self.action_low,
            high=self.action_high
        )

        # ========== 奖励权重（调整后） ==========
        self.w_pos = 0.05
        self.w_vel = 0.02
        self.w_att = 0.1
        self.w_act = 0.02
        self.alive_bonus = 0.8
        self.success_bonus = 5.0  # 或改为连续形式

        # ========== 终止阈值（可选放宽） ==========
        self.max_pos_error = 1.5  # 原1.0，稍微放宽，给无人机更多调整空间
        self.min_height = 0.15  # 保持不变（安全考虑）
        self.max_tilt = 1.2  # 原1.047，放宽到约69°，减少因轻微倾斜终止
        
        # 用于动作平滑惩罚的上一时刻动作
        self.prev_motor_cmds = None
    
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
        return self.target_pos, np.array([1.0, 0.0, 0.0])
    
    def interpret_action_e2e(self, action):
        """动作 = 悬停基准 + 小范围偏移"""
        return self.base_thrust + action
    
    def calc_reward(self, motor_cmds):
        """
        计算单步奖励。
        motor_cmds: 实际写入电机的推力指令 [0,1]（已包含 base_thrust）
        """
        pos = self.client.get_pos()
        vel = self.client.get_vel()
        roll, pitch, _ = self.client.get_euler()
        
        # 状态相关惩罚（不变）
        pos_error = np.linalg.norm(pos - self.target_pos)
        reward_pos = -self.w_pos * pos_error
        
        speed = np.linalg.norm(vel)
        reward_vel = -self.w_vel * speed
        
        attitude_error = abs(roll) + abs(pitch)
        reward_att = -self.w_att * attitude_error
        
        # ✅ 动作平滑惩罚：基于实际电机指令的变化
        if self.prev_motor_cmds is not None:
            action_diff = np.linalg.norm(motor_cmds - self.prev_motor_cmds)
            reward_act = -self.w_act * action_diff
        else:
            reward_act = 0.0
        self.prev_motor_cmds = motor_cmds.copy()  # 存储实际指令
        
        # 存活与成功奖励（不变）
        reward_alive = self.alive_bonus
        stable_condition = (pos_error < 0.2) and (speed < 0.2) and (attitude_error < 0.2)
        reward_success = self.success_bonus if stable_condition else 0.0
        
        return {
            "pos_err": reward_pos,
            "vel_err": reward_vel,
            "att_err": reward_att,
            "act_err": reward_act,
            "alive": reward_alive,
            "success": reward_success,
        }
    
    def done(self):
        self.step_count += 1
        if self.step_count <= 10:  # 前10步不终止
            return False
        
        pos = self.client.get_pos()
        roll, pitch, _ = self.client.get_euler()
        
        pos_error = np.linalg.norm(pos - self.target_pos)
        
        # if pos[2] < self.min_height:
        #     print(f"[TERM] Height: {pos[2]:.3f} < {self.min_height}")
        # elif pos_error > self.max_pos_error:
        #     print(f"[TERM] Pos error: {pos_error:.3f} > {self.max_pos_error}")
        # elif abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
        #     print(f"[TERM] Tilt: roll={roll:.3f}, pitch={pitch:.3f} > {self.max_tilt}")
        
        # 1. 掉高度（触地）
        if pos[2] < self.min_height:
            return True
        # 2. 飞得太远
        if pos_error > self.max_pos_error:
            return True
        # 3. 倾斜过度（翻机）
        if abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            return True
        if self.step_count >= 1000:
            return True
        return False
    
    def reset(self, iter_count):
        self.prev_action = None
        self.step_count = 0
        self._termination_printed = False  # 重置打印标志
