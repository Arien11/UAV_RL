from dataclasses import dataclass

import torch


@dataclass
class BatchData:
    """Typed container for batch data from trajectory collection.

    Provides a typed interface instead of anonymous dicts, enabling IDE support,
    type checking, and clearer code documentation.
    """
    
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    dones: torch.Tensor
    traj_idx: torch.Tensor
    ep_lens: torch.Tensor
    ep_rewards: torch.Tensor


class Buffer:
    def __init__(self, obs_len=1, act_len=1, gamma=0.99, size=1):
        # 数据初始化
        self.states = torch.zeros(size, obs_len, dtype=float)
        self.actions = torch.zeros(size, act_len, dtype=float)
        self.rewards = torch.zeros(size, 1, dtype=float)
        self.values = torch.zeros(size, 1, dtype=float)
        self.returns = torch.zeros(size, 1, dtype=float)
        self.dones = torch.zeros(size, 1, dtype=float)
        
        self.gamma = gamma  # 折扣因子
        self.ptr = 0  # 缓存池指针(索引位置)
        self.traj_idx = [0]
    
    def __len__(self):
        return self.ptr
    
    def store(self, state, action, reward, value, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1
    
    def finish_path(self, last_val=None):
        """Finish a trajectory and compute returns.

        Args:
            last_val: Bootstrap value for return computation (0 if truly done)
            自然终止：到达终止状态（如游戏结束此时 V(s_T) = 0（没有未来奖励）
            截断终止：达到最大步数限制此时 V(s_T) ≠ 0（状态还有未来价值）使用critic网络估计 V(s_T)
            即如果轨迹自然终止: last_val = 0
            如果轨迹被截断: last_val = critic(next_state)
        """
        self.traj_idx += [self.ptr]  # 记录新轨迹起始位置
        rewards = self.rewards[self.traj_idx[-2]: self.traj_idx[-1], 0]  # self.traj_idx[-2]到self.traj_idx[-1]就是刚结束的轨迹的最后一段
        T = len(rewards)
        
        if T == 0:
            return
        # ----------------------- Cal return --------------------------- #
        # return = 从当前时刻开始的所有未来奖励的折扣和 =  r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(T-t)·V(s_T) =
        
        # Vectorized discounted returns computation
        # Append last_val to rewards for unified computation
        last_val_scalar = last_val.squeeze(0) if last_val.dim() > 0 else last_val
        extended_rewards = torch.cat([rewards, last_val_scalar.unsqueeze(0)])
        
        # Compute discount powers: [1, γ, γ², ..., γ^T]
        discount_powers = self.gamma ** torch.arange(T + 1, dtype=rewards.dtype, device=rewards.device)
        
        # Weight rewards by discount powers 计算各个时间步加权求和后的值
        # [rt, γ*r_(t+1), γ**2 * r_(t+2), γ**3 * r_(t+3)...]
        weighted = extended_rewards * discount_powers
        
        # Reverse cumsum 反向累计和
        # 利用weighted = [rt, γ*r_(t+1), γ**2 * r_(t+2), γ**3 * r_(t+3)...] 根据 Gt = r_t + γ * (G_t+1) = r_t + γ * (r_t+1 + γ G_t+2)
        # -> rev_cumum = [G0, γ * G1, γ ** 2 * G2, ...]
        # at position i, gives sum_{j=i}^{T} γ^j * r_j
        rev_cumsum = weighted.flip(0).cumsum(0).flip(0)
        
        # Divide by position discount to get actual returns 除以当前位置的折扣因子
        # 即从原完整加权求和后的序列中取出各个时间步的return, return =
        # returns[i] = sum_{j=i}^{T} γ^(j-i) * r_j
        returns = rev_cumsum[:-1] / discount_powers[:-1]
        
        # 计算当前轨迹最终的reward
        self.returns[self.traj_idx[-2]: self.traj_idx[-1], 0] = returns
    
    def get_data(self, ep_lens=None, ep_rewards=None) -> BatchData:
        """Return collected data as BatchData.

        Args:
            ep_lens: List of completed episode lengths (from worker)
            ep_rewards: List of completed episode rewards (from worker)

        Returns:
            BatchData with collected trajectory data
        """
        return BatchData(
            states=self.states[: self.ptr],
            actions=self.actions[: self.ptr],
            rewards=self.rewards[: self.ptr],
            values=self.values[: self.ptr],
            returns=self.returns[: self.ptr],
            dones=self.dones[: self.ptr],
            traj_idx=torch.tensor(self.traj_idx),
            ep_lens=torch.tensor(ep_lens if ep_lens else []),
            ep_rewards=torch.tensor(ep_rewards if ep_rewards else []),
        )


if __name__ == '__main__':
    # test
    def test_data_collect():
        # 测试用例1：简单轨迹
        buffer = Buffer(gamma=0.9, size=100)
    
        # 存储一个3步的轨迹
        for i in range(3):
            buffer.store(
                state=torch.tensor([i], dtype=float),
                action=torch.tensor([i], dtype=float),
                reward=torch.tensor([i + 1.0]),  # 奖励: 1, 2, 3
                value=torch.tensor([0.0]),
                done=torch.tensor([0.0])
            )
    
        # 结束轨迹，假设last_val=0
        buffer.finish_path(last_val=torch.tensor([0.0]))
    
        data = buffer.get_data()
    
        # 手动计算预期回报
        # 回报计算公式: return_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
        expected_returns = [
            1.0 + 0.9 * 2.0 + 0.9 ** 2 * 3.0,  # 1 + 1.8 + 2.43 = 5.23
            2.0 + 0.9 * 3.0,  # 2 + 2.7 = 4.7
            3.0  # 3
        ]
    
        # 验证计算结果
        computed_returns = data.returns.squeeze().tolist()
    
        print(f"计算得到的回报: {computed_returns}")
        print(f"预期回报: {expected_returns}")
    
        for i, (computed, expected) in enumerate(zip(computed_returns, expected_returns)):
            assert abs(computed - expected) < 1e-6, f"第{i}步回报计算错误: {computed} != {expected}"
    
        print("折扣回报计算正确 ✓")
    test_data_collect()