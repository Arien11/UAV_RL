import sys
import os

import ray
import torch
from unittest.mock import Mock

from rl.workers.rolloutworker import RolloutWorker
from Tasks.Hover_Task import *
from envs.Simulators.MujocoSim import *
from envs.QuadEnv import QuadEnv
from QuadControl.Quad import Quadrotor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== test for sample_parallel ====================== #


# 测试前启动Ray
ray.init(local_mode=True, ignore_reinit_error=True)


class MockPolicyNetwork(torch.nn.Module):
    """模拟策略网络"""
    
    def __init__(self, state_dim=13, action_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 简单的线性策略
        self.fc = torch.nn.Linear(state_dim, action_dim)
        
        # 观察值归一化参数
        self.obs_mean = torch.zeros(state_dim)
        self.obs_std = torch.ones(state_dim)
        
        # 动作噪声（用于探索）
        self.stds = torch.nn.Parameter(torch.ones(action_dim) * 0.1)
    
    def forward(self, state, deterministic=False):
        """前向传播"""
        # 确保输入是2D张量 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        
        # 简单的归一化
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        
        # 计算网络输出
        output = self.fc(state_normalized)
        
        if deterministic:
            # 确定性策略：直接输出网络输出
            return output
        else:
            # 随机策略：添加噪声
            noise = torch.randn_like(output) * self.stds
            return output + noise
    
    def init_hidden_state(self):
        """初始化隐藏状态（用于循环网络）"""
        pass


class MockCriticNetwork(torch.nn.Module):
    """模拟价值网络"""
    
    def __init__(self, state_dim=13):
        super().__init__()
        self.state_dim = state_dim
        self.fc = torch.nn.Linear(state_dim, 1)
        
        # 观察值归一化参数
        self.obs_mean = torch.zeros(state_dim)
        self.obs_std = torch.ones(state_dim)
    
    def forward(self, state):
        """前向传播"""
        # 确保输入是2D张量 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        return self.fc(state_normalized)
    
    def init_hidden_state(self):
        """初始化隐藏状态"""
        pass


def create_single_env():
    from envs.config_builder import Configuration
    with open("../config/Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    env = QuadEnv("../config/env_config.yaml", cfg)
    return env


def make_env_fc():
    return create_single_env()


from rl.storage.rollout_storage import BatchData


def _aggregate_results(result) -> BatchData:
    """Aggregate results from multiple workers into a single BatchData.
        聚合数据为训练可用格式
    Args:
        result: List of BatchData from worker sample() calls

    Returns:
        BatchData with concatenated tensors from all workers
    """
    
    # Aggregate trajectory data - handle traj_idx specially for recurrent policies
    # (indices need to be offset to reference correct positions in concatenated data)
    states = torch.cat([r.states for r in result])
    actions = torch.cat([r.actions for r in result])
    rewards = torch.cat([r.rewards for r in result])
    values = torch.cat([r.values for r in result])
    returns = torch.cat([r.returns for r in result])
    dones = torch.cat([r.dones for r in result])
    ep_lens = torch.cat([r.ep_lens for r in result])
    ep_rewards = torch.cat([r.ep_rewards for r in result])
    
    # Fix traj_idx: offset each worker's indices by cumulative sample count
    # if self.recurrent:
    #     traj_idx_list = []
    #     offset = 0
    #     for r in result:
    #         # Skip the first 0 from subsequent workers (it's redundant)
    #         worker_traj_idx = r.traj_idx
    #         if offset > 0:
    #             worker_traj_idx = worker_traj_idx[1:]  # Skip leading 0
    #         traj_idx_list.append(worker_traj_idx + offset)
    #         offset += len(r.states)
    #     traj_idx = torch.cat(traj_idx_list)
    # else:
    traj_idx = torch.cat([r.traj_idx for r in result])
    
    return BatchData(
        states=states,
        actions=actions,
        rewards=rewards,
        values=values,
        returns=returns,
        dones=dones,
        traj_idx=traj_idx,
        ep_lens=ep_lens,
        ep_rewards=ep_rewards,
    )


def test_single_sample():
    """测试单次采样"""
    print("\n=== 测试单次采样 ===")
    
    # 创建模拟网络
    policy_template = MockPolicyNetwork()
    critic_template = MockCriticNetwork()
    
    # 创建RolloutWorker
    worker = RolloutWorker.remote(
        env_fn=make_env_fc,
        policy_template=policy_template,
        critic_template=critic_template,
        seed=42,
        worker_id=0
    )
    
    # 同步状态
    sync_future = worker.sync_state.remote(
        policy_template.state_dict(),
        critic_template.state_dict(),
        torch.zeros(13),
        torch.ones(13),
        iteration_count=0)
    ray.get(sync_future)
    
    # 执行采样
    max_steps = 50
    max_traj_len = 10
    gamma = 0.99
    
    sample_future = worker.sample.remote(
        gamma=gamma,
        max_steps=max_steps,
        max_traj_len=max_traj_len,
        deterministic=False
    )
    
    try:
        batch_data = ray.get(sample_future)
        
        # 验证返回的数据结构
        assert hasattr(batch_data, 'states'), "缺少states属性"
        assert hasattr(batch_data, 'actions'), "缺少actions属性"
        assert hasattr(batch_data, 'rewards'), "缺少rewards属性"
        assert hasattr(batch_data, 'values'), "缺少values属性"
        assert hasattr(batch_data, 'returns'), "缺少returns属性"
        assert hasattr(batch_data, 'dones'), "缺少dones属性"
        assert hasattr(batch_data, 'traj_idx'), "缺少traj_idx属性"
        assert hasattr(batch_data, 'ep_lens'), "缺少ep_lens属性"
        assert hasattr(batch_data, 'ep_rewards'), "缺少ep_rewards属性"
        
        print(f"✓ 采样成功，收集了{len(batch_data.states)}个时间步")
        print(f"✓ 完成了{len(batch_data.ep_lens)}个回合")
        print(f"✓ 轨迹索引: {batch_data.traj_idx.tolist()}")
        
        return batch_data
    
    except Exception as e:
        print(f"✗ 采样失败: {e}")
        raise


def test_sample_parallel():
    """sample traj using persistent worker actors
    """
    max_steps = 32 // 8
    policy = MockPolicyNetwork()
    critic = MockCriticNetwork()
    
    # 创建多个工作进程
    num_workers = 4
    workers = []
    print(f"创建{num_workers}个工作进程...")
    for i in range(num_workers):
        worker = RolloutWorker.remote(
            env_fn=make_env_fc,
            policy_template=policy,
            critic_template=critic,
            seed=42 + i,  # 每个worker不同种子
            worker_id=i
        )
        workers.append(worker)
    
    # Get state dicts and obs normalization, move to CPU for workers
    # (Workers always run on CPU, even if main process is on GPU)
    policy_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    critic_state_dict = {k: v.cpu() for k, v in critic.state_dict().items()}
    obs_mean_cpu = policy.obs_mean.cpu()
    obs_std_cpu = policy.obs_std.cpu()
    
    # Use ray.put() to store in object store once, avoiding redundant
    # serialization when broadcasting to multiple workers
    policy_ref = ray.put(policy_state_dict)
    critic_ref = ray.put(critic_state_dict)
    obs_mean_ref = ray.put(obs_mean_cpu)
    obs_std_ref = ray.put(obs_std_cpu)
    
    # 在一个回调中同步所有worker的state(weights, normalization, iteration)
    sync_futures = [
        w.sync_state.remote(policy_ref, critic_ref, obs_mean_ref, obs_std_ref, iteration_count=0)
        for w in workers
    ]
    ray.get(sync_futures)
    
    # 所有worker并行采样数据
    sample_futures = [
        w.sample.remote(gamma=0.99,
                        max_steps=500,  # 每个worker收集1000步
                        max_traj_len=50,
                        deterministic=False)
        for w in workers
    ]
    result = ray.get(sample_futures)
    res = _aggregate_results(result=result)
    return result


#
#
# def test_deterministic_vs_stochastic():
#     """测试确定性与随机性策略"""
#     print("\n=== 测试确定性与随机性策略 ===")
#
#     # 创建模拟网络
#     policy_template = MockPolicyNetwork()
#     critic_template = MockCriticNetwork()
#
#     # 创建RolloutWorker
#     worker = RolloutWorker.remote(
#         env_fn=mock_env_fn,
#         policy_template=policy_template,
#         critic_template=critic_template,
#         seed=42,
#         worker_id=0
#     )
#
#     # 同步状态
#     ray.get(worker.sync_state.remote(
#         policy_template.state_dict(),
#         critic_template.state_dict(),
#         torch.zeros(4),
#         torch.ones(4),
#         iteration_count=0
#     ))
#
#     # 测试确定性策略
#     det_future = worker.sample.remote(
#         gamma=0.99,
#         max_steps=20,
#         max_traj_len=5,
#         deterministic=True
#     )
#     det_data = ray.get(det_future)
#
#     # 测试随机性策略
#     stoch_future = worker.sample.remote(
#         gamma=0.99,
#         max_steps=20,
#         max_traj_len=5,
#         deterministic=False
#     )
#     stoch_data = ray.get(stoch_future)
#
#     # 验证：确定性策略应该在不同次采样中产生相同的动作（给定相同种子）
#     print("✓ 确定性策略采样完成")
#     print("✓ 随机性策略采样完成")
#
#     return det_data, stoch_data
#

if __name__ == '__main__':
    # test_worker_initialization()
    # test_single_sample()
    # test_performance()
    test_sample_parallel()
    # test_single_sample()
