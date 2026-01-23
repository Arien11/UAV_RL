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

# 测试前启动Ray
ray.init(local_mode=True, ignore_reinit_error=True)


class MockEnv:
    """模拟环境，用于测试RolloutWorker"""
    
    def __init__(self, max_steps=10, reward_scale=1.0):
        self.state_dim = 4
        self.action_dim = 2
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.step_count = 0
        self.robot = Mock()  # 模拟robot属性
        self.robot.iteration_count = 0
    
    def reset(self):
        """重置环境"""
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(self, action):
        """环境步进"""
        self.step_count += 1
        
        # 模拟状态转移
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # 模拟奖励
        reward = self.reward_scale * (1.0 - 0.1 * np.linalg.norm(action))
        reward = float(reward)
        
        # 模拟终止条件
        done = self.step_count >= self.max_steps
        truncated = False
        
        return next_state, reward, done, {}
    
    def seed(self, seed=None):
        """设置环境种子"""
        np.random.seed(seed)
        return [seed]
    
    def close(self):
        """关闭环境"""
        pass


def mock_env_fn():
    """创建模拟环境的工厂函数"""
    return MockEnv(max_steps=5, reward_scale=1.0)


class MockPolicyNetwork(torch.nn.Module):
    """模拟策略网络"""
    
    def __init__(self, state_dim=4, action_dim=2):
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
        # 简单的归一化
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        
        # 计算动作均值
        action_mean = self.fc(state_normalized)
        
        if deterministic:
            # 确定性策略：直接输出均值
            return action_mean
        else:
            # 随机策略：添加噪声
            noise = torch.randn_like(action_mean) * self.stds
            return action_mean + noise
    
    def init_hidden_state(self):
        """初始化隐藏状态（用于循环网络）"""
        pass


class MockCriticNetwork(torch.nn.Module):
    """模拟价值网络"""
    
    def __init__(self, state_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.fc = torch.nn.Linear(state_dim, 1)
        
        # 观察值归一化参数
        self.obs_mean = torch.zeros(state_dim)
        self.obs_std = torch.ones(state_dim)
    
    def forward(self, state):
        """前向传播"""
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        return self.fc(state_normalized)
    
    def init_hidden_state(self):
        """初始化隐藏状态"""
        pass


def test_worker_initialization():
    """测试工作进程初始化"""
    print("=== 测试工作进程初始化 ===")
    
    # 创建模拟网络
    policy_template = MockPolicyNetwork()
    critic_template = MockCriticNetwork()
    
    # 创建RolloutWorker
    worker = RolloutWorker.remote(
        env_fn=mock_env_fn,
        policy_template=policy_template,
        critic_template=critic_template,
        seed=42,
        worker_id=0
    )
    
    # 测试同步状态
    policy_state = policy_template.state_dict()
    critic_state = critic_template.state_dict()
    obs_mean = torch.zeros(4)
    obs_std = torch.ones(4)
    
    # 调用同步方法
    ray.get(worker.sync_state.remote(
        policy_state, critic_state, obs_mean, obs_std, iteration_count=0
    ))
    
    print("✓ 工作进程初始化成功")
    print("✓ 同步状态方法工作正常")
    
    return worker


def create_single_env():
    import yaml

    env_config_path = "../config/env_config.yaml"
    task_config_path = "../config/Task_config.yaml"
    
    # =========================== 仿真器 =========================== #
    XML_loader = XMLModelLoader()
    env_config = {}
    if env_config_path:
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
    model = XML_loader.load(env_config.get('model', {}))
    Mujoco_simulator = MuJoCoSimulator(model)
    
    # =========================== 任务 =========================== #
    # 加载配置
    task_config = {}
    if task_config_path:
        with open(task_config_path, 'r') as f:
            task_config = yaml.safe_load(f)
    # 处理配置
    task = Hover(task_config)

    Quad = Quadrotor()
    # =========================== 环境 =========================== #
    env = QuadEnv(
        # observation_space=observation_builder,
        simulator=Mujoco_simulator,
        task=task,
    )
    return env


def make_env_fc():
    return create_single_env()


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
    
    sync_future = worker.sync_state.remote(
        policy_template.state_dict(),
        critic_template.state_dict(),
        torch.zeros(4),
        torch.ones(4),
        iteration_count=0)
    # 同步状态
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
        
        # 验证数据形状
        assert len(batch_data.states) == max_steps, f"状态数量应为{max_steps}"
        assert len(batch_data.actions) == max_steps, f"动作数量应为{max_steps}"
        
        # 验证状态维度
        assert batch_data.states.shape[1] == 4, f"状态维度应为4"
        assert batch_data.actions.shape[1] == 2, f"动作维度应为2"
        
        print(f"✓ 采样成功，收集了{len(batch_data.states)}个时间步")
        print(f"✓ 完成了{len(batch_data.ep_lens)}个回合")
        print(f"✓ 轨迹索引: {batch_data.traj_idx.tolist()}")
        
        return batch_data
    
    except Exception as e:
        print(f"✗ 采样失败: {e}")
        raise


def test_performance():
    """测试性能基准"""
    print("\n=== 性能测试 ===")
    
    import time
    
    policy_template = MockPolicyNetwork()
    critic_template = MockCriticNetwork()
    
    # 创建多个工作进程
    num_workers = 4
    workers = []
    
    print(f"创建{num_workers}个工作进程...")
    for i in range(num_workers):
        worker = RolloutWorker.remote(
            env_fn=mock_env_fn,
            policy_template=policy_template,
            critic_template=critic_template,
            seed=42 + i,  # 每个worker不同种子
            worker_id=i
        )
        workers.append(worker)
    
    # 同步所有worker
    print("同步所有worker状态...")
    sync_futures = []
    for w in workers:
        future = w.sync_state.remote(
            policy_template.state_dict(),
            critic_template.state_dict(),
            torch.zeros(4),
            torch.ones(4),
            iteration_count=0
        )
        sync_futures.append(future)
    
    ray.get(sync_futures)
    
    # 并行采样测试
    print("开始并行采样测试...")
    start_time = time.time()
    
    sample_futures = []
    for w in workers:
        future = w.sample.remote(
            gamma=0.99,
            max_steps=1000,  # 每个worker收集1000步
            max_traj_len=50,
            deterministic=False
        )
        sample_futures.append(future)
    
    # 收集所有结果
    all_data = ray.get(sample_futures)
    
    end_time = time.time()
    
    total_steps = sum(len(data.states) for data in all_data)
    total_time = end_time - start_time
    
    print(f"性能结果:")
    print(f"  总时间: {total_time:.2f}秒")
    print(f"  总步数: {total_steps}步")
    print(f"  平均速度: {total_steps / total_time:.1f} 步/秒")
    print(f"  每个worker平均速度: {(total_steps / total_time) / num_workers:.1f} 步/秒")
    
    return all_data


def test_deterministic_vs_stochastic():
    """测试确定性与随机性策略"""
    print("\n=== 测试确定性与随机性策略 ===")
    
    # 创建模拟网络
    policy_template = MockPolicyNetwork()
    critic_template = MockCriticNetwork()
    
    # 创建RolloutWorker
    worker = RolloutWorker.remote(
        env_fn=mock_env_fn,
        policy_template=policy_template,
        critic_template=critic_template,
        seed=42,
        worker_id=0
    )
    
    # 同步状态
    ray.get(worker.sync_state.remote(
        policy_template.state_dict(),
        critic_template.state_dict(),
        torch.zeros(4),
        torch.ones(4),
        iteration_count=0
    ))
    
    # 测试确定性策略
    det_future = worker.sample.remote(
        gamma=0.99,
        max_steps=20,
        max_traj_len=5,
        deterministic=True
    )
    det_data = ray.get(det_future)
    
    # 测试随机性策略
    stoch_future = worker.sample.remote(
        gamma=0.99,
        max_steps=20,
        max_traj_len=5,
        deterministic=False
    )
    stoch_data = ray.get(stoch_future)
    
    # 验证：确定性策略应该在不同次采样中产生相同的动作（给定相同种子）
    print("✓ 确定性策略采样完成")
    print("✓ 随机性策略采样完成")
    
    return det_data, stoch_data


if __name__ == '__main__':
    # test_worker_initialization()
    # test_single_sample()
    # test_performance()

    test_myEnv_init()