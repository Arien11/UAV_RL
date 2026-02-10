from copy import deepcopy
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from rl.network.actor import *
from rl.network.critic import *


class PPO:
    def __init__(self, env_fn, args=None):
        # ppo参数
        self.clip = 1
        self.ent_coef = 1
        self.gamma = 1
        self.lr = 1
        self.eps = 1
        self.grad_clip = 1
        base_policy = None
        # if args.imitate:
        #     base_policy = torch.load(args.imitate, weights_only=False)
        env_instance = env_fn()  # single env instance for initialization queries
        obs_dim = env_instance.observation_space.shape[0]
        action_dim = env_instance.action_space.shape[0]
        
        policy = FF_Actor(obs_dim, action_dim)
        critic = FF_Critic(obs_dim)
        
        # Device setup (from args or auto-detect)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print("Moving policy and critic to GPU...")
                policy = policy.to(self.device)
                critic = critic.to(self.device)
                # Also move non-parameter tensors to GPU
                policy.obs_mean = policy.obs_mean.to(self.device)
                policy.obs_std = policy.obs_std.to(self.device)
                critic.obs_mean = critic.obs_mean.to(self.device)
                critic.obs_std = critic.obs_std.to(self.device)
                # Move stds if it's a plain tensor (not nn.Parameter)
                if not isinstance(policy.stds, torch.nn.Parameter):
                    policy.stds = policy.stds.to(self.device)
        if self.device.type == "cpu":
            print("Using CPU for training")
        
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic
        self.base_policy = base_policy
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
    
    def update_actor_critic(
            self,
            obs_batch,
            action_batch,
            return_batch,
            advantage_batch,
            mask,
            mirror_observation=None, mirror_action=None
    ):
        """
        
        :param obs_batch: 观测数据 (batch_size, obs_dim)
        :param action_batch: 动作数据
        :param return_batch: 回报数据
        :param advantage_batch: 优势函数数据
        :param mask:  掩码，用于处理序列长度不同的情况
        :param mirror_observation:
        :param mirror_action:
        :return:
        """
        
        # ================== Importance Sampling(计算新旧策略的概率比率) ================== #
        # 用旧策略收集的数据来估计新策略的梯度 E_(x~p)[f(x)] = E_(x~q)[(p(x)/q(x))f(x)], ratio = (p(x)/q(x)) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        # 数据复用提高样本效率
        pdf = self.policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 当前策略采取动作的概率
        
        old_pdf = self.old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 过去策略采取动作的概率
        
        # 转换为概率比率(重要性权重)
        ratio = (log_probs - old_log_probs).exp()
        
        # ================== Clipping Loss(降低方差，防止更新步长过大) ================== #
        # clip_coe=0.2,
        # advantage=1, ratio=1.5, cpi_loss=1 * 1.5, clip_loss=1 * 1.2=1.2, actor_loss=min(1.5, 1.2)=1.2 限制增强幅度
        # advantage=-1, ratio=0.6, cpi_loss=-1 * 0.6, clip_loss=-1 * 0.8=-0.8, actor_loss=min(-0.6,-0.8)=-0.8 限制削减幅度
        # L_clip(θ) = E_t[min(ratio * advantage, clip(rati, 1-epsilon, 1+epsilon) * advantage)
        cpi_loss = ratio * advantage_batch * mask  # 未加约束
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask  # 加入约束的保守值
        if isinstance(mask, torch.Tensor):
            num_valid = mask.sum()
            actor_loss = -torch.min(cpi_loss, clip_loss).sum() / num_valid
        else:
            actor_loss = -torch.min(cpi_loss, clip_loss).mean()
        
        # only used for logging
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()
        
        # ================== Value Loss ================== #
        values = self.critic(obs_batch)
        # For recurrent policies, mask out padded positions from critic loss
        if isinstance(mask, torch.Tensor):
            value_error = (return_batch - values).pow(2) * mask
            critic_loss = value_error.sum() / num_valid
        else:
            critic_loss = F.mse_loss(return_batch, values)  # 价值网络的均方误差损失
        
        # ================== Entropy Loss ================== #
        if isinstance(mask, torch.Tensor):
            action_dim = pdf.mean.shape[-1]
            entropy_penalty = -(pdf.entropy() * mask).sum() / (num_valid * action_dim)
        else:
            entropy_penalty = -(pdf.entropy() * mask).mean()
        # ================== Imitation Loss ================== #
        # 模仿损失（imitation_loss）：让当前策略模仿一个专家策略（base_policy),加速学习过程或保持策略稳定性
        # 如果存在一个基础策略（base_policy），则计算当前策略输出的动作（deterministic_actions）与基础策略输出的动作之间的均方误差。
        # 这可以看作是一种模仿学习，让当前策略去模仿基础策略的行为。
        # 如果没有基础策略，则模仿损失为零。
        # ================== 近似KL散度计算 ================== #
        # 监控策略更新的幅度，防止策略更新过大 KL ≈ (ratio - 1) - log_ratio
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)  # 近似的kl散度
        
        # ================== Total Loss ================== #
        actor_total_loss = (
                actor_loss
                # + self.mirror_coeff * mirror_loss
                # + self.imitate_coeff * imitation_loss
                + self.ent_coef * entropy_penalty
        )
        total_loss = actor_total_loss + critic_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            # mirror_loss,
            # imitation_loss,
            # clip_fraction,
        )


if __name__ == '__main__':
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
    
    
    def create_single_env():
        from envs.config_builder import Configuration
        with open("E:\\UAV_RL\config\Quad_config.yaml", 'r') as f:
            config_data = yaml.safe_load(f)
        cfg = Configuration(**config_data)
        env = QuadEnv("E:\\UAV_RL\config\env_config.yaml", cfg)
        return env
    
    
    def make_env_fc():
        return create_single_env()
    
    
    temp_ppo = PPO(make_env_fc)
    print()
