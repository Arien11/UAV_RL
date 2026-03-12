from copy import deepcopy
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from algos.network.actor import *
from algos.network.critic import *
from algos.utils.normalize import *


class PPO:
    def __init__(self, env_fn, args=None):
        # =========================== ppo参数 =========================== #
        self.clip = 0.2  # PPO裁剪系数
        self.ent_coef = 0.01  # 熵系数（鼓励探索）
        self.gamma = 0.99  # 折扣因子
        self.lam = 0.95  # GAE参数
        # =========================== 网络设置 =========================== #
        base_policy = None
        # if args.imitate:
        #     base_policy = torch.load(args.imitate, weights_only=False)
        env_instance = env_fn()  # single env instance for initialization queries
        obs_dim = env_instance.observation_space.shape[0]
        action_dim = env_instance.action_space.shape[0]
        # ================== 创建网络 ================== #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = FF_Actor(obs_dim, action_dim, layers=(256, 256), init_std=0.5, bounded=True, learn_std=True).to(self.device)
        policy.action_scale = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        policy.action_bias = torch.zeros(3, device=self.device)
        
        critic = FF_Critic(obs_dim, layers=(256, 256)).to(self.device)
        # ================== 设置观测归一化参数 ================== #
        if hasattr(env_instance, "obs_mean") and hasattr(env_instance, "obs_std"):
            # 直接从环境获取固定参数，并创建在目标设备上
            obs_mean = torch.tensor(env_instance.obs_mean, dtype=torch.float32, device=self.device)
            obs_std = torch.tensor(env_instance.obs_std, dtype=torch.float32, device=self.device)
            self.obs_rms = None
            print("Using fixed observation normalization from environment.")
        else:
            # 使用运行时统计，初始化后也放在目标设备
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
            obs_mean = torch.tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
            obs_std = torch.tensor(self.obs_rms.std, dtype=torch.float32, device=self.device)
            print("Using running observation normalization (will update during training).")
        
        policy.obs_mean = obs_mean
        policy.obs_std = obs_std
        critic.obs_mean = obs_mean
        critic.obs_std = obs_std
        
        # ================== 处理 stds（如果它不是 Parameter） ==================
        # 注意：如果 policy.stds 是普通 tensor，它不会随 .to(device) 移动，需要手动处理
        if not isinstance(policy.stds, torch.nn.Parameter):
            policy.stds = policy.stds.to(self.device)  # 确保它在正确设备上
        
        # ================== 创建旧策略 ==================
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic
        self.base_policy = None  # 根据实际情况设置
    
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
        assert not torch.isnan(obs_batch).any(), "obs_batch contains NaN"
        # ================== Importance Sampling(计算新旧策略的概率比率) ================== #
        # 用旧策略收集的数据来估计新策略的梯度 E_(x~p)[f(x)] = E_(x~q)[(p(x)/q(x))f(x)], ratio = (p(x)/q(x)) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        # 数据复用提高样本效率
        pdf = self.policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 当前策略采取动作的概率
        if torch.isinf(log_probs).any():
            print("log_probs contains inf!")
        old_pdf = self.old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 过去策略采取动作的概率
        # print(
        #     f"[LOG_PROB] current: mean={log_probs.mean().item():.2f}, min={log_probs.min().item():.2f}, max={log_probs.max().item():.2f}")
        # print(
        #     f"[LOG_PROB] old: mean={old_log_probs.mean().item():.2f}, min={old_log_probs.min().item():.2f}, max={old_log_probs.max().item():.2f}")
        # 转换为概率比率(重要性权重)
        ratio = (log_probs - old_log_probs).exp()
        # print(f"[RATIO] mean={ratio.mean():.3f}, std={ratio.std():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}")
        # ================== Clipping Loss(降低方差，防止更新步长过大) ================== #
        # clip_coe=0.2,
        # advantage=1, ratio=1.5, cpi_loss=1 * 1.5, clip_loss=1 * 1.2=1.2, actor_loss=min(1.5, 1.2)=1.2 限制增强幅度
        # advantage=-1, ratio=0.6, cpi_loss=-1 * 0.6, clip_loss=-1 * 0.8=-0.8, actor_loss=min(-0.6,-0.8)=-0.8 限制削减幅度
        # L_clip(θ) = E_t[min(ratio * advantage, clip(rati, 1-epsilon, 1+epsilon) * advantage)
        # cpi_loss = ratio * advantage_batch * mask  # 未加约束
        # clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask  # 加入约束的保守值
        cpi_loss = ratio * advantage_batch  # 未加约束
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch  # 加入约束的保守值
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
            entropy_penalty = -(pdf.entropy()).mean()
        
        # ================== Mirror Symmetry Loss ================== #
        # Reuse mean from distribution instead of redundant forward pass
        # deterministic_actions = pdf.mean
        # if mirror_observation is not None and mirror_action is not None:
        #     if self.recurrent:
        #         mir_obs = torch.stack([mirror_observation(obs_batch[i, :, :]) for i in range(obs_batch.shape[0])])
        #         mirror_actions = self.policy(mir_obs)
        #     else:
        #         mir_obs = mirror_observation(obs_batch)
        #         mirror_actions = self.policy(mir_obs)
        #     mirror_actions = mirror_action(mirror_actions)
        #     mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        # else:
        #     mirror_loss = torch.zeros_like(actor_loss)
        
        # ================== Imitation Loss ================== #
        # 模仿损失（imitation_loss）：让当前策略模仿一个专家策略（base_policy),加速学习过程或保持策略稳定性
        # 如果存在一个基础策略（base_policy），则计算当前策略输出的动作（deterministic_actions）与基础策略输出的动作之间的均方误差。
        # 这可以看作是一种模仿学习，让当前策略去模仿基础策略的行为。
        # 如果没有基础策略，则模仿损失为零。
        # if self.base_policy is not None:
        #     imitation_loss = (self.base_policy(obs_batch) - deterministic_actions).pow(2).mean()
        # else:
        #     imitation_loss = torch.zeros_like(actor_loss)
        
        # ================== Total Loss ================== #
        actor_total_loss = (
                actor_loss
                # + self.mirror_coeff * mirror_loss
                # + self.imitate_coeff * imitation_loss
                + self.ent_coef * entropy_penalty
        )
        total_loss = actor_total_loss + critic_loss
        
        # ================== 近似KL散度计算 ================== #
        # 监控策略更新的幅度，防止策略更新过大 KL ≈ (ratio - 1) - log_ratio
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)  # 近似的kl散度
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_penalty': entropy_penalty,
            'actor_total_loss': actor_total_loss,
            'approx_kl_div': approx_kl_div,
            'clip_fraction': clip_fraction,
            'values': values,
            'ratio': ratio,
            'log_probs': log_probs,
        }


if __name__ == '__main__':
    import sys
    import os
    
    import ray
    import torch
    from unittest.mock import Mock
    
    from algos.workers.rolloutworker import RolloutWorker
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
