import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from copy import deepcopy
from pathlib import Path
import ray
import csv
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import datetime
from envs.QuadEnv import QuadEnv
from envs.Simulators.MujocoSim import *
from Tasks.Hover_Task import *
from rl.workers.rolloutworker import RolloutWorker
from rl.storage.rollout_storage import BatchData
from rl.algos.ppo import PPO
from train.logger import *
import mujoco


def create_single_env():
    from envs.config_builder import Configuration
    with open("E:\\UAV_RL\config\Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    env = QuadEnv("E:\\UAV_RL\config\env_config.yaml", cfg)
    return env


def make_env_fc():
    return create_single_env()


class Training:
    def __init__(self, env_fn, algo, args=None, seed=None):
        
        self.seed = 1
        # ----------------------- train param ----------------------- #
        self.actor_lr = 3e-4
        self.critic_lr = 1e-4
        self.eps = 1e-6
        self.grad_clip = 0.5  # 梯度裁剪阈值
        
        self.minibatch_size = 512
        self.epochs = 10
        self.max_traj_len = 500
        self.n_proc = 4
        
        self.eval_freq = 10
        self.recurrent = None
        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len
        
        self.total_steps = 0
        
        # counter for training iterations
        self.iteration_count = 0
        
        # directory for saving model weights
        # self.save_path = Path(args.logdir)
        
        #  ----------------------- used algo  ----------------------- #
        self.algo = algo
        
        # ----------------------- Device setup (from args or auto-detect)  ----------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Store env_fn for later use
        self.env_fn = env_fn
        
        print(f"Creating {self.n_proc} persistent rollout workers...")
        
        # Create CPU copies for workers (deepcopy to avoid reference issues)
        if self.device.type == "cuda":
            # Networks are on GPU, need CPU copies for workers
            policy_cpu = deepcopy(algo.policy).cpu()
            critic_cpu = deepcopy(algo.critic).cpu()
            # Move non-parameter tensors to CPU
            policy_cpu.obs_mean = policy_cpu.obs_mean.cpu()
            policy_cpu.obs_std = policy_cpu.obs_std.cpu()
            critic_cpu.obs_mean = critic_cpu.obs_mean.cpu()
            critic_cpu.obs_std = critic_cpu.obs_std.cpu()
            if not isinstance(policy_cpu.stds, torch.nn.Parameter):
                policy_cpu.stds = policy_cpu.stds.cpu()
        else:
            # Already on CPU
            policy_cpu = algo.policy
            critic_cpu = algo.critic
        
        self.workers = [
            RolloutWorker.remote(
                env_fn,
                policy_cpu,
                critic_cpu,
                seed=self.seed + i,
                worker_id=i,
            )
            for i in range(self.n_proc)
        ]
        
        # ----------------------- Optim setup ----------------------- #
        self.actor_optimizer = optim.Adam(algo.policy.parameters(), lr=self.actor_lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(algo.critic.parameters(), lr=self.critic_lr, eps=self.eps)
        
        # used for plot
        self.data_path = Path("E:\\UAV_RL\\train\\data_logs")
        self.data_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.data_path / f"training_log_{timestamp}.csv"
        
        # CSV文件头
        self.csv_headers = [
            'iteration', 'total_steps', 'mean_episode_reward', 'mean_episode_length',
            'actor_loss', 'critic_loss', 'kl_divergence', 'entropy', 'clip_fraction',
            'mean_noise_std'
        ]
        self.df_log = pd.DataFrame(columns=self.csv_headers)
        print(f"Training log will be saved to {self.csv_file}")
        
        # used for tensorboard
        self.logs_path = Path("E:\\UAV_RL\\train\logs")
        self.logger = TrainingLogger(self.logs_path, flush_secs=10)
        
        # model_ckpts save
        self.ckpts_path = Path("E:\\UAV_RL\\train\\checkpoints")
        self.ckpts_path.mkdir(parents=True, exist_ok=True)
    
    def _sync_obs_normalization(self, obs_mean, obs_std, include_old_policy=True):
        """Sync observation normalization params to all networks.

        This is the single point of truth for updating normalization parameters,
        avoiding scattered manual synchronization throughout the codebase.

        Args:
            obs_mean: Observation mean tensor
            obs_std: Observation std tensor
            include_old_policy: Whether to also sync to old_policy (for PPO ratio)
        """
        self.algo.policy.obs_mean = obs_mean
        self.algo.policy.obs_std = obs_std
        self.algo.critic.obs_mean = obs_mean
        self.algo.critic.obs_std = obs_std
        if include_old_policy:
            self.algo.old_policy.obs_mean = obs_mean.clone()
            self.algo.old_policy.obs_std = obs_std.clone()
    
    def _aggregate_results(self, result) -> BatchData:
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
        if self.recurrent:
            traj_idx_list = []
            offset = 0
            for r in result:
                # Skip the first 0 from subsequent workers (it's redundant)
                worker_traj_idx = r.traj_idx
                if offset > 0:
                    worker_traj_idx = worker_traj_idx[1:]  # Skip leading 0
                traj_idx_list.append(worker_traj_idx + offset)
                offset += len(r.states)
            traj_idx = torch.cat(traj_idx_list)
        else:
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
    
    def sample_parallel_with_workers(self, deterministic=False):
        """sample traj using persistent worker actors
        """
        max_steps = self.batch_size // self.n_proc
        
        # Get state dicts and obs normalization, move to CPU for workers
        # (Workers always run on CPU, even if main process is on GPU)
        policy_state_dict = {k: v.cpu() for k, v in self.algo.policy.state_dict().items()}
        critic_state_dict = {k: v.cpu() for k, v in self.algo.critic.state_dict().items()}
        obs_mean_cpu = self.algo.policy.obs_mean.cpu()
        obs_std_cpu = self.algo.policy.obs_std.cpu()
        
        # Use ray.put() to store in object store once, avoiding redundant
        # serialization when broadcasting to multiple workers
        policy_ref = ray.put(policy_state_dict)
        critic_ref = ray.put(critic_state_dict)
        obs_mean_ref = ray.put(obs_mean_cpu)
        obs_std_ref = ray.put(obs_std_cpu)
        
        # 在一个回调中同步所有worker的state(weights, normalization, iteration)
        sync_futures = [
            w.sync_state.remote(policy_ref, critic_ref, obs_mean_ref, obs_std_ref, self.iteration_count)
            for w in self.workers
        ]
        ray.get(sync_futures)
        
        # 所有worker并行采样数据
        # sample_futures = [
        #     w.sample.remote(self.algo.gamma, max_steps, self.max_traj_len, deterministic) for w in self.workers
        # ]
        sample_futures = [
            w.sample.remote(self.algo.gamma, self.algo.lam, max_steps, self.max_traj_len, deterministic) for w in
            self.workers
        ]
        result = ray.get(sample_futures)
        
        return self._aggregate_results(result)
    
    def _optimization_step(self, loss_dict):
        """执行优化步骤（反向传播、梯度裁剪、更新参数）"""
        total_loss = loss_dict['total_loss']
        
        # 清空梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.algo.policy.parameters(),
            max_norm=self.grad_clip
        )
        torch.nn.utils.clip_grad_norm_(
            self.algo.critic.parameters(),
            max_norm=self.grad_clip
        )
        
        # 更新参数
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        if self.algo.policy.learn_std:
            self.algo.policy.stds.data.clamp_(min=0.1)  # 防止噪声过小
        # # 可选：添加学习率调度
        # self._update_learning_rate()
    
    def save_checkpoint(self, itr):
        """保存当前模型和优化器状态"""
        checkpoint = {
            'iteration': itr,
            'policy_state_dict': self.algo.policy.state_dict(),
            'critic_state_dict': self.algo.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'obs_mean': self.algo.policy.obs_mean.cpu(),
            'obs_std': self.algo.policy.obs_std.cpu(),
            'args': vars(self) if hasattr(self, 'args') else None,  # 可选，保存参数
        }
        # 如果使用了 running normalization，也保存其状态
        if self.algo.obs_rms is not None:
            checkpoint['obs_rms_mean'] = self.algo.obs_rms.mean
            checkpoint['obs_rms_std'] = self.algo.obs_rms.std
            checkpoint['obs_rms_count'] = self.algo.obs_rms.count
        
        path = self.ckpts_path / f"checkpoint_iter{itr + 1}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, model_path):
        """加载检查点并恢复训练状态"""
        checkpoint = torch.load(model_path, map_location='cpu')  # 先加载到CPU
        itr = checkpoint['iteration']
        
        # 加载模型参数
        self.algo.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.algo.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 移动模型到目标设备
        self.algo.policy.to(self.device)
        self.algo.critic.to(self.device)
        
        # 加载优化器状态
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 恢复观测标准化参数
        obs_mean = checkpoint['obs_mean'].to(self.device)
        obs_std = checkpoint['obs_std'].to(self.device)
        self._sync_obs_normalization(obs_mean, obs_std, include_old_policy=True)
        
        # 如果使用 running normalization，也恢复
        if self.algo.obs_rms is not None and 'obs_rms_mean' in checkpoint:
            self.algo.obs_rms.mean = checkpoint['obs_rms_mean']
            self.algo.obs_rms.std = checkpoint['obs_rms_std']
            self.algo.obs_rms.count = checkpoint['obs_rms_count']
        
        print(f"Checkpoint loaded from {model_path}, resumed from iteration {itr}")
        return itr  # 返回已训练的迭代次数，以便从下一轮开始
    
    def evaluate(self, env_fn, nets, itr):
        ...
    
    def train(self, n_itr, model_path=None):
        train_start_time = time.time()
        obs_mirr, act_mirr = None, None
        
        # =========================== warmup =========================== #
        # Warmup phase for running observation normalization
        if self.algo.obs_rms is not None:
            print("Warming up observation normalization...")
            print(
                f"  Initial policy norm - mean: {self.algo.policy.obs_mean[:3]}..., std: {self.algo.policy.obs_std[:3]}...")
            warmup_batches = 5
            for i in range(warmup_batches):
                batch = self.sample_parallel_with_workers()
                self.algo.obs_rms.update(batch.states.numpy())
                print(
                    f"  Warmup batch {i + 1}: {len(batch.states)} samples, obs_rms count: {self.algo.obs_rms.count:.0f}")
            # Sync warmed-up normalization to all networks
            with torch.no_grad():
                obs_mean = torch.from_numpy(self.algo.obs_rms.mean).float().to(self.device)
                obs_std = torch.from_numpy(self.algo.obs_rms.std).float().to(self.device)
                self._sync_obs_normalization(obs_mean, obs_std)
            print(f"Normalization initialized with {self.algo.obs_rms.count:.0f} samples")
            print(f"  obs_mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
            print(f"  obs_std range: [{obs_std.min():.4f}, {obs_std.max():.4f}]")
            print("warmup finished ...")
        # =========================== training =========================== #
        start_itr = 0
        # 载入模型
        if model_path:
            start_itr = self.load_checkpoint(model_path) + 1
        for itr in range(start_itr, n_itr):
            print(f"********** Iteration {itr} ************")
            
            self.algo.policy.train()
            self.algo.critic.train()
            
            # 后续用于课程学习
            self.iteration_count = itr
            
            # ----------------------- Sample parallel (worker process) ----------------------- #
            sample_start_time = time.time()
            
            batch = self.sample_parallel_with_workers()
            
            # 同步 old_policy（PPO核心）
            self.algo.old_policy.load_state_dict(self.algo.policy.state_dict())
            # Sync obs normalization to old_policy (not in state_dict, policy/critic already correct)
            self.algo.old_policy.obs_mean = self.algo.policy.obs_mean.clone()
            self.algo.old_policy.obs_std = self.algo.policy.obs_std.clone()
            # print(f"stds:{self.algo.policy.obs_std}")
            
            # # 更新观测标准化
            # if self.algo.obs_rms is not None:
            #     # batch.states 是 CPU 上的 torch.Tensor，转为 numpy
            #     self.algo.obs_rms.update(batch.states.cpu().numpy())
            #     new_mean = torch.from_numpy(self.algo.obs_rms.mean).float().to(self.device)
            #     new_std = torch.from_numpy(self.algo.obs_rms.std).float().to(self.device)
            #     self._sync_obs_normalization(new_mean, new_std)
            
            # ----------------------- master process ----------------------- #
            # Move batch to device for training
            observations = batch.states.float().to(self.device)
            actions = batch.actions.float().to(self.device)
            returns = batch.returns.float().to(self.device)
            returns = returns / 100
            values = batch.values.float().to(self.device)
            
            num_samples = len(observations)
            sample_time = time.time() - sample_start_time
            print(f"Sampling took {sample_time:.2f}s for {num_samples} steps.")
            
            # 归一化优势函数 (on device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            # print(f"[ADV] mean={advantages.mean():.3f}, std={advantages.std():.3f}, min={advantages.min():.3f}, max={advantages.max():.3f}")
            
            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples
            
            optimizer_start_time = time.time()
            
            # ----------------------- train proc ----------------------- #
            total_losses = []
            actor_losses = []
            entropies = []
            critic_losses = []
            kls = []
            mirror_losses = []
            imitation_losses = []
            clip_fractions = []
            action_means = []
            action_stds = []
            action_offsets = []
            for epoch in range(self.epochs):
                # ----------------------- random seed generate ----------------------- #
                if self.seed is not None:  # Create seeded generator for deterministic batch sampling
                    g = torch.Generator()
                    g.manual_seed(self.seed + itr * self.epochs + epoch)
                else:
                    g = None
                
                # ----------------------- Sampler ----------------------- #
                random_indices = SubsetRandomSampler(range(num_samples), generator=g)
                sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)
                for indices in sampler:
                    obs_batch = observations[indices]
                    action_batch = actions[indices]
                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]
                    mask = None
                    # print(
                    #     f"Return: mean={return_batch.mean():.2f}, std={return_batch.std():.2f}, min={return_batch.min():.2f}, max={return_batch.max():.2f}")
                    # print(
                    #     f"Value: mean={values.mean():.2f}, std={values.std():.2f}, min={values.min():.2f}, max={values.max():.2f}")
                    loss_dict = self.algo.update_actor_critic(
                        obs_batch,
                        action_batch,
                        return_batch,
                        advantage_batch,
                        mask,
                        mirror_observation=obs_mirr,
                        mirror_action=act_mirr,
                    )
                    
                    # 执行优化步骤
                    self._optimization_step(loss_dict)
                    if torch.isnan(next(self.algo.policy.parameters()).any()):
                        print("NaN detected in policy parameters after update!")
                        torch.save(self.algo.policy.state_dict(), "nan_policy.pth")
                        raise RuntimeError("NaN in policy parameters")
                    # ----------------------- data record ----------------------- #
                    total_losses.append(loss_dict["total_loss"].item())
                    actor_losses.append(loss_dict["actor_loss"].item())
                    entropies.append(loss_dict["entropy_penalty"].item())
                    critic_losses.append(loss_dict["critic_loss"].item())
                    kls.append(loss_dict["approx_kl_div"].item())
                    action_means.append(action_batch.mean())
                    action_stds.append(action_batch.std())
                    action_offsets.append(np.sum(action_batch.cpu().numpy()[:, :3], axis=0))
                    # mirror_losses.append(mirror_loss.item())
                    # imitation_losses.append(imitation_loss.item())
                    # clip_fractions.append(clip_fraction)
            
            # 保存模型
            if (itr+1) % 10 == 0:
                self.save_checkpoint(itr)
            
            # 日志记录
            optimize_time = time.time() - optimizer_start_time
            print(f"Optimizer took: {optimize_time:.2f}s")
            
            action_noise = self.algo.policy.stds.data.tolist()
            
            # ----------------------- data vis(system output & csv) ----------------------- #
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write(f"| {'Mean Eprew':>15} | {torch.mean(batch.ep_rewards):>15.5g} |\n")
            sys.stdout.write(f"| {'Mean Eplen':>15} | {torch.mean(batch.ep_lens.float()):>15.5g} |\n")
            sys.stdout.write(f"| {'Actor loss':>15} | {np.mean(actor_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Critic loss':>15} | {np.mean(critic_losses):>15.3g} |\n")
            # sys.stdout.write(f"| {'Mirror loss':>15} | {np.mean(mirror_losses):>15.3g} |\n")
            # sys.stdout.write(f"| {'Imitation loss':>15} | {np.mean(imitation_losses):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean KL Div':>15} | {np.mean(kls):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean Entropy':>15} | {np.mean(entropies):>15.3g} |\n")
            # sys.stdout.write(f"| {'Clip Fraction':>15} | {np.mean(clip_fractions):>15.3g} |\n")
            sys.stdout.write(f"| {'Mean noise std':>15} | {np.mean(action_noise):>15.3g} |\n")
            sys.stdout.write(f"| {'actions mean':>15} | {np.mean(action_means):>15.3g} |\n")
            sys.stdout.write(f"| {'actions std':>15} | {np.mean(action_stds):>15.3g} |\n")
            sys.stdout.write(f"| {'action offsets':>15} | {np.mean(action_offsets):>15.3g} |\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()
            
            total_time = time.time() - train_start_time
            fps = self.total_steps / total_time
            iter_avg = total_time / (itr + 1)
            ETA = round((n_itr - itr) * iter_avg)
            print(
                f"Total time elapsed: {total_time:.2f}s. Total steps: {self.total_steps} "
                f"(fps={fps:.2f}. iter-avg={iter_avg:.2f}s. "
                f"ETA={datetime.timedelta(seconds=ETA)})"
            )
            self.df_log.loc[len(self.df_log)] = [
                itr, self.total_steps,
                float(torch.mean(batch.ep_rewards)),
                float(torch.mean(batch.ep_lens.float())),
                np.mean(actor_losses),
                np.mean(critic_losses),
                np.mean(kls),
                np.mean(entropies),
                np.mean(clip_fractions) if clip_fractions else 0.0,
                np.mean(action_noise),
            ]
            
            # ----------------------- data vis(tensorboard) ----------------------- #

            # tensorboard logging for training
            self.logger.log_training_metrics(
                actor_loss=np.mean(actor_losses),
                critic_loss=np.mean(critic_losses),
                # mirror_loss=np.mean(mirror_losses),
                # imitation_loss=np.mean(imitation_losses),
                mean_reward=float(torch.mean(batch.ep_rewards)),
                mean_ep_len=float(torch.mean(batch.ep_lens.float())),
                mean_noise_std=np.mean(action_noise),
                step=itr,
            )
            
            # tensorboard logging for timing/performance metrics
            self.logger.log_timing_metrics(
                fps=fps,
                sample_time=sample_time,
                optimize_time=optimize_time,
                total_time=total_time,
                step=itr,
            )
            
            # ----------------------- eval interval ----------------------- #
            # if itr == 0 or (itr + 1) % self.eval_freq == 0:
            #     nets = {"actor": self.policy, "critic": self.critic}
            #
            #     evaluate_start = time.time()
            #     eval_batches = self.evaluate(env_fn, nets, itr)
            #     eval_time = time.time() - evaluate_start
            #
            #     eval_ep_lens = [float(i) for b in eval_batches for i in b.ep_lens]
            #     eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
            #     avg_eval_ep_lens = np.mean(eval_ep_lens)
            #     avg_eval_ep_rewards = np.mean(eval_ep_rewards)
            #     print("====EVALUATE EPISODE====")
            #     print(
            #         f"(Episode length:{avg_eval_ep_lens:.3f}. Reward:{avg_eval_ep_rewards:.3f}. "
            #         f"Time taken:{eval_time:.2f}s)"
            #     )
            #
            #     # tensorboard logging for evaluation
            #     self.logger.log_eval_metrics(avg_eval_ep_rewards, avg_eval_ep_lens, itr)
        
        # 训练结束后保存数据
        self.df_log.to_csv(self.csv_file, index=False)


def load_policy_for_inference(checkpoint_path, device='cpu'):
    from rl.network.actor import FF_Actor  # 你的Actor类
    """加载训练好的策略网络用于推理"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建策略网络实例（参数需与训练时一致）
    policy = FF_Actor(
        state_dim=13,  # 根据你的环境实际观测维度
        action_dim=9,
        layers=(512, 512),
        init_std=0.5,
        learn_std=True,  # 应与训练时一致
        bounded=True
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()  # 切换到评估模式
    
    # 恢复观测标准化参数
    if 'obs_mean' in checkpoint and 'obs_std' in checkpoint:
        policy.obs_mean = checkpoint['obs_mean'].to(device)
        policy.obs_std = checkpoint['obs_std'].to(device)
    
    return policy


def test_policy(policy, env_fn, num_episodes=5, deterministic=True, render=True):
    """用训练好的策略在环境中测试，并可视化"""
    env = env_fn()  # 创建环境
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # 启动 MuJoCo viewer（如果启用渲染）
        if render:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while not done:
                    # 将观测转为 tensor，并移至正确设备
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.obs_mean.device)
                    
                    # 用策略选择动作（deterministic 通常用于测试）
                    with torch.no_grad():
                        action = policy(obs_tensor, deterministic=deterministic).squeeze(0).cpu().numpy()
                    
                    # 执行一步
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
                    
                    # 更新 viewer（渲染）
                    viewer.sync()
                    
                    step += 1
                    # print(f"step {step}, action Δz = {action[2]:.4f}")
                    if step % 100 == 0:
                        print(f"obs:{obs[2]:.2f}")
                        # print(f"action:{action}")
                        # print(f"Episode {ep + 1}, step {step}, reward {total_reward:.2f}")
                    
                    # 可以按需添加延迟（否则可能跑得太快）
                    # import time; time.sleep(0.01)
        else:
            # 不渲染时快速测试
            while True:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.obs_mean.device)
                with torch.no_grad():
                    action = policy(obs_tensor, deterministic=deterministic).squeeze(0).cpu().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step += 1
            print(f"Episode {ep + 1} finished after {step} steps, total reward: {total_reward:.2f}")
    
    env.close()


