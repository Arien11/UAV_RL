from copy import deepcopy
from pathlib import Path

import ray

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from envs.QuadEnv import QuadEnv
from envs.Simulators.MujocoSim import *
from QuadControl.Quad import Quadrotor
from Tasks.Hover_Task import *
import mujoco.viewer as viewer

from rl.workers.rolloutworker import RolloutWorker
from rl.storage.rollout_storage import BatchData
from rl.algos.ppo import PPO


class Training:
    def __init__(self, env_fn, algo, args=None, seed=None):
        
        self.seed = 1
        self.gamma = 0.99
        # ----------------------- train param ----------------------- #
        self.lr = 0.01
        self.eps = 1
        
        self.minibatch_size = 10
        self.epochs = 500
        self.max_traj_len = 50
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
        
        #  ----------------------- create networks or load up pretrained  ----------------------- #
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
        self.actor_optimizer = optim.Adam(algo.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(algo.critic.parameters(), lr=self.lr, eps=self.eps)
        
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
        sample_futures = [
            w.sample.remote(self.gamma, 500, self.max_traj_len, deterministic) for w in self.workers
        ]
        result = ray.get(sample_futures)
        
        return self._aggregate_results(result)
    
    def train(self, n_itr):
        train_start_time = time.time()
        obs_mirr, act_mirr = None, None
        # =========================== warmup =========================== #
        
        # =========================== training process(A2C) =========================== #
        for itr in range(n_itr):
            print(f"********** Iteration {itr} ************")
            
            self.algo.policy.train()
            self.algo.critic.train()
            
            # set iteration count (could be used for curriculum training)
            self.iteration_count = itr
            
            sample_start_time = time.time()
            # ----------------------- sample parallel (worker process) ----------------------- #
            batch = self.sample_parallel_with_workers()
            
            # ----------------------- master process ----------------------- #
            # Move batch to device for training
            observations = batch.states.float().to(self.device)
            actions = batch.actions.float().to(self.device)
            returns = batch.returns.float().to(self.device)
            values = batch.values.float().to(self.device)
            
            num_samples = len(observations)
            sample_time = time.time() - sample_start_time
            print(f"Sampling took {sample_time:.2f}s for {num_samples} steps.")
            
            # 归一化优势函数 (on device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            
            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples
            
            optimizer_start_time = time.time()
            
            # ----------------------- data record ----------------------- #
            actor_losses = []
            entropies = []
            critic_losses = []
            kls = []
            mirror_losses = []
            imitation_losses = []
            clip_fractions = []
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
                    mask = 1
                    
                    scalars = self.algo.update_actor_critic(
                        obs_batch,
                        action_batch,
                        return_batch,
                        advantage_batch,
                        mask,
                        mirror_observation=obs_mirr,
                        mirror_action=act_mirr,
                    )
                    (
                        actor_loss,
                        entropy_penalty,
                        critic_loss,
                        approx_kl_div,
                        mirror_loss,
                        imitation_loss,
                        clip_fraction,
                    ) = scalars
                    
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    imitation_losses.append(imitation_loss.item())
                    clip_fractions.append(clip_fraction)
                    
                    optimize_time = time.time() - optimizer_start_time
                    print(f"Optimizer took: {optimize_time:.2f}s")
                
                # ----------------------- data vis(system) ----------------------- #
                
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
                
                # ----------------------- data vis(tensorboard) ----------------------- #
    
    def evaluate(self, env_fn, nets, itr):
        ...


if __name__ == '__main__':
    def create_single_env():
        from envs.config_builder import Configuration
        with open("E:\\UAV_RL\config\Quad_config.yaml", 'r') as f:
            config_data = yaml.safe_load(f)
        cfg = Configuration(**config_data)
        env = QuadEnv("E:\\UAV_RL\config\env_config.yaml", cfg)
        return env
    
    
    def make_env_fc():
        return create_single_env()
    
    _ppo = PPO(make_env_fc)
    train_proc = Training(make_env_fc, _ppo)
    train_proc.train(5)
    print()
