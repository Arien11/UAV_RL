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


class Training:
    def __init__(self, env_fn, args, seed=None):
        self.seed = seed
        self.gamma = args.gamma
        self.lr = args.lr
        
        # ----------------------- train param ----------------------- #
        self.eps = args.eps
        self.ent_coeff = args.entropy_coeff
        self.clip = args.clip
        self.minibatch_size = args.minibatch_size
        self.epochs = args.epochs
        self.max_traj_len = args.max_traj_len
        self.n_proc = args.num_procs
        self.grad_clip = args.max_grad_norm
        self.mirror_coeff = args.mirror_coeff
        self.eval_freq = args.eval_freq
        self.recurrent = args.recurrent
        self.imitate_coeff = args.imitate_coeff
        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len
        
        self.total_steps = 0
        
        # counter for training iterations
        self.iteration_count = 0
        
        # directory for saving model weights
        self.save_path = Path(args.logdir)
        
        #  ----------------------- create networks or load up pretrained  ----------------------- #
        env_instance = env_fn()  # single env instance for initialization queries
        obs_dim = env_instance.observation_space.shape[0]
        action_dim = env_instance.action_space.shape[0]
        if args.continued:
            path_to_actor = args.continued
            path_to_critic = Path(args.continued.parent, "critic" + str(args.continued).split("actor")[1])
            policy = torch.load(path_to_actor, weights_only=False)
            critic = torch.load(path_to_critic, weights_only=False)
            # policy action noise parameters are initialized from scratch and not loaded
            if args.learn_std:
                policy.stds = torch.nn.Parameter(args.std_dev * torch.ones(action_dim))
            else:
                policy.stds = args.std_dev * torch.ones(action_dim)
            print("Loaded (pre-trained) actor from: ", path_to_actor)
            print("Loaded (pre-trained) critic from: ", path_to_critic)
            # Pretrained models already have obs normalization embedded
            self.obs_rms = None
        # else:
        #     if args.recurrent:
        #         policy = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std)
        #         critic = LSTM_V(obs_dim)
        #     else:
        #         policy = Gaussian_FF_Actor(
        #             obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std, bounded=False
        #         )
        #         critic = FF_V(obs_dim)
        #
        #     # Setup observation normalization (reuse env_instance from above)
        #     if hasattr(env_instance, "obs_mean") and hasattr(env_instance, "obs_std"):
        #         # Use fixed normalization params from environment
        #         obs_mean, obs_std = env_instance.obs_mean, env_instance.obs_std
        #         self.obs_rms = None  # No running stats needed
        #         print("Using fixed observation normalization from environment.")
        #     else:
        #         # Use running mean/std that will be updated during training
        #         self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        #         obs_mean, obs_std = self.obs_rms.mean, self.obs_rms.std
        #         print("Using running observation normalization (will update during training).")
        #
        #     with torch.no_grad():
        #         policy.obs_mean = torch.tensor(obs_mean, dtype=torch.float32)
        #         policy.obs_std = torch.tensor(obs_std, dtype=torch.float32)
        #         critic.obs_mean = policy.obs_mean
        #         critic.obs_std = policy.obs_std
        
        # ----------------------- Device setup (from args or auto-detect)  ----------------------- #
        device_arg = getattr(args, "device", "auto")
        if device_arg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_arg)
        
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
        
        base_policy = None
        if args.imitate:
            base_policy = torch.load(args.imitate, weights_only=False)
        
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic
        self.base_policy = base_policy
        # Store env_fn for later use
        self.env_fn = env_fn
        
        # Create persistent worker actors - this is the key optimization.
        # Each worker creates its environment ONCE and reuses it across all iterations,
        # avoiding expensive MuJoCo model recompilation.
        # Workers always use CPU (they do single-sample inference, no batching benefit)
        print(f"Creating {self.n_proc} persistent rollout workers...")
        
        # Create CPU copies for workers (deepcopy to avoid reference issues)
        if self.device.type == "cuda":
            # Networks are on GPU, need CPU copies for workers
            policy_cpu = deepcopy(self.policy).cpu()
            critic_cpu = deepcopy(self.critic).cpu()
            # Move non-parameter tensors to CPU
            policy_cpu.obs_mean = policy_cpu.obs_mean.cpu()
            policy_cpu.obs_std = policy_cpu.obs_std.cpu()
            critic_cpu.obs_mean = critic_cpu.obs_mean.cpu()
            critic_cpu.obs_std = critic_cpu.obs_std.cpu()
            if not isinstance(policy_cpu.stds, torch.nn.Parameter):
                policy_cpu.stds = policy_cpu.stds.cpu()
        else:
            # Already on CPU
            policy_cpu = self.policy
            critic_cpu = self.critic
        
        self.workers = [
            RolloutWorker.remote(
                env_fn,
                policy_cpu,
                critic_cpu,
                seed=42 + i,
                worker_id=i,
            )
            for i in range(self.n_proc)
        ]
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
    
    def _aggregate_results(self, result: list[BatchData]) -> BatchData:
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
        policy_state_dict = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        critic_state_dict = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        obs_mean_cpu = self.policy.obs_mean.cpu()
        obs_std_cpu = self.policy.obs_std.cpu()
        
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
            w.sample.remote(self.gamma, max_steps, self.max_traj_len, deterministic) for w in self.workers
        ]
        result = ray.get(sample_futures)
        
        return self._aggregate_results(result)
    
    def train(self, env_fn, n_itr):
        train_start_time = time.time()
        obs_mirr, act_mirr = None, None
        
        # =========================== warmup =========================== #
        
        # =========================== training proc =========================== #
        for itr in range(n_itr):
            print(f"********** Iteration {itr} ************")
            
            self.policy.train()
            self.critic.train()
            
            # set iteration count (could be used for curriculum training)
            self.iteration_count = itr
            
            sample_start_time = time.time()
            # ----------------------- sample parallel & process ----------------------- #
            batch = self.sample_parallel_with_workers()
            
            # Move batch to device for training
            observations = batch.states.float().to(self.device)
            actions = batch.actions.float().to(self.device)
            returns = batch.returns.float().to(self.device)
            values = batch.values.float().to(self.device)
            
            num_samples = len(observations)
            sample_time = time.time() - sample_start_time
            print(f"Sampling took {sample_time:.2f}s for {num_samples} steps.")
            
            # Normalize advantage (on device)
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
                    
                    scalars = self.update_actor_critic(
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
        with open("../config/Quad_config.yaml", 'r') as f:
            config_data = yaml.safe_load(f)
        cfg = Configuration(**config_data)
        env = QuadEnv("../config/env_config.yaml", cfg)
        return env
    
    
    def make_env_fc():
        return create_single_env()
    
    Training(make_env_fc, None)
