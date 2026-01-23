from envs.Simulators.MujocoSim import *
import collections
import copy
import yaml
from envs.config_builder import Configuration


class QuadBaseEnv(MuJoCoSimulator):
    def __init__(self, model_path, cfg=None):
        MuJoCoSimulator.__init__(self, model_path)
        self.action_space = None
        self.cfg = cfg
        
        self.history_len = self.cfg.obs_history_len  # 历史帧的数量
        self._action_smoothing = self.cfg.action_smoothing
        
        self.MOTOR_NAME = ['MOTOR_1', 'MOTOR_2', 'MOTOR_3', 'MOTOR_4']
        
        # 设置特定机器人的组件(交互接口interface, 任务task, 机器人robot)
        self._setup_robot()
        
        # 存储默认的模型用于域随机化
        self.default_model = copy.deepcopy(self.model)
        
        # 设置域随机化
        # self._setup_domain_randomization()
        
        # 初始化动作/观测空间
        self._setup_spaces()
        
        # 初始化观测历史帧
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.prev_prediction = np.zeros_like(self.action_space)
    
    @abstractmethod
    def _setup_robot(self):
        """Setup robot interface, task, and RobotBase.

        Must set:
        - self.interface: RobotInterface instance
        - self.task: Task instance
        - self.robot: RobotBase instance
        - self.nominal_pose: List of nominal joint positions
        - self.actuators or self.leg_names: List of actuator names
        - self.half_sitting_pose: Robot's half-sitting pose
        """
        pass
    
    @abstractmethod
    def _setup_spaces(self):
        pass
    
    @abstractmethod
    def _get_robot_state(self):
        pass
    
    @abstractmethod
    def _get_external_state(self):
        pass
    
    def get_obs(self):
        robot_state = self._get_robot_state()
        ext_state = self._get_external_state()
        state = np.concatenate([robot_state, ext_state])
        # assert state.shape == (self.base_obs_len,), (
        #     f"State vector length expected to be: {self.base_obs_len} but is {len(state)}"
        # )
        
        # Manage observation history
        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)
        
        return np.array(self.observation_history).flatten()
    
    def step(self, action: np.ndarray):
    
        # Get offsets from nominal pose
        # offsets = self._get_action_offsets()
        
        # Execute robot step
        # rewards, done = self.robot.step(targets, np.asarray(offsets))
        rewards, done = self.robot.step(action)
        obs = self.get_obs()
        self.prev_prediction = action
        
        # Apply domain randomization if enabled
        # if self.dynrand_interval > 0 and np.random.randint(self.dynrand_interval) == 0:
        #     self._randomize_dynamics()
        #
        # if self.perturb_interval > 0 and np.random.randint(self.perturb_interval) == 0:
        #     self._apply_perturbation()
        
        return obs, sum(rewards.values()), done, rewards
    
    # def step(self, action=None):
    #     # action = np.clip(action, 0.0, 1.0) # 确保动作在有效范围内
    #     state = self.simulator.step(action)
    #     self.last_state = state
    #     # ============= calculate reward ============= #
    #     reward = 0
    #     if self.task:
    #         reward = self.task.calc_reward(state, action)
    #
    #     # ============= get observation ============= #
    #     if self.observation_builder:
    #         obs = self.observation_builder.get_observation(state)
    #
    #     # ============= check end subjects ============= #
    #     terminated = 0
    #     truncated = 0
    #     done = terminated
    #     # terminated, termination_reason = self.task.check_termination(sim_state)
    #     # truncated, truncation_reason = self.task.check_truncation(sim_state)
    #
    #     # ============= create traj_info ============= #
    #     return obs, reward, done, info
    #
    
    def reset_model(self):
        """Reset the environment to initial state."""
        pass
    
    def _apply_init_noise(self):
        pass
    
    def _apply_observation_noise(self):
        pass


if __name__ == '__main__':
    with open("../config/Quad_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    temp = QuadBaseEnv("../config/env_config.yaml", cfg)
    print()
