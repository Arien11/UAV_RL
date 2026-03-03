from envs.Simulators.MujocoSim import *
import collections
import copy
import yaml
from envs.config_builder import Configuration


class QuadBaseEnv(MuJoCoSimulator):
    def __init__(self, model_path, cfg=None):
        MuJoCoSimulator.__init__(self, model_path)
        self.cfg = cfg
        
        self.history_len = self.cfg.obs_history_len  # 历史帧的数量
        self._action_smoothing = self.cfg.action_smoothing
        
        # 设置特定机器人的组件(交互接口interface, 任务task, 机器人robot, 初始状态init_state)
        self.interface = None
        self.nominal_pose = None
        self.task = None
        self.robot = None
        self._setup_robot()
        
        # 存储默认的模型用于域随机化
        self.default_model = copy.deepcopy(self.model)
        
        # 设置域随机化
        # self._setup_domain_randomization()
        
        # 初始化动作/观测空间
        self.action_space = None
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
    
    def get_obs(self) -> np.array:
        """获取完整的观测量"""
        robot_state = self._get_robot_state()
        ext_state = self._get_external_state()
        if ext_state is not None:
            state = np.concatenate([robot_state, ext_state])
        else:
            state = robot_state
        # assert state.shape == (self.base_obs_len,), (
        #     f"State vector length expected to be: {self.base_obs_len} but is {len(state)}"
        # )
        
        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)
        
        return np.array(self.observation_history).flatten()
    
    @abstractmethod
    def step(self, action: np.ndarray):
        pass
    
    def reset_model(self):
        """Reset the environment to initial state."""
        init_qpos = self.nominal_pose.copy()
        init_qvel = [0] * self.interface.nv()
        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        
        self.task.reset(iter_count=self.robot.iteration_count)
        
        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        
        return self.get_obs()
    
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
