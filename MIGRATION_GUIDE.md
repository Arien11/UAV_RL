# 架构迁移路径指南

## 📋 当前架构状态

### 现有架构

```
QuadEnv (环境层)
    │
    └──继承──→ MuJoCoSimulator (具体仿真器)
            │
            └──包含相机功能
```

### 问题

❌ QuadEnv 与 MuJoCoSimulator 强耦合  
❌ 难以切换到其他仿真器或实机  
❌ 违反"组合优于继承"原则  

---

## 🎯 目标架构

### 组合架构（未来）

```
QuadEnvV2 (新环境层)
    │
    ├─组合──→ BaseSimulator (抽象仿真器接口)
    │               │
    │               ├─实现──→ MuJoCoSimulator
    │               ├─实现──→ PyBulletSimulator (未来)
    │               └─实现──→ RealRobot (未来)
    │
    └─组合──→ SensorManager (传感器管理器)
```

---

## 🚀 迁移策略

### 阶段1：保持现状（当前）

**不进行大规模重构**

✅ **QuadEnv 保持不变** - 继续使用继承MuJoCoSimulator的版本  
✅ **当前功能完全可用** - 传感器已集成，测试通过  
✅ **为未来准备** - 创建抽象接口但不立即使用  

**原因：**
- 风险可控 - 不影响现有功能
- 成本效益 - 当前架构已能工作
- 渐进式改进 - 可以在需要时再迁移

---

### 阶段2：创建抽象接口（准备阶段）

创建新架构的接口，但不立即使用：

#### 2.1 创建仿真器抽象接口

**文件：** `envs/Simulators/BaseSimulator.py` ✅ (已创建)

```python
class BaseSimulator(ABC):
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def set_state(self, qpos, qvel):
        pass
    
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def get_time(self):
        pass
    
    @abstractmethod
    def get_dt(self):
        pass
    
    @abstractmethod
    def get_camera_depth(self, camera_name=None):
        pass
    
    @abstractmethod
    def get_camera_rgb(self, camera_name=None):
        pass
```

#### 2.2 创建新的环境基类

**文件：** `envs/QuadBaseEnvV2.py` (新创建)

```python
class QuadBaseEnvV2:
    """使用组合而非继承的新环境基类"""
    
    def __init__(self, simulator: BaseSimulator, cfg=None):
        self.simulator = simulator  # 组合，不是继承
        self.cfg = cfg
        
        self.history_len = self.cfg.obs_history_len
        self._action_smoothing = self.cfg.action_smoothing
        
        self.interface = None
        self.nominal_pose = None
        self.task = None
        self.robot = None
        self._setup_robot()
        
        self.action_space = None
        self._setup_spaces()
        
        self.observation_history = deque(maxlen=self.history_len)
        self.prev_prediction = np.zeros_like(self.action_space)
    
    @abstractmethod
    def _setup_robot(self):
        pass
    
    @abstractmethod
    def _setup_spaces(self):
        pass
    
    def get_obs(self):
        robot_state = self._get_robot_state()
        ext_state = self._get_external_state()
        # ... 其余逻辑与原QuadBaseEnv相同
        pass
    
    def reset(self):
        return self.simulator.reset()
    
    def step(self, action):
        return self.simulator.step(action)
    
    def render(self):
        self.simulator.render()
    
    def close(self):
        self.simulator.close()
```

#### 2.3 创建仿真器工厂类

**文件：** `envs/Simulators/SimulatorFactory.py` (新创建)

```python
class SimulatorFactory:
    """仿真器工厂类"""
    
    @staticmethod
    def create(simulator_type: str, config: dict):
        """
        创建仿真器实例
        
        Args:
            simulator_type: 仿真器类型 ('mujoco', 'pybullet', 'real')
            config: 仿真器配置
            
        Returns:
            BaseSimulator 实例
        """
        if simulator_type == 'mujoco':
            return MuJoCoSimulatorV2(config)
        elif simulator_type == 'pybullet':
            return PyBulletSimulator(config)
        elif simulator_type == 'real':
            return RealRobotSimulator(config)
        else:
            raise ValueError(f"未知的仿真器类型: {simulator_type}")
```

---

### 阶段3：创建新架构的实现（可选，当前不使用）

#### 3.1 创建 MuJoCoSimulatorV2

**文件：** `envs/Simulators/MuJoCoSimulatorV2.py` (新创建)

```python
class MuJoCoSimulatorV2(BaseSimulator):
    """实现 BaseSimulator 接口的 MuJoCo 仿真器"""
    
    def __init__(self, config: dict):
        model_path = config.get('model_path')
        # 加载模型...
        self.model = ...
        self.data = ...
        # ... 其余初始化
        
    def reset(self):
        # 重置实现...
        pass
    
    def step(self, action):
        # 步进实现...
        pass
    
    def get_state(self):
        # 获取状态...
        pass
    
    def set_state(self, qpos, qvel):
        # 设置状态...
        pass
    
    def render(self):
        # 渲染...
        pass
    
    def close(self):
        # 关闭...
        pass
    
    def get_time(self):
        return self.data.time
    
    def get_dt(self):
        return self.model.opt.timestep
    
    def get_camera_depth(self, camera_name=None):
        # 获取深度相机...
        pass
    
    def get_camera_rgb(self, camera_name=None):
        # 获取RGB相机...
        pass
```

#### 3.2 创建 QuadEnvV2

**文件：** `envs/QuadEnvV2.py` (新创建)

```python
class QuadEnvV2(QuadBaseEnvV2):
    """使用新架构的环境"""
    
    def _setup_robot(self):
        # 使用 self.simulator 而不是继承
        self.interface = RobotInterface(self.simulator.model, self.simulator.data)
        self.frame_skip = int(control_dt / self.interface.sim.sim_dt())
        
        self._setup_task(control_dt)
        self.robot = Quadrotor(self.task, self.interface)
        
        # 设置传感器管理器（使用self.simulator）
        sensor_config = self._get_sensor_config()
        self.sensor_manager = SensorManager(sensor_config, sim=self.simulator)
        
        self.nominal_pose = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    
    # ... 其余方法
```

---

### 阶段4：测试新架构（可选）

创建测试文件验证新架构：

**文件：** `test_new_architecture.py` (新创建)

```python
def test_new_architecture():
    """测试新架构"""
    
    # 1. 使用工厂创建仿真器
    simulator_config = {
        'model_path': 'config/env_config.yaml'
    }
    simulator = SimulatorFactory.create('mujoco', simulator_config)
    
    # 2. 创建新环境
    with open('config/Quad_config.yaml') as f:
        config_data = yaml.safe_load(f)
    cfg = Configuration(**config_data)
    
    env = QuadEnvV2(simulator, cfg)
    
    # 3. 测试
    obs = env.reset()
    for _ in range(10):
        action = np.zeros_like(env.action_space)
        obs, reward, done, info = env.step(action)
    
    env.close()
    print("✅ 新架构测试通过！")
```

---

## 📅 迁移计划

### 时间表

| 阶段 | 时间 | 任务 | 状态 |
|------|------|------|------|
| 阶段1 | 现在 | 保持现有架构，创建抽象接口 | ✅ 进行中 |
| 阶段2 | 需要时 | 创建新架构的实现 | ⏳ 待开始 |
| 阶段3 | 需要时 | 测试新架构 | ⏳ 待开始 |
| 阶段4 | 需要时 | 逐步迁移功能 | ⏳ 待开始 |
| 阶段5 | 完成后 | 弃用旧架构 | ⏳ 待开始 |

---

## 🔑 关键决策点

### 何时开始迁移？

**触发条件：**
- ✅ 需要支持 PyBullet 或其他仿真器
- ✅ 需要迁移到实机
- ✅ 需要更好的可测试性
- ✅ 当前架构成为开发瓶颈

**不触发条件：**
- ❌ 当前架构能满足需求
- ❌ 没有明确的多仿真器需求
- ❌ 时间/资源有限

---

## 🛠️ 迁移检查表

### 迁移前检查

- [ ] 明确需要支持的仿真器类型
- [ ] 评估迁移成本和收益
- [ ] 制定详细的迁移计划
- [ ] 准备完整的测试用例
- [ ] 备份当前代码
- [ ] 安排足够的时间进行迁移

### 迁移中检查

- [ ] 保持两个架构并存
- [ ] 逐步迁移，不要一次性重写
- [ ] 每步都进行充分测试
- [ ] 保留回滚能力
- [ ] 及时更新文档

### 迁移后检查

- [ ] 所有功能正常工作
- [ ] 性能不低于原架构
- [ ] 测试覆盖率足够
- [ ] 文档更新完整
- [ ] 团队培训完成
- [ ] 旧架构标记为弃用

---

## 📚 相关文件

### 已创建文件

- ✅ `envs/Simulators/BaseSimulator.py` - 仿真器抽象接口
- ✅ `ARCHITECTURE_GUIDE.md` - 传感器架构使用指南
- ✅ `MIGRATION_GUIDE.md` - 本迁移指南

### 待创建文件（需要时）

- ⏳ `envs/QuadBaseEnvV2.py` - 新的环境基类
- ⏳ `envs/QuadEnvV2.py` - 新的环境实现
- ⏳ `envs/Simulators/MuJoCoSimulatorV2.py` - 新的MuJoCo仿真器
- ⏳ `envs/Simulators/SimulatorFactory.py` - 仿真器工厂
- ⏳ `test_new_architecture.py` - 新架构测试

---

## 🎯 总结

### 当前决策

✅ **保持现有架构不变** - QuadEnv 继续继承 MuJoCoSimulator  
✅ **创建抽象接口** - 为未来迁移做准备  
✅ **不立即重构** - 风险可控，成本效益最佳  

### 未来迁移路径

1. **触发条件**：需要支持多仿真器或实机
2. **创建新架构**：使用组合模式实现
3. **渐进式迁移**：两个架构并存，逐步迁移功能
4. **完成迁移**：新架构稳定后，弃用旧架构

---

**此文档将在需要时指导架构迁移！** 🚀
