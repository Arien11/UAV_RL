# 传感器架构设计指南

## 📋 当前架构概述

### 架构图

```
QuadEnv (环境层)
    │
    ├── SensorManager (传感器管理器)
    │       ├── MujocoDepthCamera (MuJoCo深度相机)
    │       └── MujocoRGBCamera (MuJoCo RGB相机)
    │
    ├── RobotInterface (机器人接口)
    │
    └── Task (任务层)

MuJoCoSimulator (仿真器层)
    ├── 物理引擎
    ├── 渲染器
    └── (独立于传感器)
```

---

## ✅ 当前架构的高内聚低耦合特性

### 1. **BaseSensor - 纯粹的抽象接口**

**高内聚**：只定义传感器必须实现的方法，没有任何实现细节

**低耦合**：完全独立于任何具体的仿真器或硬件

```python
class BaseSensor(ABC):
    @abstractmethod
    def get_observation(self):
        pass
    
    @abstractmethod
    def preprocess_observation(self, raw_data):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_observation_shape(self):
        pass
```

### 2. **MujocoCamera - 只负责MuJoCo相机**

**高内聚**：所有代码都只与MuJoCo相机相关

**低耦合**：通过 `set_simulator()` 注入仿真器依赖，不硬编码

```python
class MujocoDepthCamera(BaseSensor):
    def __init__(self, config):
        self.camera_name = config.get('camera_name', 'drone_depth_camera')
        self._sim = None  # 通过外部注入
    
    def set_simulator(self, sim):
        self._sim = sim  # 依赖注入，低耦合
    
    def get_observation(self):
        # 只使用sim的接口，不关心sim的具体实现
        return self._sim.get_camera_depth()
```

### 3. **SensorManager - 统一管理，低耦合**

**高内聚**：只负责传感器的生命周期管理

**低耦合**：通过配置创建传感器，不依赖具体传感器实现

```python
class SensorManager:
    def __init__(self, config, sim=None):
        self.sensors = []
        for sensor_config in config.get('sensors', []):
            # 通过配置动态创建传感器
            sensor_class = self._get_sensor_class(sensor_config['type'])
            sensor = sensor_class(sensor_config)
            if sim:
                sensor.set_simulator(sim)
            self.sensors.append(sensor)
    
    def get_all_observations(self):
        # 统一获取所有传感器数据
        observations = []
        for sensor in self.sensors:
            obs = sensor.get_processed_observation()
            observations.append(obs.flatten())
        return np.concatenate(observations)
```

### 4. **QuadEnv - 通过配置使用传感器**

**高内聚**：环境只负责环境逻辑

**低耦合**：通过 `_get_sensor_config()` 配置传感器，不硬编码

```python
class QuadEnv(QuadBaseEnv):
    def _setup_robot(self):
        # 创建传感器管理器（通过配置）
        sensor_config = self._get_sensor_config()
        self.sensor_manager = SensorManager(sensor_config, sim=self)
    
    def _get_sensor_config(self):
        # 配置驱动，低耦合
        return {
            'sensors': [
                {
                    'type': 'mujoco_depth_camera',
                    'enabled': True,
                    'camera_name': 'drone_depth_camera',
                    'width': 160,
                    'height': 120,
                    'max_depth': 5.0,
                    'min_depth': 0.1
                }
            ]
        }
    
    def _get_external_state(self):
        # 使用传感器管理器获取数据
        return self.sensor_manager.get_all_observations()
```

---

## 🎯 架构设计原则

### 1. **单一职责原则（SRP）**

- **BaseSensor**: 定义传感器接口
- **MujocoDepthCamera**: 实现MuJoCo深度相机
- **SensorManager**: 管理传感器生命周期
- **QuadEnv**: 处理环境逻辑

### 2. **依赖倒置原则（DIP）**

- **依赖抽象**：SensorManager依赖BaseSensor，不依赖具体实现
- **注入依赖**：通过`set_simulator()`注入仿真器

### 3. **开闭原则（OCP）**

- **对扩展开放**：可以轻松添加新传感器类型
- **对修改关闭**：添加新传感器不需要修改SensorManager

### 4. **接口隔离原则（ISP）**

- **最小接口**：BaseSensor只定义必要的方法
- **按需实现**：具体传感器只实现需要的方法

---

## 📖 使用指南

### 1. 添加新传感器类型

```python
# 1. 在 envs/Sensors/ 中创建新文件
# envs/Sensors/RealDepthCamera.py

from .BaseSensor import BaseSensor

class RealDepthCamera(BaseSensor):
    """实机深度相机"""
    
    def __init__(self, config):
        super().__init__(config)
        self.device_id = config.get('device_id', 0)
        self.camera = None
    
    def set_hardware(self, hardware):
        """设置硬件接口（实机特有）"""
        self.hardware = hardware
    
    def get_observation(self):
        """从实机获取深度图像"""
        return self.hardware.get_depth_image()
    
    def preprocess_observation(self, raw_data):
        """实机特定的预处理"""
        return raw_data / 1000.0  # 转换为米
    
    def reset(self):
        """重置相机"""
        if self.camera:
            self.camera.reset()
    
    def get_observation_shape(self):
        return (self.height, self.width)
```

### 2. 在配置中添加新传感器

```python
# 在 QuadEnv._get_sensor_config() 中添加

def _get_sensor_config(self):
    return {
        'sensors': [
            {
                'type': 'mujoco_depth_camera',
                'enabled': True,
                'camera_name': 'drone_depth_camera',
                'width': 160,
                'height': 120
            },
            # 新添加的传感器
            {
                'type': 'real_depth_camera',
                'enabled': False,  # 可以通过配置启用/禁用
                'device_id': 0,
                'width': 640,
                'height': 480
            }
        ]
    }
```

### 3. 在 SensorManager 中注册新传感器

```python
# 在 SensorsManager._get_sensor_class() 中添加

def _get_sensor_class(self, sensor_type):
    sensor_classes = {
        'mujoco_depth_camera': MujocoDepthCamera,
        'mujoco_rgb_camera': MujocoRGBCamera,
        # 新添加的传感器
        'real_depth_camera': RealDepthCamera,
    }
    return sensor_classes[sensor_type]
```

---

## 🚀 未来迁移到实机的架构

### 迁移方案

```
仿真环境                    实机环境
    │                            │
    ├─ QuadEnv (共用)         ├─ QuadEnv (共用)
    │       │                    │       │
    │       ├─ SensorManager    │       ├─ SensorManager
    │       │       │            │       │       │
    │       │       └─ MujocoDepthCamera  │       │       └─ RealDepthCamera
    │       │
    │       └─ MuJoCoSimulator │       └─ RealHardwareInterface
    │
    └─ MuJoCo (仿真器)         └─ RealRobot (实机)
```

### 迁移步骤

1. **保持 QuadEnv 不变** - 环境逻辑完全共用
2. **创建 RealHardwareInterface** - 实现实机硬件接口
3. **创建 RealDepthCamera** - 实现实机传感器
4. **修改配置** - 切换到实机传感器
5. **运行！** - 环境逻辑完全不需要改动

---

## ✅ 总结

### 当前架构的优势

1. **传感器完全独立** - BaseSensor 是纯粹的抽象
2. **配置驱动** - 通过配置而非硬编码
3. **易于测试** - 每个组件都可以独立测试
4. **易于扩展** - 添加新传感器不需要修改现有代码
5. **为实机预留** - 接口设计考虑了实机需求

### 不需要大规模重构的原因

1. **当前架构已满足要求** - 传感器部分已经是高内聚低耦合
2. **风险可控** - 大规模重构可能引入新bug
3. **成本效益** - 当前架构已经可以正常工作
4. **渐进式改进** - 可以在需要时逐步优化

### 建议

- ✅ **使用当前架构** - 它已经很好了
- ✅ **保持接口稳定** - 为未来实机预留空间
- ✅ **通过配置切换** - 需要时只需修改配置
- ✅ **添加文档** - 记录如何使用和扩展

---

## 📞 总结

**当前架构已经是高内聚低耦合的设计！** 

传感器部分完全独立，通过配置管理，QuadEnv使用组合而非硬编码。唯一的"耦合"是QuadEnv继承MuJoCoSimulator，但这是环境与仿真器的正常关系，不是问题。

**建议：使用当前架构，通过配置和接口扩展来满足需求！** 🎉
