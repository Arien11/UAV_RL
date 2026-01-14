from abc import ABC, abstractmethod
from typing import Dict, Any
import mujoco
import numpy as np
import mujoco.viewer as viewer
import time


# class BaseModelLoader(ABC):
#     """模型加载器抽象类"""
#
#     @abstractmethod
#     def load(self, model_info: Dict[str, Any]) -> mujoco.MjModel:
#         """加载模型"""
#         pass
#
#     @abstractmethod
#     def validate(self, model: mujoco.MjModel) -> bool:
#         """验证模型"""
#         pass

class XMLModelLoader:
    def __init__(self, asset_resolver=None):
        self.asset_resolver = asset_resolver
    
    def load(self, model_info):
        # 利用string进行加载
        if 'xml_string' in model_info:
            return mujoco.MjModel.from_xml_string(
                model_info['xml_string'],
                assets=self.asset_resolver
            )
        # 利用文件读取路径进行加载
        elif 'xml_path' in model_info:
            return mujoco.MjModel.from_xml_path(
                model_info['xml_path']
            )
        else:
            raise ValueError("No valid model source provided")
    
    def validate(self, model):
        return model is not None and model.nq > 0


# # 从文件加载
# loader = XMLModelLoader()
# # 加载模型
# try:
#     model_info = {'xml_path': r'E:\UAV_RL\envs\crazyfile\scene.xml'}
#     model = loader.load(model_info)
#
#     # 验证模型
#     if loader.validate(model):
#         print("模型加载成功！")
#         # print(f"自由度数量: {model.nq}")
#     else:
#         print("模型验证失败！")
#
# except ValueError as e:
#     print(f"参数错误: {e}")
# except FileNotFoundError:
#     print("XML文件不存在，请检查路径！")
# except Exception as e:
#     print(f"加载失败: {e}")

class BaseSimulator(ABC):
    @abstractmethod
    def render(self):
        """不同仿真器的可视化"""
        pass


class MuJoCoSimulator(BaseSimulator):
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        # 传感器ID映射
        # self.sensor_ids = {
        #     'gyro': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro"),
        #     'accel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc"),
        #     'quat': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat"),
        # }
        
        # 智能体ID(无人机)
        self.robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    
    def step(self, control_action=None):
        if control_action is not None:
            # 这里就是给到控制输出的位置
            self.data.ctrl[:] = control_action
        
        mujoco.mj_step(self.model, self.data)
        
        return self._get_state()
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        return self._get_state()
    
    def render(self):
        viewer.launch(loader=self.load_callback)
    
    def load_callback(self):
        mujoco.set_mjcb_control(None)
        m = self.model
        d = self.data
        if m is not None:
            mujoco.set_mjcb_control(lambda m, d: self.control_callback(m, d))  # 设置控制回调函数
        return m, d
    
    def control_callback(self, m, d):
        ...
    
    def _get_state(self):
        """获取纯物理状态"""
        return {
            'qpos': self.data.qpos.copy(),  # [x, y, z, qw, qx, qy, qz]
            'qvel': self.data.qvel.copy(),  # [vx, vy, vz, wx, wy, wz]
            'time': self.data.time,
            'ctrl': self.data.ctrl.copy()
        }
    
    def _get_sensor_state(self):
        """获取传感器读数（模拟真实传感器）"""
        return {
            'gyro': self.data.sensordata[0:3],  # 陀螺仪
            'accel': self.data.sensordata[3:6],  # 加速度计
            'quat': self.data.sensordata[6:10],  # 姿态四元数
        }
    
    def _isVis(self):
        self.vis = True


# sim = MuJoCoSimulator()

if __name__ == '__main__':
    ...
