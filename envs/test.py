import mujoco
import mujoco.viewer
import time
import numpy as np


class ModelSensor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 传感器ID映射
        self.sensor_ids = {
            'gyro': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro"),
            'accel': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc"),
            'quat': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat"),
        }
        
        # 无人机主体ID
        self.robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    
    def read_all(self):
        """读取所有传感器数据"""
        sensors = {}
        
        # 从传感器读取
        sensors['gyro'] = self._read_sensor('gyro')  # 角速度
        sensors['accel'] = self._read_sensor('accel')  # 加速度
        sensors['quat'] = self._read_sensor('quat')  # 四元数
        return sensors
    
    def _read_sensor(self, sensor_type):
        """读取指定类型的传感器数据"""
        sensor_id = self.sensor_ids[sensor_type]
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        # print(sensor_id, adr, dim)
        return self.data.sensordata[adr:adr + dim].copy()


# 加载模型
model = mujoco.MjModel.from_xml_path('E:\\UAV_RL\envs\crazyfile\scene.xml')
data = mujoco.MjData(model)
sensor = ModelSensor(model, data)
# print(sensor.sensor_ids, sensor.robot_id)
if __name__ == '__main__':
    ...
    # # 仿真循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(1000):
            # 设置初始控制信号（悬停状态）
            data.ctrl = np.array([0.26487, 0.26487, 0.26487, 0.26487])
            step_start = time.time()
            
            # 前向动力学
            mujoco.mj_step(model, data)
            
            # 读取传感器数据
            if i % 50 == 0:
                sensor_data = sensor.read_all()
                print(f"\ntime: {data.time:.2f}s")
                print(f"Robot_State:")
                print(
                    f"gyro: {sensor_data['gyro'][0]:.2f}, {sensor_data['gyro'][1]:.2f}, {sensor_data['gyro'][2]:.2f} ")
                print(
                    f"acc: {sensor_data['accel'][0]:.2f}, {sensor_data['accel'][1]:.2f}, {sensor_data['accel'][2]:.2f}")
                print(f"quat: {sensor_data['quat'][0]:.2f}, {sensor_data['quat'][1]:.2f}, {sensor_data['quat'][2]:.2f}")
            # 同步渲染
            viewer.sync()
            time.sleep(0.01)
    #     # 控制帧率
    #     # time_until_next_step = model.opt.timestep - (time.time() - step_start)
    #     # if time_until_next_step > 0:
    #     #     time.sleep(time_until_next_step)
