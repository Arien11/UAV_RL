import mujoco
from QuadControl.mixer.linear_mixer import LinearMixer


def test_mixer():
    model = mujoco.MjModel.from_xml_path("/envs/crazyfile/scene.xml")
    data = mujoco.MjData(model)
    
    mixer = LinearMixer()
    
    # 设置初始姿态水平
    data.qpos[3:7] = [1, 0, 0, 0]
    
    # 测试工况：悬停推力 + 小滚转力矩
    thrust = 0.3237
    Mx = 0.001
    My = 0.0
    Mz = 0.0
    
    ctrl = mixer.calculate(thrust, Mx, My, Mz)
    data.ctrl[:] = ctrl
    
    mujoco.mj_forward(model, data)
    
    print("控制量:", ctrl)
    print("作动器力/力矩 (qfrc_actuator):", data.qfrc_actuator)
    print("姿态四元数:", data.qpos[3:7])


if __name__ == "__main__":
    test_mixer()
