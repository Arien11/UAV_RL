"""
测试新的观测空间结构
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import numpy as np

print("=" * 80)
print("测试新的观测空间结构 - 分离运动状态和传感器数据")
print("=" * 80)

try:
    from envs.QuadEnv import QuadEnv
    
    print("\n1. 读取配置文件...")
    with open(r"E:\UAV_RL\config\QuadEnv_config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    print("✅ 配置文件读取成功")
    
    print("\n2. 创建环境...")
    env = QuadEnv(config_data)
    print("✅ 环境创建成功")
    
    print("\n3. 检查观测空间定义...")
    print(f"   观测空间: {env.observation_space}")
    print(f"   运动状态形状: {env.observation_space.get('motion_shape')}")
    print(f"   传感器形状: {env.observation_space.get('sensors_shapes')}")
    
    print("\n4. 重置环境...")
    obs = env.reset()
    print("✅ 环境重置成功")
    
    print("\n5. 检查观测数据结构...")
    print(f"   观测类型: {type(obs)}")
    print(f"   观测键: {list(obs.keys())}")
    
    motion_obs = obs.get('motion')
    sensors_obs = obs.get('sensors', {})
    
    print(f"\n   运动状态:")
    print(f"     类型: {type(motion_obs)}")
    print(f"     形状: {motion_obs.shape}")
    print(f"     数据: {motion_obs}")
    
    print(f"\n   传感器数据:")
    print(f"     传感器数量: {len(sensors_obs)}")
    for sensor_name, sensor_data in sensors_obs.items():
        print(f"     {sensor_name}:")
        print(f"       类型: {type(sensor_data)}")
        print(f"       形状: {sensor_data.shape}")
        print(f"       范围: [{sensor_data.min():.4f}, {sensor_data.max():.4f}]")
    
    print("\n6. 测试一步仿真...")
    action = np.zeros_like(env.action_space)
    obs, reward, done, info = env.step(action)
    print(f"✅ 仿真步骤完成")
    print(f"   奖励: {reward:.3f}")
    print(f"   完成: {done}")
    
    print("\n7. 再次检查观测...")
    motion_obs = obs.get('motion')
    sensors_obs = obs.get('sensors', {})
    print(f"   运动状态形状: {motion_obs.shape}")
    for sensor_name, sensor_data in sensors_obs.items():
        print(f"   {sensor_name} 形状: {sensor_data.shape}")
    
    print("\n8. 关闭环境...")
    env.close()
    print("✅ 环境关闭成功")
    
    print("\n" + "=" * 80)
    print("✅ 观测空间测试完成！")
    print("=" * 80)
    

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
