"""
简单测试 primitive 和 state_transform 的核心功能
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from algos.primitive import LatticePrimitive
from algos.state_transform import StateTransform

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def test_primitive():
    """测试基元生成"""
    print("=" * 70)
    print("测试1: 基元生成")
    print("=" * 70)
    
    lattice = LatticePrimitive.get_instance()
    
    print(f"\n基元数量: {lattice.traj_num}")
    print(f"垂直基元数: {lattice.vertical_num}")
    print(f"水平基元数: {lattice.horizon_num}")
    
    # 获取基元位置
    pos = lattice.getStateLattice()
    print(f"\n基元位置形状: {pos.shape}")
    print(f"前5个基元位置:")
    for i in range(min(5, pos.shape[0])):
        print(f"  基元 {i}: {pos[i]}")
    
    # 可视化基元
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=100, alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('基元位置 (3D)')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pos[:, 0], pos[:, 1], c='blue', s=100, alpha=0.7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('基元位置 (俯视图)')
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('test_primitive_positions.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ 基元位置可视化已保存到: test_primitive_positions.png")
    
    return lattice


def test_state_transform(lattice):
    """测试状态变换"""
    print("\n" + "=" * 70)
    print("测试2: 状态变换 (pred_to_endstate)")
    print("=" * 70)
    
    transform = StateTransform()
    
    # 模拟网络输出
    batch_size = 1
    V = lattice.vertical_num
    H = lattice.horizon_num
    
    # 简单的测试输入：选择中间的基元
    endstate_pred = np.zeros((batch_size, 9, V, H))
    
    # 设置中间基元的参数
    v_center = V // 2
    h_center = H // 2
    endstate_pred[0, 0, v_center, h_center] = 0.0  # delta_yaw
    endstate_pred[0, 1, v_center, h_center] = 0.0  # delta_pitch
    endstate_pred[0, 2, v_center, h_center] = 0.0  # radio (标准化后)
    
    # 速度和加速度
    endstate_pred[0, 3:6, v_center, h_center] = [0.5, 0.0, 0.0]
    endstate_pred[0, 6:9, v_center, h_center] = [0.1, 0.0, 0.0]
    
    print(f"\n模拟网络输出形状: {endstate_pred.shape}")
    
    # 转换为终端状态
    import torch
    endstate = transform.pred_to_endstate(torch.FloatTensor(endstate_pred))
    endstate = endstate.numpy()
    
    print(f"终端状态形状: {endstate.shape}")
    
    # 查看中间基元的终端状态
    center_idx = v_center * H + h_center
    print(f"\n中间基元 (ID: {center_idx}) 的终端状态:")
    print(f"  位置: {endstate[0, 0:3, v_center, h_center]}")
    print(f"  速度: {endstate[0, 3:6, v_center, h_center]}")
    print(f"  加速度: {endstate[0, 6:9, v_center, h_center]}")
    
    return endstate


def visualize_trajectory(lattice, endstate):
    """可视化基元与轨迹"""
    print("\n" + "=" * 70)
    print("可视化: 基元与终端位置")
    print("=" * 70)
    
    pos = lattice.getStateLattice()
    V = lattice.vertical_num
    H = lattice.horizon_num
    
    # 展平终端状态: [1, 9, 3, 5] -> [15, 9]
    # endstate[0] 是 (9, 3, 5)，直接 reshape 到 (9, 15)，再转置
    endstate_flat = endstate[0].reshape(9, V * H).T
    
    # 提取所有基元的终端位置
    end_positions = endstate_flat[:, 0:3]
    
    # fig = plt.figure(figsize=(15, 5))
    
    # # 子图1: 基元位置 (原始)
    # ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=100, alpha=0.7, label='预设基元')
    # ax1.set_xlabel('X (m)')
    # ax1.set_ylabel('Y (m)')
    # ax1.set_zlabel('Z (m)')
    # ax1.set_title('预设基元位置')
    # ax1.legend()
    # ax1.grid(True)
    
    # # 子图2: 终端位置 (网络输出)
    # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # ax2.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
    #            c='red', s=100, alpha=0.7, label='终端位置')
    # ax2.set_xlabel('X (m)')
    # ax2.set_ylabel('Y (m)')
    # ax2.set_zlabel('Z (m)')
    # ax2.set_title('网络输出的终端位置')
    # ax2.legend()
    # ax2.grid(True)
    
    # # 子图3: 对比
    # ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=100, alpha=0.5, label='预设基元')
    # ax3.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
    #            c='red', s=100, alpha=0.5, label='终端位置')
    # ax3.set_xlabel('X (m)')
    # ax3.set_ylabel('Y (m)')
    # ax3.set_zlabel('Z (m)')
    # ax3.set_title('预设基元 vs 终端位置')
    # ax3.legend()
    # ax3.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('test_primitive_vs_endstate.png', dpi=100, bbox_inches='tight')
    # print(f"✅ 可视化已保存到: test_primitive_vs_endstate.png")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Primitive 和 StateTransform 简单测试")
    print("=" * 70)
    
    # 测试1: 基元
    lattice = test_primitive()
    
    # 测试2: 状态变换
    endstate = test_state_transform(lattice)
    
    # 可视化
    visualize_trajectory(lattice, endstate)
    
    print("\n" + "=" * 70)
    print("✅ 简单测试完成！")
    print("=" * 70)
