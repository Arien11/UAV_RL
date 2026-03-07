"""
测试ESDF（Euclidean Signed Distance Field）功能
包含距离计算和梯度验证
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import torch

from envs.Simulators.MujocoSim import MuJoCoSimulator
from envs.maps.esdf_generate import ESDFGenerator, create_esdf_from_mujoco, DifferentiableESDF

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_config():
    """加载配置文件"""
    config_path = os.path.join(project_root, 'config', 'QuadEnv_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def test_single_point_distance():
    """测试单个点的距离计算"""
    print("=" * 70)
    print("测试1: 单个点距离计算")
    print("=" * 70)
    
    # 加载配置和仿真器
    config = load_config()
    simulator = MuJoCoSimulator(config)
    
    # 创建ESDF生成器
    generator = ESDFGenerator(simulator.model, simulator.data)
    
    # 采样障碍物点
    generator.sample_obstacle_points(num_points_per_geom=100)
    
    # 测试几个点
    test_points = [
        np.array([0, 0, 0.5]),      # 起点附近
        np.array([2.5, 2.5, 1.0]),   # 中间
        np.array([5, 5, 1.0]),       # 终点
    ]
    
    print("\n测试点距离:")
    for i, point in enumerate(test_points):
        dist_kdtree, closest_kdtree = generator.distance_kdtree(point)
        
        print(f"\n  点 {i+1}: {point}")
        print(f"    KDTree距离: {dist_kdtree:.3f}m, 最近点: {closest_kdtree}")
    
    simulator.close()
    print("\n✅ 测试1完成")


def test_esdf_grid():
    """测试ESDF网格计算"""
    print("\n" + "=" * 70)
    print("测试2: ESDF网格计算")
    print("=" * 70)
    
    # 加载配置和仿真器
    config = load_config()
    simulator = MuJoCoSimulator(config)
    
    # 定义边界（根据你的地图调整）
    bounds = (
        np.array([-5, -5, -1]),   # xmin, ymin, zmin
        np.array([10, 10, 5])    # xmax, ymax, zmax
    )
    resolution = 0.2  # 分辨率
    
    print(f"\n边界: {bounds}")
    print(f"分辨率: {resolution}")
    
    # 创建ESDF
    esdf_data = create_esdf_from_mujoco(
        simulator.model, simulator.data,
        bounds=bounds,
        resolution=resolution
    )
    
    esdf = esdf_data['esdf']
    grid_points = esdf_data['grid_points']
    
    # 可视化ESDF切片
    print("\n生成可视化...")
    visualize_esdf_slices(esdf, bounds, resolution)
    
    simulator.close()
    print("\n✅ 测试2完成")


def visualize_esdf_slices(esdf, bounds, resolution):
    """可视化ESDF切片"""
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]
    
    # 创建坐标网格
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymin, ymax, resolution)
    z = np.arange(zmin, zmax, resolution)
    
    # 选择几个Z轴切片
    z_slices = [len(z) // 4, len(z) // 2, 3 * len(z) // 4]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, z_idx in enumerate(z_slices):
        slice_z = z[z_idx]
        esdf_slice = esdf[:, :, z_idx]
        
        # 绘制热力图
        im = axes[i].imshow(
            esdf_slice.T,
            extent=[xmin, xmax, ymin, ymax],
            origin='lower',
            cmap='viridis',
            aspect='auto'
        )
        axes[i].set_title(f'Z = {slice_z:.1f}m')
        axes[i].set_xlabel('X (m)')
        axes[i].set_ylabel('Y (m)')
        plt.colorbar(im, ax=axes[i], label='距离 (m)')
    
    plt.tight_layout()
    plt.savefig('esdf_slices.png', dpi=150, bbox_inches='tight')
    print("✅ ESDF切片图已保存到: esdf_slices.png")
    plt.show()


def test_gradient_validation():
    """测试梯度计算的正确性"""
    print("\n" + "=" * 70)
    print("测试3: 梯度计算验证")
    print("=" * 70)
    
    # 加载配置和仿真器
    config = load_config()
    simulator = MuJoCoSimulator(config)
    
    # 创建ESDF数据
    print("\n创建ESDF数据...")
    generator = ESDFGenerator(simulator.model, simulator.data)
    generator.sample_obstacle_points(num_points_per_geom=100)
    esdf_module = DifferentiableESDF(generator)


    # 测试几个点的梯度
    test_points = [
        np.array([0.5, 0.5, 0.5]),
        np.array([2.5, 2.5, 1.0]),
        np.array([4.5, 4.5, 0.8]),
    ]
    
    print("\n梯度验证:")
    print("-" * 70)
    
    all_gradients_numerical = []
    all_gradients_autograd = []
    all_points = []
    
    for i, point_np in enumerate(test_points):
        print(f"\n测试点 {i+1}: {point_np}")
        
        # ========== 方法1: PyTorch自动微分 ==========
        point_torch = torch.tensor(point_np, dtype=torch.float32, requires_grad=True)
        distance = esdf_module(point_torch)
        distance.backward() # 反向传播计算梯度
        grad_autograd = point_torch.grad.numpy().copy()
        
        # ========== 方法2: 数值梯度（有限差分） ==========
        eps = 1e-4  # 微小扰动
        grad_numerical = np.zeros_like(point_np)
        
        for dim in range(3):
            # 正向扰动
            point_plus = point_np.copy()
            point_plus[dim] += eps
            dist_plus, _ = generator.distance_kdtree(point_plus)
            
            # 负向扰动
            point_minus = point_np.copy()
            point_minus[dim] -= eps
            dist_minus, _ = generator.distance_kdtree(point_minus)
            
            # 中心差分
            grad_numerical[dim] = (dist_plus - dist_minus) / (2 * eps)
        
        # ========== 验证梯度正确性 ==========
        grad_error = np.linalg.norm(grad_autograd - grad_numerical)
        
        print(f"  自动微分梯度: {grad_autograd}")
        print(f"  数值梯度:     {grad_numerical}")
        print(f"  梯度误差:     {grad_error:.2e}")
        
        if grad_error < 1e-2:
            print("  ✅ 梯度验证通过！")
        else:
            print("  ⚠️  梯度误差较大")
        
        all_points.append(point_np)
        all_gradients_numerical.append(grad_numerical)
        all_gradients_autograd.append(grad_autograd)
    
    # ========== 可视化梯度 ==========
    print("\n生成梯度可视化...")
    visualize_gradients(
        all_points, 
        all_gradients_numerical, 
        all_gradients_autograd,
        generator
    )
    
    simulator.close()
    print("\n✅ 测试3完成")


def visualize_gradients(points, gradients_numerical, gradients_autograd, generator):
    """可视化梯度向量"""
    fig = plt.figure(figsize=(15, 6))
    
    # ========== 子图1: 数值梯度 ==========
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 绘制障碍物点（采样点）
    obstacle_points = generator._obstacle_points
    ax1.scatter(
        obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2],
        c='gray', s=1, alpha=0.3, label='障碍物点'
    )
    
    # 绘制测试点和梯度向量
    for point, grad in zip(points, gradients_numerical):
        # 绘制点
        ax1.scatter(point[0], point[1], point[2], c='red', s=100, marker='o')
        # 绘制梯度向量
        ax1.quiver(
            point[0], point[1], point[2],
            grad[0], grad[1], grad[2],
            color='blue', length=0.5, normalize=True,
            label='数值梯度' if point is points[0] else ""
        )
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('数值梯度（有限差分）')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim([-2, 7])
    ax1.set_ylim([-2, 7])
    ax1.set_zlim([0, 3])
    
    # ========== 子图2: 自动微分梯度 ==========
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 绘制障碍物点
    ax2.scatter(
        obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2],
        c='gray', s=1, alpha=0.3, label='障碍物点'
    )
    
    # 绘制测试点和梯度向量
    for point, grad in zip(points, gradients_autograd):
        # 绘制点
        ax2.scatter(point[0], point[1], point[2], c='red', s=100, marker='o')
        # 绘制梯度向量
        ax2.quiver(
            point[0], point[1], point[2],
            grad[0], grad[1], grad[2],
            color='green', length=0.5, normalize=True,
            label='自动微分梯度' if point is points[0] else ""
        )
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('自动微分梯度（PyTorch）')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim([-2, 7])
    ax2.set_ylim([-2, 7])
    ax2.set_zlim([0, 3])
    
    plt.tight_layout()
    plt.savefig('gradient_validation.png', dpi=150, bbox_inches='tight')
    print("✅ 梯度验证图已保存到: gradient_validation.png")
    plt.show()



if __name__ == "__main__":
    print("=" * 70)
    print("ESDF功能测试")
    print("=" * 70)
    
    try:
        # 测试1: 单个点距离
        # test_single_point_distance()
        
        # # 测试2: ESDF网格
        # test_esdf_grid()
        
        # 测试3: 梯度验证
        test_gradient_validation()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
