"""
测试深度图像处理网络
"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from vision.depth_cnn import DepthCNN, DepthAutoencoder

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("测试深度图像处理神经网络")
print("=" * 70)

try:
    # ========== 1. 测试 DepthCNN ==========
    print("\n" + "=" * 70)
    print("测试 1: DepthCNN - 特征提取网络")
    print("=" * 70)
    
    cnn = DepthCNN(
        input_shape=(120, 160),
        num_features=256
    )
    print(f"\n✅ DepthCNN 创建成功")
    print(f"   输入形状: {cnn.input_shape}")
    print(f"   输出特征: {cnn.num_features}")
    print(f"   参数量: {sum(p.numel() for p in cnn.parameters()):,}")
    
    # 生成测试数据
    print(f"\n生成测试数据...")
    num_samples = 5
    depth_images = np.zeros((num_samples, 120, 160))
    
    # 创建一些有结构的深度图
    for i in range(num_samples):
        x = np.linspace(-1, 1, 160)
        y = np.linspace(-1, 1, 120)
        xx, yy = np.meshgrid(x, y)
        
        # 模拟场景：中心有一个障碍物
        depth_images[i] = 3.0 + np.sin(xx * 3 + i) * 0.5
        obstacle = np.exp(-((xx)**2 + (yy + 0.3)**2) / 0.1)
        depth_images[i] -= obstacle * 1.5
        depth_images[i] = np.clip(depth_images[i], 0.1, 5.0)
    
    print(f"   样本数量: {num_samples}")
    print(f"   图像形状: {depth_images.shape}")
    print(f"   深度范围: [{depth_images.min():.2f}, {depth_images.max():.2f}] m")
    
    # 提取特征
    print(f"\n提取特征...")
    features = cnn.extract_features(depth_images)
    print(f"✅ 特征提取成功")
    print(f"   特征形状: {features.shape}")
    print(f"   特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # ========== 2. 测试 DepthAutoencoder ==========
    print("\n" + "=" * 70)
    print("测试 2: DepthAutoencoder - 自编码器")
    print("=" * 70)
    
    autoencoder = DepthAutoencoder(
        input_shape=(120, 160),
        latent_dim=64
    )
    print(f"\n✅ DepthAutoencoder 创建成功")
    print(f"   输入形状: {autoencoder.input_shape}")
    print(f"   潜在维度: {autoencoder.latent_dim}")
    print(f"   参数量: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    # 测试自编码器
    print(f"\n测试自编码器...")
    import torch
    import torch.nn.functional as F
    
    x = torch.FloatTensor(depth_images).unsqueeze(1)
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    
    with torch.no_grad():
        x_recon, z = autoencoder(x_norm)
    
    print(f"✅ 自编码成功")
    print(f"   输入形状: {x.shape}")
    print(f"   重建形状: {x_recon.shape}")
    print(f"   潜在向量: {z.shape}")
    
    recon_loss = F.mse_loss(x_recon, x_norm)
    print(f"   重建MSE损失: {recon_loss.item():.6f}")
    
    # ========== 3. 可视化 ==========
    print("\n" + "=" * 70)
    print("可视化结果")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # 原始深度图
        im1 = axes[0, i].imshow(depth_images[i], cmap='jet', vmin=0, vmax=5)
        axes[0, i].set_title(f'原始深度图 {i+1}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 归一化后的输入
        im2 = axes[1, i].imshow(x_norm[i, 0], cmap='gray')
        axes[1, i].set_title(f'归一化输入')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # 重建的深度图
        im3 = axes[2, i].imshow(x_recon[i, 0], cmap='gray')
        axes[2, i].set_title(f'自编码器重建')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('depth_network_test.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存到: depth_network_test.png")
    
    # ========== 4. 特征可视化 ==========
    print(f"\n特征可视化...")
    fig2, ax = plt.subplots(figsize=(12, 5))
    
    # 绘制特征向量
    feature_img = ax.imshow(features, aspect='auto', cmap='viridis')
    ax.set_title('提取的特征向量 (5个样本 x 256维)')
    ax.set_xlabel('特征维度')
    ax.set_ylabel('样本索引')
    plt.colorbar(feature_img, ax=ax)
    
    plt.tight_layout()
    plt.savefig('features_visualization.png', dpi=100, bbox_inches='tight')
    print(f"✅ 特征可视化已保存到: features_visualization.png")
    
    plt.close('all')
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)
    
    print("\n📊 网络架构总结:")
    print("  DepthCNN:")
    print(f"    - 输入: (1, 120, 160)")
    print(f"    - 输出: (256,) 特征向量")
    print(f"    - 用途: 从深度图提取视觉特征")
    print("\n  DepthAutoencoder:")
    print(f"    - 输入: (1, 120, 160)")
    print(f"    - 潜在空间: (64,)")
    print(f"    - 输出: (1, 120, 160) 重建图像")
    print(f"    - 用途: 压缩、去噪、预训练")
    
    print("\n🚀 下一步建议:")
    print("  1. 使用真实深度图数据进行预训练")
    print("  2. 添加分类/回归头用于特定任务")
    print("  3. 集成到强化学习框架中")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
