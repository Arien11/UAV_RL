"""
简单测试深度CNN网络
"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from vision.depth_cnn import DepthCNN

print("=" * 60)
print("测试深度CNN网络")
print("=" * 60)

try:
    print("\n1. 创建DepthCNN...")
    cnn = DepthCNN(
        input_shape=(120, 160),
        num_features=256
    )
    print(f"✅ 模型创建成功！")
    print(f"   输入形状: {cnn.input_shape}")
    print(f"   输出特征: {cnn.num_features}")
    print(f"   参数量: {sum(p.numel() for p in cnn.parameters()):,}")
    
    print("\n2. 生成测试数据...")
    depth_image = np.random.rand(120, 160) * 5.0
    print(f"✅ 测试数据生成")
    print(f"   图像形状: {depth_image.shape}")
    print(f"   深度范围: [{depth_image.min():.2f}, {depth_image.max():.2f}] m")
    
    print("\n3. 提取特征...")
    features = cnn.extract_features(depth_image)
    print(f"✅ 特征提取成功！")
    print(f"   特征形状: {features.shape}")
    print(f"   特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
    
    print("\n📋 文件位置:")
    print("  - vision/depth_cnn.py - 网络定义")
    print("  - test_depth_network.py - 完整测试（包含可视化）")
    print("  - test_simple_cnn.py - 简单测试（本文件）")
    
    print("\n🚀 使用方法:")
    print("  from vision.depth_cnn import DepthCNN")
    print("  cnn = DepthCNN(input_shape=(120, 160), num_features=256)")
    print("  features = cnn.extract_features(depth_image)")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
