"""
简单的深度图像处理神经网络
用于处理无人机采集的深度图像
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthCNN(nn.Module):
    """
    简单的深度图像CNN网络
    用于从深度图像中提取特征
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int] = (120, 160),
                 num_features: int = 256,
                 num_channels: int = 1):
        """
        初始化深度CNN
        
        Args:
            input_shape: 输入图像形状 (height, width)
            num_features: 输出特征维度
            num_channels: 输入通道数（深度图为1）
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_features = num_features
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # 计算卷积后的特征图大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, *input_shape)
            conv_out = self.conv_layers(dummy_input)
            self.flatten_size = int(np.prod(conv_out.shape[1:]))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_features),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入深度图像 [batch, channels, height, width]
            
        Returns:
            提取的特征 [batch, num_features]
        """
        # 卷积特征提取
        x = self.conv_layers(x)
        
        # 展平
        x = x.flatten(1)
        
        # 全连接
        x = self.fc_layers(x)
        
        return x
    
    def extract_features(self, depth_image: np.ndarray) -> np.ndarray:
        """
        从numpy数组中提取特征
        
        Args:
            depth_image: 深度图像 [height, width] 或 [batch, height, width]
            
        Returns:
            提取的特征 [num_features] 或 [batch, num_features]
        """
        self.eval()
        
        # 预处理
        if len(depth_image.shape) == 2:
            # 单张图像 [height, width] -> [1, 1, height, width]
            x = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0)
        elif len(depth_image.shape) == 3:
            # 批量图像 [batch, height, width] -> [batch, 1, height, width]
            x = torch.FloatTensor(depth_image).unsqueeze(1)
        else:
            raise ValueError(f"不支持的输入形状: {depth_image.shape}")
        
        # 归一化
        x = (x - x.mean()) / (x.std() + 1e-8)
        
        # 前向传播
        with torch.no_grad():
            features = self.forward(x)
        
        return features.numpy()


class DepthAutoencoder(nn.Module):
    """
    深度图像自编码器
    用于压缩和重建深度图像
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int] = (120, 160),
                 latent_dim: int = 64):
        """
        初始化自编码器
        
        Args:
            input_shape: 输入图像形状 (height, width)
            latent_dim: 潜在空间维度
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 计算编码器输出大小
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]
            self.flatten_size = int(np.prod(self.enc_shape))
        
        # 潜在层
        self.fc_enc = nn.Linear(self.flatten_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_size)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc_enc(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        x = self.fc_dec(z)
        x = x.view(-1, *self.enc_shape)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ==================== 示例使用 ====================

def example_usage():
    """
    示例：如何使用深度CNN网络
    """
    print("=" * 60)
    print("深度图像处理网络 - 示例使用")
    print("=" * 60)
    
    # 1. 创建CNN模型
    print("\n1. 创建深度CNN模型...")
    cnn = DepthCNN(
        input_shape=(120, 160),
        num_features=256
    )
    print(f"✅ 模型创建成功")
    print(f"   输入形状: {cnn.input_shape}")
    print(f"   输出特征维度: {cnn.num_features}")
    print(f"   总参数量: {sum(p.numel() for p in cnn.parameters()):,}")
    
    # 2. 生成模拟深度图像
    print("\n2. 生成模拟深度图像...")
    batch_size = 4
    depth_images = np.random.rand(batch_size, 120, 160) * 5.0  # 0-5米
    print(f"✅ 生成模拟数据")
    print(f"   批量大小: {batch_size}")
    print(f"   图像形状: {depth_images.shape}")
    print(f"   深度范围: [{depth_images.min():.2f}, {depth_images.max():.2f}] 米")
    
    # 3. 提取特征
    print("\n3. 提取特征...")
    features = cnn.extract_features(depth_images)
    print(f"✅ 特征提取成功")
    print(f"   特征形状: {features.shape}")
    
    # 4. 创建自编码器
    print("\n4. 创建深度自编码器...")
    autoencoder = DepthAutoencoder(
        input_shape=(120, 160),
        latent_dim=64
    )
    print(f"✅ 自编码器创建成功")
    print(f"   潜在空间维度: {autoencoder.latent_dim}")
    print(f"   总参数量: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    # 5. 测试自编码器
    print("\n5. 测试自编码器...")
    x = torch.FloatTensor(depth_images).unsqueeze(1)
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    with torch.no_grad():
        x_recon, z = autoencoder(x)
    
    print(f"✅ 自编码器测试成功")
    print(f"   输入形状: {x.shape}")
    print(f"   重建形状: {x_recon.shape}")
    print(f"   潜在向量形状: {z.shape}")
    
    # 计算重建误差
    recon_loss = F.mse_loss(x_recon, x)
    print(f"   重建MSE损失: {recon_loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("✅ 示例运行完成！")
    print("=" * 60)
    
    print("\n📋 使用说明:")
    print("  1. DepthCNN - 用于特征提取")
    print("  2. DepthAutoencoder - 用于压缩和重建")
    print("\n🔧 可扩展方向:")
    print("  - 添加分类头用于障碍物检测")
    print("  - 添加回归头用于距离估计")
    print("  - 预训练后迁移到RL任务")


if __name__ == "__main__":
    example_usage()
