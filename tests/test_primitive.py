"""
测试 primitive 和 state_transform 以及轨迹生成
模拟观测量，不接入真实项目
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rl.primitive import LatticePrimitive
from rl.state_transform import StateTransform
from QuadControl.poly_solver import Poly5Solver
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TestPrimitive:
    def __init__(self) -> None:
        print("=" * 70)
        print("初始化测试环境")
        print("=" * 70)
        
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        
        # 测试参数
        self.traj_time = 1.0  # 轨迹时间
        self.batch_size = 1
        
        # 世界坐标系旋转（单位矩阵，测试用）
        self.Rotation_wc = np.eye(3)
        
        print(f"\n✅ 初始化完成")
        print(f"   基元数量: {self.lattice_primitive.traj_num}")
        print(f"   垂直基元数: {self.lattice_primitive.vertical_num}")
        print(f"   水平基元数: {self.lattice_primitive.horizon_num}")
        print(f"   轨迹时间: {self.traj_time}s")
    
    def simulate_odom(self, batch_size: int = 1):
        """
        模拟里程计观测
        
        Returns:
            obs: [batch, 9] - 机体坐标系中的观测
                 [vx, vy, vz, ax, ay, az, gx, gy, gz]
        """
        obs = np.zeros((batch_size, 9))
        
        for b in range(batch_size):
            # 模拟速度（机体坐标系）
            obs[b, 0:3] = np.array([0.5, 0.0, 0.0])  # 向前0.5 m/s
            
            # 模拟加速度
            obs[b, 3:6] = np.array([0.1, 0.0, 0.0])  # 向前0.1 m/s²
            
            # 模拟重力向量（机体坐标系，近似向下）
            obs[b, 6:9] = np.array([0.0, 0.0, -1.0])    
        
        return torch.tensor(obs, dtype=torch.float32)
    
    def simulate_policy(self, obs_input: np.ndarray):
        """
        模拟策略网络输出
        
        Args:
            obs_input: [batch, 9, V, H] - 基元坐标系中的观测
            
        Returns:
            endstate_pred: [batch, 9, V, H] - 预测的终端状态
            score_pred: [batch, V, H] - 预测的分数
        """
        batch_size = obs_input.shape[0]
        V = self.lattice_primitive.vertical_num
        H = self.lattice_primitive.horizon_num
        
        # 模拟终端状态预测
        endstate_pred = np.zeros((batch_size, 9, V, H))
        
        # 前3维：primitive参数 (delta_yaw, delta_pitch, radio)
        endstate_pred[:, 0:3, :, :] = np.random.randn(batch_size, 3, V, H) * 0.1
        
        # 中间3维：速度
        endstate_pred[:, 3:6, :, :] = np.random.randn(batch_size, 3, V, H) * 0.5
        
        # 后3维：加速度
        endstate_pred[:, 6:9, :, :] = np.random.randn(batch_size, 3, V, H) * 0.2
        
        # 模拟分数（中间的基元分数较高）
        score_pred = np.zeros((batch_size, V, H))
        for v in range(V):
            for h in range(H):
                # 距离中心越近，分数越高（越小越好）
                dv = v - (V - 1) / 2
                dh = h - (H - 1) / 2
                score_pred[:, v, h] = np.sqrt(dv**2 + dh**2) + np.random.randn(batch_size) * 0.1
        
        return endstate_pred, score_pred
    
    def test(self):
        """
        完整测试流程
        """
        print("\n" + "=" * 70)
        print("开始测试")
        print("=" * 70)
        
        # ========== 1. 模拟观测 ==========
        print("\n1. 模拟里程计观测...")
        obs_norm = self.simulate_odom(self.batch_size)
        print(f"✅ 观测模拟完成")
        print(f"   观测形状: {obs_norm.shape}")
        print(f"   观测范围: [{obs_norm.min():.3f}, {obs_norm.max():.3f}]")
        
        # ========== 2. 准备输入 ==========
        print("\n2. 准备网络输入...")
        obs_input = self.state_transform.prepare_input(obs_norm)
        print(f"✅ 输入准备完成")
        print(f"   输入形状: {obs_input.shape}")
        
        # ========== 3. 模拟策略 ==========
        print("\n3. 模拟策略网络输出...")
        # endstate_pred：[B, 9, 3, 5]，score_pred：[B, 3, 5] 每个基元的分数
        endstate_pred, score_pred = self.simulate_policy(obs_input)
        print(f"✅ 策略模拟完成")
        print(f"   终端状态形状: {endstate_pred.shape}")
        print(f"   分数形状: {score_pred.shape}")
        
        # ========== 4. 处理输出 ==========
        print("\n4. 处理网络输出...")
        import torch
        # [B, 9, 3, 5]
        endstate = self.state_transform.pred_to_endstate(torch.FloatTensor(endstate_pred))
        endstate = endstate.numpy()
        print(f"✅ 输出处理完成")
        print(f"   终端状态形状: {endstate.shape}")
        
        # ========== 5. 选择最佳基元 ==========
        print("\n5. 选择最佳基元...")
        batch_idx = 0  # 测试第一个样本
        action_id = np.argmin(score_pred[batch_idx])        # 15条轨迹中选择最优轨迹
        print(f"✅ 最佳基元选择完成")
        print(f"   最佳基元ID: {action_id}")
        print(f"   最佳分数: {score_pred[batch_idx].flatten()[action_id]:.3f}")
        
        # ========== 6. 转换到世界坐标系 ==========
        print("\n6. 转换到世界坐标系...")
        # 重塑终端状态
        N = self.lattice_primitive.traj_num
        # [px,py,pz, vx,vy,vz, ax,ay, az]->[px vx ax, py vy ay, pz vz az]   
        endstate_c = endstate[batch_idx].reshape(N, 3, 3).transpose(0, 2, 1)    #  [9, 3, 5]—>[15, 3, 3]
        print(f"   终端状态形状 (机体): {endstate_c.shape}")
        
        # 转换到世界坐标系
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)
        print(f"✅ 世界坐标系转换完成")
        print(f"   终端状态形状 (世界): {endstate_w.shape}")
        
        # ========== 7. 生成轨迹 ==========
        print("\n7. 生成轨迹...")
        
        # 模拟起点
        start_pos = np.array([0.0, 0.0, 0.5])  # 初始位置
        start_vel = np.array([0.5, 0.0, 0.0])  # 初始速度
        start_acc = np.array([0.1, 0.0, 0.0])  # 初始加速度
        
        # 终点（相对于起点）
        end_pos_rel = endstate_w[action_id, :, 0]  # 相对位置
        end_vel_rel = endstate_w[action_id, :, 1]  # 相对速度
        end_acc_rel = endstate_w[action_id, :, 2]  # 相对加速度
        
        print(f"   起点: pos={start_pos}, vel={start_vel}, acc={start_acc}")
        print(f"   相对终点: pos={end_pos_rel}, vel={end_vel_rel}, acc={end_acc_rel}")
        
        # 绝对终点
        end_pos = start_pos + end_pos_rel
        end_vel = end_vel_rel
        end_acc = end_acc_rel
        
        print(f"   绝对终点: pos={end_pos}, vel={end_vel}, acc={end_acc}")
        
        # 生成5次多项式轨迹
        print(f"\n   生成5次多项式轨迹...")
        traj_t = np.linspace(0, self.traj_time, 100)
        
        poly_x = Poly5Solver(
            start_pos[0], start_vel[0], start_acc[0],
            end_pos[0], end_vel[0], end_acc[0],
            self.traj_time
        )
        poly_y = Poly5Solver(
            start_pos[1], start_vel[1], start_acc[1],
            end_pos[1], end_vel[1], end_acc[1],
            self.traj_time
        )
        poly_z = Poly5Solver(
            start_pos[2], start_vel[2], start_acc[2],
            end_pos[2], end_vel[2], end_acc[2],
            self.traj_time
        )
        
        # 计算轨迹
        traj_pos = np.zeros((len(traj_t), 3))
        traj_vel = np.zeros((len(traj_t), 3))
        traj_acc = np.zeros((len(traj_t), 3))
        
        for i, t in enumerate(traj_t):
            traj_pos[i, 0] = poly_x.get_position(t)
            traj_pos[i, 1] = poly_y.get_position(t)
            traj_pos[i, 2] = poly_z.get_position(t)
            
            traj_vel[i, 0] = poly_x.get_velocity(t)
            traj_vel[i, 1] = poly_y.get_velocity(t)
            traj_vel[i, 2] = poly_z.get_velocity(t)
            
            traj_acc[i, 0] = poly_x.get_acceleration(t)
            traj_acc[i, 1] = poly_y.get_acceleration(t)
            traj_acc[i, 2] = poly_z.get_acceleration(t)
        
        print(f"✅ 轨迹生成完成")
        print(f"   轨迹点数: {len(traj_t)}")
        print(f"   起点: {traj_pos[0]}")
        print(f"   终点: {traj_pos[-1]}")
        
        # # ========== 8. 可视化 ==========
        print("\n8. 生成可视化...")
        self.visualize_results(
            traj_t, traj_pos, traj_vel, traj_acc,
            start_pos, end_pos, action_id, endstate_w
        )
        
        print("\n" + "=" * 70)
        print("✅ 测试完成！")
        print("=" * 70)
    
    def visualize_results(self, traj_t, traj_pos, traj_vel, traj_acc, 
                          start_pos, end_pos, action_id, endstate_w):
        """
        可视化测试结果 - 论文级别
        
        显示内容：
        - 实际轨迹（蓝色实线）
        - 预设基元方向（黑色虚线）
        - 网络输出的终端位置（红色点线）
        - 起点（绿色圆点）
        - 终点（红色圆点）
        """
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取基元信息
        lattice_pos = self.lattice_primitive.getStateLattice()  # [15, 3] 机体坐标系
        N = self.lattice_primitive.traj_num
        
        # ========== 将基元转换到世界坐标系 ==========
        lattice_pos_world = np.matmul(self.Rotation_wc, lattice_pos.T).T  # [15, 3]
        end_pos_world = endstate_w[:, :, 0]  # [15, 3]
        
        # ========== 绘制预设基元方向（黑色虚线） ==========
        for i in range(N):
            ax.plot([start_pos[0], start_pos[0] + lattice_pos_world[i, 0]], 
                   [start_pos[1], start_pos[1] + lattice_pos_world[i, 1]], 
                   [start_pos[2], start_pos[2] + lattice_pos_world[i, 2]], 
                   color='#808080', linestyle='--', alpha=0.4, linewidth=1.0)
        
        # ========== 绘制网络输出的终端位置（红色点线） ==========
        for i in range(N):
            ax.plot([start_pos[0], start_pos[0] + end_pos_world[i, 0]], 
                   [start_pos[1], start_pos[1] + end_pos_world[i, 1]], 
                   [start_pos[2], start_pos[2] + end_pos_world[i, 2]], 
                   color='#FF6B6B', linestyle=':', alpha=0.3, linewidth=0.8)
        
        # 标记所有基元终点
        ax.scatter(start_pos[0] + end_pos_world[:, 0], 
                  start_pos[1] + end_pos_world[:, 1], 
                  start_pos[2] + end_pos_world[:, 2], 
                  c='#FF6B6B', s=30, alpha=0.4, marker='o', 
                  label='Terminal Positions')
        
        # ========== 绘制实际轨迹（蓝色实线） ==========
        ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2], 
               color='#2E86AB', linewidth=2.5, label='Generated Trajectory', 
               zorder=10)
        
        # ========== 标记起点和终点 ==========
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                  c='#28A745', s=200, marker='o', edgecolors='white', 
                  linewidths=2, label='Start Position', zorder=11)
        
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                  c='#DC3545', s=200, marker='o', edgecolors='white', 
                  linewidths=2, label='End Position', zorder=11)
        
        # ========== 设置坐标轴标签 ==========
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        
        # ========== 设置标题 ==========
        ax.set_title(f'Motion Primitive Trajectory\n(Primitive ID: {action_id})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # ========== 设置图例 ==========
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                 edgecolor='gray', fancybox=True)
        
        # ========== 设置网格 ==========
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # ========== 设置坐标轴范围 ==========
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_zlim([-3, 6])
        
        # ========== 设置视角 ==========
        ax.view_init(elev=25, azim=45)
        
        # ========== 设置背景颜色 ==========
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        
        # ========== 设置刻度字体大小 ==========
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        
        # ========== 调整布局 ==========
        plt.tight_layout()
        
        # # ========== 保存图片 ==========
        # plt.savefig('primitive_trajectory.png', dpi=300, bbox_inches='tight', 
        #            facecolor='white', edgecolor='none')
        # print(f"✅ 可视化已保存到: primitive_trajectory.png")
        
        plt.show()


if __name__ == "__main__":
    test = TestPrimitive()
    test.test()
