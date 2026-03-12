import os
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import open3d as o3d
from scipy.ndimage import distance_transform_edt


class SafetyLoss(nn.Module):
    def __init__(self, L) -> None:
        super(SafetyLoss, self).__init__()
        self.d0 = 1.2
        self.r = 6
        self._L = L
        self.sgm_time = 5/3
        self.eval_points = 30
        self.voxel_size = 0.2
        self.map_expand_min =  [0, 0, 0.2]  # shape: (N, 3)
        self.map_expand_max =  [0, 0, 1]  # shape: (N, 3)
        self.sdf_shapes = None  # shape: (N, 3)
        self.device = self._L.device
        self.time_integral = True
        self.traj_num = 15
        print("Building ESDF map...")
        pcd_path = "envs/crazyfile/seed_6.pcd"
        sdf_tensor, min_bound, max_bound, sdf_shape = self._get_sdf_from_pcd(pcd_path)
        self.sdf_maps = [sdf_tensor]
        self.min_bounds = [min_bound]
        self.max_bounds = [max_bound]
        self.sdf_shapes = [sdf_shape]
        self.num_maps = 1
        print("Map built!")

    def forward(self, Df, Dp, map_id):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            map_id: (batch_size) which esdf map to query
        Returns:
            cost_colli: (batch_size) → safety loss
        """
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)     # AC=D加入映射和置换后-> LC = D, L = (A^(-1)) * C
        coe = self.get_coefficient_from_derivative(Dp, Df, L)   # 根据LC = D, 反解得到多项式系数 [batch_size, 18]

        dt = self.sgm_time / self.eval_points
        t_list = torch.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        # get pos from coeff [B*H*V, N, 3] -> [B, H*V*N, 3]
        pos_coe = self.get_position_from_coeff(coe, t_list)     # [B, H*V*N, 3]
        pos_batch = pos_coe.reshape(-1, self.traj_num * pos_coe.shape[1], 3)

        # get info from sdf_map
        cost, dist = self.get_distance_cost(pos_batch, map_id)

        if self.time_integral:
            # Compute average time integral of trajectory cost
            # Issue: uneven eval points may undercut cost by quickly crossing obstacles
            cost_colli = cost.reshape(-1, pos_coe.shape[1]).mean(dim=-1)  # [B*H*V, N]
        else:
            # Compute average line integral of trajectory cost
            vel_coe = self.get_velocity_from_coeff(coe, t_list)
            vel_coe = vel_coe.norm(dim=-1)
            line_integral_cost = (cost.reshape(-1, pos_coe.shape[1]) * vel_coe * dt).sum(dim=1)  # [B*H*V, N] -> [B*H*V]
            line_length = (vel_coe * dt).sum(dim=1)  # [B*H*V]
            cost_colli = line_integral_cost / line_length  # [B*H*V]

        return cost_colli

    # ------------------------------- 反解系数 ------------------------------- # 
    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = torch.zeros(Dp.shape[0], 18, device=self.device)   # 三轴的五次多项式对应18个系数(由于映射矩阵的存在，多项式系数是与约束量相关的)

        for i in range(3):
            # i = 0时，取出的时x轴的所有约束，即[px0, vx0, ax0; pxt, vxt, axt]
            d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)  # [batch_size, num_dp + num_df, 1]
            coe = (L @ d).squeeze()   # [batch_size, 6] L = A^(-1) * C, p = L * d
            coefficient[:, 6 * i: 6 * (i + 1)] = coe    # 得到三轴的五次多项式系数

        return coefficient  # [batch_size, 18]

    def get_position_from_coeff(self, coe, t):
        # p = c0 + c1 * t + c2 * t^2 + ... + c5 * t^5

        # [batch_size, eval_points, 6], 每个点的时间系数t
        t_power = torch.stack([torch.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        coe_x = coe[:, 0: 6]
        coe_y = coe[:, 6:12]
        coe_z = coe[:, 12:18]
        
        # 得到每个点的位置 [batch_size, eval_points]
        x = torch.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = torch.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = torch.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        pos = torch.stack([x, y, z], dim=-1) # [batch_size, eval_points, 3]
        return pos

    def get_velocity_from_coeff(self, coe, t):
        
        t_power = torch.stack([torch.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4], dim=-1).squeeze(-2)

        coe_x = coe[:, 1:6]
        coe_y = coe[:, 7:12]
        coe_z = coe[:, 13:18]
        # 
        vx = torch.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        vy = torch.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        vz = torch.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        vel = torch.stack([vx, vy, vz], dim=-1)     
        return vel

    def _get_sdf_from_pcd(self, pcd_path: str):
        """
        从PCD生成完整的sdf(初始化使用)
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
        max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
        points = np.asarray(pcd.points)

        print(f"点云范围:")
        print(f"  x: [{min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}]")
        print(f"  y: [{min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}]")
        print(f"  z: [{min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f}]")

        # 得到sdf地图尺寸并将点云转换为栅格
        sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
        voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)
        # 地图尺寸内的有效掩码生成
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
        voxel_indices = voxel_indices[valid_mask]
        # 占用栅格生成
        occupancy = np.zeros(sdf_shape, dtype=np.uint8)
        occupancy[tuple(voxel_indices.T)] = 1

        print(f"占用网格生成完成，占用点数: {np.sum(occupancy)}")
        # 划分自由空间与障碍物空间的掩码
        obstacle_mask = occupancy == 1
        free_mask = occupancy == 0

        print(f"计算距离变换...")
        # 距离符号计算
        dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
        dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size
        dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

        # 得到最终sdf地图并进行类型转换
        sdf_tensor = torch.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2).to(self.device)  # (1, 1, D, H, W)

        sdf_shape_stored = sdf_tensor.shape[-3:][::-1]  # D, H, W -> X, Y, Z

        return sdf_tensor, torch.tensor(min_bound, dtype=torch.float32, device=self.device), torch.tensor(max_bound, dtype=torch.float32, device=self.device), torch.tensor(sdf_shape_stored, dtype=torch.float32, device=self.device)
    
    def get_batch_sdf(self, pos, map_id=None):
        """
        从完整SDF地图中裁剪出轨迹所在的局部SDF
        
        Args:
            pos: (B, N, 3) - 点在世界坐标系下的位置，n个坐标点(轨迹)
            map_id: (B) - 每个 batch 使用哪张 sdf_map，如果为 None 则默认使用第一张地图
            
        Returns:
            sdf_maps: 裁剪后的SDF地图 (B, 1, 1, D, H, W)
            local_origin: 局部地图原点 (B, 3)
            local_shape: 局部地图形状 (B, 3)
        """
        B, N, _ = pos.shape
        device = pos.device
        
        # 如果 map_id 为 None，默认使用第一张地图
        if map_id is None:
            map_id = torch.zeros(B, dtype=torch.long, device=device)
        
        # 获取局部SDF地图
        min_bounds = torch.stack([self.min_bounds[mid] for mid in map_id])  # [B, 3]
        sdf_shapes = torch.stack([self.sdf_shapes[mid] for mid in map_id])  # [B, 3]
        
        # 从轨迹中得到局部地图的边界
        min_pos = pos.amin(dim=1)  # [B, 3]
        max_pos = pos.amax(dim=1)  # [B, 3]
        
        # 世界坐标系转局部地图索引
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices  # [B, 3]
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2  # [B, 3]
        min_indices = centers - max_spans // 2 - 5  # [B, 3]
        max_indices = centers + max_spans // 2 + 5  # [B, 3]
        
        # 裁剪最小值
        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount
        
        # 裁剪最大值
        new_max_indices = torch.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount
        
        # 检查越界
        if (min_indices < 0).any():
            min_underflow = torch.minimum(min_indices, torch.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift
        
        # 裁剪SDF地图
        sdf_maps = torch.stack([
            self.sdf_maps[map_idx][0, :,
                             min_idx[2]:max_idx[2],
                             min_idx[1]:max_idx[1],
                             min_idx[0]:max_idx[0]]
                             for map_idx, min_idx, max_idx in zip(map_id.tolist(), min_indices.tolist(), max_indices.tolist())
                             ])
        
        # 计算局部地图的原点和形状
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        
        return sdf_maps, local_origin, local_shape

    def get_distance_cost(self, pos: torch.Tensor, map_id: int = 0) -> torch.Tensor:
        """
        查询距离（在局部范围内查询，与 safety_loss.py 保持一致）

        Args:
            pos: 查询点张量，形状 (B, N, 3)
            map_id: 使用的地图索引，默认为 0（第一张地图）

        Returns:
            dist: 有符号距离，形状 (B, N)
        """
        B, N, _ = pos.shape
        device = pos.device

        map_id_tensor = torch.full((B,), map_id, dtype=torch.long, device=device)

        # 从局部SDF地图中查询距离
        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id_tensor)

        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size

        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0

        grid_point = grid_point.view(B, 1, 1, N, 3)
        grid_point = torch.clamp(grid_point, min=-0.99, max=0.99)

        dist = F.grid_sample(
            sdf_maps,
            grid_point,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        dist = dist.view(B, N)
        cost = self.cost_function(dist)
        return cost, dist
        
    def cost_function(self, d):
        return torch.exp(-(d - self.d0) / self.r)