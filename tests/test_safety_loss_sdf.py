"""
测试 safety_loss.py 的 SDF 功能
从 seed_6.pcd 点云生成 SDF 并查询
"""
import os
import sys
import numpy as np
import torch 
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt


class SDFTester:
    """
    简化版 SDF 测试类
    只保留从 PLY 生成 SDF 和查询的功能
    """

    def __init__(self,
                 pcd_path: str,
                 voxel_size: float = 0.2,
                 map_expand_min: np.ndarray = None,
                 map_expand_max: np.ndarray = None,
                 device: str = 'cpu'):
        """
        初始化 SDF 测试器

        Args:
            pcd_patorch: PCD 点云文件路径
            voxel_size: 体素大小
            map_expand_min: 地图边界扩张（最小值方向）
            map_expand_max: 地图边界扩张（最大值方向）
            device: 设备
        """
        self.voxel_size = voxel_size
        self.map_expand_min = map_expand_min if map_expand_min is not None else np.array([0.0, 0.0, 0.0])
        self.map_expand_max = map_expand_max if map_expand_max is not None else np.array([0.0, 0.0, 0.0])
        self.device = device
        # d0 ：安全距离阈值 
        # 当 d > d0 时， (d - d0) 为正， exp(-正数) 趋近于 0，表示安全
        # 当 d < d0 时， (d - d0) 为负， exp(-负数) 指数增长，表示危险
        self.d0 = 1.2       # 安全距离阈值
        self.r = 0.6        # 敏感度参数, 用于控制 exp(-(d-d0)/r) 的增长速度    
        # 从 PCD 生成 SDF
        print(f"加载点云: {pcd_path}")
        sdf_tensor, min_bound, max_bound, sdf_shape = self._get_sdf_from_pcd(pcd_path)

        # 存储为列表形式，支持多地图扩展接口
        self.sdf_maps = [sdf_tensor]
        self.min_bounds = [min_bound]
        self.max_bounds = [max_bound]
        self.sdf_shapes = [sdf_shape]
        self.num_maps = 1

        print(f"✅ SDF 生成完成")
        print(f"   SDF 形状: {sdf_shape}")
        print(f"   Tensor 形状: {sdf_tensor.shape}")

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

    def query_distance(self, pos: torch.Tensor, map_id: int = 0) -> torch.Tensor:
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

        dist_query = F.grid_sample(
            sdf_maps,
            grid_point,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        dist_query = dist_query.view(B, N)
        cost = self.cost_function(dist_query)
        return cost, dist_query
        
    def cost_function(self, d):
        return torch.exp(-(d - self.d0) / self.r)

def test_sdf_functionality(tester, device='cpu'):
    """
    测试 SDF 功能的完整函数，可在 test_mujoco.py 中调用
    
    Args:
        tester: SDFTester 实例
        device: 设备类型
        
    Returns:
        test_results: 测试结果字典
    """
    print("\n" + "=" * 60)
    print("测试 SDF 功能")
    print("=" * 60)
    
    test_results = {}
    
    # 测试 1: 单点查询
    print("\n测试 1: 单点查询")
    test_pos1 = torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    dist1 = tester.query_distance(test_pos1)
    print(f"  测试点: {test_pos1[0, 0].cpu().numpy()}")
    print(f"  距离: {dist1[0, 0].item():.6f}")
    test_results['single_point'] = dist1[0, 0].item()
    
    # 测试 2: 批量查询
    print("\n测试 2: 批量查询")
    test_pos_batch = torch.tensor([
        [[0.0, 0.0, 0.5], [1.0, 0.0, 1.0], [2.0, 0.0, 0.5]]
    ], dtype=torch.float32, device=device)
    dist_batch = tester.query_distance(test_pos_batch)
    for i in range(3):
        print(f"  点 {i+1}: {test_pos_batch[0, i].cpu().numpy()}, 距离: {dist_batch[0, i].item():.6f}")
    test_results['batch_query'] = dist_batch[0, :].cpu().numpy()
    
    # 测试 3: get_batch_sdf 方法（单地图）
    print("\n测试 3: get_batch_sdf 方法（单地图）")
    test_pos_sdf = torch.tensor([
        [[0.5, 0.5, 0.8], [1.0, 1.0, 1.2], [1.5, 1.5, 1.5]]
    ], dtype=torch.float32, device=device)
    sdf_maps, local_origin, local_shape = tester.get_batch_sdf(test_pos_sdf)
    print(f"  输入点形状: {test_pos_sdf.shape}")
    print(f"  裁剪后 SDF 形状: {sdf_maps.shape}")
    print(f"  局部原点: {local_origin[0].cpu().numpy()}")
    print(f"  局部形状: {local_shape[0].cpu().numpy()}")
    test_results['batch_sdf'] = {
        'sdf_shape': sdf_maps.shape,
        'local_origin': local_origin[0].cpu().numpy(),
        'local_shape': local_shape[0].cpu().numpy()
    }
    
    # 测试 4: get_batch_sdf 方法（显式指定 map_id）
    print("\n测试 4: get_batch_sdf 方法（显式指定 map_id=0）")
    map_id = torch.zeros(1, dtype=torch.long, device=device)
    sdf_maps2, local_origin2, local_shape2 = tester.get_batch_sdf(test_pos_sdf, map_id)
    print(f"  裁剪后 SDF 形状: {sdf_maps2.shape}")
    print(f"  局部原点: {local_origin2[0].cpu().numpy()}")
    print(f"  局部形状: {local_shape2[0].cpu().numpy()}")
    test_results['batch_sdf_witorch_map_id'] = {
        'sdf_shape': sdf_maps2.shape,
        'local_origin': local_origin2[0].cpu().numpy(),
        'local_shape': local_shape2[0].cpu().numpy()
    }
    
    # 测试 5: 多 batch 测试
    print("\n测试 5: 多 batch 测试")
    test_pos_multi = torch.tensor([
        [[0.5, 0.5, 0.8], [1.0, 1.0, 1.2]],
        [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]]
    ], dtype=torch.float32, device=device)
    dist_multi = tester.query_distance(test_pos_multi)
    print(f"  输入形状: {test_pos_multi.shape}")
    print(f"  输出形状: {dist_multi.shape}")
    for b in range(2):
        for n in range(2):
            print(f"  Batch {b}, Point {n}: {test_pos_multi[b, n].cpu().numpy()}, 距离: {dist_multi[b, n].item():.6f}")
    test_results['multi_batch'] = dist_multi.cpu().numpy()
    
    # 测试 6: 边界测试
    print("\n测试 6: 边界测试（地图边界附近）")
    min_bound = tester.min_bounds[0].cpu().numpy()
    max_bound = tester.max_bounds[0].cpu().numpy()
    test_pos_boundary = torch.tensor([[
        [min_bound[0] + 0.1, min_bound[1] + 0.1, min_bound[2] + 0.1],
        [max_bound[0] - 0.1, max_bound[1] - 0.1, max_bound[2] - 0.1]
    ]], dtype=torch.float32, device=device)
    dist_boundary = tester.query_distance(test_pos_boundary)
    print(f"  地图最小边界: {min_bound}")
    print(f"  地图最大边界: {max_bound}")
    print(f"  测试点 1（靠近最小边界）: {test_pos_boundary[0, 0].cpu().numpy()}, 距离: {dist_boundary[0, 0].item():.6f}")
    print(f"  测试点 2（靠近最大边界）: {test_pos_boundary[0, 1].cpu().numpy()}, 距离: {dist_boundary[0, 1].item():.6f}")
    test_results['boundary_test'] = dist_boundary[0, :].cpu().numpy()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    
    return test_results

def main():
    """测试脚本主函数"""
    # 配置
    pcd_path = "envs/crazyfile/seed_6.pcd"
    voxel_size = 0.2
    map_expand_min = np.array([0, 0, 0])
    map_expand_max = np.array([0, 0, 0])
    device = 'cpu'

    # 创建 SDF 测试器
    print("=" * 60)
    print("测试 safety_loss.py 的 SDF 功能")
    print("=" * 60)
    tester = SDFTester(
        pcd_path=pcd_path,
        voxel_size=voxel_size,
        map_expand_min=map_expand_min,
        map_expand_max=map_expand_max,
        device=device
    )

    # 运行完整测试
    # test_results = test_sdf_functionality(tester, device)
    
    # 运行方向因子测试
    direction_results = test_direction_factor(tester, device)
    
    # 运行梯度传播测试
    gradient_results = test_gradient_propagation(tester, device)
    
    # 打印测试结果摘要
    print("\n测试结果摘要:")
    print(f"  单点查询距离: {test_results['single_point']:.6f}")
    print(f"  批量查询距离: {test_results['batch_query']}")
    print(f"  裁剪 SDF 形状: {test_results['batch_sdf']['sdf_shape']}")
    print(f"  多 batch 查询形状: {test_results['multi_batch'].shape}")
    print(f"\n方向因子测试结果:")
    print(f"  朝向障碍物 (危险): {direction_results['approaching']:.6f}")
    print(f"  远离障碍物 (安全): {direction_results['receding']:.6f}")
    print(f"  垂直运动: {direction_results['perpendicular']:.6f}")
    print(f"  批量测试: {direction_results['batch']}")
    print(f"\n梯度传播测试结果:")
    print(f"  梯度存在: {'✅' if gradient_results.get('gradient_exists', False) else '❌'}")
    print(f"  梯度范数: {gradient_results.get('gradient_norm', 0):.6f}")
    print(f"  梯度与相对位置夹角: {gradient_results.get('gradient_angle', 0):.2f}°")
    print(f"  批量梯度范数: {gradient_results.get('batch_gradient_norm', 0):.6f}")


if __name__ == '__main__':
    main()
