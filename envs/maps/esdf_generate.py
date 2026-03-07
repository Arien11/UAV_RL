"""
ESDF (Euclidean Signed Distance Field) 工具模块
提供多种在MuJoCo中计算距离场的方法，支持可微分计算
"""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
import mujoco
from scipy.spatial import KDTree
import warnings
import torch
import torch.nn as nn


class ESDFGenerator:
    """
    ESDF生成器
    1. MuJoCo碰撞检测法(无用)
    2. KDTree法
    3. PyTorch法
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        初始化ESDF生成器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
        """
        self.model = model
        self.data = data
        self._collision_geoms = []
        self._obstacle_points = None
        self._obstacle_points_torch = None  # PyTorch版本的点云
        self._kd_tree = None
        self._extract_collision_geoms()     # 提取碰撞几何体
    
    #------------------------------------- 提取方法 -------------------------------------
    def _extract_collision_geoms(self):
        """提取模型中的碰撞几何体"""
        self._collision_geoms = []
        for geom_id in range(self.model.ngeom):
            geom_type = self.model.geom_type[geom_id]
            geom_pos = self.model.geom_pos[geom_id]
            geom_quat = self.model.geom_quat[geom_id]
            geom_size = self.model.geom_size[geom_id]
            
            # 提取所有几何体，不过滤碰撞组
            self._collision_geoms.append({
                'id': geom_id,
                'type': geom_type,
                'pos': geom_pos.copy(),
                'quat': geom_quat.copy(),
                'size': geom_size.copy(),
            })
        
        print(f"✅ 提取了 {len(self._collision_geoms)} 个碰撞几何体")
    
    def sample_obstacle_points(self, num_points_per_geom: int = 100) -> np.ndarray:
        """
        采样障碍物表面点
        
        Args:
            num_points_per_geom: 每个几何体采样的点数
            
        Returns:
            obstacle_points: 障碍物点云 [N, 3]
        """
        all_points = []
        
        for geom in self._collision_geoms:
            geom_type = geom['type']
            pos = geom['pos']
            size = geom['size']
            
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                # 球体采样
                r = size[0]
                theta = np.random.uniform(0, 2 * np.pi, num_points_per_geom)
                phi = np.random.uniform(0, np.pi, num_points_per_geom)
                x = pos[0] + r * np.sin(phi) * np.cos(theta)
                y = pos[1] + r * np.sin(phi) * np.sin(theta)
                z = pos[2] + r * np.cos(phi)
                points = np.column_stack([x, y, z])
                all_points.append(points)
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                # 盒子采样（表面点）
                hx, hy, hz = size
                for _ in range(num_points_per_geom):
                    # 随机选择一个面
                    face = np.random.randint(6)
                    if face == 0:  # +x
                        px, py, pz = hx, np.random.uniform(-hy, hy), np.random.uniform(-hz, hz)
                    elif face == 1:  # -x
                        px, py, pz = -hx, np.random.uniform(-hy, hy), np.random.uniform(-hz, hz)
                    elif face == 2:  # +y
                        px, py, pz = np.random.uniform(-hx, hx), hy, np.random.uniform(-hz, hz)
                    elif face == 3:  # -y
                        px, py, pz = np.random.uniform(-hx, hx), -hy, np.random.uniform(-hz, hz)
                    elif face == 4:  # +z
                        px, py, pz = np.random.uniform(-hx, hx), np.random.uniform(-hy, hy), hz
                    else:  # -z
                        px, py, pz = np.random.uniform(-hx, hx), np.random.uniform(-hy, hy), -hz
                    all_points.append(pos + np.array([px, py, pz]))
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # 圆柱体采样
                r, h = size[0], size[1]
                for _ in range(num_points_per_geom):
                    theta = np.random.uniform(0, 2 * np.pi)
                    z = np.random.uniform(-h, h)
                    px, py = r * np.cos(theta), r * np.sin(theta)
                    all_points.append(pos + np.array([px, py, z]))
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # 胶囊体采样
                r, h = size[0], size[1]
                for _ in range(num_points_per_geom):
                    if np.random.random() < 0.5:
                        # 圆柱部分
                        theta = np.random.uniform(0, 2 * np.pi)
                        z = np.random.uniform(-h, h)
                        px, py = r * np.cos(theta), r * np.sin(theta)
                    else:
                        # 半球部分
                        theta = np.random.uniform(0, 2 * np.pi)
                        phi = np.random.uniform(0, np.pi / 2)
                        if np.random.random() < 0.5:
                            z = h + r * np.cos(phi)
                        else:
                            z = -h - r * np.cos(phi)
                        px, py = r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta)
                    all_points.append(pos + np.array([px, py, z]))
        
        if all_points:
            self._obstacle_points = np.vstack(all_points)
            self._kd_tree = KDTree(self._obstacle_points)
            # 同时转换为PyTorch张量以便可微分计算
            self._obstacle_points_torch = torch.tensor(
                self._obstacle_points, 
                dtype=torch.float32
            )
            print(f"✅ 采样了 {self._obstacle_points.shape[0]} 个障碍物点")
        
        return self._obstacle_points
    
    #------------------------------------- 距离获取方法 -------------------------------------
    def distance_mujoco_collision(self, point: np.ndarray, body_id: int = -1) -> Tuple[float, np.ndarray]:
        """
        使用MuJoCo碰撞检测计算距离（备用方法，推荐使用KDTree）
        
        Args:
            point: 查询点 [x, y, z]
            body_id: 要排除的刚体ID（-1表示不排除）
            
        Returns:
            (distance, closest_point): 距离和最近点
        """
        warnings.warn("MuJoCo碰撞检测方法暂不推荐使用，请使用KDTree方法", UserWarning)
        # 直接返回一个大值，主要使用KDTree方法
        return 1000.0, point
    
    def distance_kdtree(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        使用KDTree计算到障碍物的距离
        
        Args:
            point: 查询点 [x, y, z]
            
        Returns:
            (distance, closest_point): 距离和最近点
        """
        if self._kd_tree is None:
            self.sample_obstacle_points()
        
        dist, idx = self._kd_tree.query(point, k=1)
        closest_point = self._obstacle_points[idx]
        
        return dist, closest_point
    
    def distance_torch(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用PyTorch计算到障碍物的距离
        
        Args:
            point: 查询点 [x, y, z]，torch.Tensor [3] 或 [B, 3]
            
        Returns:
            (distance, closest_point): 距离和最近点
                distance: torch.Tensor [1] 或 [B]
                closest_point: torch.Tensor [3] 或 [B, 3]
        """
        if self._obstacle_points_torch is None:
            self.sample_obstacle_points()
        
        # 确保点在正确的设备上
        device = point.device
        obstacle_points = self._obstacle_points_torch.to(device)
        
        # 计算所有距离
        if point.dim() == 1:
            # 单个点: [3]
            point = point.unsqueeze(0)  # [1, 3]
        
        # 计算欧氏距离: [B, N]
        diff = point.unsqueeze(1) - obstacle_points.unsqueeze(0)  # [B, N, 3]
        dist_sq = torch.sum(diff**2, dim=-1)  # [B, N]
        dist = torch.sqrt(dist_sq)  # [B, N]
        
        # 找到最近点
        min_dist, min_idx = torch.min(dist, dim=-1)  # [B]
        closest_point = obstacle_points[min_idx]  # [B, 3]
        
        # 如果输入是单个点，压缩维度
        if point.shape[0] == 1:
            min_dist = min_dist.squeeze(0)
            closest_point = closest_point.squeeze(0)
        
        return min_dist, closest_point
    
    def get_distance(self, point: Union[np.ndarray, torch.Tensor], 
                    method: str = 'kdtree') -> Union[float, torch.Tensor]:
        """
        获取单个点的距离
        
        Args:
            point: 查询点 [x, y, z]，numpy数组或torch张量
            method: 计算方法 ('kdtree', 'mujoco', 'torch')
            
        Returns:
            distance: 到最近障碍物的距离
        """
        if method == 'torch':
            if isinstance(point, np.ndarray):
                point = torch.tensor(point, dtype=torch.float32)
            dist, _ = self.distance_torch(point)
            return dist
        elif method == 'kdtree':
            if isinstance(point, torch.Tensor):
                point = point.detach().cpu().numpy()
            dist, _ = self.distance_kdtree(point)
            return dist
        elif method == 'mujoco':
            if isinstance(point, torch.Tensor):
                point = point.detach().cpu().numpy()
            dist, _ = self.distance_mujoco_collision(point)
            return dist
        else:
            raise ValueError(f"未知方法: {method}")

    #------------------------------------- �算方法 -------------------------------------
    def compute_esdf_grid(self, 
                          bounds: Tuple[np.ndarray, np.ndarray],
                          resolution: float = 0.1,
                          method: str = 'kdtree') -> Tuple[np.ndarray, np.ndarray]:
        """
        计算网格ESDF
        
        Args:
            bounds: 边界 [(xmin, ymin, zmin), (xmax, ymax, zmax)]
            resolution: 网格分辨率
            method: 计算方法 ('kdtree' 或 'mujoco')
            
        Returns:
            (esdf, grid_points): ESDF数组和对应的网格点
        """
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]
        
        # 创建网格
        x = np.arange(xmin, xmax, resolution)
        y = np.arange(ymin, ymax, resolution)
        z = np.arange(zmin, zmax, resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # 计算ESDF
        esdf = np.zeros(grid_points.shape[0])
        
        if method == 'kdtree':
            if self._kd_tree is None:
                self.sample_obstacle_points()
            
            dists, _ = self._kd_tree.query(grid_points, k=1)
            esdf = dists
        
        elif method == 'mujoco':
            for i, point in enumerate(grid_points):
                dist, _ = self.distance_mujoco_collision(point)
                esdf[i] = dist
        
        # 重塑为3D
        esdf = esdf.reshape(xx.shape)
        
        return esdf, grid_points
    



class DifferentiableESDF(nn.Module):
    """
    可微分的ESDF模块，用于PyTorch自动微分
    
    使用方法：
        esdf_module = DifferentiableESDF(generator)
        distance = esdf_module(query_point)  # 支持梯度
    """
    
    def __init__(self, generator: ESDFGenerator):
        """
        初始化可微分ESDF模块
        
        Args:
            generator: ESDFGenerator实例（已采样好障碍物点）
        """
        super().__init__()
        self.generator = generator
        
        # 确保障碍物点已采样
        if generator._obstacle_points_torch is None:
            generator.sample_obstacle_points()
        
        # 注册障碍物点为缓冲区（不参与训练，但会保存到模型）
        self.register_buffer(
            'obstacle_points', 
            generator._obstacle_points_torch.clone()
        )
    
    def forward(self, query_point: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算查询点到障碍物的距离
        
        Args:
            query_point: 查询点 [3] 或 [B, 3]
            
        Returns:
            distance: 距离 [1] 或 [B]
        """
        # 计算所有距离
        if query_point.dim() == 1:
            query_point = query_point.unsqueeze(0)  # [1, 3]
        
        # 计算欧氏距离:d = ||x - p||= sqrt(∑(xi - pi)^2)，[B, N] 
        diff = query_point.unsqueeze(1) - self.obstacle_points.unsqueeze(0)  # [B, N, 3]
        dist_sq = torch.sum(diff**2, dim=-1)  # ∑(xi - pi)^2  [B, N]
        dist = torch.sqrt(dist_sq)  # [B, N]     

        # 找到最近距离
        min_dist, _ = torch.min(dist, dim=-1)  # [B]
        
        # 如果输入是单个点，压缩维度
        if query_point.shape[0] == 1:
            min_dist = min_dist.squeeze(0)
        
        return min_dist


def create_esdf_from_mujoco(model: mujoco.MjModel, 
                            data: mujoco.MjData,
                            bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            resolution: float = 0.1) -> Dict[str, Any]:
    """
    从MuJoCo模型创建ESDF
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        bounds: 边界 [(xmin, ymin, zmin), (xmax, ymax, zmax)]
        resolution: 网格分辨率
        
    Returns:
        esdf_data: 包含ESDF数据的字典
    """
    generator = ESDFGenerator(model, data)
    
    # 如果没有指定边界，使用模型范围
    if bounds is None:
        extent = model.stat.extent
        center = np.array([0, 0, 0])  # 可以根据需要调整
        bounds = (center - extent, center + extent)
    
    print(f"📊 计算ESDF，边界: {bounds}")
    print(f"📊 分辨率: {resolution}")
    
    # 采样障碍物点
    generator.sample_obstacle_points(num_points_per_geom=200)
    
    # 计算ESDF
    esdf, grid_points = generator.compute_esdf_grid(bounds, resolution, method='kdtree')
    
    esdf_data = {
        'esdf': esdf,
        'grid_points': grid_points,
        'bounds': bounds,
        'resolution': resolution,
        'generator': generator,
        'differentiable': DifferentiableESDF(generator)
    }
    
    print(f"✅ ESDF计算完成")
    print(f"   ESDF形状: {esdf.shape}")
    print(f"   距离范围: [{esdf.min():.3f}, {esdf.max():.3f}]")
    
    return esdf_data


if __name__ == "__main__":
    print("=" * 70)
    print("ESDF工具模块")
    print("=" * 70)
    print("\n使用方法:")
    print("1. 创建ESDFGenerator实例")
    print("2. 采样障碍物点")
    print("3. 计算距离或ESDF网格")
    print("\n可微分使用（PyTorch）:")
    print("""
    # 从MuJoCo创建ESDF
    esdf_data = create_esdf_from_mujoco(model, data)
    
    # 获取可微分模块
    esdf_module = esdf_data['differentiable']
    
    # 查询距离（支持梯度）
    query_point = torch.tensor([x, y, z], requires_grad=True)
    distance = esdf_module(query_point)
    
    # 反向传播
    distance.backward()
    gradient = query_point.grad
    """)
