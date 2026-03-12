"""
ESDF (Euclidean Signed Distance Field) PyTorch 版本
支持批量查询和反向传播，接口兼容 safety_loss.py
完全照搬 safety_loss.py 的高效架构：预计算完整 ESDF + 局部裁剪查询
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import mujoco
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESDFTorch(nn.Module):
    """
    PyTorch 版本的 ESDF 生成器
    支持批量查询和反向传播
    架构：预计算完整 ESDF + 局部裁剪查询
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 voxel_size: float = 0.1,
                 map_expand_min: Optional[np.ndarray] = None,
                 map_expand_max: Optional[np.ndarray] = None):
        """
        初始化 ESDF 生成器

        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
            voxel_size: 体素大小
            map_expand_min: 地图边界扩张（最小值方向）
            map_expand_max: 地图边界扩张（最大值方向）
        """
        super().__init__()
        self.model = model
        self.data = data
        self.voxel_size = voxel_size
        self.map_expand_min = map_expand_min if map_expand_min is not None else np.array([0.0, 0.0, 0.0])
        self.map_expand_max = map_expand_max if map_expand_max is not None else np.array([0.0, 0.0, 0.0])

        self._collision_geoms = []
        self._full_esdf_tensor = None
        self._min_bound = None
        self._max_bound = None
        self._esdf_shape = None

        self._extract_collision_geoms()
        self._build_complete_esdf()

    def _extract_collision_geoms(self):
        """提取模型中的碰撞几何体 - 只保留障碍物（box），过滤掉无人机自身零件"""
        self._collision_geoms = []

        # 先更新运动学，确保 data.geom_xpos 是最新的
        mujoco.mj_forward(self.model, self.data)

        for geom_id in range(self.model.ngeom):
            geom_type = self.model.geom_type[geom_id]

            # 只保留 box 类型的几何体（真正的障碍物）
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self._collision_geoms.append({
                    'id': geom_id,
                    'type': geom_type,
                    'pos': self.data.geom_xpos[geom_id].copy(),
                    'quat': self.model.geom_quat[geom_id].copy(),
                    'size': self.model.geom_size[geom_id].copy(),
                })
        # print(f"✅ 提取了 {len(self._collision_geoms)} 个碰撞几何体（仅 box）")

    def _build_complete_esdf(self):
        """
        预计算完整 ESDF（从所有障碍物自动计算边界）
        完全照搬 safety_loss.py 中 get_sdf_from_ply 的逻辑
        """
        # 从所有障碍物自动计算边界
        all_pos = np.array([g['pos'] for g in self._collision_geoms])
        all_sizes = np.array([g['size'] for g in self._collision_geoms])

        min_bound = all_pos.min(axis=0) - all_sizes.max(axis=0) - self.map_expand_min
        max_bound = all_pos.max(axis=0) + all_sizes.max(axis=0) + self.map_expand_max

        # 生成完整网格
        x = np.arange(min_bound[0], max_bound[0], self.voxel_size)
        y = np.arange(min_bound[1], max_bound[1], self.voxel_size)
        z = np.arange(min_bound[2], max_bound[2], self.voxel_size)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # print(f"生成完整占用网格...")
        # print(f"  网格形状: {xx.shape}")
        # print(f"  网格点数: {grid_points.shape[0]}")
        occupancy = np.zeros(xx.shape, dtype=np.uint8)

        for geom in self._collision_geoms:
            pos = geom['pos']
            quat = geom['quat']
            size = geom['size']

            rot = R.from_quat(quat)
            rot_inv = rot.inv()
            points_local = rot_inv.apply(grid_points - pos)
            points_local_reshaped = points_local.reshape(xx.shape + (3,))
            hx, hy, hz = size
            mask = (np.abs(points_local_reshaped[..., 0]) <= hx) & \
                   (np.abs(points_local_reshaped[..., 1]) <= hy) & \
                   (np.abs(points_local_reshaped[..., 2]) <= hz)
            occupancy[mask] = 1

        # print(f"占用网格生成完成，占用点数: {np.sum(occupancy)}")

        obstacle_mask = occupancy == 1
        free_mask = occupancy == 0

        # print(f"计算距离变换...")
        dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
        dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

        esdf = dist_to_obstacle.copy()
        esdf[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

        # 转换为 PyTorch 张量
        # 注意：不进行 permute，保持 (1, 1, X, Y, Z) 顺序（用户验证过不 permute 才对）
        esdf_tensor = torch.from_numpy(esdf).float().unsqueeze(0).unsqueeze(0)

        self._full_esdf_tensor = esdf_tensor
        self._min_bound = torch.tensor(min_bound, dtype=torch.float32)
        self._max_bound = torch.tensor(max_bound, dtype=torch.float32)
        self._esdf_shape = torch.tensor(esdf.shape, dtype=torch.float32)

        # print(f"✅ 完整 ESDF 预计算完成")
        # print(f"   ESDF 形状: {esdf.shape}")
        # print(f"   Tensor 形状: {esdf_tensor.shape}")
        # print(f"   距离范围: [{esdf.min():.3f}, {esdf.max():.3f}]")
        # print(f"   边界: min={min_bound}, max={max_bound}")

    def get_local_esdf(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据查询点位置裁剪局部 ESDF 区域
        完全照搬 safety_loss.py.get_batch_sdf() 的逻辑

        Args:
            pos: 查询点张量，形状 (B, N, 3)

        Returns:
            sdf_maps: 裁剪后的局部 ESDF，形状 (B, 1, D_local, H_local, W_local)
            local_origin: 局部区域原点，形状 (B, 3)
            local_shape: 局部区域形状，形状 (B, 3) - (X, Y, Z) 顺序
        """
        B, N, _ = pos.shape
        device = pos.device

        min_bound = self._min_bound.to(device)
        esdf_shape = self._esdf_shape.to(device)

        # 计算查询点的 min 和 max pos
        min_pos = pos.amin(dim=1)  # [B, 3]
        max_pos = pos.amax(dim=1)  # [B, 3]

        # 转换为 voxel 索引
        min_indices = ((min_pos - min_bound.unsqueeze(0)) / self.voxel_size).int()
        max_indices = ((max_pos - min_bound.unsqueeze(0)) / self.voxel_size).int()

        # 计算 span 并扩展 +5 个 voxel 边距
        spans = max_indices - min_indices  # [B, 3]
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2  # [B, 3]
        min_indices = centers - max_spans // 2 - 5  # [B, 3]
        max_indices = centers + max_spans // 2 + 5  # [B, 3]

        # Crop minimum value
        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        # Crop maximum value
        new_max_indices = torch.minimum(max_indices, esdf_shape.int().unsqueeze(0))
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        # Check for out-of-bounds indices
        if (min_indices < 0).any():
            min_underflow = torch.minimum(min_indices, torch.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        # 裁剪局部 ESDF
        # 注意：索引顺序为 Z, Y, X（因为 tensor 形状是 (1, 1, X, Y, Z)，不！等一下...
        # 实际上我们没有 permute，所以 tensor 形状是 (1, 1, X, Y, Z)
        # 所以索引顺序应该是 X, Y, Z？
        # 不对，让我们照搬 safety_loss.py 的方式
        # safety_loss.py 用 permute 后是 (1, 1, Z, Y, X)，所以索引是 [z1:z2, y1:y2, x1:x2]
        # 但我们没有 permute，是 (1, 1, X, Y, Z)，所以索引应该是 [x1:x2, y1:y2, z1:z2]
        # 但用户验证过不 permute 才对，所以让我们调整索引顺序

        sdf_maps = []
        for b in range(B):
            min_idx = min_indices[b]
            max_idx = max_indices[b]
            # tensor 形状 (1, 1, X, Y, Z)，所以切片是 [:, :, x1:x2, y1:y2, z1:z2]
            local_sdf = self._full_esdf_tensor[0, :,
                                                 min_idx[0]:max_idx[0],
                                                 min_idx[1]:max_idx[1],
                                                 min_idx[2]:max_idx[2]]
            sdf_maps.append(local_sdf.unsqueeze(0))

        sdf_maps = torch.cat(sdf_maps, dim=0)
        local_origin = min_indices.float() * self.voxel_size + min_bound.unsqueeze(0)
        local_shape = max_indices - min_indices

        return sdf_maps, local_origin, local_shape

    def query_distance(self, pos: torch.Tensor) -> torch.Tensor:
        """
        批量查询有符号距离（使用局部裁剪优化）

        Args:
            pos: 查询点张量，形状 (B, N, 3) - (batch_size, num_points, 3)

        Returns:
            dist: 有符号距离，形状 (B, N)
        """
        if self._full_esdf_tensor is None:
            raise ValueError("完整 ESDF 未预计算")

        B, N, _ = pos.shape
        device = pos.device

        # 获取局部 ESDF
        sdf_maps, local_origin, local_shape = self.get_local_esdf(pos)

        # 将 pos 转为局部 voxel 坐标
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size  # (B, N, 3) - X,Y,Z 顺序

        # 归一化 grid 到 [-1, 1]
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0  # (B, N, 3) - X,Y,Z 顺序

        # 注意：因为我们没有 permute，tensor 形状是 (B, 1, X, Y, Z)
        # grid_sample 期望的坐标顺序是 (D, H, W) 对应 tensor 最后三个维度 (X, Y, Z)
        # 所以 grid_point 不需要变换顺序！
        grid_point = grid_point.view(B, 1, 1, N, 3)
        grid_point = torch.clamp(grid_point, min=-0.99, max=0.99)

        # 使用 F.grid_sample 进行三线性插值
        dist_query = F.grid_sample(
            sdf_maps,
            grid_point,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B, 1, 1, 1, N)

        dist_query = dist_query.view(B, N)

        return dist_query

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        PyTorch Module 前向传播接口

        Args:
            pos: 查询点张量，形状 (B, N, 3)

        Returns:
            dist: 有符号距离，形状 (B, N)
        """
        return self.query_distance(pos)


def create_esdf_torch_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    voxel_size: float = 0.1,
    map_expand_min: Optional[np.ndarray] = None,
    map_expand_max: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    从 MuJoCo 模型创建 PyTorch 版本的 ESDF

    Args:
        model: MuJoCo 模型
        data: MuJoCo 数据
        voxel_size: 体素大小
        map_expand_min: 地图边界扩张（最小值方向）
        map_expand_max: 地图边界扩张（最大值方向）

    Returns:
        esdf_data: 包含 ESDF 数据的字典
    """
    esdf_torch = ESDFTorch(
        model, data,
        voxel_size=voxel_size,
        map_expand_min=map_expand_min,
        map_expand_max=map_expand_max
    )

    esdf_data = {
        'esdf_torch': esdf_torch,
        'full_esdf_tensor': esdf_torch._full_esdf_tensor,
        'min_bound': esdf_torch._min_bound,
        'max_bound': esdf_torch._max_bound,
        'voxel_size': voxel_size,
    }

    return esdf_data
