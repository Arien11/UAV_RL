"""
ESDF (Euclidean Signed Distance Field) 工具模块
使用距离变换（distance_transform_edt）实现精确的有符号距离场
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
import mujoco
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R


class ESDFGenerator:
    """
    ESDF生成器

    使用距离变换法计算精确的有符号距离场
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
        self._esdf_grid = None
        self._esdf_bounds = None
        self._esdf_resolution = None
        self._extract_collision_geoms()

    def _extract_collision_geoms(self):
        """提取模型中的碰撞几何体 - 只保留障碍物（box），过滤掉无人机自身零件"""
        self._collision_geoms = []

        # 先更新运动学，确保 data.geom_xpos 是最新的
        mujoco.mj_forward(self.model, self.data)

        for geom_id in range(self.model.ngeom):
            geom_type = self.model.geom_type[geom_id]

            # 只保留 box 类型的几何体（真正的障碍物）
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                # 使用 data.geom_xpos 获取世界坐标系下的位置
                # quat 用 model.geom_quat（因为 data 里没有 geom_xquat）
                self._collision_geoms.append({
                    'id': geom_id,
                    'type': geom_type,
                    'pos': self.data.geom_xpos[geom_id].copy(),
                    'quat': self.model.geom_quat[geom_id].copy(),
                    'size': self.model.geom_size[geom_id].copy(),
                })
        print(f"✅ 提取了 {len(self._collision_geoms)} 个碰撞几何体（仅 box）")

    def compute_esdf_grid(self,
                          bounds: Tuple[np.ndarray, np.ndarray],
                          resolution: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算网格ESDF（有符号距离场）- 使用距离变换

        Args:
            bounds: 边界 [(xmin, ymin, zmin), (xmax, ymax, zmax)]
            resolution: 网格分辨率

        Returns:
            (esdf, grid_points): ESDF数组和对应的网格点
        """
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]

        x = np.arange(xmin, xmax, resolution)
        y = np.arange(ymin, ymax, resolution)
        z = np.arange(zmin, zmax, resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        print(f"生成占用网格...")
        print(f"网格点数: {grid_points.shape[0]}")
        occupancy = np.zeros(xx.shape, dtype=np.uint8)

        for geom in self._collision_geoms:
            geom_type = geom['type']
            pos = geom['pos']
            quat = geom['quat']
            size = geom['size']

            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = size[0]
                dx = xx - pos[0]
                dy = yy - pos[1]
                dz = zz - pos[2]
                dist_sq = dx**2 + dy**2 + dz**2
                mask = dist_sq <= r**2
                occupancy[mask] = 1

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                rot = R.from_quat(quat)
                rot_inv = rot.inv()
                points_local = rot_inv.apply(grid_points - pos)
                points_local_reshaped = points_local.reshape(xx.shape + (3,))
                hx, hy, hz = size
                mask = (np.abs(points_local_reshaped[..., 0]) <= hx) & \
                       (np.abs(points_local_reshaped[..., 1]) <= hy) & \
                       (np.abs(points_local_reshaped[..., 2]) <= hz)
                occupancy[mask] = 1

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                rot = R.from_quat(quat)
                rot_inv = rot.inv()
                points_local = rot_inv.apply(grid_points - pos)
                points_local_reshaped = points_local.reshape(xx.shape + (3,))
                r, h = size[0], size[1]
                radial_dist_sq = points_local_reshaped[..., 0]**2 + points_local_reshaped[..., 1]**2
                mask = (radial_dist_sq <= r**2) & \
                       (np.abs(points_local_reshaped[..., 2]) <= h)
                occupancy[mask] = 1

            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                rot = R.from_quat(quat)
                rot_inv = rot.inv()
                points_local = rot_inv.apply(grid_points - pos)
                points_local_reshaped = points_local.reshape(xx.shape + (3,))
                r, h = size[0], size[1]
                z_clamped = np.clip(points_local_reshaped[..., 2], -h, h)
                closest_points = np.zeros_like(points_local_reshaped)
                closest_points[..., 2] = z_clamped
                dist = np.linalg.norm(points_local_reshaped - closest_points, axis=-1)
                mask = dist <= r
                occupancy[mask] = 1

        print(f"占用网格生成完成，占用点数: {np.sum(occupancy)}")

        obstacle_mask = occupancy == 1
        free_mask = occupancy == 0

        print(f"计算距离变换...")
        dist_to_obstacle = distance_transform_edt(free_mask) * resolution
        dist_inside_obstacle = distance_transform_edt(obstacle_mask) * resolution

        esdf = dist_to_obstacle.copy()
        esdf[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

        self._esdf_grid = esdf
        self._esdf_bounds = bounds
        self._esdf_resolution = resolution

        # 返回ESDF网格和对应的网格点
        return esdf, grid_points 

    def query_esdf_grid(self, point: np.ndarray, interpolate: bool = True) -> float:
        """
        从预计算的ESDF网格中查询有符号距离

        Args:
            point: 查询点 [x, y, z]
            interpolate: 是否使用三线性插值

        Returns:
            signed_distance: 有符号距离
        """
        if self._esdf_grid is None:
            raise ValueError("ESDF网格未计算，请先调用compute_esdf_grid()")

        xmin, ymin, zmin = self._esdf_bounds[0]
        res = self._esdf_resolution

        x_idx = (point[0] - xmin) / res
        y_idx = (point[1] - ymin) / res
        z_idx = (point[2] - zmin) / res

        if not interpolate:
            x_idx = int(np.clip(x_idx, 0, self._esdf_grid.shape[0] - 1))
            y_idx = int(np.clip(y_idx, 0, self._esdf_grid.shape[1] - 1))
            z_idx = int(np.clip(z_idx, 0, self._esdf_grid.shape[2] - 1))
            return self._esdf_grid[x_idx, y_idx, z_idx]

        x0 = int(np.floor(x_idx))
        y0 = int(np.floor(y_idx))
        z0 = int(np.floor(z_idx))
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        x0_clamped = np.clip(x0, 0, self._esdf_grid.shape[0] - 1)
        y0_clamped = np.clip(y0, 0, self._esdf_grid.shape[1] - 1)
        z0_clamped = np.clip(z0, 0, self._esdf_grid.shape[2] - 1)
        x1_clamped = np.clip(x1, 0, self._esdf_grid.shape[0] - 1)
        y1_clamped = np.clip(y1, 0, self._esdf_grid.shape[1] - 1)
        z1_clamped = np.clip(z1, 0, self._esdf_grid.shape[2] - 1)

        xd = np.clip(x_idx - x0, 0.0, 1.0)
        yd = np.clip(y_idx - y0, 0.0, 1.0)
        zd = np.clip(z_idx - z0, 0.0, 1.0)

        c000 = self._esdf_grid[x0_clamped, y0_clamped, z0_clamped]
        c001 = self._esdf_grid[x0_clamped, y0_clamped, z1_clamped]
        c010 = self._esdf_grid[x0_clamped, y1_clamped, z0_clamped]
        c011 = self._esdf_grid[x0_clamped, y1_clamped, z1_clamped]
        c100 = self._esdf_grid[x1_clamped, y0_clamped, z0_clamped]
        c101 = self._esdf_grid[x1_clamped, y0_clamped, z1_clamped]
        c110 = self._esdf_grid[x1_clamped, y1_clamped, z0_clamped]
        c111 = self._esdf_grid[x1_clamped, y1_clamped, z1_clamped]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd

        return c

    def get_distance(self, point: np.ndarray) -> float:
        """
        获取单个点的有符号距离

        Args:
            point: 查询点 [x, y, z]

        Returns:
            distance: 到最近障碍物的有符号距离
        """
        return self.query_esdf_grid(point, interpolate=True)

    def print_geoms(self):
        """打印提取的几何体信息（调试用）"""
        print("\n提取的碰撞几何体列表:")
        for i, geom in enumerate(self._collision_geoms):
            geom_type = geom['type']
            # 获取类型名称
            type_id = int(geom_type)
            type_name = mujoco.mju_type2Str(type_id) if hasattr(mujoco, 'mju_type2Str') else str(type_id)
            print(f"  {i:2d}: {type_name:10} pos={geom['pos']}, size={geom['size']}")
        print()


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

    if bounds is None:
        extent = model.stat.extent
        center = np.array([0, 0, 0])
        bounds = (center - extent, center + extent)

    # print(f"📊 计算ESDF，边界: {bounds}")
    # print(f"📊 分辨率: {resolution}")

    esdf, grid_points = generator.compute_esdf_grid(bounds, resolution)

    esdf_data = {
        'esdf': esdf,
        'grid_points': grid_points,
        'bounds': bounds,
        'resolution': resolution,
        'generator': generator,
    }

    # print(f"✅ ESDF计算完成")
    # print(f"   ESDF形状: {esdf.shape}")
    # print(f"   距离范围: [{esdf.min():.3f}, {esdf.max():.3f}]")

    return esdf_data

