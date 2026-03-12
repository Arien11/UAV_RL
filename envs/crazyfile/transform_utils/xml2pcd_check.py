"""
check_pointcloud_alignment.py

可视化对比 MuJoCo box 几何体与采样点云的对齐情况。
用法：
    python check_pointcloud_alignment.py <xml_path> [--density DENSITY]
"""

import numpy as np
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import argparse


def sample_box_surface(box_pos: np.ndarray,
                       box_quat: np.ndarray,   # 四元数 [x, y, z, w] (scipy 格式)
                       box_half_size: np.ndarray,
                       sample_density: float = 0.05) -> np.ndarray:
    """采样单个 box 表面的点云（修正版）"""
    hx, hy, hz = box_half_size

    # 使用 linspace 确保边界被包含
    x = np.linspace(-hx, hx, max(1, int(2 * hx / sample_density) + 1))
    y = np.linspace(-hy, hy, max(1, int(2 * hy / sample_density) + 1))
    z = np.linspace(-hz, hz, max(1, int(2 * hz / sample_density) + 1))

    # 前后面 (y 方向)
    xx, zz = np.meshgrid(x, z)
    yy_pos = np.ones_like(xx) * hy
    yy_neg = np.ones_like(xx) * (-hy)
    face1 = np.stack([xx.ravel(), yy_pos.ravel(), zz.ravel()], axis=1)
    face2 = np.stack([xx.ravel(), yy_neg.ravel(), zz.ravel()], axis=1)

    # 左右面 (x 方向)
    yy, zz = np.meshgrid(y, z)
    xx_pos = np.ones_like(yy) * hx
    xx_neg = np.ones_like(yy) * (-hx)
    face3 = np.stack([xx_pos.ravel(), yy.ravel(), zz.ravel()], axis=1)
    face4 = np.stack([xx_neg.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # 上下面 (z 方向)
    xx, yy = np.meshgrid(x, y)
    zz_pos = np.ones_like(xx) * hz
    zz_neg = np.ones_like(xx) * (-hz)
    face5 = np.stack([xx.ravel(), yy.ravel(), zz_pos.ravel()], axis=1)
    face6 = np.stack([xx.ravel(), yy.ravel(), zz_neg.ravel()], axis=1)

    points_local = np.vstack([face1, face2, face3, face4, face5, face6])

    # 旋转变换到世界坐标系
    rot = R.from_quat(box_quat)  # 输入应为 [x,y,z,w]
    points_world = rot.apply(points_local) + box_pos

    return points_world


def extract_pointcloud_from_mujoco(model: mujoco.MjModel,
                                    data: mujoco.MjData,
                                    sample_density: float = 0.05) -> np.ndarray:
    """从 MuJoCo 模型提取所有 box 的点云（自动处理四元数转换）"""
    all_points = []
    mujoco.mj_forward(model, data)

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            pos = data.geom_xpos[geom_id].copy()
            quat_mj = model.geom_quat[geom_id].copy()        # [w, x, y, z]
            quat_xyzw = quat_mj[[1, 2, 3, 0]]                # 转为 [x, y, z, w]
            size = model.geom_size[geom_id].copy()
            points = sample_box_surface(pos, quat_xyzw, size, sample_density)
            all_points.append(points)

    if all_points:
        return np.vstack(all_points)
    else:
        return np.empty((0, 3))


def create_box_mesh(pos: np.ndarray,
                    quat: np.ndarray,      # [x, y, z, w]
                    half_size: np.ndarray,
                    color: list = [1, 0, 0]) -> o3d.geometry.TriangleMesh:
    """创建用于可视化的 box 网格（半透明）"""
    # 生成立方体网格（中心在原点，边长为2*half_size）
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=2 * half_size[0],
        height=2 * half_size[1],
        depth=2 * half_size[2]
    )
    # 将网格中心移至原点，因为 create_box 的默认角点在 (0,0,0)
    mesh.translate(-half_size)

    # 应用旋转
    rot_matrix = R.from_quat(quat).as_matrix()
    mesh.rotate(rot_matrix, center=(0, 0, 0))

    # 应用平移
    mesh.translate(pos)

    # 设置颜色和透明度
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()  # 改善光照效果

    return mesh


def visualize_alignment(xml_path: str, density: float = 0.05):
    """主可视化函数"""
    # 加载 MuJoCo 模型
    print(f"加载模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 提取点云
    print("提取点云...")
    points = extract_pointcloud_from_mujoco(model, data, density)
    print(f"点云点数: {points.shape[0]}")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])  # 蓝色

    # 创建每个 box 的网格
    meshes = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            pos = data.geom_xpos[geom_id].copy()
            quat_mj = model.geom_quat[geom_id].copy()
            quat_xyzw = quat_mj[[1, 2, 3, 0]]
            size = model.geom_size[geom_id].copy()
            mesh = create_box_mesh(pos, quat_xyzw, size, color=[1, 0, 0])
            # 创建线框版本也可以，但半透明网格更直观
            meshes.append(mesh)

    # 可视化
    print("显示可视化窗口（红色网格 = MuJoCo box，蓝色点云 = 采样点）")
    o3d.visualization.draw_geometries([pcd] + meshes,
                                      window_name="Pointcloud vs MuJoCo Boxes",
                                      width=1024, height=768)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查点云与 MuJoCo box 对齐情况")
    parser.add_argument("xml_path", type=str, help="MuJoCo XML 文件路径")
    parser.add_argument("--density", type=float, default=0.05, help="采样密度（米），默认 0.05")
    args = parser.parse_args()

    visualize_alignment(args.xml_path, args.density)