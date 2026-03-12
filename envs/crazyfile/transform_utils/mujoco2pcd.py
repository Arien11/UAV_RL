"""
从 MuJoCo 模型的 box 几何体采集点云
确保点云和障碍物完全对应
"""
import numpy as np
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def sample_box_surface(box_pos: np.ndarray,
                       box_quat: np.ndarray,
                       box_half_size: np.ndarray,
                       sample_density: float = 0.05) -> np.ndarray:
    """
    采样 box 表面的点云

    Args:
        box_pos: box 位置 (3,)
        box_quat: box 四元数 (4,)
        box_half_size: box 半长 (3,)
        sample_density: 采样密度（点之间的距离）

    Returns:
        points: 点云数组 (N, 3)
    """
    hx, hy, hz = box_half_size

    # 在每个面上生成网格点
    # 前后面 (y 方向)
    x = np.linspace(-hx, hx, max(1, int(2*hx / sample_density) + 1))
    y = np.linspace(-hy, hy, max(1, int(2*hy / sample_density) + 1))
    z = np.linspace(-hz, hz, max(1, int(2*hz / sample_density) + 1))
    xx, zz = np.meshgrid(x, z)
    yy_pos = np.ones_like(xx) * hy
    yy_neg = np.ones_like(xx) * (-hy)

    face1 = np.column_stack([xx.ravel(), yy_pos.ravel(), zz.ravel()])
    face2 = np.column_stack([xx.ravel(), yy_neg.ravel(), zz.ravel()])

    # 左右面 (x 方向)
    yy, zz = np.meshgrid(y, z)
    xx_pos = np.ones_like(yy) * hx
    xx_neg = np.ones_like(yy) * (-hx)

    face3 = np.column_stack([xx_pos.ravel(), yy.ravel(), zz.ravel()])
    face4 = np.column_stack([xx_neg.ravel(), yy.ravel(), zz.ravel()])

    # 上下面 (z 方向)
    xx, yy = np.meshgrid(x, y)
    zz_pos = np.ones_like(xx) * hz
    zz_neg = np.ones_like(xx) * (-hz)

    face5 = np.column_stack([xx.ravel(), yy.ravel(), zz_pos.ravel()])
    face6 = np.column_stack([xx.ravel(), yy.ravel(), zz_neg.ravel()])

    # 合并所有面
    points_local = np.vstack([face1, face2, face3, face4, face5, face6])

    # 转换到世界坐标系
    rot = R.from_quat(box_quat)
    points_world = rot.apply(points_local) + box_pos

    return points_world


def extract_pointcloud_from_mujoco(model: mujoco.MjModel,
                                    data: mujoco.MjData,
                                    sample_density: float = 0.05) -> np.ndarray:
    """
    从 MuJoCo 模型提取 box 障碍物的点云

    Args:
        model: MuJoCo 模型
        data: MuJoCo 数据
        sample_density: 采样密度

    Returns:
        points: 点云数组 (N, 3)
    """
    all_points = []

    # 更新运动学
    mujoco.mj_forward(model, data)

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]

        # 只处理 box 类型
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            pos = data.geom_xpos[geom_id].copy()
            quat_mj = model.geom_quat[geom_id].copy()
            size = model.geom_size[geom_id].copy()
            quat = quat_mj[[1, 2, 3, 0]]
            # print(f"处理 box {geom_id}: pos={pos}, size={size}")

            # 采样表面点
            points = sample_box_surface(pos, quat, size, sample_density)
            all_points.append(points)

    # 合并所有点
    if all_points:
        points = np.vstack(all_points)
        print(f"\n总点数: {points.shape[0]}")
    else:
        points = np.empty((0, 3))
        print("警告：没有找到 box 几何体！")

    return points


def main():
    """主函数"""
    import sys
    import os

    # 加载 MuJoCo 模型
    xml_path = "envs/crazyfile/seed_6.xml"
    print(f"加载模型: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 提取点云
    print("\n提取点云...")
    sample_density = 0.05  # 5cm 采样密度
    points = extract_pointcloud_from_mujoco(model, data, sample_density)

    # 保存点云
    output_path = "envs/crazyfile/seed_6.pcd"
    print(f"\n保存点云: {output_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"✅ 点云保存完成！")
    print(f"   点数: {points.shape[0]}")

    # 打印点云范围
    print(f"\n点云范围:")
    print(f"  x: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # 可选：可视化点云
    # print("\n显示点云...")
    # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
