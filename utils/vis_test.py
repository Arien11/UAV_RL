import sys
import os

# 将RaceUtils目录添加到Python路径中
race_utils_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RaceUtils')
if race_utils_root not in sys.path:
    sys.path.insert(0, race_utils_root)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from race_utils.RaceVisualizer.RacePlotter import RacePlotter
from race_utils.RaceGenerator.GenerationTools import create_state, create_gate

print("=" * 70)
print("创建门...")
print("=" * 70)

"""
gate_type 门类型
    SingleBall: 单球范围
    TrianglePrisma: 三角形门
    RectanglePrisma: 矩形门
    PentagonPrisma: 五边形门
    HexagonPrisma: 六边形门
"""
gate = create_gate(
    gate_type="RectanglePrisma",
    position=[2.5, 2.5, 1.0],
    stationary=True,
    shape_kwargs={"rpy": [45, 90, 90], 
                    "length": 2.0, 
                    "midpoints": [[0, 0, 0]], 
                    "width": 1.0, 
                    "height": 2.0, 
                    "marginW": 0.1, 
                    "marginH": 0.1},
    name="Gate_1",
)
print("✅ 门创建完成")
print(f"   门信息: {gate.to_dict()}")

print("\n" + "=" * 70)
print("创建赛道...")
print("=" * 70)
init_state = create_state({"pos": [0, 0, 1]})
end_state = create_state({"pos": [5, 5, 1]})
race_track = RaceTrack(init_state=init_state, end_state=end_state, race_name="MyTrack")
race_track.add_gate(gate, gate.name)
print("✅ 赛道创建完成")
print(f"   赛道字典: {race_track.to_dict()}")

print("\n" + "=" * 70)
print("创建模拟轨迹...")
print("=" * 70)
t = np.linspace(0, 10, 100)

# 简单的轨迹：从起点到终点
p_x = np.linspace(0, 5, 100)
p_y = np.linspace(0, 5, 100)
p_z = np.linspace(1, 1, 100)

# 变化的姿态（让无人机在运动时旋转）
# 使用欧拉角生成四元数：roll, pitch, yaw 都随时间变化
roll = np.sin(t) * 0.2    # 横滚角变化
pitch = np.cos(t) * 0.15   # 俯仰角变化
yaw = np.linspace(0, np.pi, 100)  # 偏航角从0到π

# 将欧拉角转换为四元数 (ZYX顺序)
q = np.zeros((len(t), 4))
for i in range(len(t)):
    r = Rotation.from_euler('ZYX', [yaw[i], pitch[i], roll[i]], degrees=False)
    q[i] = r.as_quat()  # [x, y, z, w]

q_x = q[:, 0]
q_y = q[:, 1]
q_z = q[:, 2]
q_w = q[:, 3]

# 变化的速度（让速度有变化，避免 vt_norm 变成标量）
v_x = 0.5 * np.sin(t) + 0.5
v_y = 0.5 * np.sin(t) + 0.5
v_z = np.zeros_like(t)

# 创建结构化数组（CSV格式）
dtype = [('t', float), ('p_x', float), ('p_y', float), ('p_z', float),
         ('q_w', float), ('q_x', float), ('q_y', float), ('q_z', float),
         ('v_x', float), ('v_y', float), ('v_z', float)]

your_trajectory_data = np.zeros(len(t), dtype=dtype)
your_trajectory_data['t'] = t
your_trajectory_data['p_x'] = p_x
your_trajectory_data['p_y'] = p_y
your_trajectory_data['p_z'] = p_z
your_trajectory_data['q_w'] = q_w
your_trajectory_data['q_x'] = q_x
your_trajectory_data['q_y'] = q_y
your_trajectory_data['q_z'] = q_z
your_trajectory_data['v_x'] = v_x
your_trajectory_data['v_y'] = v_y
your_trajectory_data['v_z'] = v_z
print("✅ 模拟轨迹创建完成")
print(f"   轨迹点数: {len(t)}")
print(f"   起点: ({p_x[0]:.2f}, {p_y[0]:.2f}, {p_z[0]:.2f})")
print(f"   终点: ({p_x[-1]:.2f}, {p_y[-1]:.2f}, {p_z[-1]:.2f})")
print(f"   速度范围: v_x [{v_x.min():.2f}, {v_x.max():.2f}], v_y [{v_y.min():.2f}, {v_y.max():.2f}]")
print(f"   姿态变化: roll [{roll.min():.2f}, {roll.max():.2f}], pitch [{pitch.min():.2f}, {pitch.max():.2f}], yaw [{yaw.min():.2f}, {yaw.max():.2f}]")

print("\n" + "=" * 70)
print("创建绘图器...")
print("=" * 70)
raceplotter = RacePlotter(
    traj_file=your_trajectory_data,
    track_file=race_track,
)
print("✅ 绘图器创建完成")
print(f"   track_file 是否为None: {raceplotter.track_file is None}")

# print("\n" + "=" * 70)
# print("绘制2D图...")
# print("=" * 70)
# raceplotter.plot(
#     fig_title="2D Trajectory",
#     radius=1.0,
#     margin=0.5,
# )
# print("✅ 2D图绘制完成")

# print("\n" + "=" * 70)
# print("绘制3D图...")
# print("=" * 70)
raceplotter.plot3d(
    fig_title="3D Trajectory",
    radius=1.0,
    margin=0.5,
    gate_alpha=0.8,
    gate_color="blue",
)
# print("✅ 3D图绘制完成")

print("\n" + "=" * 70)
print("创建动画（不保存视频）...")
print("=" * 70)
anim = raceplotter.create_animation(
    save_path=None,  # 不保存视频，只显示
    fps=30,
    dpi=200,
    traj_history=0.0,
    follow_drone=False,
    show_title=True,
    show_time=True,
    plot_colorbar=True,
    cmap=plt.get_cmap("cool_r"),
)
print("✅ 动画创建完成")

print("\n" + "=" * 70)
print("显示所有图像和动画...")
print("=" * 70)
raceplotter.plot_show()
print("✅ 所有图像和动画已显示")
