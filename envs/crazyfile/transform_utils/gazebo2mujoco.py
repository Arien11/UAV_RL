
"""
将 Gazebo/SDFormat 格式的 world 文件转换为 MuJoCo 格式的 XML
主要保留：
- 地面平面
- 所有障碍物模型（box, block等）
- CF2无人机模型和执行器
"""
import xml.etree.ElementTree as ET
import numpy as np

def parse_poses(sdf):
    """解析poses属性"""
    elements = sdf.split()
    poses = np.zeros(6)  # x, y, z, r, p, y
    for i in range(min(len(elements), 6)):
        try:
            poses[i] = float(elements[i])
        except:
            pass
    return poses

def convert_to_mujoco(world_path, output_path):
    tree = ET.parse(world_path)
    root = tree.getroot()
    
    mujoco_str = """
<mujoco model="seed_6_map">
  <include file="cf2.xml"/>

  <statistic center="0 0 0" extent="20"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20" ellipsoidinertia="true"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
    
    <material name="obstacle_red" rgba="0.8 0.2 0.2 1"/>
    <material name="obstacle_green" rgba="0.2 0.8 0.2 1"/>
    <material name="obstacle_blue" rgba="0.2 0.2 0.8 1"/>
    <material name="obstacle_yellow" rgba="0.9 0.9 0.2 1"/>
    <material name="obstacle_purple" rgba="0.6 0.2 0.8 1"/>
    <material name="obstacle_grey" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.05" material="groundplane"/>
    
    <light pos="10 -10 20" dir="-1 1 -1" diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <light pos="-10 10 15" dir="1 -1 -1" diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2"/>
"""

    # 解析模型
    model_index = 0
    for model in root.findall('.//model'):
        model_name = model.get('name')
        
        if model_name == 'ground_plane':
            continue
            
        static = False
        if model.find('./static') is not None:
            static = model.findtext('./static').lower() == 'true' or model.findtext('./static') == '1'
        
        if not static:
            continue
            
        # 获取姿态
        poses = np.zeros(6)
        if model.find('./pose') is not None:
            poses = parse_poses(model.findtext('./pose'))
            
        x, y, z, r, p, yaw = poses
        r, p, yaw = np.rad2deg(r), np.rad2deg(p), np.rad2deg(yaw)
        # 处理block类型（比赛穿框）
        if 'block_' in model_name:
            mujoco_str += "    <!-- " + model_name + " (block frame) -->\n"
            mujoco_str += f"    <body name=\"{model_name}\" pos=\"{x} {y} {z}\" euler=\"{r} {p} {yaw}\">\n"
            
            # block包含多个link，需要逐一处理
            links = model.findall('./link')
            for link in links:
                link_name = link.get('name')
                
                if 'right' in link_name or 'left' in link_name or 'top' in link_name or 'bottom' in link_name:
                    # 获取link的本地位置
                    link_pose = np.zeros(6)
                    if link.find('./pose') is not None:
                        link_pose = parse_poses(link.findtext('./pose'))
                        
                    collision = link.find('./collision')
                    if collision is not None:
                        geometry = collision.find('./geometry')
                        
                        if geometry is not None and geometry.find('./box') is not None:
                            box_size = geometry.findtext('./box/size')
                            sx, sy, sz = [float(x) for x in box_size.split()]
                            
                            lx, ly, lz, lr, lp, lyaw = link_pose
                            mujoco_str += f"      <geom name=\"{model_name}_{link_name}\" type=\"box\" "
                            mujoco_str += f"pos=\"{lx} {ly} {lz}\" "
                            mujoco_str += f"size=\"{sx/2} {sy/2} {sz/2}\" "
                            mujoco_str += f"material=\"obstacle_blue\"/>\n"
                        
            mujoco_str += "    </body>\n"
            
        else:
            # 处理普通box/sphere/cylinder
            link = model.find('./link')
            if link is None:
                continue
                
            collision = link.find('./collision')
            if collision is None:
                continue
                
            geometry = collision.find('./geometry')
            
            if geometry is not None:
                mujoco_str += "    <!-- " + model_name + " -->\n"
                
                # 处理不同类型的几何体
                if geometry.find('./box') is not None:
                    box_size = geometry.findtext('./box/size')
                    sx, sy, sz = [float(x) for x in box_size.split()]
                    
                    mujoco_str += f"    <body name=\"{model_name}\" pos=\"{x} {y} {z}\" euler=\"{r} {p} {yaw}\">\n"
                    mujoco_str += f"      <geom type=\"box\" size=\"{sx/2} {sy/2} {sz/2}\" material=\"obstacle_grey\"/>\n"
                    mujoco_str += "    </body>\n"
                    
                elif geometry.find('./sphere') is not None:
                    radius = float(geometry.findtext('./sphere/radius'))
                    mujoco_str += f"    <body name=\"{model_name}\" pos=\"{x} {y} {z}\" euler=\"{r} {p} {yaw}\">\n"
                    mujoco_str += f"      <geom type=\"sphere\" size=\"{radius}\" material=\"obstacle_red\"/>\n"
                    mujoco_str += "    </body>\n"
                    
                elif geometry.find('./cylinder') is not None:
                    radius = float(geometry.findtext('./cylinder/radius'))
                    length = float(geometry.findtext('./cylinder/length'))
                    mujoco_str += f"    <body name=\"{model_name}\" pos=\"{x} {y} {z}\" euler=\"{r} {p} {yaw}\">\n"
                    mujoco_str += f"      <geom type=\"cylinder\" size=\"{radius} {length/2}\" material=\"obstacle_green\"/>\n"
                    mujoco_str += "    </body>\n"
                
            model_index += 1
    
    # 添加CF2无人机
    mujoco_str += """
        </worldbody>
    </mujoco>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mujoco_str)
        
    print("转换完成！生成", output_path)
    
    # 统计block数量
    block_count = 0
    box_count = 0
    for model in root.findall('.//model'):
        if 'block_' in model.get('name'):
            block_count += 1
        if 'box_' in model.get('name'):
            box_count += 1
    print(f"成功转换 {box_count} 个box模型, 成功转换 {block_count} 个block穿框模型")

if __name__ == '__main__':
    import sys
    world_file = "envs/crazyfile/seed_6.world"
    output_file = "envs/crazyfile/seed_6.xml"
    
    convert_to_mujoco(world_file, output_file)
