
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time

def create_demo_scene_xml():
    """创建带深度相机的演示场景XML"""
    return """
<mujoco model="depth_camera_demo">
  <option timestep="0.02">
    <flag eulerdamp="disable"/>
  </option>

  <statistic center="0 0 0.5" extent="2"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.5 0.5 0.5"/>
    <global azimuth="-120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.2 0.3" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
    
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
    <material name="blue" rgba="0.2 0.2 0.8 1"/>
    <material name="yellow" rgba="0.9 0.9 0.2 1"/>
    <material name="purple" rgba="0.6 0.2 0.8 1"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <light pos="2 -2 2" dir="-1 1 -1"/>
    <light pos="-2 -2 2" dir="1 1 -1"/>
    
    <body name="obstacle_1" pos="0.8 0 0.5">
      <joint name="joint_1" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
      <geom type="box" size="0.2 0.2 0.2" material="red"/>
    </body>
    
    <body name="obstacle_2" pos="-0.8 0.5 0.4">
      <joint name="joint_2" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.15 0.3" material="green"/>
    </body>
    
    <body name="obstacle_3" pos="0 -1 0.3">
      <joint name="joint_3" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.2" material="blue"/>
    </body>
    
    <body name="obstacle_4" pos="1.5 1 0.8">
      <geom type="box" size="0.1 0.8 0.6" material="yellow"/>
    </body>
    
    <body name="obstacle_5" pos="-1.5 1 0.6">
      <geom type="capsule" size="0.1 0.4" material="purple"/>
    </body>
    
    <body name="drone" pos="0 0 1.5">
      <freejoint name="drone_joint"/>
      <inertial mass="0.033" pos="0 0 0" diaginertia="1.395e-5 1.395e-5 2.173e-5"/>
      <geom type="box" size="0.05 0.05 0.01" rgba="0.3 0.3 0.3 1"/>
      
      <camera name="drone_camera" pos="0 0.05 0" xyaxes="1 0 0 0 0 1" fovy="60"/>
    </body>
    
    <body name="static_camera" pos="0 -2 1.5">
      <camera name="fixed_camera" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="45"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint_1" gear="1"/>
    <motor name="motor2" joint="joint_2" gear="1"/>
    <motor name="motor3" joint="joint_3" gear="1"/>
  </actuator>
</mujoco>
"""

class DepthCameraRenderer:
    """MuJoCo深度相机渲染器"""
    
    def __init__(self, model, data, camera_name="drone_camera", width=640, height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found!")
        
        self.renderer = mujoco.Renderer(model, height, width)
        
        self.depth_buffer = np.zeros((height, width), dtype=np.float32)
        self.rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
    def update(self):
        """更新相机数据"""
        self.renderer.update_scene(self.data, camera=self.cam_id)
        
        self.rgb_buffer = self.renderer.render()
        
        depth_renderer = mujoco.Renderer(self.model, self.height, self.width)
        depth_renderer.update_scene(self.data, camera=self.cam_id)
        
        depth = depth_renderer.render()
        
        depth_gray = np.mean(depth, axis=2).astype(np.float32)
        depth_normalized = depth_gray / 255.0
        
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        
        depth_linear = near / (1 - depth_normalized * (1 - near / far))
        self.depth_buffer = depth_linear
        
        depth_renderer.close()
        
        return self.rgb_buffer, self.depth_buffer
    
    def get_depth(self):
        """获取深度图"""
        return self.depth_buffer
    
    def get_rgb(self):
        """获取RGB图像"""
        return self.rgb_buffer
    
    def close(self):
        """关闭渲染器"""
        self.renderer.close()

def demo_simple():
    """简单演示：显示单帧RGB和深度图"""
    print("创建MuJoCo模型...")
    xml = create_demo_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    print("初始化深度相机渲染器...")
    renderer = DepthCameraRenderer(model, data, camera_name="drone_camera", width=640, height=480)
    
    print("渲染图像...")
    rgb, depth = renderer.update()
    
    print("显示结果...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    depth_im = axes[1].imshow(depth, cmap='jet')
    axes[1].set_title('Depth Image')
    axes[1].axis('off')
    plt.colorbar(depth_im, ax=axes[1], label='Depth (m)')
    
    plt.tight_layout()
    plt.show()
    
    renderer.close()

def demo_with_viewer():
    """带Viewer的实时演示"""
    print("创建MuJoCo模型...")
    xml = create_demo_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    print("初始化深度相机渲染器...")
    renderer = DepthCameraRenderer(model, data, camera_name="drone_camera", width=480, height=360)
    
    drone_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_joint")
    drone_qpos_start = model.jnt_qposadr[drone_joint_id]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im_rgb = axes[0].imshow(np.zeros((360, 480, 3), dtype=np.uint8))
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    im_depth = axes[1].imshow(np.zeros((360, 480)), cmap='jet', vmin=0, vmax=5)
    axes[1].set_title('Depth Image')
    axes[1].axis('off')
    cbar = plt.colorbar(im_depth, ax=axes[1], label='Depth (m)')
    
    plt.tight_layout()
    plt.ion()
    plt.show()
    
    print("启动MuJoCo Viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        sim_time = 0
        frame_count = 0
        while viewer.is_running() and sim_time < 30:
            t = data.time
            
            data.ctrl[0] = np.sin(t * 2) * 0.3
            data.ctrl[1] = np.cos(t * 1.5) * 0.5
            data.ctrl[2] = np.sin(t * 0.8) * 0.5
            
            data.qpos[drone_qpos_start:drone_qpos_start+3] = [0, np.sin(t) * 0.3, 1.5 + np.cos(t * 0.5) * 0.2]
            data.qpos[drone_qpos_start+3:drone_qpos_start+7] = [1, 0, 0, 0]
            
            mujoco.mj_step(model, data)
            
            if frame_count % 10 == 0:
                rgb, depth = renderer.update()
                
                im_rgb.set_data(rgb)
                im_depth.set_data(depth)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            viewer.sync()
            sim_time = t
            frame_count += 1
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("演示被用户中断")
    finally:
        viewer.close()
        renderer.close()
        plt.ioff()
        plt.close(fig)

def compare_cameras():
    """比较不同相机的视角"""
    print("创建MuJoCo模型...")
    xml = create_demo_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    print("初始化多个相机渲染器...")
    renderers = [
        DepthCameraRenderer(model, data, camera_name="drone_camera", width=320, height=240),
        DepthCameraRenderer(model, data, camera_name="fixed_camera", width=320, height=240)
    ]
    
    camera_names = ["Drone Camera", "Fixed Camera"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, (renderer, name) in enumerate(zip(renderers, camera_names)):
        rgb, depth = renderer.update()
        
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'{name} - RGB')
        axes[0, i].axis('off')
        
        depth_im = axes[1, i].imshow(depth, cmap='jet')
        axes[1, i].set_title(f'{name} - Depth')
        axes[1, i].axis('off')
        plt.colorbar(depth_im, ax=axes[1, i], label='Depth (m)')
    
    plt.tight_layout()
    plt.show()
    
    for renderer in renderers:
        renderer.close()

def main():
    print("=" * 60)
    print("MuJoCo 深度相机演示")
    print("=" * 60)
    print("\n请选择演示模式:")
    print("1. 简单演示 - 显示单帧RGB和深度图")
    print("2. 实时演示 - 带Viewer的动态场景")
    print("3. 相机对比 - 比较不同视角")
    
    choice = input("\n请输入选项 (1/2/3, 默认2): ").strip()
    
    if choice == "1":
        demo_simple()
    elif choice == "3":
        compare_cameras()
    else:
        demo_with_viewer()

if __name__ == "__main__":
    main()

