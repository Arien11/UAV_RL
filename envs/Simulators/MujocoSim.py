from abc import ABC, abstractmethod
import mujoco
import numpy as np
import mujoco.viewer
import yaml


class XMLModelLoader:
    def __init__(self, asset_resolver=None):
        self.asset_resolver = asset_resolver
    
    def load(self, model_info):
        # 利用string进行加载
        if 'xml_string' in model_info:
            return mujoco.MjModel.from_xml_string(
                model_info['xml_string'],
                assets=self.asset_resolver
            )
        # 利用文件读取路径进行加载
        elif 'xml_path' in model_info:
            return mujoco.MjModel.from_xml_path(
                model_info['xml_path']
            )
        else:
            raise ValueError("No valid model source provided")
    
    def validate(self, model):
        return model is not None and model.nq > 0


class MuJoCoSimulator(ABC):
    def __init__(self, config):
        """
        初始化MuJoCo仿真器
        
        Args:
            config: 配置字典，包含 'model' 键
        """
        XML_loader = XMLModelLoader()
        self.model = XML_loader.load(config.get('model', {}))
        self.data = mujoco.MjData(self.model)
        self.viewer = None  # 添加viewer引用
        self._viewer_paused = False
        self._marker_drawer = None
        
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        
        self._camera_renderer = None
        self._camera_name = "drone_depth_camera"
        self._camera_width = 160
        self._camera_height = 120
        self._last_camera_frame = 0
        self._cached_rgb = None
        self._cached_depth = None
    
    def reset_model(self):
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
        """重置robot到其特定的初始状态"""
        # raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        with self.viewer.lock():
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.lookat[2] = 1.5
            self.viewer.cam.lookat[0] = 2.0
            self.viewer.cam.elevation = -20
            self.viewer.opt.geomgroup[2] = 0
    
    def _key_callback(self, keycode):
        """按键事件触发"""
        if keycode == 32:
            self._viewer_paused = not self._viewer_paused
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        obs = self.reset_model()
        return obs
    
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,), f"qpos shape {qpos.shape} is expected to be {(self.model.nq,)}"
        assert qvel.shape == (self.model.nv,), f"qvel shape {qvel.shape} is expected to be {(self.model.nv,)}"
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act = []
        self.data.plugin_state = []
        # Disable actuation since we don't yet have meaningful control inputs.
        # with self.disable("actuation"):
        #     mujoco.mj_forward(self.model, self.data)
    
    def draw_markers(self, marker_drawer):
        """Draw task-specific markers in the viewer.

        Override this method in subclasses to draw custom visualizations
        (e.g., step targets, goal positions, debug info).

        Args:
            marker_drawer: MarkerDrawer instance for adding geometries
        """
        pass
    
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._key_callback)
            # self._marker_drawer = MarkerDrawer(self.viewer)
            self.viewer_setup()
        
        # Draw markers if we have a marker drawer
        # if self._marker_drawer is not None:
        #     self._marker_drawer.reset()
        #     self.draw_markers(self._marker_drawer)
        #     self._marker_drawer.finalize()
        
        # Block while paused, but keep viewer responsive
        while self._viewer_paused and self.viewer.is_running():
            self.viewer.sync()
        self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._close_camera()
    
    def _init_camera(self, camera_name=None, width=None, height=None):
        """初始化深度相机渲染器"""
        if camera_name is not None:
            self._camera_name = camera_name
        if width is not None:
            self._camera_width = width
        if height is not None:
            self._camera_height = height
        
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{self._camera_name}' not found!")
        
        self._camera_renderer = mujoco.Renderer(self.model, self._camera_height, self._camera_width)
        self._camera_renderer._cam_id = cam_id
        
    def _close_camera(self):
        """关闭深度相机渲染器"""
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
    
    def get_camera_rgb(self, force_update=False):
        """获取相机RGB图像"""
        if self._camera_renderer is None:
            self._init_camera()
        
        current_frame = int(self.data.time * 100)
        if force_update or current_frame != self._last_camera_frame:
            self._camera_renderer.update_scene(self.data, camera=self._camera_name)
            self._cached_rgb = self._camera_renderer.render()
            self._last_camera_frame = current_frame
        
        return self._cached_rgb
    
    def get_camera_depth(self, force_update=False):
        """获取相机深度图（优化版）"""
        if self._camera_renderer is None:
            self._init_camera()
        
        current_frame = int(self.data.time * 100)
        if force_update or current_frame != self._last_camera_frame or self._cached_depth is None:
            self._camera_renderer.update_scene(self.data, camera=self._camera_name)
            rgb = self._camera_renderer.render()
            
            depth_gray = np.mean(rgb, axis=2).astype(np.float32)
            depth_normalized = depth_gray / 255.0
            
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent
            
            self._cached_depth = near / (1 - depth_normalized * (1 - near / far))
            self._last_camera_frame = current_frame
        
        return self._cached_depth
    
    def get_camera_data(self, force_update=False):
        """同时获取RGB图像和深度图"""
        rgb = self.get_camera_rgb(force_update=force_update)
        depth = self.get_camera_depth(force_update=force_update)
        return rgb, depth

