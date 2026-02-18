from abc import ABC, abstractmethod
from typing import Dict, Any
import mujoco
import numpy as np
import mujoco.viewer
import time
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


class MuJoCoSimulator:
    def __init__(self, model_path):
        XML_loader = XMLModelLoader()
        env_config = {}
        if model_path:
            with open(model_path, 'r') as f:
                env_config = yaml.safe_load(f)
        self.model = XML_loader.load(env_config.get('model', {}))
        self.data = mujoco.MjData(self.model)
        self.viewer = None  # 添加viewer引用
        self._viewer_paused = False
        self._marker_drawer = None
        
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
    
    def reset_model(self):
        """重置robot到其特定的初始状态"""
        raise NotImplementedError
    
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


if __name__ == '__main__':
    ...
