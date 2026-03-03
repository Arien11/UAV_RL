import numpy as np


class LinearMixer:
    def __init__(self, max_thrust_per_motor=0.1573, L=0.065 / 2.0, kappa=3.842e-3 / 0.1573):
        self.max_thrust = max_thrust_per_motor
        self.L = L
        self.kappa = kappa
        # 混控矩阵（假设电机顺序：前左、前右、后左、后右，与cf2.xml一致）
        self.mixer = np.array([
            [1, 1, 1, 1],  # 总推力
            [ L, -L, -L,  L],   # τx = f1*y1 + f2*y2 + f3*y3 + f4*y4, y=±L
            [-L, -L,  L,  L],   # τy = -f1*x1 -f2*x2 -f3*x3 -f4*x4, x=±L
            [-kappa, kappa, -kappa, kappa]  # τz
        ])
        self.inv_mixer = np.linalg.inv(self.mixer)

    def calculate(self, thrust, Mx, My, Mz):
        # 期望电机推力（未饱和）
        f = self.inv_mixer @ np.array([thrust, Mx, My, Mz])
    
        # 饱和处理：缩放力矩以保证推力不超限，同时保持总推力不变
        eps = 1e-12
        T_avg = thrust / 4.0
        delta = f - T_avg
        scale = 1.0

        if np.max(delta) > eps:
            available_upper = max(0.0, self.max_thrust - T_avg)
            scale_upper = available_upper / (np.max(delta) + eps)
            scale = min(scale, scale_upper)

        if np.min(delta) < -eps:
            available_lower = max(0.0, T_avg)
            scale_lower = available_lower / (-np.min(delta) + eps)
            scale = min(scale, scale_lower)

        f_sat = T_avg + scale * delta
        f_sat = np.clip(f_sat, 0, self.max_thrust)  # 二次保险
        ctrl = f_sat / self.max_thrust
        return ctrl, scale
