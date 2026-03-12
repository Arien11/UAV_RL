import math
import torch.nn as nn
import torch 
from .safety_loss import SafetyLoss
from .guidance_loss import GuidanceLoss
from .smooth_loss import SmoothnessLoss


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.sgm_time = 5/3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # _C: 置换矩阵，_B: 边界条件矩阵A的逆，_L: A^(-1) * C 置换矩阵加映射矩阵的效果，
        # _RJ: Jerk的海森矩阵，_RA: Accel的海森矩阵
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()    
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L)
        self.goal_loss = GuidanceLoss()

        self.smoothness_weight = 10.0
        self.safety_weight = 1.0
        self.goal_weight = 0.15
        self.acceleration_weight = 0.1


    def qp_generation(self):
        """
            生成二次规划的映射矩阵A和需要的海森矩阵Q(H)
        """
        # 边界条件映射矩阵，将约束条件映射到多项式系数中
        A = torch.zeros((6, 6))
        # 对应的的是 b:[p0, pT, v0, vT, a0, aT]
        for i in range(3):      # 表示为某一阶，2是因为起点终点两个位置
            A[2 * i, i] = math.factorial(i)     # 起点约束： [0,0] [2,1] [4,2]
            for j in range(i, 6):
                # 根据当前阶数，计算终点约束的系数
                # i = 0时，A[1, 0] = 1, A[1, 1] = sgm_time, A[1, 2] = sgm_time ** 2...
                # i = 1时，A[3, 1] = 1, A[3, 2] = 2 * sgm_time, A[3, 3] = 3 * sgm_time ** 2...
                # i = 2时，A[5, 2] = 1, A[5, 3] = 3 * 2 * sgm_time, A[5, 4] = 4 * 3 * sgm_time ** 2...
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))    # 终点约束

        # 二次规划的对称阵
        # H海森矩阵，对应Jerk的对称阵
        H = torch.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Q海森矩阵，对应Accel的对称阵
        Q = torch.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)

    def stack_opt_dep(self, A, H, Q):
        """
            生成二次规划的置换矩阵C和最终合并计算的矩阵R(Jerk)、R(Accel)
        """
        # 置换矩阵分离自由变量与决策变量
        Ct = torch.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1      # 调整位置

       
        _C = torch.transpose(Ct, 0, 1)
        B = torch.inverse(A)
        B_T = torch.transpose(B, 0, 1) # (A^(-1))^T
        _L = B @ Ct

        # R = (A^-1 * Ct)^T * H(Q) * (A^-1 * Ct)
        _R_Jerk = _C @ (B_T) @ H @ B @ Ct

        _R_Acc = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R_Jerk, _R_Acc
    
    def forward(self, state, prediction, goal, map_id):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        safety_cost = self.safety_loss(Df, Dp, map_id)
        # safety_cost = 0
        goal_cost = self.goal_loss(Df, Dp, goal)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, self.goal_weight * goal_cost, self.acceleration_weight * acceleration_cost