import torch
import numpy as np
from algos.primitive import LatticePrimitive

class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
    
    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
        endstate_pred: [batch_size; pred_state; primitive_v; primitive_h]
        pred_state: delta_yaw, delta_pitch, radio, vx, vy, vz, ax, ay, az
        primitive_v(h): 表示基元框的尺寸
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]
        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        # [B, primitive, traj_state]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)

        # ----------------- 偏移量处理 ----------------- # 
        # 各个预设基元对应的yaw和pitch变化(.flip: 由于lattice和grid的顺序相反)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15] 
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        # 各个预设基元的旋转矩阵
        Rbp = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3] 
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        # 处理网络输出的部分，网络输出通过tanh后得到primitive部分是[-1, 1]
        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]，[-yaw_diff, yaw_diff]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff  # [-pitch_diff, pitch_diff]
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range  # [0, 2 * radio_range]

        # 加入网络输出的yaw、pitch、radio偏移量到预设基元中
        # f_ij(T)=r′_ij*（cos(θ′_ij)*cos(φ′_ij)，cos(θ′_ij)*sin(φ′_ij)，sin(θ′_ij))
        # pitch + delta_pitch = θ′_ij， yaw + delta_yaw = φ′_ij
        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        # 基元是以球坐标系表示的，本身就是位置坐标(px, py, pz)
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]
        
        # ----------------- 物理量部分处理 ----------------- # 
        # 轨迹状态 -> 运动基元坐标系 → 机体坐标系
        # vel / acc 转换到 运动基元坐标系
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v/a 变换到 机体坐标系
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)
        # 
        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)

    def unnormalize_obs(self, vel_acc):
        vel_acc[:, 0:3] = vel_acc[:, 0:3] * self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] * self.lattice_primitive.acc_max
        return vel_acc

    def normalize_obs(self, vel_acc_goal):
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.lattice_primitive.acc_max

        # Clamp the goal direction to unit length
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return vel_acc_goal

    def prepare_input(self, obs):
        """
            机体坐标系 → 运动基元坐标系
            obs: [batch; vx, vy, yz, ax, ay, az, gx, gy, gz] in body frame
            :return [batch; vx, vy, yz, ax, ay, az, gx, gy, gz; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num

        # 获得所有基元对应的旋转矩阵
        Rbp_all = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3] 
        
        obs = obs.view(B, 3, 3)  # [B, 3, 3]

        # 扩展 obs 和 Rbp 到 [B, N, 3, 3]是为了服务坐标变换
        obs_exp = obs[:, None, :, :].expand(B, N, 3, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # 右乘转到运动基元坐标系
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 3, 3]
        
        # 转换为原观测量向量形式，每个样本的每个时间步的9个状态量按时间步排序
        transformed_flat = transformed.view(B, N, 9)  # [B, N, 9]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, 9, N]
        out = out.view(B, 9, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)  # [B, 9, V, H]
        return out

    
    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T # [9, traj_num] -> [traj_num, 9]
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)    # 轨迹状态的分数，用于选择最优轨迹状态

        # 选择最优轨迹状态
        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        # 返回所有轨迹状态
        else:       
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score  

    
