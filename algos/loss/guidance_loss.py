import torch.nn as nn
import torch 
import torch.nn.functional as F

class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.goal_length = 1.0
        self.vel_dir_weight = 0  # 5
        
    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        Returns:
            guidance_loss: (batch_size) → guidance loss

        GuidanceLoss: distance_loss (for straighter flight) or similarity_loss (for faster flight in large scenario)
        """ 
        cur_pos = Df[:, :, 0]       # 轨迹起点
        end_pos = Dp[:, :, 0]       # 轨迹终点
        end_vel = Dp[:, :, 1]       

        traj_dir = end_pos - cur_pos  # [B, 3]  轨迹方向：从当前位置到末端位置
        goal_dir = goal - cur_pos  # [B, 3]     目标方向：从当前位置到目标点

        # guidance_loss = self.distance_loss(traj_dir, goal_dir)
        guidance_loss = self.similarity_loss(traj_dir, goal_dir)

        # if self.vel_dir_weight > 0:
        #     vel_dir_loss = self.derivative_similarity_loss(end_vel, goal_dir)
        #     guidance_loss += self.vel_dir_weight * vel_dir_loss
        return guidance_loss


    def similarity_loss(self, traj_dir, goal_dir):
        """
        Returns:
            similarity: (batch_size) → guidance loss

        SimilarityLoss: Projection length of the trajectory onto the goal direction:
                        higher cosine similarity and longer trajectory are preferred.

        Adjust perp_weight to penalize deviation perpendicular to the goal; equals the distance_loss() when perp_weight = 1.
        """
        # 目标方向归一化
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]

        # projection length of trajectory on goal direction 计算轨迹在目标方向上的投影长度
        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)  # [B]
        goal_length = goal_dir.norm(dim=1)  # [B]

        # length difference along goal direction (cosine similarity) 计算轨迹在目标方向上的投影长度与目标长度的差异
        parallel_diff = F.smooth_l1_loss(goal_length, traj_along, reduction='none')

        # length perpendicular to goal direction 除了goal_dir外其他分量的差异
        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm  # [B, 3]
        perp_diff = traj_perp.norm(dim=1)  # [B]

        # distance weighting (reduce perpendicular constraint, allow lateral exploration)
        perp_weight = 0.5   # the given weight is trained with perp_weight = 0, for higher speed in large-scale scenario
        similarity_loss = parallel_diff + perp_weight * perp_diff
        return similarity_loss