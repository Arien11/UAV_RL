import torch
from scipy.spatial.transform import Rotation as R


class LatticeParam:
    def __init__(self):
        # ratio = 1.0 if cfg["train"] else cfg["velocity"] / cfg["vel_max_train"]
        self.vel_max = 6
        self.acc_max = 6
        # self.segment_time = cfg["sgm_time"] / ratio
        self.horizon_num = 5
        self.vertical_num = 3
        self.radio_num = 1
        self.traj_num = 15
        self.horizon_fov = 90.0
        self.vertical_fov = 60.0
        self.horizon_anchor_fov = 30.0
        self.vertical_anchor_fov = 30.0
        self.radio_range = 5


class LatticePrimitive(LatticeParam):
    _instance = None

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # direction_diff: 水平方向范围采样间隔
        if self.horizon_num == 1:
            direction_diff = 0
        else:
            direction_diff = (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num
        # altitude_diff: 垂直方向范围采样间隔
        if self.vertical_num == 1:
            altitude_diff = 0
        else:
            altitude_diff = (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
        # radio_diff: 规划范围半径采样间隔
        radio_diff = self.radio_range / self.radio_num

        lattice_pos_list = []
        lattice_angle_list = []
        lattice_Rbp_list = []

        # 生成运动基元 p_ij = r *（cos(dir_diff)*cos(alt_diff), cos(dir_diff)*sin(alt_diff), sin(dir_diff)）
        # 生成顺序：自下而上，从左到右，覆盖所有可能的运动基元
        for h in range(0, self.radio_num):
            for i in range(0, self.vertical_num):
                for j in range(0, self.horizon_num):
                    search_radio = (h + 1) * radio_diff # 规划范围半径
                    alpha = torch.tensor(-direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff)  # -direction_diff * (self.horizon_num - 1) / 2为最左  j * direction_diff 横向移动
                    beta = torch.tensor(-altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff)    # -altitude_diff * (self.vertical_num - 1) / 2为最下  i * altitude_diff 纵向移动
                   # 运动基元表示
                    pos_node = torch.tensor([torch.cos(beta) * torch.cos(alpha) * search_radio,
                                            torch.cos(beta) * torch.sin(alpha) * search_radio,
                                            torch.sin(beta) * search_radio])
                    # 运动基元表示（位置）
                    lattice_pos_list.append(pos_node)
                    # 运动基元表示（角度）alpha->yaw, beta->pitch
                    lattice_angle_list.append(torch.tensor([alpha, beta]))
                    # 运动基元表示（旋转矩阵）ZYX顺序 基元坐标系相对于机体坐标系的旋转
                    Rotation = R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False)  # inner rotation: yaw-pitch-roll
                    lattice_Rbp_list.append(torch.tensor(Rotation.as_matrix()))

        # 转换为张量并设置数据类型和设备
        self.lattice_pos_node = torch.stack(lattice_pos_list).to(dtype=torch.float32, device=device)  # shape: [N, 3]
        self.lattice_angle_node = torch.stack(lattice_angle_list).to(dtype=torch.float32, device=device)  # shape: [N, 2]
        self.lattice_Rbp_node = torch.stack(lattice_Rbp_list).to(dtype=torch.float32, device=device)  # shape: [N, 3, 3]
        # 锚定视野的半角度
        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    def getStateLattice(self, id=None):
        """获取所有或指定的运动基元位置状态"""
        if id is not None:
            return self.lattice_pos_node[id, :]
        else:
            return self.lattice_pos_node

    def getAngleLattice(self, id=None):
        """获取所有或指定的运动基元角度状态"""
        if id is not None:
            return self.lattice_angle_node[id, 0], self.lattice_angle_node[id, 1]  # yaw, pitch
        else:
            return self.lattice_angle_node[:, 0], self.lattice_angle_node[:, 1]  # yaw, pitch

    def getRotation(self, id=None):
        """获取所有或指定的运动基元旋转矩阵状态"""
        if id is not None:
            return self.lattice_Rbp_node[id]
        else:
            return self.lattice_Rbp_node

    def convert_ImageGrid_LatticeID(self, id):
        return self.traj_num - id - 1

    @classmethod
    def get_instance(self):
        if self._instance is None: self._instance = self()
        return self._instance       
