import torch
import torch.nn as nn

from rl.network.base import Net


class Critic(Net):
    def __init__(self):
        super().__init__()
    
    def forward(self, state):
        raise NotImplementedError


class FF_Critic(Critic):
    def __init__(
            self,
            state_dim,
            layers=(256, 256),
            nonlinearity=torch.nn.functional.relu,  # 激活函数默认为relu
            normc_init=True,  # 是否使用归一化列初始化（稳定训练）
            obs_std=None,
            obs_mean=None,
    ):
        super().__init__()
        # 网络层构建
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)
        
        self.nonlinearity = nonlinearity
        
        if obs_std is None:
            self.obs_std = torch.ones(state_dim)
        else:
            self.obs_std = obs_std
            
        if obs_mean is None:
            self.obs_mean = torch.zeros(state_dim)
        else:
            self.obs_mean = obs_mean
            
        self.normc_init = normc_init
        
        self.init_parameters()
    
    def forward(self, state):
        # 估计当前状态价值
        
        # 确保 obs_mean 和 obs_std 已设置
        if self.obs_mean is None or self.obs_std is None:
            # 如果没有设置，使用默认值
            if len(state.shape) == 1:
                state_dim = state.shape[0]
            else:
                state_dim = state.shape[1]
            device = state.device
            self.obs_mean = torch.zeros(state_dim, device=device)
            self.obs_std = torch.ones(state_dim, device=device)
        
        # 确保 obs_mean 和 obs_std 在正确的设备上
        device = state.device
        if self.obs_mean.device != device:
            self.obs_mean = self.obs_mean.to(device)
        if self.obs_std.device != device:
            self.obs_std = self.obs_std.to(device)
        
        state = (state - self.obs_mean) / self.obs_std
        
        x = state
        for layer in self.critic_layers:
            x = self.nonlinearity(layer(x))
        value = self.network_out(x)
        
        return value
