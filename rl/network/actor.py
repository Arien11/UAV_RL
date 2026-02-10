import torch
import torch.nn as nn
from rl.network.base import *


class ActorNetwork(torch.nn.Module):
    """模拟策略网络"""
    
    def __init__(self, state_dim=13, action_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 简单的线性策略
        self.fc = torch.nn.Linear(state_dim, action_dim)
        
        # 观察值归一化参数
        self.obs_mean = torch.zeros(state_dim)
        self.obs_std = torch.ones(state_dim)
        
        # 动作噪声（用于探索）
        self.stds = torch.nn.Parameter(torch.ones(action_dim) * 0.1)
    
    def forward(self, state, deterministic=False):
        """前向传播"""
        # 确保输入是2D张量 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        
        # 简单的归一化
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        
        # 计算网络输出
        output = self.fc(state_normalized)
        
        if deterministic:
            # 确定性策略：直接输出网络输出
            return output
        else:
            # 随机策略：添加噪声
            noise = torch.randn_like(output) * self.stds
            return output + noise
    
    def init_hidden_state(self):
        """初始化隐藏状态（用于循环网络）"""
        pass


class CriticNetwork(torch.nn.Module):
    """模拟价值网络"""
    
    def __init__(self, state_dim=13):
        super().__init__()
        self.state_dim = state_dim
        self.fc = torch.nn.Linear(state_dim, 1)
        
        # 观察值归一化参数
        self.obs_mean = torch.zeros(state_dim)
        self.obs_std = torch.ones(state_dim)
    
    def forward(self, state):
        """前向传播"""
        # 确保输入是2D张量 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        state_normalized = (state - self.obs_mean) / (self.obs_std + 1e-8)
        return self.fc(state_normalized)
    
    def init_hidden_state(self):
        """初始化隐藏状态"""
        pass


class Actor(Net):
    def __init__(self):
        super().__init__()
    
    def forward(self, state, deterministic=True):
        raise NotImplementedError


class FF_Actor(Actor):
    def __init__(
            self,
            state_dim,
            action_dim,
            layers=(256, 256),
            nonlinearity=torch.nn.functional.relu,  # 激活函数默认为relu
            init_std=0.2,  # 初始标准差
            learn_std=False,  # 是否学习标准差（可训练参数）
            bounded=False,  # 是否限制动作范围（如使用tanh）
            normc_init=True,  # 是否使用归一化列初始化（稳定训练）
    ):
        super().__init__()
        # 网络层构建(均值参数)
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.means = nn.Linear(layers[-1], action_dim)  # 输出动作分布的均值
        
        # 标准差参数
        self.learn_std = learn_std
        if self.learn_std:
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            self.stds = init_std * torch.ones(action_dim)
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.nonlinearity = nonlinearity
        
        self.obs_std = 1.0
        self.obs_mean = 0.0
        
        self.bounded = bounded
        
        self.normc_init = normc_init
        self.init_parameters(self.means)
    
    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std
        
        # 输出均值mean
        x = state
        for layer in self.actor_layers:
            x = self.nonlinearity(layer(x))
        mean = self.means(x)
        
        if self.bounded:
            mean = torch.tanh(mean)
        
        # 输出标准差std
        sd = torch.zeros_like(mean)
        if hasattr(self, "stds"):
            sd = self.stds
        return mean, sd
    
    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)
        
        # 根据输出的概率分布参数得到对应的动作
        if not deterministic:
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            action = mu
        
        return action
    
    def distribution(self, inputs):
        # 得到动作属于的概率分布
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class Temp_Actor(Actor):
    def __init__(self):
        super().__init__()
        self.obs_std = 1.0
        self.obs_mean = 0.0
    
    def _get_device(self):
        """Get device from network parameters."""
        return next(self.parameters()).device
    
    def init_hidden_state(self, batch_size=1, device=None):
        if device is None:
            device = self._get_device()
        self.hidden = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]
        self.cells = [torch.zeros(batch_size, layer.hidden_size, device=device) for layer in self.actor_layers]
    
    def _get_dist_params(self, state):
        """
        根据输入状态计算连续动作分布的参数
        :param state: (seq_len, batch_size, feature_dim) 支持单时间步、批量单时间步、批量轨迹三种形态
        :return: mu, std (均值与方差)
        """
        # 状态标准化
        state = (state - self.obs_mean) / self.obs_std
        
        dims = len(state.size())
        
        x = state
        if dims == 3:  # 处理一批完整轨迹
            self.init_hidden_state(batch_size=x.size(1), device=x.device)
            y = []
            # 手动时间步循环：逐时间步、逐RNN层前向传播
            for _t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])  # 收集每步输出，堆叠为 [seq_len, batch_size, hidden_dim]
        
        else:  # 处理单时间步或批量单时间步
            if dims == 1:  # 单时间步(feature_size,) 当做批时间步处理(batch_size=1, feature_size)
                x = x.view(1, -1)
            
            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            
            if dims == 1:
                x = x.view(-1)
        
        mu = self.network_out(x)
        if self.bounded:  # 限制范围
            mu = torch.tanh(mu)
        sd = self.stds
        return mu, sd
    
    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)  # 正态分布采样的动作
