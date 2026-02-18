import torch
import torch.nn as nn
from rl.network.base import *
import torch.distributions as D


class ScaledTanhNormal(D.TransformedDistribution):
    def __init__(self, loc, scale, action_scale, action_bias):
        base_dist = D.Normal(loc, scale)
        transforms = [
            D.TanhTransform(),
            D.AffineTransform(loc=action_bias, scale=action_scale)  # y = bias + scale * x
        ]
        super().__init__(base_dist, transforms)
        self.loc = loc
        self.scale = scale
    
    def log_prob(self, value):
        return super().log_prob(value)
    
    def entropy(self):
        # Entropy of transformed distribution (can approximate with base entropy)
        return self.base_dist.entropy() + torch.log(self.transforms[-1].scale).sum()


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
            init_std=0.5,  # 初始标准差
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
        
        # 动作边界映射
        self.action_scale = None
        self.action_bias = None
        
        self.normc_init = normc_init
        self.init_parameters(self.means)
    
    def set_action_range(self, low, high, device):
        """设置动作的实际范围，并计算 scale 和 bias"""
        self.action_scale = torch.as_tensor((high - low) / 2, dtype=torch.float32, device=device)
        self.action_bias = torch.as_tensor((high + low) / 2, dtype=torch.float32, device=device)
    
    def _get_dist_params(self, state):
        if torch.isnan(state).any():
            print("Input state contains NaN!")
        state = (state - self.obs_mean) / self.obs_std
        state = torch.clamp(state, -10.0, 10.0)  # 限制输入范围
 
        # 输出均值mean
        x = state
        for i, layer in enumerate(self.actor_layers):
            x = layer(x)
            x = self.nonlinearity(x)

        mean = self.means(x)
        if torch.isnan(mean).any():
            print("NaN in mean!")
            # 可以暂时用 0 替换，但最好找出原因
            mean = torch.nan_to_num(mean, nan=0.0)
        # 输出标准差std
        sd = self.stds
        return mean, sd

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)  # mu 无界
    
        if not deterministic:
            dist = self.distribution(state)  # 返回已包含缩放的分布
            action = dist.sample()  # 直接得到最终动作
        else:
            if self.bounded:
                action = torch.tanh(mu) * self.action_scale + self.action_bias
            else:
                action = mu
        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        if self.bounded:
            # Ensure action_scale and action_bias are on same device as mu
            device = mu.device
            action_scale = self.action_scale.to(device)
            action_bias = self.action_bias.to(device)
            return ScaledTanhNormal(mu, sd, action_scale, action_bias)
        else:
            return D.Normal(mu, sd)


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
