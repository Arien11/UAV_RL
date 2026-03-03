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
            mean = torch.nan_to_num(mean, nan=0.0)
        # 输出标准差std
        sd = self.stds
        return mean, sd
    
    # 直接输出动作均值，在均值上加高斯探索
    def forward(self, state, deterministic=False):
        mu, sd = self._get_dist_params(state)  # mu 无界
        
        # stochastic 高斯分布探索得到动作
        if not deterministic:
            dist = self.distribution(state)  # 返回已包含缩放的分布
            action = dist.sample()  # 直接得到最终动作
        # deterministic 直接输出为动作
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


class TrajParam_Actor(Actor):
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

    # 先输出虚拟轨迹参数，在参数上加探索，再映射到动作
    def forward(self, state, deterministic=False):
        # 1. 输出「虚拟轨迹参数」（比如：期望速度、期望加速度）
        traj_params = self.net(state)  # 输出维度不变，还是4维，但物理意义变了
    
        # 2. 探索噪声加在「虚拟轨迹参数」上，而不是最终动作上
        if not deterministic:
            std = self.stds.exp() if self.learn_std else self.stds
            dist = torch.distributions.Normal(traj_params, std)
            traj_params = dist.rsample()  # 重参数化采样，噪声加在轨迹参数上
            
        return traj_params


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_state = torch.randn(1, 13).to(device)
    policy = FF_Actor(13, 9, bounded=True, learn_std=True).to(device)
    print("stds requires_grad:", policy.stds.requires_grad)
    # policy.action_scale = torch.tensor([2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0], device=device)
    # policy.action_bias = torch.zeros(9, device=device)
    # with torch.no_grad():
    #     action = policy(test_state, deterministic=True)
    #     print(action)  # 应该在 [-2,2], [-5,5], [-10,10] 范围内