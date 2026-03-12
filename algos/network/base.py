import torch
import torch.nn as nn


def normc_fn(m):  # "归一化列"（normalized columns）
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  # 只对线性层进行初始化
        # 1. 先用标准正态分布初始化权重
        m.weight.data.normal_(0, 1)
        # 2. 对每一行进行归一化（除以该行的L2范数）
        #    pow(2)是平方，sum(1, keepdim=True)对每行求和
        #    这样每行的权重向量长度变为1
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        # 3. 偏置初始化为0
        if m.bias is not None:
            m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
    
    def init_parameters(self, output_layer=None):
        # 检查是否使用normc_init（默认True）
        if getattr(self, "normc_init", True):
            # 对所有模块应用normc_fn初始化
            self.apply(normc_fn)
            
            # 如果指定了输出层，将其权重缩小100倍
            if output_layer is not None:
                output_layer.weight.data.mul_(0.01)
