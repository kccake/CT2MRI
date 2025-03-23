import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===== 辅助函数 =====
# 0
# ===== 辅助函数 =====

# ===== 0.1 权重初始化（带均衡学习率）
def get_weight(shape, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(shape[:-1])  # [kernel, kernel, in, out]
    he_std = gain / np.sqrt(fan_in)
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul
    weight = torch.randn(*shape) * init_std
    return nn.Parameter(weight * runtime_coef)

# ===== 生成器 G_main =====
# 1
# ===== 生成器 G_main =====
class GMain(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

# ==== 生成器 G_main含有的网络结构 =====

# ==== 1.1 映射网络 G_mapping
# 1.1.1 全连接层（带均衡学习率）
class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gain=1, use_wscale=True, lrmul=1):
        super().__init__()
        self.weight = get_weight([in_dim, out_dim], gain, use_wscale, lrmul)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.lrmul = lrmul

    def forward(self, x):
        x = F.linear(x, self.weight * self.lrmul, self.bias * self.lrmul)
        return x
# 1.1.2 映射网络 G_mapping
class GMapping(nn.Module):
    def __init__(self, latent_size=512, dlatent_size=512, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(dlatent_size, dlatent_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        w = z
        for layer in self.layers:
            w = layer(w)
            w = F.leaky_relu(w, 0.2)
        return w
    
# 合成网络 G_synthesis_stylegan2
class GSynthesisStylegan2(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

# ===== 判别器 D_stylegan2 =====
# 2
# ===== 判别器 D_stylegan2 =====
class DStylegan2(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass