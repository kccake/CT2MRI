import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2, sqrt
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== 1.噪声映射网络 =====
# ===== 1.噪声映射网络 =====
# ===== 1.噪声映射网络 =====
# 1.0 噪声映射网络
class MappingNetwork(nn.Module):
    """
    MappingNetwork: 噪声映射网络，将输入的z_dim维 噪声 映射到w_dim维 潜在空间
    latent z -> w
    
    Args:
        z_dim: 噪声维度(256)
        w_dim: 潜在空间维度(256)
        num_layers: 映射网络层数(8)
    """
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = []
        # 第 1 层: z_dim → w_dim
        layers += [
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            PixelNorm()
        ]
        # 中间层: w_dim → w_dim (共 num_layers-2 层)
        for _ in range(num_layers-2):
            layers += [
                EqualizedLinear(w_dim, w_dim),
                nn.ReLU(),
                PixelNorm()
            ]
        # 最后一层: w_dim → w_dim (无激活/归一化)
        layers += [EqualizedLinear(w_dim, w_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 1.1 像素归一化
class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
# 1.2 权重缩放
class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias_init=0.):
        super().__init__()
        self.weight = EqualizedWeight([out_dim, in_dim])
        self.bias = nn.Parameter(torch.ones(out_dim) * bias_init)

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias)
# 1.3 权重初始化
class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        in_features = shape[1]  # [out, in]
        self.scale = 1 / sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(shape))
    
    def forward(self):
        return self.weight * self.scale  # 缩放至稳定区间

# ===== 2.生成器 =====
# ===== 2.生成器 =====
# ===== 2.生成器 =====
# 2.0 生成器
class StyleGAN2Generator(nn.Module):
    """
    完整生成器结构（示例）
    
    :param style_dim: 风格向量维度
    :param n_blocks: 生成器块数量（决定输出分辨率）
    """
    def __init__(self, style_dim=512, n_blocks=5):
        super().__init__()
        # 初始常量输入
        self.initial_constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # 生成器块序列
        self.blocks = nn.ModuleList()
        channels = [512, 512, 256, 128, 64, 32]  # 示例通道数配置
        for i in range(n_blocks):
            in_c = channels[i]
            out_c = channels[i+1]
            self.blocks.append(
                GeneratorBlock(in_c, out_c, style_dim, upsample=(i>0))
            )
        
        # 最终 ToRGB 层
        self.to_rgb = ModulatedConv2d(channels[-1], 3, 1, style_dim, demodulate=False)
        
    def forward(self, style_w, noise=None):
        x = self.initial_constant.repeat(style_w.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, style_w, noise)
        x = self.to_rgb(x, style_w)
        return torch.tanh(x)  # 输出归一化到 [-1, 1]

# 2.1.生成器块
class GeneratorBlock(nn.Module):
    """
    StyleGAN2 生成器块
    
    :param in_channel: 输入通道数
    :param out_channel: 输出通道数
    :param style_dim: 风格向量维度
    :param upsample: 是否上采样输入
    """
    def __init__(self, in_channel, out_channel, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        # 调制卷积层（带解调）
        self.conv1 = ModulatedConv2d(in_channel, out_channel, 3, style_dim)
        self.conv2 = ModulatedConv2d(out_channel, out_channel, 3, style_dim)
        
        # 噪声注入
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))
        
        # AdaIN 层
        self.adain1 = AdaIN(out_channel, style_dim)
        self.adain2 = AdaIN(out_channel, style_dim)
        
        # 激活函数
        self.act = nn.LeakyReLU(0.2)
        
        # 上采样层（使用双线性插值避免棋盘效应）
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x, style_w, noise=None):
        """
        :param x: 输入特征图 (B, in_channel, H, W)
        :param style_w: 风格向量 (B, style_dim)
        :param noise: 可选噪声张量 (B, 1, H, W)
        :return: 输出特征图 (B, out_channel, 2H, 2W)（如果上采样）
        """
        if self.upsample:
            x = self.up(x)
            
        # 第一个卷积 + AdaIN
        x = self.conv1(x, style_w)
        x = self._inject_noise(x, noise, self.noise_scale1)
        x = self.act(x)
        x = self.adain1(x, style_w)
        
        # 第二个卷积 + AdaIN
        x = self.conv2(x, style_w)
        x = self._inject_noise(x, noise, self.noise_scale2)
        x = self.act(x)
        x = self.adain2(x, style_w)
        
        return x
    
    def _inject_noise(self, x, noise, scale):
        """ 注入可学习的缩放噪声 """
        if noise is None:
            B, _, H, W = x.shape
            noise = torch.randn(B, 1, H, W, device=x.device)
        return x + noise * scale
# 2.1.1 Mod std: 权重调制和解调卷积
class ModulatedConv2d(nn.Module):
    """
    调制卷积层（包含解调步骤）
    
    :param in_channel: 输入通道数
    :param out_channel: 输出通道数
    :param kernel_size: 卷积核大小
    :param style_dim: 风格向量 w 的维度
    :param demodulate: 是否进行解调（默认为 True）
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.eps = 1e-8
        self.demodulate = demodulate
        
        # 创建卷积权重（使用均衡初始化）
        self.weight = EqualizedWeight([out_channel, in_channel, kernel_size, kernel_size])
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size//2)
        
        # 调制网络（将风格向量 w 转换为缩放因子）
        self.modulation = EqualizedLinear(style_dim, in_channel, bias_init=1.0)
        
    def forward(self, x, style):
        """
        :param x: 输入特征图 (batch, in_channel, H, W)
        :param style: 风格向量 (batch, style_dim)
        :return: 输出特征图 (batch, out_channel, H, W)
        """
        batch, in_channel, H, W = x.shape
        
        # Step 1: 调制（Modulation）—— 用风格向量缩放权重
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  # (B, 1, in_c, 1, 1)
        weight = self.weight()[None, :, :, :, :]  # (1, out_c, in_c, k, k)
        weight = weight * style  # (B, out_c, in_c, k, k)
        
        # Step 2: 解调（Demodulation）—— 归一化权重防止激活爆炸
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum(dim=[2,3,4], keepdim=True) + self.eps)
            weight = weight * d
        
        # 重塑权重以进行分组卷积
        weight = weight.view(batch * self.conv.out_channels, in_channel, 
                           self.conv.kernel_size[0], self.conv.kernel_size[1])
        
        # 使用分组卷积实现调制后的卷积
        x = x.view(1, batch * in_channel, H, W)  # (1, B*in_c, H, W)
        x = F.conv2d(x, weight, padding=self.conv.padding[0], groups=batch)
        x = x.view(batch, self.conv.out_channels, H, W)
        
        return x
# 2.1.2 AdaIN: 自适应实例归一化
class AdaIN(nn.Module):
    """
    自适应实例归一化（Adaptive Instance Normalization）
    
    :param channels: 输入特征图的通道数
    :param style_dim: 风格向量 w 的维度
    """
    def __init__(self, channels, style_dim):
        super().__init__()
        # 用全连接层从 w 生成缩放因子（scale）和偏移（bias）
        self.scale = EqualizedLinear(style_dim, channels)
        self.bias = EqualizedLinear(style_dim, channels)
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        
    def forward(self, x, style):
        """
        :param x: 输入特征图 (B, C, H, W)
        :param style: 风格向量 (B, style_dim)
        :return: 归一化后的特征图 (B, C, H, W)
        """
        scale = self.scale(style).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        bias = self.bias(style).unsqueeze(2).unsqueeze(3)    # (B, C, 1, 1)
        x_norm = self.norm(x)
        return x_norm * scale + bias

# ===== 3.判别器 =====
# ===== 3.判别器 =====
# ===== 3.判别器 =====
# 3.0 判别器
class Discriminator(nn.Module):
    """
    StyleGAN2 判别器（适配灰度图像）
    
    :param in_channels: 输入图像通道数（灰度图为 1）
    :param base_channels: 基础通道数（默认 64，可随分辨率调整）
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # 初始 FromGray 层（将灰度图转换为特征图）
        self.from_gray = FromGray(in_channels, base_channels*8)
        
        # 残差块序列（分辨率逐步降低）
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(base_channels*8, base_channels*8, downsample=True),  # 64x64 → 32x32
            DiscriminatorBlock(base_channels*8, base_channels*4, downsample=True),  # 32x32 → 16x16
            DiscriminatorBlock(base_channels*4, base_channels*2, downsample=True),  # 16x16 → 8x8
            DiscriminatorBlock(base_channels*2, base_channels*1, downsample=True),  # 8x8 → 4x4
        ])
        
        # 最终迷你批标准差层（MiniBatch StdDev）
        self.final_conv = nn.Conv2d(base_channels*1 + 1, base_channels*1, 3, padding=1)  # +1 通道用于 MiniBatchStd
        self.final_fc = nn.Linear(base_channels*1 * 4 * 4, 1)
    
    # MiniBatchStdDev：迷你批标准差层（增强批多样性）
    def minibatch_stddev(self, x):
        """
        计算 MiniBatch 标准差并拼接至特征图
        """
        batch, channels, height, width = x.shape
        # 计算每个样本的标准差
        std = torch.std(x, dim=0, unbiased=False).mean()  # (1,)
        # 扩展为 (B, 1, H, W) 并拼接
        std = std.repeat(batch, 1, height, width)
        return torch.cat([x, std], dim=1)  # (B, C+1, H, W)
        
    def forward(self, x):
        """
        :param x: 输入灰度图 (B, 1, H, W)
        :return: 判别结果 (B, 1)
        """
        x = self.from_gray(x)  # (B, 512, H, W)
        
        for block in self.blocks:
            x = block(x)  # 逐步下采样
        
        # 添加 MiniBatch 标准差
        x = self.minibatch_stddev(x)  # (B, C+1, 4, 4)
        
        x = self.final_conv(x)        # (B, C, 4, 4)
        x = x.view(x.size(0), -1)     # (B, C*4*4)
        x = self.final_fc(x)          # (B, 1)
        return x

# 3.0.1 fromGray：灰度图到特征图的转换层
class FromGray(nn.Module):
    """
    灰度图到特征图的转换层（替代 FromRGB）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)  # 1x1 卷积
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.act(self.conv(x))  # (B, out_c, H, W)

# 3.1 判别器块
class DiscriminatorBlock(nn.Module):
    """
    判别器残差块
    
    :param in_ch: 输入通道数
    :param out_ch: 输出通道数
    :param downsample: 是否下采样
    """
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        
        # 下采样层（使用 BlurPool 抗锯齿） # （详见论文附录 B）
        self.downsample = nn.Sequential(
            BlurPool(out_ch), # 低通滤波
            nn.AvgPool2d(2)
        ) if downsample else None
        
        # 跳跃连接（匹配维度）
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        if self.downsample is not None:
            x = self.downsample(x)
            residual = self.downsample(self.skip(residual))
        else:
            residual = self.skip(residual)
            
        return (x + residual) / sqrt(2)  # 归一化残差连接
    
# 3.1.1 BlurPool：模糊池化层（抗锯齿下采样）
class BlurPool(nn.Module):
    """ 通过低通滤波后下采样，减少棋盘效应 """
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel[None, None, :, :] / 16.0  # 归一化
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1))
        self.pad = nn.ReplicationPad2d(1)
        
    def forward(self, x):
        # return F.conv2d(self.pad(x), self.kernel, groups=x.size(1), stride=2)
        return F.conv2d(self.pad(x), self.kernel, groups=x.size(1), stride=1)

