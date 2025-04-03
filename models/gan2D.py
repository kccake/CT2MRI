# gan2D.py
# -*- coding: utf-8 -*-
# @Time    : 2025/4/2 20:00
# @Author  : XingYueChenFu

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== 1. 生成器 =====
class ResidualBlock2D(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock2D, self).__init__()
        
        conv_block = [nn.ReplicationPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReplicationPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
        
        def forward(self, x):
            return x + self.conv_block(x)

class Generator2D(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator2D, self).__init__()
        
        # Initial convolution block
        model_head = [
            nn.ReplicationPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock2D(in_features)]
        
        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model_tail += [nn.ReplicationPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]
        
        # self.model = nn.Sequential(*model_head + model_body + model_tail)
        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)
        
    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        
        return x
        # return self.model(x)
        
    def get_feature(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        
        return x

# ====== 2. 判别器 =====
class Discriminator2D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator2D, self).__init__()
        
        # Convolutional layers
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(3): # 64 -> 128 -> 256 -> 512
            model += [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(0.2, inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        
        model += [nn.Conv2d(in_features, out_features, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(out_features),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Output layer
        model += [nn.Conv2d(in_features, 1, 4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)