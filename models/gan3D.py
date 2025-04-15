from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# local
from .layers3D import DownBlock3d, Conv3d, ResnetTransformer3d


# ===== 1. 生成器 =====
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model_head = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(input_nc, 64, 7),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm3d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm3d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReplicationPad3d(3),
                       nn.Conv3d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x
    
    def get_feature(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        
        return x

# ===== 2. 判别器 =====
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv3d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv3d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm3d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv3d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm3d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv3d(256, 512, 4, padding=1),
                  nn.InstanceNorm3d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv3d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)

# ==== 2.2 对应16切片的判别器 ====
class NewDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(NewDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 根据输入深度决定是否使用第三层之后的卷积
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, 4, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv5 = nn.Conv3d(512, 1, 4, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 只在深度维度足够大时应用第三层及之后的卷积
        if x.size(2) >= 4:
            x = self.conv3(x)
        if x.size(2) >= 4:
            x = self.conv4(x)
        if x.size(2) >= 4:  
            x = self.conv5(x)
        
        return F.adaptive_avg_pool3d(x, (1,1,1)).view(x.size()[0], -1)

