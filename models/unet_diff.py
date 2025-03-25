# unet_diff.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        return self.time_mlp(t.unsqueeze(-1))


class ConditionalEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)

class ConditionalDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DiffusionUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入通道调整
        self.init_conv = nn.Conv3d(2, 64, 3, padding=1)
        
        # 时间嵌入（输出通道调整为64）
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # 编码器
        self.encoder1 = ConditionalEncoderBlock(64, 64)
        self.encoder2 = ConditionalEncoderBlock(64, 128)
        self.encoder3 = ConditionalEncoderBlock(128, 256)
        self.encoder4 = ConditionalEncoderBlock(256, 512)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.GroupNorm(16, 1024),
            nn.SiLU(),
            nn.Conv3d(1024, 1024, 3, padding=1),
            nn.GroupNorm(16, 1024),
            nn.SiLU()
        )
        
        # 解码器
        self.decoder1 = ConditionalDecoderBlock(1024, 512, 512)  # 输入1024, skip512 → 输出512
        self.decoder2 = ConditionalDecoderBlock(512, 256, 256)   # 输入512, skip256 → 输出256
        self.decoder3 = ConditionalDecoderBlock(256, 128, 128)   # 输入256, skip128 → 输出128
        self.decoder4 = ConditionalDecoderBlock(128, 64, 64)     # 输入128, skip64 → 输出64
        
        # 最终输出
        self.final_conv = nn.Conv3d(64, 1, 1)

    def forward(self, x, ct, t):
        # 拼接输入
        x = torch.cat([x, ct], dim=1)  # [B,2,D,H,W]
        x = self.init_conv(x)          # [B,64,D,H,W]
        
        # 时间嵌入处理
        t_emb = self.time_embed(t.view(-1,1)).view(-1,64,1,1,1)  # 调整为64通道
        x = x + t_emb  # 现在维度匹配
        
        # 编码过程
        s1, x = self.encoder1(x)  # s1: [B,64,D,H,W]
        s2, x = self.encoder2(x)  # s2: [B,128,D/2,H/2,W/2]
        s3, x = self.encoder3(x)  # s3: [B,256,D/4,H/4,W/4]
        s4, x = self.encoder4(x)  # s4: [B,512,D/8,H/8,W/8]
        
        # 瓶颈层
        x = self.bottleneck(x)    # [B,1024,D/16,H/16,W/16]
        
        # 解码过程
        x = self.decoder1(x, s4)  # [B,512,D/8,H/8,W/8]
        x = self.decoder2(x, s3)  # [B,256,D/4,H/4,W/4]
        x = self.decoder3(x, s2)  # [B,128,D/2,H/2,W/2]
        x = self.decoder4(x, s1)  # [B,64,D,H,W]
        
        return self.final_conv(x) # [B,1,D,H,W]