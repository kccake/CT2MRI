import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class ConditionedUNet3D(nn.Module):
    """带时间步和CT条件输入的3D U-Net"""
    def __init__(self):
        super().__init__()
        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        
        # 编码器（修改为接收2通道输入：混合图像+CT）
        self.encoder1 = EncoderBlock(2, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        # 解码器（添加时间条件）
        self.decoder1 = DecoderBlockWithCond(512, 256)
        self.decoder2 = DecoderBlockWithCond(256, 128)
        self.decoder3 = DecoderBlockWithCond(128, 64)
        self.decoder4 = DecoderBlockWithCond(64, 32)
        
        # 最终输出层
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x, ct, t):
        # 拼接混合图像和CT作为输入
        x = torch.cat([x, ct], dim=1)  # (B,2,D,H,W)
        
        # 时间嵌入
        t_emb = self.time_embed(t[:, None].float())  # (B,256)
        t_emb = rearrange(t_emb, "b c -> b c 1 1 1")
        
        # 编码过程
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        
        # 解码过程（注入时间信息）
        x = self.decoder1(x + t_emb, s4)
        x = self.decoder2(x + t_emb, s3)
        x = self.decoder3(x + t_emb, s2)
        x = self.decoder4(x + t_emb, s1)
        
        return self.final_conv(x)

class DiffusionCT2MR:
    def __init__(self):
        self.model = ConditionedUNet3D()
        self.num_timesteps = 1000
        self.alpha_schedule = self._create_schedule()
    
    def _create_schedule(self):
        """创建混合系数调度表"""
        return torch.linspace(0, 1, self.num_timesteps)
    
    def forward_process(self, mr, ct, t):
        """正向混合过程"""
        alpha = self.alpha_schedule[t]
        mixed = (1-alpha)*mr + alpha*ct
        return mixed
    
    def train_step(self, mr, ct):
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (mr.shape[0],))
        
        # 生成混合图像
        mixed = self.forward_process(mr, ct, t)
        
        # 网络预测
        pred_mr = self.model(mixed, ct, t/self.num_timesteps)
        
        # 计算损失
        loss = F.l1_loss(pred_mr, mr)
        return loss
    
    @torch.no_grad()
    def sample(self, ct, steps=100):
        # 初始化从CT开始
        x_t = ct.clone()
        
        # 反向过程
        for t in reversed(range(steps)):
            alpha = self.alpha_schedule[t]
            pred = self.model(x_t, ct, torch.tensor([t/steps]))
            
            # 逐步去除CT成分
            x_t = (x_t - alpha*ct) / (1 - alpha + 1e-8) * (1 - alpha_prev)
            x_t = x_t + alpha_prev * ct
            
        return x_t

# 改进的解码器块（带条件注入）
class DecoderBlockWithCond(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels*2, out_channels, 3, padding=1),
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