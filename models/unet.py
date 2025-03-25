import torch
import torch.nn as nn
import torch.nn.functional as F
# ===== 生成器 ===== 
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels//2 + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet3DGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        # Final output
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder1(x, s4)
        x = self.decoder2(x, s3)
        x = self.decoder3(x, s2)
        x = self.decoder4(x, s1)
        
        # Output
        x = self.final_conv(x)
        return self.tanh(x)

#  ===== 判别器 =====
class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
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

class StarDiscriminator3D(nn.Module):
    """Discriminator network with C×D×H×W input format."""
    def __init__(self, conv_dim=64, repeat_num=5):
        super(StarDiscriminator3D, self).__init__()
        layers = []
        # Initial layer (halves H, W, D)
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        # Second layer (halves H, W, D again)
        layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim *= 2

        # Subsequent layers (adjust kernel/stride for D×H×W format)
        for _ in range(2, repeat_num):
            # Kernel: (D=3, H=4, W=4), Stride: (D=1, H=2, W=2)
            layers.append(nn.Conv3d(
                curr_dim, curr_dim*2, 
                kernel_size=(3, 4, 4),    # Adjusted for D×H×W
                stride=(1, 2, 2),          # Stride applied to H/W only
                padding=(1, 1, 1)          # Padding matches kernel
            ))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        self.main = nn.Sequential(*layers)
        # Final layer (no spatial reduction)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Input shape: (batch, C=1, D, H, W)
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src
