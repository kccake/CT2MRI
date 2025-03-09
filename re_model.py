import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class StarGenerator3D(nn.Module):
    """Generator network with C×D×H×W input/output format."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(StarGenerator3D, self).__init__()
        layers = []
        # Initial layer (D, H, W are preserved due to padding)
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers (halves D, H, W each time)
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # Bottleneck layers (no spatial change)
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers (doubles D, H, W each time)
        for _ in range(2):
            layers.append(nn.ConvTranspose3d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        # Output layer
        layers.append(nn.Conv3d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, C=1, D, H, W)
        return self.model(x)

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