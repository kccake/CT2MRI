# system
import os

# torch
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
# local
from .layer2D import DownBlock2d, Conv2d, ResnetTransformer2d
sampling_align_corners = False

ndf = {'A': [32, 64, 64, 64, 64], }

nuf = {'A': [64, 64, 64, 64, 32], }

user_down_resblocks = {'A': True, }

resnet_nblocks = {'A': 3, }

refine_output = {'A': True, }

down_activation = {'A': 'leaky_relu', }

up_activation = {'A': 'leaky_relu', }

class ResUnet2d(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet2d, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock2d(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=user_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if user_down_resblocks[cfg]:
            self.c1 = Conv2d(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                            use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer2d(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv2d(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                            use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv2d(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                         use_resnet=False, use_norm=False))
            in_nf = out_nf
            conv_num -= 1 # += 1？
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer2d(in_nf, 1, init_func),
                                        Conv2d(in_nf, nc_a, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                                               use_resnet=False, use_norm=False))
        else:
            self.refine = lambda x: x
        self.output = Conv2d(in_nf, 3, 3, 1, 1, use_resnet=False, bias=True,
                            init_func=('zeros' if init_to_identity else init_func), activation=None,
                            use_norm=False)
    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=sampling_align_corners)
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        return self.output(x)
    
class Reg2D(nn.Module):
    def __init__(self, height, width, in_channels_a, in_channels_b):
        super(Reg2D, self).__init__()
        init_func = 'kaiming'
        init_to_identity = True
        
        self.oh, self.ow = height, width
        self.nc_a = in_channels_a
        self.nc_b = in_channels_b
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.offset_map = ResUnet2d(self.nc_a, self.nc_b, cfg='A', init_func=init_func, init_to_identity=init_to_identity).to(self.device)
        self.identity_grid = self.get_identity_grid()
        
    def get_identity_grid(self):
        y = torch.linspace(-1.0, 1.0, self.oh)
        x = torch.linspace(-1.0, 1.0, self.ow)
        yy, xx = torch.meshgrid([y, x])
        yy = yy.unsqueeze(dim=0)
        xx = xx.unsqueeze(dim=0)
        identity = torch.cat((xx, yy), dim=0).unsqueeze(0)
        return identity
    
    def forward(self, img_a, img_b, apply_on=None):
        deformations = self.offset_map(img_a, img_b)
        return deformations