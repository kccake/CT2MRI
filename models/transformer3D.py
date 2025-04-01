# torch
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

class Transformer_3D(nn.Module):
    def __init__(self):
        super(Transformer_3D, self).__init__()

    def forward(self, src, flow):
        b = flow.shape[0]
        d = flow.shape[2]
        h = flow.shape[3]
        w = flow.shape[4]

        size = (d, h, w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1).cuda()
        new_locs = grid + flow

        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]  # Adjust for 3D coordinates
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")

        return warped



