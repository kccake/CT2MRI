import glob
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImagesDataset2D(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.preload = config['preload']
        if not self.preload:
            print(f'\033[1;34m[INFO]\033[0m ImagesDataset2D 暂时不支持 非preload 模式。将自动转为\033[34m preload \033[0m模式')
            self.preload = True
        if self.train:
            self.CT_roots = config['trainCT_root']
            self.MR_roots = config['trainMR_root']
        else:
            self.CT_roots = config['testCT_root']
            self.MR_roots = config['testMR_root']
        
        self.CT_paths = []
        self.MR_paths = []
        for root in self.CT_roots:
            self.CT_paths += glob.glob(f'{root}/*')
        for root in self.MR_roots:
            self.MR_paths += glob.glob(f'{root}/*')
        
        # Preload images
        if self.preload:
            self.CT_images = []
            self.MR_images = []
            for path in self.CT_paths:
                CT_image = torch.from_numpy(np.load(path).astype(np.float32))
                self.CT_images.append(CT_image)

            # print(f'\033[1;33m[DEBUG]\033[0m {self.CT_images[0].shape}') # torch.Size([32, 256, 256])
            # concat 到一起，变成 len*32, 256, 256
            self.CT_images = torch.cat(self.CT_images, dim=0)
            for path in self.MR_paths:
                MR_image = torch.from_numpy(np.load(path).astype(np.float32))
                self.MR_images.append(MR_image)
            self.MR_images = torch.cat(self.MR_images, dim=0)
            print(f'\033[1;34m[INFO]\033[0m \033[32m{len(self.CT_images)}\033[0m CT images and {len(self.MR_images)} MR images are\033[34m preloaded\033[0m.')
        # not Preload images
        else:
            print(f'\033[1;34m[INFO]\033[0m \033[32m{len(self.CT_paths)}\033[0m CT images and {len(self.MR_paths)} MR images are\033[34m founded.')
    
    def __getitem__(self, index):
        if self.preload:
            CT_image = self.CT_images[index]
            MR_image = self.MR_images[index]
        else:
            CT_image = torch.from_numpy(np.load(self.CT_paths[index]).astype(np.float32))
            MR_image = torch.from_numpy(np.load(self.MR_paths[index]).astype(np.float32))
        
        return {'CT': CT_image, 'MR': MR_image} # 每个都是 torch.Size([256, 256])
    
    def __len__(self):
        return max(len(self.CT_paths), len(self.MR_paths))