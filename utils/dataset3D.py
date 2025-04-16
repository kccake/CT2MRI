import glob

import numpy as np
import torch
from torch.utils.data import Dataset

class ImagesDataset3D(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.preload = config['preload']
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
                if len(CT_image.shape) == 3: # 如果是3D图像，增加一个维度
                    CT_image = CT_image.unsqueeze(0)
                self.CT_images.append(CT_image) # torch.Size([1, 32, 256, 256])
            for path in self.MR_paths:
                MR_image = torch.from_numpy(np.load(path).astype(np.float32))
                # 如果是3D图像，增加一个维度
                if len(MR_image.shape) == 3:
                    MR_image = MR_image.unsqueeze(0)
                self.MR_images.append(MR_image) # torch.Size([1, 32, 256, 256])
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_images)}\033[0m CT images and \033[32m{len(self.MR_images)}\033[0m MR images are\033[34m preloaded\033[0m.')
        # not Preload images
        else:
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_paths)}\033[0m CT images and \033[32m{len(self.MR_paths)}\033[0m  MR images are\033[34m founded.')
    
    def __getitem__(self, index):
        if self.preload:
            CT_image = self.CT_images[index]
            MR_image = self.MR_images[index]
        else:
            CT_image = torch.from_numpy(np.load(self.CT_paths[index]).astype(np.float32))
            MR_image = torch.from_numpy(np.load(self.MR_paths[index]).astype(np.float32))
        
        return {'CT': CT_image, 'MR': MR_image} # 每个都是 torch.Size([1, 32, 256, 256])
    
    def __len__(self):
        return max(len(self.CT_paths), len(self.MR_paths))


class CycleGANDataset3D(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.preload = config['preload']
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
            self.CT_images_2 = []
            self.MR_images_255 = []
            
            for path in self.CT_paths:
                CT_image = torch.from_numpy(np.load(path).astype(np.float32))
                if len(CT_image.shape) == 3: # 如果是3D图像，增加一个维度
                    CT_image = CT_image.unsqueeze(0)
                self.CT_images.append(CT_image) # torch.Size([1, 32, 256, 256])
                # 从(0~255)放缩到(-1~1)
                # CT_image_2 = (CT_image - 127.5) / 127.5
                # CT_image_2 = CT_image_2.clamp(-1, 1)
                CT_image_2 = (CT_image - CT_image.min()) / (CT_image.max() - CT_image.min()) * 2 - 1 # 这个需要特殊一点,分布排满-1~1
                self.CT_images_2.append(CT_image_2) # torch.Size([1, 32, 256, 256])
            for path in self.MR_paths:
                MR_image = torch.from_numpy(np.load(path).astype(np.float32))
                # 如果是3D图像，增加一个维度
                if len(MR_image.shape) == 3:
                    MR_image = MR_image.unsqueeze(0)
                self.MR_images.append(MR_image) # torch.Size([1, 32, 256, 256])
                # 从(-1~1)放缩到(0~255)
                MR_image_255 = (MR_image + 1) * 127.5
                MR_image_255 = MR_image_255.clamp(0, 255)
                self.MR_images_255.append(MR_image_255)

            # 输出每种CT与MR[0]的最小最大值
            print(f'CT_255[0] [{self.CT_images[0].min()} ~ {self.CT_images[0].max()}]')
            print(f'MR_255[0] [{self.MR_images_255[0].min()} ~ {self.MR_images_255[0].max()}]')
            print(f'CT_2[0] [{self.CT_images_2[0].min()} ~ {self.CT_images_2[0].max()}]')
            print(f'MR_2[0] [{self.MR_images[0].min()} ~ {self.MR_images[0].max()}]')
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_images)}\033[0m CT images and \033[32m{len(self.MR_images)}\033[0m MR images are\033[34m preloaded\033[0m.')
        # not Preload images
        else:
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_paths)}\033[0m CT images and \033[32m{len(self.MR_paths)}\033[0m  MR images are\033[34m founded.')
    
    def __getitem__(self, index):
        if self.preload:
            CT_image = self.CT_images[index]
            MR_image = self.MR_images[index]
            CT_image_2 = self.CT_images_2[index]
            MR_image_255 = self.MR_images_255[index]
        else:
            CT_image = torch.from_numpy(np.load(self.CT_paths[index]).astype(np.float32))
            MR_image = torch.from_numpy(np.load(self.MR_paths[index]).astype(np.float32))
            # CT_image_2 = (CT_image - 127.5) / 127.5
            # CT_image_2 = CT_image_2.clamp(-1, 1)
            CT_image_2 = (CT_image - CT_image.min()) / (CT_image.max() - CT_image.min()) * 2 - 1 # 这个需要特殊一点,分布排满-1~1
            MR_image_255 = (MR_image + 1) * 127.5
            MR_image_255 = MR_image_255.clamp(0, 255)
        
        # return {'CT': CT_image, 'MR': MR_image} # 每个都是 torch.Size([1, 32, 256, 256])
        return {'CT': CT_image, 'MR': MR_image, 'CT_2': CT_image_2, 'MR_255': MR_image_255}
    
    def __len__(self):
        return max(len(self.CT_paths), len(self.MR_paths))
    
class NewImagesDataset3D(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.preload = config['preload']
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
                CT_image = (CT_image - 127.5) / 127.5
                # CT_image = (CT_image- CT_image.min()) / (CT_image.max() - CT_image.min()) * 2 - 1 # 这个需要特殊一点,分布排满-1~1
                CT_image = np.tanh(CT_image-127.5)

                if len(CT_image.shape) == 3: # 如果是3D图像，增加一个维度
                    CT_image = CT_image.unsqueeze(0)
                self.CT_images.append(CT_image) # torch.Size([1, 32, 256, 256])
            for path in self.MR_paths:
                MR_image = torch.from_numpy(np.load(path).astype(np.float32))
                # 如果是3D图像，增加一个维度
                if len(MR_image.shape) == 3:
                    MR_image = MR_image.unsqueeze(0)
                self.MR_images.append(MR_image) # torch.Size([1, 32, 256, 256])
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_images)}\033[0m CT images and \033[32m{len(self.MR_images)}\033[0m MR images are\033[34m preloaded\033[0m.')
        # not Preload images
        else:
            print(f'\033[1;34m[info]\033[0m \033[32m{len(self.CT_paths)}\033[0m CT images and \033[32m{len(self.MR_paths)}\033[0m  MR images are\033[34m founded.')
    
    def __getitem__(self, index):
        if self.preload:
            CT_image = self.CT_images[index]
            MR_image = self.MR_images[index]
        else:
            CT_image = torch.from_numpy(np.load(self.CT_paths[index]).astype(np.float32))
            CT_image = (CT_image - 127.5) / 127.5
            # CT_image = (CT_image- CT_image.min()) / (CT_image.max() - CT_image.min()) * 2 - 1
            CT_image = np.tanh(CT_image-127.5) # tanh放缩到-1~1
            MR_image = torch.from_numpy(np.load(self.MR_paths[index]).astype(np.float32))
            if len(CT_image.shape) == 3: # 如果是3D图像，增加一个维度
                CT_image = CT_image.unsqueeze(0)
            
            if len(MR_image.shape) == 3:
                MR_image = MR_image.unsqueeze(0)
        
        return {'CT': CT_image, 'MR': MR_image} # 每个都是 torch.Size([1, 32, 256, 256])
    
    def __len__(self):
        return max(len(self.CT_paths), len(self.MR_paths))