from models import *
from utils import *

import os
import time
import itertools
from collections import defaultdict
from pathlib import Path

import json
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

# from models import Reg, Generator, Discriminator, Transformer_3D
from models import Reg2D, Generator2D, Discriminator2D, Transformer_2D
from utils import *

class RegGAN2DSolver(object):
    def __init__(self, config):
        super().__init__()
    # 1.保留参数
        self.global_config = config
        self.config = config['solver']
        self.misc = config['misc']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_SSIM = None
        # 2.定义模型
        # base
        self.netG = Generator2D(self.config['input_nc'], self.config['output_nc']).to(self.device)
        self.netD = Discriminator2D(self.config['input_nc']).to(self.device)
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        # regist
        self.netR = Reg2D(self.config['size'], self.config['size'], self.config['input_nc'], self.config['input_nc']).to(self.device)
        self.spatial_transformer = Transformer_2D().to(self.device)
        self.optimizer_R = optim.Adam(self.netR.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        
        # 3.断点恢复
        if self.config['start_epoch'] > 0:
            checkpoint_root = Path(self.misc['log_root']) / self.global_config['name'] / self.misc['log_dir_names']['checkpoint']
            filepath = checkpoint_root / f'epoch_{self.config["start_epoch"]}.ckpt'
            try:
                checkpoint = torch.load(filepath)
                self.start_SSIM = checkpoint['best_SSIM']
                self.netG.load_state_dict(checkpoint['netG'])
                self.netD.load_state_dict(checkpoint['netD'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                if 'R' in checkpoint: # 之前代码的坑
                    self.netR.load_state_dict(checkpoint['R'])
                else:
                    self.netR.load_state_dict(checkpoint['netR'])
                self.spatial_transformer.load_state_dict(checkpoint['spatial_transformer'])
                self.optimizer_R.load_state_dict(checkpoint['optimizer_R'])
            except FileNotFoundError:
                print(f'\033[1;31m[error]\033[0m 未能找到断点恢复文件，从0开始训练')
                self.config['start_epoch'] = 0
            except:
                print(f'\033[1;31m[error]\033[0m 断点恢复文件损坏或加载异常，从0开始训练')
                self.config['start_epoch'] = 0
            else:
                print(f'\033[1;32m[success]\033[0m 断点恢复成功，从 {self.config['start_epoch']}代 继续训练')             
        # (*Φ皿Φ*)
        
        # 4.损失函数
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        
        self.ssim = SSIM(data_range=2.0).to(self.device)
        self.psnr = PSNR(data_range=2.0).to(self.device)

        # 5.内/显存分配
        # self._allocate_memory() # 不做了，有点复杂
        Tensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False).to(self.device)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False).to(self.device)
        
        # 6.数据加载
        self.train_loader = DataLoader(ImagesDataset2D(config['dataset'], train=True))
        print(f'\033[1;34m[info]\033[0m train_loader已加载 \033[32m{len(self.train_loader)}\033[0m 个batch')
        self.val_loader = DataLoader(ImagesDataset2D(config['dataset'], train=False))
        print(f'\033[1;34m[info]\033[0m val_loader已加载 \033[32m{len(self.val_loader)}\033[0m 个batch')
        
    def _allocate_memory(self):
        Tensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        self.input_A = Tensor(self.config['batch_size'], self.config['input_nc'], self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.input_B = Tensor(self.config['batch_size'], self.config['output_nc'], self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False).to(self.device)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False).to(self.device)
        
    def train(self):
        best_SSIM = self.start_SSIM if self.start_SSIM is not None else 0.0
        self.start_time = time.time()
        
        self.epoch_bar = tqdm(total=self.config['n_epochs'], desc='Training Progress', unit='epoch', position=0)
        self.epoch_bar.update(self.config['start_epoch']) # 更新到start_epoch
        
        for self.epoch in range(self.config['start_epoch'], self.config['n_epochs']):
            self.batch_bar = tqdm(total=len(self.train_loader), desc='Batch Progress', unit='batch', position=1)
            for batch_idx, batch_data in enumerate(self.train_loader):
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                
                # ===== regist & generator 训练 =====
                self.netR.train()
                self.netG.train()
                self.netD.eval()
                self.optimizer_R.zero_grad()
                self.optimizer_G.zero_grad()
                
                #
                # print(f'\033[1;33m [DEBUG]\033[0m batch_idx: {batch_idx}, real_A: {real_A.shape}, real_B: {real_B.shape}')
                fake_B = self.netG(real_A)
                Trans = self.netR(fake_B, real_B)
                SysRegist_A2B = self.spatial_transformer(fake_B, Trans)
                SR_loss = self.L1_loss(SysRegist_A2B, real_B) * self.config['SR_lambda']
                
                pred_fake = self.netD(fake_B)
                
                Adv_loss = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lambda']
                
                SM_loss = smooothing_loss2D(Trans) * self.config['SM_lambda']
                
                total_loss = SR_loss + Adv_loss + SM_loss
                if torch.isnan(total_loss).any():
                    continue
                
                total_loss.backward()
                self.optimizer_R.step()
                self.optimizer_G.step()
                
                # ===== discriminator training =====
                self.netD.train()
                self.netG.eval()
                self.optimizer_D.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG(real_A)
                pred_fake = self.netD(fake_B)
                pred_real = self.netD(real_B)
                loss_D = (self.MSE_loss(pred_real, self.target_real) + self.MSE_loss(pred_fake, self.target_fake)) * self.config['Adv_lambda']
                if torch.isnan(loss_D).any():
                    continue
                loss_D.backward()
                self.optimizer_D.step()
                
                self.batch_bar.update(1)
            self.batch_bar.close()
                
            self.epoch_bar.update(1)
            
            # 评估
            
        self.epoch_bar.close()