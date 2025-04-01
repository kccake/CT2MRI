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

from models import Reg, Generator, Discriminator, Transformer_3D
from utils import *

class Solver(object):
    def __init__(self, config):
        super().__init__()
        # 1.保留参数
        self.global_config = config
        self.config = config['solver']
        self.misc = config['misc']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2.定义模型
        # base
        self.netG_A2B = Generator(self.config['input_nc'], self.config['output_nc']).to(self.device)
        self.netD_B = Discriminator(self.config['input_nc']).to(self.device)
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        # regist
        self.R_A = Reg(32, self.config['size'], self.config['size'], self.config['input_nc'], self.config['input_nc']).to(self.device)
        self.spatial_transformer = Transformer_3D().to(self.device)
        self.optimizer_R_A = optim.Adam(self.R_A.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        # 备用
        # self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999)).to(self.device)
        # 3.断点恢复
        # TODO ↓ 定义断点恢复
        # (*Φ皿Φ*)
        # TODO ↑ 定义断点恢复
        
        # 4.损失函数
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        
        self.ssim = SSIM(data_range=255.0).to(self.device)
        self.psnr = PSNR(data_range=255.0).to(self.device)

        # 5.内/显存分配
        # self._allocate_memory() # 不做了，有点复杂
        Tensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False).to(self.device)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False).to(self.device)
        
        # 6.数据加载
        self.train_loader = DataLoader(ImagesDataset3D(config['dataset'], train=True))
        print(f'\033[1;34m[info]\033[0m train_loader已加载{len(self.train_loader)}个batch')
        self.val_loader = DataLoader(ImagesDataset3D(config['dataset'], train=True))
        print(f'\033[1;34m[info]\033[0m val_loader已加载{len(self.val_loader)}个batch')
        
    def _allocate_memory(self):
        Tensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        self.input_A = Tensor(self.config['batch_size'], self.config['input_nc'], 32, self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.input_B = Tensor(self.config['batch_size'], self.config['output_nc'], 32, self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False).to(self.device)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False).to(self.device)
    
    def train(self):
        self.start_time = time.time()
        best_SSIM = 0
        self.epoch_bar = tqdm(range(self.config['n_epochs']), desc='Training Progress', unit='epoch')
        
        for epoch in range(self.config['start_epoch'], self.config['n_epochs']):
            for batch_idx, batch_data in enumerate(self.train_loader):
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                # reggan training
                self.R_A.train()
                self.optimizer_R_A.zero_grad()
                # self.optimizer_G.zero_grad() # 备用
                
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B, real_B)
                SysRegist_A2B = self.spatial_transformer(fake_B, Trans)
                SR_loss = self.L1_loss(SysRegist_A2B, real_B) * self.config['SR_lambda']
                
                pred_fake = self.netD_B(fake_B)
                
                Adv_loss = self.MSE_loss(pred_fake, self.target_real)
                
                SM_loss = smooothing_loss(Trans) * self.config['Smooth_lambda']
                
                total_loss = SR_loss + Adv_loss + SM_loss
                if torch.isnan(total_loss).any():
                    continue
                
                total_loss.backward()
                self.optimizer_R_A.step()
                # self.optimizer_G.step() # 备用
                
                # discriminator training
                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = (self.MSE_loss(pred_real, self.target_real) + self.MSE_loss(pred_fake, self.target_fake)) * self.config['Adv_lambda']
                if torch.isnan(loss_D_B).any():
                    continue
                loss_D_B.backward()
                self.optimizer_D_B.step()
                
            self.epoch_bar.update(1)
            # TODO 评估 未完善
            if epoch % self.misc['eval_interval'] == 0:
                self.epoch_bar.write(f'\033[1;34m[info]\033[0m Evaluating...')
                train_metrics = self._evaluate(self.train_loader)
                self.epoch_bar.write(f'\033[1;34m[info]\033[0m epoch: {epoch}, train_metrics: {train_metrics}')
                # print(f'\033[1;34m[info]\033[0m epoch: {epoch}, train_metrics: {train_metrics}')
                # val_metrics = self._evaluate(self.val_loader)
                # print(f'\033[1;34m[info]\033[0m epoch: {epoch}, train_metrics: {train_metrics}, val_metrics: {val_metrics}')
                # if val_metrics['metrics']['ssim'] > best_SSIM:
                #     best_SSIM = val_metrics['metrics']['ssim']
                #     self.save_checkpoint(epoch, best_SSIM)
                self.epoch_bar.write(f'\033[1;34m[info]\033[0m Training...')
                
        self.epoch_bar.close()
        
    def _evaluate(self, dataloader):
        self.netG_A2B.eval()
        self.netD_B.eval()
        self.R_A.eval()
        
        metrics = defaultdict(list)
        losses = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                
                self.optimizer_R_A.zero_grad()
                
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B, real_B)
                SysRegist_A2B = self.spatial_transformer(fake_B, Trans)
                SR_loss = self.L1_loss(SysRegist_A2B, real_B) * self.config['SR_lambda']
                pred_fake = self.netD_B(fake_B)
                Adv_loss = self.MSE_loss(pred_fake, self.target_real)
                SM_loss = smooothing_loss(Trans) * self.config['Smooth_lambda']
                total_loss = SR_loss + Adv_loss + SM_loss
                if torch.isnan(total_loss).any():
                    continue
                losses['SR_loss'].append(SR_loss)
                losses['adv_loss'].append(Adv_loss)
                losses['SM_loss'].append(SM_loss)
                losses['total_loss'].append(total_loss)
                
                metrics['ssim'].append(self.ssim(SysRegist_A2B, real_B))
                metrics['psnr'].append(self.psnr(SysRegist_A2B, real_B))
        # 对字典中的所有元素取平均
        losses = {key: torch.mean(torch.stack(value)).item() for key, value in losses.items()}
        metrics = {key: torch.mean(torch.stack(value)).item() for key, value in metrics.items()}
        return {
            'losses' : losses,
            'metrics' : metrics
        }