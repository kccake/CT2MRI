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

class RegGAN3DSolver(object):
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
        self.netG = Generator(self.config['input_nc'], self.config['output_nc']).to(self.device)
        self.netD = Discriminator(self.config['input_nc']).to(self.device)
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        # regist
        self.netR = Reg(32, self.config['size'], self.config['size'], self.config['input_nc'], self.config['input_nc']).to(self.device)
        self.spatial_transformer = Transformer_3D().to(self.device)
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
        self.train_loader = DataLoader(ImagesDataset3D(config['dataset'], train=True))
        print(f'\033[1;34m[info]\033[0m train_loader已加载 \033[32m{len(self.train_loader)}\033[0m 个batch')
        self.val_loader = DataLoader(ImagesDataset3D(config['dataset'], train=False))
        print(f'\033[1;34m[info]\033[0m val_loader已加载 \033[32m{len(self.val_loader)}\033[0m 个batch')
        
    def _allocate_memory(self):
        Tensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        self.input_A = Tensor(self.config['batch_size'], self.config['input_nc'], 32, self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.input_B = Tensor(self.config['batch_size'], self.config['output_nc'], 32, self.config['size'], self.config['size']) # torch.Size([1, 1, 32, 256, 256])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False).to(self.device)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False).to(self.device)
    
    def train(self):
        best_SSIM = self.start_SSIM if self.start_SSIM is not None else 0.0
        self.start_time = time.time()
        
        self.epoch_bar = tqdm(total=self.config['n_epochs'], desc='Training Progress', unit='epoch', position=0)
        self.epoch_bar.update(self.config['start_epoch']) # 更新到start_epoch
        
        for self.epoch in range(self.config['start_epoch'], self.config['n_epochs']):
            self.batch_bar = tqdm(total=len(self.train_loader), desc='Batch Progress', unit='batch', position=1, leave=False)
            for batch_idx, batch_data in enumerate(self.train_loader):
                # print(f'\033[1;33m[debug]\033[0m batch_idx: {batch_idx}, real_A.shape: {batch_data["CT"].shape}, real_B.shape: {batch_data["MR"].shape}')
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                # ===== reggan & generator training =====
                self.netR.train()
                self.netG.train()
                self.netD.eval()
                self.optimizer_R.zero_grad()
                self.optimizer_G.zero_grad()
                
                fake_B = self.netG(real_A)
                Trans = self.netR(fake_B, real_B)
                SysRegist_A2B = self.spatial_transformer(fake_B, Trans)
                SR_loss = self.L1_loss(SysRegist_A2B, real_B) * self.config['SR_lambda']
                
                pred_fake = self.netD(fake_B)
                
                Adv_loss = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lambda']
                
                SM_loss = smooothing_loss(Trans) * self.config['SM_lambda']
                
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
            # TODO 评估 未完善
            if self.epoch % self.misc['eval_interval'] == 0:
                tqdm.write(f'\033[1;34m[info]\033[0m Evaluating...')
                
                train_metrics = self._evaluate(self.val_loader)
                tqdm.write(f'\033[1;34m[info]\033[0m epoch: {self.epoch}, train_metrics: {train_metrics}')
                # 保存最好的模型
                if train_metrics['metrics']['ssim'] > best_SSIM:
                    best_SSIM = train_metrics['metrics']['ssim']
                    self.save_checkpoint(self.epoch, best_SSIM)
                
                tqdm.write(f'\033[1;34m[info]\033[0m Training...')
                
        self.epoch_bar.close()
        
    def _evaluate(self, dataloader):
        self.netG.eval()
        self.netD.eval()
        self.netR.eval()
        
        metrics = defaultdict(list)
        losses = defaultdict(list)
        
        with torch.no_grad():
            self.eval_bar = tqdm(total=len(dataloader), desc='Eval Progress', unit='batch', position=1, leave=False)
            for batch_idx, batch_data in enumerate(dataloader):
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                
                self.optimizer_R.zero_grad()
                
                fake_B = self.netG(real_A)
                Trans = self.netR(fake_B, real_B)
                SysRegist_A2B = self.spatial_transformer(fake_B, Trans)
                SR_loss = self.L1_loss(SysRegist_A2B, real_B) * self.config['SR_lambda']
                pred_fake = self.netD(fake_B)
                Adv_loss = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lambda']
                SM_loss = smooothing_loss(Trans) * self.config['SM_lambda']
                total_loss = SR_loss + Adv_loss + SM_loss
                if torch.isnan(total_loss).any():
                    continue
                losses['SR_loss'].append(SR_loss)
                losses['adv_loss'].append(Adv_loss)
                losses['SM_loss'].append(SM_loss)
                losses['total_loss'].append(total_loss)
                
                metrics['ssim'].append(self.ssim(fake_B, real_B))
                metrics['psnr'].append(self.psnr(fake_B, real_B))
                self.eval_bar.update(1)
            self.eval_bar.close()
            
            # 保存最后一个的real_A, real_B, fake_B
            sample_dir = Path(self.misc['log_root']) / self.global_config['name'] / self.misc['log_dir_names']['sample']
            os.makedirs(sample_dir, exist_ok=True)
            # 数据是(B,1,32,256,256)的
            real_A = real_A[0, 0].cpu().numpy()
            real_B = real_B[0, 0].cpu().numpy()
            fake_B = fake_B[0, 0].cpu().numpy()
            # 保存为npy
            np.save(sample_dir / f'real_A_{self.epoch}_{batch_idx}.npy', real_A)
            np.save(sample_dir / f'real_B_{self.epoch}_{batch_idx}.npy', real_B)
            np.save(sample_dir / f'fake_B_{self.epoch}_{batch_idx}.npy', fake_B)
            
        # 对字典中的所有元素取平均
        losses = {key: torch.mean(torch.stack(value)).item() for key, value in losses.items()}
        metrics = {key: torch.mean(torch.stack(value)).item() for key, value in metrics.items()}
        metric_loss = {
            'losses' : losses,
            'metrics' : metrics
        }
        
        log_dir = Path(self.misc['log_root']) / self.global_config['name'] / self.misc['log_dir_names']['loss']
        os.makedirs(log_dir, exist_ok=True)
        # 保存metric_loss为json
        with open(log_dir / f'metric_loss_{self.epoch}.json', 'w') as f:
            json.dump(metric_loss, f)
        
        return metric_loss
    
    def save_checkpoint(self, epoch, best_SSIM):
        checkpoint_root = Path(self.misc['log_root']) / self.global_config['name'] / self.misc['log_dir_names']['checkpoint']
        os.makedirs(checkpoint_root, exist_ok=True)
        filepath = checkpoint_root / f'epoch_{epoch}.ckpt'
        torch.save({
            'best_SSIM': best_SSIM,
            'epoch': epoch,
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            
            'netR': self.netR.state_dict(),
            'spatial_transformer': self.spatial_transformer.state_dict(),
            'optimizer_R': self.optimizer_R.state_dict(),
        }, filepath)
        tqdm.write(f'\033[1;32m[success]\033[0m 保存模型到 {filepath}')
        