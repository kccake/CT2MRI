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

from models import Reg, Generator, NewDiscriminator, Transformer_3D
from utils import *

class CycleGAN3DSolver(object):
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
        self.netG_A2B = Generator(self.config['input_nc'], self.config['output_nc']).to(self.device)
        self.netD_B = NewDiscriminator(self.config['output_nc']).to(self.device)
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        # cycle
        self.netG_B2A = Generator(self.config['output_nc'], self.config['input_nc']).to(self.device)
        self.netD_A = NewDiscriminator(self.config['input_nc']).to(self.device)
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netD_B.parameters()), lr=self.config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        

        # 3.断点恢复
        if self.config['start_epoch'] > 0:
            checkpoint_root = Path(self.misc['log_root']) / self.global_config['name'] / self.misc['log_dir_names']['checkpoint']
            filepath = checkpoint_root / f'epoch_{self.config["start_epoch"]}.ckpt'
            try:
                checkpoint = torch.load(filepath)
                self.start_SSIM = checkpoint['best_SSIM']
                self.config['start_epoch'] = checkpoint['epoch']
                self.netG_A2B.load_state_dict(checkpoint['netG_A2B'])
                self.netG_B2A.load_state_dict(checkpoint['netG_B2A'])
                self.netD_A.load_state_dict(checkpoint['netD_A'])
                self.netD_B.load_state_dict(checkpoint['netD_B'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
                self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
                
            except FileNotFoundError:
                print(f'\033[1;31m[error]\033[0m 未能找到断点恢复文件，从0开始训练')
                self.config['start_epoch'] = 0
            except:
                print(f'\033[1;31m[error]\033[0m 断点恢复文件损坏或加载异常，从0开始训练')
                self.config['start_epoch'] = 0
            else:
                print(f'\033[1;32m[success]\033[0m 断点恢复成功，从 {self.config["start_epoch"]}代 继续训练')             
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
        
        self.epoch_bar = tqdm(range(self.config['start_epoch']+1, self.config['n_epochs']+1), desc='Training Progress', unit='epoch', position=0)
        
        for self.epoch in range(self.config['start_epoch']+1, self.config['n_epochs']+1):
            self.batch_bar = tqdm(total=len(self.train_loader), desc='Batch Progress', unit='batch', position=1, leave=False)
            for batch_idx, batch_data in enumerate(self.train_loader):
                # print(f'\033[1;33m[debug]\033[0m batch_idx: {batch_idx}, real_A.shape: {batch_data["CT"].shape}, real_B.shape: {batch_data["MR"].shape}')
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                # ===== Generator training =====
                self.netG_A2B.train()
                self.netG_B2A.train()
                self.netD_A.eval()
                self.netD_B.eval()
                self.optimizer_G.zero_grad()
                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.config['Adv_lambda']*self.MSE_loss(pred_fake, self.target_real)

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.config['Cyc_lambda'] * self.L1_loss(recovered_A, real_A)

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.config['Cyc_lambda'] * self.L1_loss(recovered_B, real_B)

                # Total loss
                loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_Total.backward()
                self.optimizer_G.step()

                ###### NewDiscriminator A ######
                self.netD_A.train()
                self.optimizer_D_A.zero_grad()
                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.config['Adv_lambda'] * self.MSE_loss(pred_real, self.target_real)
                # Fake loss
                # fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### NewDiscriminator B ######
                self.netD_B.train()
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.config['Adv_lambda'] * self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                # fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)
                loss_D_B.backward()

                self.optimizer_D_B.step()
                    ###################################
                
                self.batch_bar.update(1)
            self.batch_bar.close()
                
            self.epoch_bar.update(1)
            # TODO 评估 未完善
            if self.epoch % self.misc['eval_interval'] == 1: # 期望第1次就输出评估结果
                tqdm.write(f'\033[1;34m[info]\033[0m Evaluating...')
                
                train_metrics = self._evaluate(self.val_loader)
                tqdm.write(f'\033[1;34m[info]\033[0m epoch: {self.epoch}, train_metrics: {train_metrics}')
                # 保存最好的模型
                if train_metrics['metrics']['ssim'] > best_SSIM:
                    best_SSIM = train_metrics['metrics']['ssim']
                    tqdm.write(f'\033[1;32m[success]\033[0m epoch: {self.epoch}, best_SSIM: {best_SSIM}')
                    self.save_checkpoint(self.epoch, best_SSIM)
                
                tqdm.write(f'\033[1;34m[info]\033[0m Training...')
                
        self.epoch_bar.close()
        
    def _evaluate(self, dataloader):
        self.netG_A2B.eval()
        self.netG_B2A.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        
        metrics = defaultdict(list)
        losses = defaultdict(list)
        
        with torch.no_grad():
            self.eval_bar = tqdm(total=len(dataloader), desc='Eval Progress', unit='batch', position=1, leave=False)
            for batch_idx, batch_data in enumerate(dataloader):
                real_A = batch_data['CT'].to(self.device)
                real_B = batch_data['MR'].to(self.device)
                
                
                # ===== Generator training =====
                
                
                self.optimizer_G.zero_grad()
                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.config['Adv_lambda']*self.MSE_loss(pred_fake, self.target_real)

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.config['Cyc_lambda'] * self.L1_loss(recovered_A, real_A)

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.config['Cyc_lambda'] * self.L1_loss(recovered_B, real_B)

                # Total loss
                loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                ###### NewDiscriminator A ######
                self.optimizer_D_A.zero_grad()
                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.config['Adv_lambda'] * self.MSE_loss(pred_real, self.target_real)
                # Fake loss
                # fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)

                ###################################

                ###### NewDiscriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.config['Adv_lambda'] * self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                # fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)

                losses['loss_G_A2B'].append(loss_GAN_A2B)
                losses['loss_G_B2A'].append(loss_GAN_B2A)
                losses['loss_cycle_ABA'].append(loss_cycle_ABA)
                losses['loss_cycle_BAB'].append(loss_cycle_BAB)
                
                losses['loss_G'].append(loss_Total)
                losses['loss_D_A'].append(loss_D_A)
                losses['loss_D_B'].append(loss_D_B)
                
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
            'metrics' : metrics,
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
            'netG_A2B' : self.netG_A2B.state_dict(),
            'netG_B2A' : self.netG_B2A.state_dict(),
            'netD_A' : self.netD_A.state_dict(),
            'netD_B' : self.netD_B.state_dict(),
            'optimizer_G' : self.optimizer_G.state_dict(),
            'optimizer_D_A' : self.optimizer_D_A.state_dict(),
            'optimizer_D_B' : self.optimizer_D_B.state_dict(),
            
        }, filepath)
        tqdm.write(f'\033[1;32m[success]\033[0m 保存模型到 {filepath}')

if __name__ == '__main__':
    from utils import *
    import argparse
    
    print(f'\033[1;34m[info]\033[0m main.py \033[1;32mstart\033[0m')
    
    # load config
    parser = argparse.ArgumentParser(
        description='main.py'
    )
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = load_yaml(opts.config)
    
    # print config
    # print(f'\033[1;34m[info]\033[0m config: {config}')
    
    bugfree = BugFree(config['bugfree'])
    bugfree()
    
    # make log directory
    log_root = f"{config['misc']['log_root']}/{config['name']}"
    for dirtype in config['misc']['log_dir_names'].keys():
        os.makedirs(f"{log_root}/{config['misc']['log_dir_names'][dirtype]}", exist_ok=True)


    solver = CycleGAN3DSolver(config)
    solver.train()
    
    # id3d = ImagesDataset3D(config['dataset'], train=True)
    # print(id3d[0]['CT'].shape)