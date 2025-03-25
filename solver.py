from models import *
from utils import *

import os
import time
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


class Solver(object):
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.train_config = config['training']
        self.misc = config['misc']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 解析config函数
        self._parse_config()
        
        self._build_model() # 构建模型、优化器、损失函数
        # self._load_checkpoint() # 加载模型
        # self._memory_allocation() # 内存分配（暂时不实现）
        
        self.dataloader = train_loader
        self.val_data = val_loader
        
        self.ssim_calculator = SSIM(data_range=2.0).to(self.device)
        self.psnr_calculator = PSNR(data_range=2.0).to(self.device)

    def _parse_config(self):
        log_root = f"{self.config['log_root']}/{self.config['name']}"
        self.misc['loss_path'] = f"{log_root}/{self.config['log_dir_names']['loss']}"
        self.misc['sample_path'] = f"{log_root}/{self.config['log_dir_names']['sample']}"
        self.misc['result_path'] = f"{log_root}/{self.config['log_dir_names']['result']}"
        self.misc['checkpoint_path'] = f"{log_root}/{self.config['log_dir_names']['checkpoint']}"
        
        self.d_interval = self.config['training']['d_update_interval']
        self.g_interval = self.config['training']['g_update_interval']
    
    def _build_model(self):
        self.G = UNet3DGenerator().to(self.device)
        # self.D = Discriminator().to(self.device)
        self.D = StarDiscriminator3D().to(self.device)
        # 优化器
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.train_config['lr'], betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.train_config['lr'], betas=(0.5, 0.999))
        # 损失函数
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
    
    def _memory_allocation(self):
        pass
        
    def train(self):
        self.start_time = time.time()
        best_SSIM = 0
        epoch_bar = tqdm(
            total=self.train_config['n_epochs'],
            desc="Training Progress",
            unit="epoch"
        )
        real_CT, real_MR = None, None
        for epoch in range(self.train_config['epoch'], self.train_config['n_epochs']):
            for batch_idx, batch_data in enumerate(self.dataloader):
                real_CT = batch_data['CT'].float().to(self.device)
                real_MR = batch_data['MR'].float().to(self.device)
                
                if batch_idx % self.d_interval == 0:
                    self.D.train()
                    self.optim_D.zero_grad()
                    
                    # 生成假图像
                    self.G.eval()
                    with torch.no_grad():
                        fake_MR = self.G(real_CT)
                    self.G.train()
                    
                    # 真实图像的损失
                    pred_real = self.D(real_MR)
                    d_loss_real = - torch.mean(pred_real)
                    # 生成图像的损失
                    pred_fake = self.D(fake_MR.detach())
                    d_loss_fake = torch.mean(pred_fake)
                    # 计算损失
                    loss_gp = self._compute_gradient_penalty(real_MR, fake_MR)
                    d_loss = d_loss_fake + d_loss_real + loss_gp
                    d_loss.backward()
                    self.optim_D.step()
                
                if batch_idx % self.g_interval == 0:
                    self.G.train()
                    self.optim_G.zero_grad()
                    
                    # 生成假图像
                    fake_MR = self.G(real_CT)
                    # 生成图像的损失
                    pred_fake = self.D(fake_MR)
                    g_loss_adv = - torch.mean(pred_fake)
                    g_loss_l1 = self.L1_loss(fake_MR, real_MR)
                    
                    g_loss = g_loss_adv + g_loss_l1 * self.train_config['lambda_rec']
                    g_loss.backward()
                    self.optim_G.step()
                    
                # 更新进度条
                epoch_bar.set_postfix({
                    "D_loss": d_loss.item() if batch_idx % self.d_interval == 0 else None,
                    "G_loss": g_loss.item() if batch_idx % self.g_interval == 0 else None
                })
            epoch_bar.update(1)
            # 保存loss日志
            if epoch % self.misc['log_interval'] == 0:
                train_metrics = self._evaluate(self.dataloader)
                test_metrics = self._evaluate(self.val_data, is_train=False)
                self._log_metrics(epoch, train_metrics, test_metrics)
            
            if epoch % self.misc['sample_interval'] == 0:
                self.G.eval()
                fake_MR = self.G(real_CT)
                self._save_sample(real_CT, fake_MR, real_MR, epoch)
                
        pass
    
    def _save_checkpoint(self, epoch):
        pass
    
    def _load_checkpoint(self):
        pass
    
    # 计算PSNR
    def _calculate_psnr(self, real_images, fake_images):
        # 将 [B, C, D, H, W] 转换为 [B*C*D, 1, H, W]
        real_images = real_images.view(-1, 1, *real_images.shape[-2:])  # [B*C*D, 1, H, W]
        fake_images = fake_images.view(-1, 1, *fake_images.shape[-2:])  # [B*C*D, 1, H, W]

        psnr_value = self.psnr_calculator(real_images, fake_images)
        return psnr_value.item()
    
    # 计算SSIM
    def _calculate_ssim(self, real_images, fake_images):
        # 将 [B, C, D, H, W] 转换为 [B*C*D, 1, H, W]
        real_images = real_images.view(-1, 1, *real_images.shape[-2:])  # [B*C*D, 1, H, W]
        fake_images = fake_images.view(-1, 1, *fake_images.shape[-2:])  # [B*C*D, 1, H, W]

        ssim_value = self.ssim_calculator(real_images, fake_images)
        return ssim_value.item()
    
    # 计算WGAN-GP梯度惩罚
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    # 评估某代模型的效果
    def _evaluate(self, dataloader, is_train=True):
        self.G.eval()
        self.D.eval()
        
        metrics = defaultdict(list)
        losses = defaultdict(list)
        
        with torch.no_grad():
            for batch_data in dataloader:
                real_CT = batch_data['CT'].float().to(self.device)
                real_MR = batch_data['MR'].float().to(self.device)

                # 生成预测
                fake_MR = self.G(real_CT)

                # 计算真实图像的损失
                pred_real = self.D(real_MR)
                d_loss_real = - torch.mean(pred_real)

                # 计算生成图像的损失
                pred_fake = self.D(fake_MR.detach())
                d_loss_fake = torch.mean(pred_fake)
                
                # Compute loss for gradient penalty.
                # loss_gp = self._compute_gradient_penalty(real_MR, fake_MR)
                # 计算总损失
                d_loss = d_loss_fake + d_loss_real

                g_loss_adv = -torch.mean(pred_fake)
                # g_loss_l1 = self.criterionL1(fake_MR, real_MR)
                g_loss_l1= self.L1_loss(fake_MR, real_MR)
                g_loss = g_loss_adv + g_loss_l1 * self.train_config['lambda_rec']
                
                # 记录loss
                losses['D/loss_real'].append(d_loss_real.item())
                losses['D/loss_fake'].append(d_loss_fake.item())
                # losses['D/loss_gp'].append(loss_gp.item())
                losses['D/total'].append(d_loss.item())

                losses['G/loss_adv'].append(g_loss_adv.item())
                losses['G/loss_l1'].append(g_loss_l1.item())
                losses['G/total'].append(g_loss.item())

                # 计算指标
                metrics['psnr'].append(self._calculate_psnr(real_MR, fake_MR))
                metrics['ssim'].append(self._calculate_ssim(real_MR, fake_MR))

        return {
            'losses': {k: np.mean(v) for k, v in losses.items()},
            'metrics': {k: np.mean(v) for k, v in metrics.items()}
        }
    
    # 统一记录训练集和测试集指标
    def _log_metrics(self, epoch, train_metrics, test_metrics):
        log_data = {
            "epoch": epoch + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": time.time() - self.start_time,
        }

        # 训练集指标
        for loss, value in train_metrics['losses'].items():
            log_data[f"train/{loss}"] = value
        for metric, value in train_metrics['metrics'].items():
            log_data[f"train/{metric}"] = value

        # 测试集指标
        for loss, value in test_metrics['losses'].items():
            log_data[f"test/{loss}"] = value
        for metric, value in test_metrics['metrics'].items():
            log_data[f"test/{metric}"] = value

        # 保存日志
        self._save_log(log_data, epoch)

        # TensorBoard记录
        # if self.use_tensorboard:
        #     for k, v in log_data.items():
        #         if isinstance(v, (int, float)):
        #             self.writer.add_scalar(k, v, epoch)
    
    # 保存日志
    def _save_log(self, log_data: dict, epoch: int):
        # 1. 路径类型安全转换
        log_dir = Path(self.misc['loss_path']) if not isinstance(self.misc['loss_path'], Path) else self.misc['loss_path']

        # 2. 创建目录（含父目录）
        log_dir.mkdir(parents=True, exist_ok=True)

        # 3. 构造文件路径
        file_path = log_dir / f"epoch_{epoch+1:04d}.json"

        # 4. 安全写入流程
        with open(file_path, 'w', encoding='utf-8') as f:  # 显式指定编码
            json.dump(
                log_data, f,
                indent=2,
                ensure_ascii=False,  # 允许保存中文等非ASCII字符
                default=str  # 处理无法序列化的对象
            )

        print(f"\033[1;34m[Info]\033[0m 日志成功保存至：{file_path}")
        return True
    
    # 保存样片
    def _save_sample(self, real_CT, fake_MR, real_MR, epoch):
        sample_dir = Path(self.misc['sample_path']) if not isinstance(self.misc['sample_path'], Path) else self.misc['sample_path']
        
        dirs = {
            'real_CT_dir': f"{sample_dir}/{epoch}/real_CT",
            'fake_MR_dir': f"{sample_dir}/{epoch}/fake_MR",
            'real_MR_dir': f"{sample_dir}/{epoch}/real_MR",
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        
        # tensor(1, 1, 32, 256, 256) -> nparray(32, 256,256)
        real_CT = np.squeeze(real_CT.detach().cpu().numpy())
        fake_MR = np.squeeze(fake_MR.detach().cpu().numpy())
        real_MR = np.squeeze(real_MR.detach().cpu().numpy())
        
        # 放缩到到0-255范围（如果用[-1,1]的数据）
        real_CT = ((real_CT + 1) * 127.5).astype(np.uint8)
        fake_MR = ((fake_MR + 1) * 127.5).astype(np.uint8)
        real_MR = ((real_MR + 1) * 127.5).astype(np.uint8)
        
        # 保存所有切片
        for slice_idx in range(real_CT.shape[0]):
            real_CT_slice = real_CT[slice_idx]
            fake_MR_slice = fake_MR[slice_idx]
            real_MR_slice = real_MR[slice_idx]
            
            # 保存图像
            real_CT_path = f"{dirs['real_CT_dir']}/{slice_idx:03d}.png"
            fake_MR_path = f"{dirs['fake_MR_dir']}/{slice_idx:03d}.png"
            real_MR_path = f"{dirs['real_MR_dir']}/{slice_idx:03d}.png"
            
            # 保存图像
            self._save_image(real_CT_slice, real_CT_path)
            self._save_image(fake_MR_slice, fake_MR_path)
            self._save_image(real_MR_slice, real_MR_path)
    
    def _save_image(self, image, path):
        # 添加对比度拉伸以适应显示
        vmin, vmax = np.percentile(image, (0.5, 99.5))
        if vmax == vmin:
            slice_norm = np.zeros_like(image) # 全黑
        else:
            slice_norm = np.clip((image - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

        # 保存图像
        # Image.fromarray(slice_norm).save(path, optimize=True, compress_level=9) # 9级压缩
        Image.fromarray(slice_norm).save(path)
        
    def _save_result(self, result):
        pass
        