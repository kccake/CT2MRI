import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim2d
from skimage.metrics import peak_signal_noise_ratio as psnr2d
from PIL import Image
from pathlib import Path
from models import StarGenerator3D, StarDiscriminator3D # ours
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

class Solver:

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        self.config = config

        # 定义dataloader
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Model configurations
        self.g_conv_dim = config['model']['g_conv_dim']
        self.d_conv_dim = config['model']['d_conv_dim']
        self.g_repeat_num = config['model']['g_repeat_num']
        self.d_repeat_num = config['model']['d_repeat_num']
        self.lambda_cls = config['model']['lambda_cls']
        self.lambda_rec = config['model']['lambda_rec']
        self.lambda_gp = config['model']['lambda_gp']

        # Training configurations
        self.batch_size = config['training']['batch_size']
        self.num_iters = config['training']['num_iters']
        self.num_iters_decay = config['training']['num_iters_decay']
        self.g_lr = config['training']['g_lr']
        self.d_lr = config['training']['d_lr']
        self.n_critic = config['training']['n_critic']
        self.beta1 = config['training']['beta1']
        self.beta2 = config['training']['beta2']
        self.resume_iters = config['training']['resume_iters']
        
        self.criterionL1 = torch.nn.L1Loss()

        # Test configurations 告诉test该用哪个模型
        self.test_iters = config['testing']['test_iters']

        # Miscellaneous.
        self.use_tensorboard = config['misc']['use_tensorboard']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config['directories']['log_dir']
        self.model_save_dir = config['directories']['model_save_dir']
        self.sample_dir = config['directories']['sample_dir']
        self.result_dir = config['directories']['result_dir']

        # Training step settings
        self.log_step = config['misc']['log_step']
        self.sample_step = config['misc']['sample_step']
        self.model_save_step = config['misc']['model_save_step']
        self.lr_update_step = config['misc']['lr_update_step']
        
        self.ssim_calculator = SSIM(data_range=2.0).to(self.device)
        self.psnr_calculator = PSNR(data_range=2.0).to(self.device)

        # Build the model and tensor_board
        self.build_model()

        if self.config['misc']['use_tensorboard']:
            self.build_tensorboard()

    # __init__的辅助函数************************************************
    # 1.
    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = StarGenerator3D(self.g_conv_dim, self.g_repeat_num).to(self.device)
        self.D = StarDiscriminator3D(self.d_conv_dim, self.d_repeat_num).to(self.device)

        self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if self.resume_iters:
            self.load_checkpoint(self.resume_iters)

    # 2.
    def load_checkpoint(self, resume_iters):
        checkpoint_path = os.path.join(self.model_save_dir, f'model_optim_epoch{resume_iters}.ckpt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
            time.sleep(5)
            exit(1)

    # 3.
    # 要单独写的logger还没写, 只能放一个SummaryWriter(但实际上logger也是一个SummeryWriter的封装)
    def build_tensorboard(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)
    # ******************************************************************************************


    def train(self):
        self.start_time = time.time()
        start_epoch = 0 if not self.resume_iters else self.resume_iters

        # Learning rate cache for decaying.
        g_lr = self.g_lr  # 0.0001
        d_lr = self.d_lr  # 0.0001

        # 创建 tqdm 进度条(不成熟的进度条)*************
        progress_bar = tqdm(total=self.num_iters, desc="Training Progress", unit="iter")
        # *********************************************

        for epoch in range(start_epoch, self.num_iters):

            # Batch_wise训练
            for batch_idx, batch in enumerate(self.train_loader):
                # 数据预处理
                real_CT = batch['CT'].to(self.device).float()
                real_MR = batch['MR'].to(self.device).float()
                
                # Train Discriminator
                # Compute loss with real images.
                self.D.train()
                self.d_optimizer.zero_grad()

                # 生成假样本
                with torch.no_grad():
                    fake_MR = self.G(real_CT)
                
                 # 真实样本损失
                pred_real = self.D(real_MR)
                d_loss_real = -torch.mean(pred_real)

                # 假样本损失
                pred_fake = self.D(fake_MR.detach())
                d_loss_fake = torch.mean(pred_fake)

                # 梯度惩罚
                loss_gp = self._compute_gradient_penalty(real_MR, fake_MR)

                # 总损失
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * loss_gp
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                if (batch_idx + 1) % self.n_critic == 0:
                    self.G.train()
                    self.g_optimizer.zero_grad()

                    # 生成假样本
                    fake_MR = self.G(real_CT)
                    
                    pred_fake = self.D(fake_MR)
                    g_loss_adv = -torch.mean(pred_fake)

                    # L1重建损失
                    g_loss_L1 = self.criterionL1(fake_MR, real_MR) * self.lambda_rec

                    # 总损失
                    g_loss = g_loss_adv + g_loss_L1

                    g_loss.backward()
                    self.g_optimizer.step()

                # 更新 tqdm 进度条 **********************************
                
                progress_bar.set_postfix(
                    D_loss=d_loss.item(),
                    G_loss=g_loss.item() if (batch_idx + 1) % self.n_critic == 0 else None
                )
            progress_bar.update(1)
                # ****************************************************
                
            # Logging
            if epoch % self.log_step == 0:
                # 在训练集和测试集上同时评估
                train_metrics = self._evaluate(self.train_loader, is_training=True)
                test_metrics = self._evaluate(self.test_loader, is_training=False)
                self._log_metrics(epoch, train_metrics, test_metrics)
                
            # Save samples
            if epoch % self.sample_step == 0:
                # 改为利用epoch来表示总step数字, 当epoch达到步数后, 就会自动利用该epoch最后的那组real_CT,real_MR,fake_MR作为采样结果
                fake_MR = self.G(real_CT)
                self._save_3d_volumes(real_CT, fake_MR, real_MR, epoch)
            
            # Save model
            if epoch % self.model_save_step == 0:
                self._save_checkpoint(epoch)

            # Learning rate decay
            if (epoch+1) % self.lr_update_step == 0 and (epoch+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        # 关闭 tqdm 进度条 ***************
        progress_bar.close()
        # ********************************
        print(f"Training finished. Total time: {time.time() - self.start_time:.2f}s")

    # train的辅助函数
    # 1. 
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        """WGAN-GP梯度惩罚计算"""
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

    # 2.
    def _evaluate(self, dataloader, is_training=True):
        """在指定数据集上评估模型性能"""
        self.G.eval()
        self.D.eval()

        metrics = defaultdict(list)
        losses = defaultdict(list)

        with torch.no_grad():
            for batch in dataloader:
                real_CT = batch['CT'].to(self.device).float()
                real_MR = batch['MR'].to(self.device).float()

                # 生成预测
                fake_MR = self.G(real_CT)

                # Compute loss with real images.
                pred_real = self.D(real_MR)
                d_loss_real = - torch.mean(pred_real)

                # Compute loss with fake images.
                pred_fake = self.D(fake_MR.detach())
                d_loss_fake = torch.mean(pred_fake)
                
                # Compute loss for gradient penalty.
                # loss_gp = self._compute_gradient_penalty(real_MR, fake_MR)
                d_loss = d_loss_fake + d_loss_real

                g_loss_adv = -torch.mean(pred_fake)
                g_loss_l1 = self.criterionL1(fake_MR, real_MR) * self.lambda_rec
                g_loss = g_loss_adv + g_loss_l1
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

    def _log_metrics(self, epoch, train_metrics, test_metrics):
        """统一记录训练集和测试集指标"""
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
        self._save_training_log(log_data, epoch)

        # TensorBoard记录
        if self.use_tensorboard:
            for k, v in log_data.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, epoch)

    def _calculate_psnr(self, real_images, fake_images):
        # 将 [B, C, D, H, W] 转换为 [B*C*D, 1, H, W]
        real_images = real_images.view(-1, 1, *real_images.shape[-2:])  # [B*C*D, 1, H, W]
        fake_images = fake_images.view(-1, 1, *fake_images.shape[-2:])  # [B*C*D, 1, H, W]

        # 批量计算 PSNR
        psnr_value = self.psnr_calculator(real_images, fake_images)
        return psnr_value.item()
        
    
    def _calculate_ssim(self, real_images, fake_images):
        # 将 [B, C, D, H, W] 转换为 [B*C*D, 1, H, W]
        real_images = real_images.view(-1, 1, *real_images.shape[-2:])  # [B*C*D, 1, H, W]
        fake_images = fake_images.view(-1, 1, *fake_images.shape[-2:])  # [B*C*D, 1, H, W]

        # 批量计算 SSIM
        ssim_value = self.ssim_calculator(real_images, fake_images)
        return ssim_value.item()

    
    def _save_training_log(self, log_data: dict, epoch: int) -> bool:
        """
        安全保存训练日志到指定目录
        参数：
            log_data: 需要保存的日志字典数据
            epoch: 当前epoch数 (从0开始计数)
        返回：
            bool: 是否保存成功
        """
        # 1. 路径类型安全转换
        log_dir = Path(self.log_dir) if not isinstance(self.log_dir, Path) else self.log_dir

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
    

    # 4
    # 这个函数写的不好, 可能需要单写一个类来继承Solver, 否则的话过于冗长了
    def _save_3d_volumes(self, real_CT, fake_MR, real_MR, epoch):
        """保存完整3D体积的所有切片(支持任意batch_size)"""
        batch_size = real_CT.size(0)
        depth = real_CT.size(2)

        # 创建epoch目录
        epoch_dir = os.path.join(self.sample_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # 遍历batch中的每个样本
        for sample_idx in range(batch_size):
            # 创建样本子目录（当batch_size>1时）
            if batch_size > 1:
                sample_dir = os.path.join(epoch_dir, f"sample_{sample_idx:03d}")
                os.makedirs(sample_dir, exist_ok=True)
            else:
                sample_dir = epoch_dir  # batch_size=1时直接使用epoch目录

            # 定义各模态保存路径
            dirs = {
                'CT': os.path.join(sample_dir, 'CT'),
                'fake_MR': os.path.join(sample_dir, 'fake_MR'),
                'real_MR': os.path.join(sample_dir, 'real_MR')
            }

            # 创建各模态目录
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)

            # 获取当前样本数据
            ct_volume = real_CT[sample_idx, 0].cpu().detach().numpy()  # [D, H, W]
            fake_volume = fake_MR[sample_idx, 0].cpu().detach().numpy()
            real_volume = real_MR[sample_idx, 0].cpu().detach().numpy()

            # 反归一化到0-255范围（假设原始归一化到[-1,1]）
            ct_volume = ((ct_volume + 1) * 127.5).astype(np.uint8)
            fake_volume = ((fake_volume + 1) * 127.5).astype(np.uint8)
            real_volume = ((real_volume + 1) * 127.5).astype(np.uint8)

            # 保存所有切片
            for slice_idx in range(depth):
                # CT切片
                self._save_slice(
                    ct_volume[slice_idx],
                    os.path.join(dirs['CT'], f"slice_{slice_idx:04d}.png")
                )

                # 生成MR切片
                self._save_slice(
                    fake_volume[slice_idx],
                    os.path.join(dirs['fake_MR'], f"slice_{slice_idx:04d}.png")
                )

                # 真实MR切片
                self._save_slice(
                    real_volume[slice_idx],
                    os.path.join(dirs['real_MR'], f"slice_{slice_idx:04d}.png")
                )

    def _save_slice(self, slice_array, save_path):
        """保存单个切片为PNG图像(优化版)"""
        # 添加对比度拉伸以适应显示
        vmin, vmax = np.percentile(slice_array, (1, 99))
        if vmax == vmin:
            slice_norm = np.zeros_like(slice_array) # 全黑
        else:
            slice_norm = np.clip((slice_array - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

        # 使用PIL保存并优化压缩
        Image.fromarray(slice_norm).save(
            save_path,
            optimize=True,
            compress_level=9
        )

    def _save_mid_slice_comparison(self, real_CT, fake_MR, real_MR, epoch):
        """保存中间切片对比图（支持多样本）"""
        batch_size = real_CT.size(0)
        cols = 3  # CT, Fake, Real
        rows = batch_size

        plt.figure(figsize=(cols*6, rows*4), dpi=100)

        for sample_idx in range(batch_size):
            # 获取中间切片
            slice_idx = real_CT.size(2) // 2
            ct_slice = (real_CT[sample_idx, 0, slice_idx].cpu().numpy() + 1) * 127.5
            fake_slice = (fake_MR[sample_idx, 0, slice_idx].cpu().numpy() + 1) * 127.5
            real_slice = (real_MR[sample_idx, 0, slice_idx].cpu().numpy() + 1) * 127.5

            # 计算质量指标
            psnr = psnr2d(real_slice, fake_slice).item()
            ssim = ssim2d(real_slice, fake_slice).item()

            # 绘制CT
            plt.subplot(rows, cols, sample_idx*cols + 1)
            plt.imshow(ct_slice, cmap='gray')
            plt.title(f"Sample {sample_idx} CT" + (
                "\nPSNR: - | SSIM: -" if sample_idx==0 else ""), fontsize=8)
            plt.axis('off')

            # 绘制生成MR
            plt.subplot(rows, cols, sample_idx*cols + 2)
            plt.imshow(fake_slice, cmap='gray')
            plt.title(f"Sample {sample_idx} Fake MR\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}", fontsize=8)
            plt.axis('off')

            # 绘制真实MR
            plt.subplot(rows, cols, sample_idx*cols + 3)
            plt.imshow(real_slice, cmap='gray')
            plt.title(f"Sample {sample_idx} Real MR" + (
                "\nReference" if sample_idx==0 else ""), fontsize=8)
            plt.axis('off')

        # 保存对比图
        compare_dir = os.path.join(self.sample_dir, "comparison")
        os.makedirs(compare_dir, exist_ok=True)
        plt.savefig(
            os.path.join(compare_dir, f"epoch_{epoch:04d}_comparison.png"),
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()

    # 5
    def _save_checkpoint(self,epoch):
        # 这个没有加is_final这个参数, 这个还是要求做实验的时候, 做好管理
        filename =  f"model_optim_epoch{epoch}.ckpt"
        checkpoint_path = os.path.join(self.model_save_dir,filename)
        checkpoint = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'epoch': epoch,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # 6
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def test(self):
        self.G.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                real_CT = batch['CT'].to(self.device)
                fake_MR = self.G(real_CT)

                # Save or evaluate results
                # Add your testing logic here