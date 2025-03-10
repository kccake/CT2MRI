from model import StarGenerator3D
from model import StarDiscriminator3D
from torchmetrics import StructuralSimilarityIndexMeasure
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm



# 重写Solver的主要目的还是改变数据输入, 原来读的方式不好(一片一片读, 然后一片一片处理), 应该改为一个3D_CT, 一个3D_CT地读, 一个MRI, 一个MRI地读
# 但可能会遇到的一个问题就是和原有的数据预处理起冲突, 我觉得为了数据的流畅性, 还是改一下数据预处理的代码
class Solver(object):

    def __init__(self, dataloader, config):
        """Initialize configurations."""

        self.config = config

        # Data loader
        # 解决好dataloader的问题**********
        # 正常应该在Solver中利用统一的config.yaml来定义
        self.dataloader = dataloader
        # ********************************

        # Model configurations.
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        self.criterionL1 = torch.nn.L1Loss()

        # Test configurations 告诉test该用哪个模型
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step 主要是save和参数更新的频率
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard. build模型实例和tensorboard
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        
    def build_model(self):
        # 这个和batch_size无关
        """Create a generator and a discriminator."""

        self.G = StarGenerator3D()
        self.D = StarDiscriminator3D()

        self.G.to(self.device)
        self.D.to(self.device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if self.resume_iters:
            checkpoint_path = os.path.join(self.model_sava_dir,f"Epoch_{self.resume_iter}.ckpt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                print(f"Loaded checkpoint from {checkpoint_path}")

            else:
                print(f"Checkpoint file {checkpoint_path} not found. Exiting.")
                exit(1) # 直接终止程序, 这样写太硬了

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        # 测试一下重写的logger.py好不好使
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        # 这个要注意, 一定要让训练时的数据属于[-1,1]
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    # train用于训练的辅助函数**********************************************
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
    
    def _train_discriminator_step(self, real_CT, real_MR):
        # 不支持batch_size > 1
        """判别器单步训练"""
        self.D.train()
        self.d_optimizer.zero_grad()

        # 生成假样本
        with torch.no_grad():
            fake_MR = self.G(real_CT)
        
        # 真实样本损失
        real_pred = self.D(real_MR)
        d_loss_real = -torch.mean(real_pred)

        # 假样本损失
        fake_pred = self.D(fake_MR.detach())
        d_loss_fake = torch.mean(fake_pred)

        # 梯度惩罚
        gp_loss = self._compute_gradient_penalty(real_MR, fake_MR)

        # 总损失
        d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'D/loss_real': d_loss_real.item(),
            'D/loss_fake': d_loss_fake.item(),
            'D/loss_gp': gp_loss.item(),
            'D/total': d_loss.item()
        }
    
    def _train_generator_step(self, real_CT, real_MR):
        # 不支持batch_size > 1
        """生成器单步训练"""
        self.G.train()
        self.g_optimizer.zero_grad()

        # 生成假样本
        fake_MR = self.G(real_CT)

        # 对抗损失
        pred_fake = self.D(fake_MR)
        g_loss_adv = -torch.mean(pred_fake)

        # L1重建损失
        g_loss_l1 = self.criterionL1(fake_MR, real_MR) * self.lambda_l1
        
        # 总损失
        g_loss = g_loss_adv + g_loss_l1
        g_loss.backward()
        self.optimizer_G.step()

        return {
            'G/loss_adv': g_loss_adv.item(),
            'G/loss_l1': g_loss_l1.item(),
            'G/total': g_loss.item()
        }
    #**************************************************************************

    # train用于记录的辅助函数**************************************************
    # 1. model与optimizer的记录
    def _save_checkpoint(self,epoch):
        # 这个没有加is_final这个参数, 这个还是要求做实验的时候, 做好管理
        checkpoint_path = os.path.join(self.model_save_dir,filename)
        checkpoint = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.optimizer_G.state_dict(),
            'opt_D': self.optimizer_D.state_dict(),
            'epoch': epoch,
        }

        filename =  f"model_optim_epoch{epoch}.ckpt"

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # 2.训练时metrics的保存, 使用json不再使用txt, 是对一整个dataloader中的所有样本进行测试(metric取平均值),并取平均值,用的是刚经过epoch轮训练的model
    # 这个还是要改, 现在只能监控一个dataset的效果, 必须要改更顶层的代码, 如果能把dataset以统一的config这件事做到了, 这个问题就能够解决
    def _log_trainning(self, epoch):
        """记录当前epoch模型在训练集上的平均损失和评估指标"""
        self.G.eval()
        self.D.eval()

        # 初始化指标存储
        metrics = defaultdict(list)
        losses = defaultdict(list)
        
        # 遍历整个数据集
        start_time = time.time()
        with torch.no_grad():
            for batch in self.dataloader:

                # 数据加载
                real_CT = batch['CT'].to(self.device).float()
                real_MR = batch['MR'].to(self.device).float()

                # 生成预测
                fake_MR = self.G(real_CT)

                # 计算损失
                # 这一部分被写到了函数里了, 但是应该有办法优化结构, 暂时还没想到
                # discriminator部分
                real_pred = self.D(real_MR)
                d_loss_real = -torch.mean(real_pred)

                fake_pred = self.D(fake_MR.detach())
                d_loss_fake = torch.mean(fake_pred)

                # 梯度惩罚
                gp_loss = self._compute_gradient_penalty(real_MR, fake_MR)

                # 总损失
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp_loss

                # generator部分
                g_loss_adv = -d_loss_fake
                g_loss_l1 = self.criterionL1(fake_MR, real_MR) * self.lambda_l1
                g_loss = g_loss_adv + g_loss_l1

                # 记录loss
                losses['D/loss_real'].append(d_loss_real.item())
                losses['D/loss_fake'].append(d_loss_fake.item())
                losses['D/loss_gp'].append(gp_loss.item())
                losses['D/total'].append(d_loss.item())
                losses['G/loss_adv'].append(g_loss_adv.item())
                losses['G/loss_l1'].append(g_loss_l1.item())
                losses['G/total'].append(g_loss.item())

                # 计算指标
                metrics['psnr'].append(self._calculate_psnr(real_MR, fake_MR)) # 这个实现的psnr支持batchsize > 1
                metrics['ssim'].append(self._calculate_ssim(real_MR, fake_MR)) # 这个是skimage的, 这个不知道是否支持batchsize > 1

        log_data = {
            "epoch": epoch + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": time.time() - start_time,
        }

        for loss, values in losses.items():
            log_data.update({
                f"{loss}_mean": float(torch.tensor(values).mean().item())
            })
        
        for metric, values in metrics.items():
            log_data.update({
                f"{metric}_mean": float(torch.tensor(values).mean().item())
            })
        
        # 保存日志
        self._save_training_log(log_data, epoch)
    
    def _save_train_log(self, log_data, epoch):
        """保存训练日志到 train 子目录"""

        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / f"epoch_{epoch+1:04d}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_psnr(y, G, data_range=2.0):
        """
        计算 PSNR (3D 医学图像专用版本)
        参数：
            y: 真实图像 [B, C, D, H, W]，范围 [-1,1]
            G: 生成图像 [B, C, D, H, W]
            data_range: 数据范围（论文中为 max(y,G)）
        返回：
            PSNR 值 (dB)
        """
        # 计算动态范围
        max_val = torch.max(torch.max(y), torch.max(G))  # 论文中公式的 max(y(x), G(x))
        
        # 计算均方误差
        mse = torch.mean((y - G) ** 2, dim=[1,2,3,4])  # 按样本计算
        
        # 避免除以零
        mse = torch.clamp(mse, min=1e-8)
        
        # 按论文公式计算
        psnr = 10 * torch.log10((max_val ** 2) / mse)
        return torch.mean(psnr).item()
    
    def _calculate_ssim(self, real_images, fake_images):

        ssim = StructuralSimilarityIndexMeasure(
            data_range=2.0,            # 数据范围 [-1, 1] -> 2.0
            kernel_size=(11, 11, 11),  # 3D 卷积核尺寸 (D, H, W)
            win_size=(11, 11, 11),     # 窗口大小需与 kernel_size 一致
            reduction='mean'           # 返回批量的均值
        ).to(self.device)   

        """
        计算 3D 医学图像的 SSIM
        :param real_images: 真实图像 [B, C, D, H, W]
        :param fake_images: 生成图像 [B, C, D, H, W]
        :return: SSIM 值 (标量)
        """
        # 验证输入维度
        if real_images.dim() != 5 or fake_images.dim() != 5:
            raise ValueError("Input must be 5D tensor: [B, C, D, H, W]")
        
        # 计算 SSIM（注意参数顺序：preds在前，target在后）
        return self.ssim(fake_images, real_images).item()


    # 3. 采样中间结果
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
            ct_volume = real_CT[sample_idx, 0].cpu().numpy()  # [D, H, W]
            fake_volume = fake_MR[sample_idx, 0].cpu().numpy()
            real_volume = real_MR[sample_idx, 0].cpu().numpy()

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
            psnr = self.psnr(fake_MR[sample_idx], real_MR[sample_idx]).item()
            ssim = self.ssim(fake_MR[sample_idx], real_MR[sample_idx]).item()

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


    # 4. update_lr, Decay learning rates 关于lr的调度逻辑
    def _update_learning_rates(self, epoch):
        if epoch % self.lr_update_step == 0 and (epoch+1) > (self.num_iters - self.num_iters_decay):
            g_lr -= (self.g_lr / float(self.num_iters_decay))
            d_lr -= (self.d_lr / float(self.num_iters_decay))
            self.update_lr(g_lr, d_lr)

    # *************************************************************************


    def train(self):
        """主循环训练"""

        # 先确保训练和日志功能完善, 然后再考虑tqdm的优化
        # if self.resume_iters:
        #     epoch_bar = tqdm(
        #         initial=self.resume_iters,
        #         total=self.num_iters,
        #         desc="[Training Progress]",
        #         unit="epoch"
        #     )
        # else:
        #     epoch_bar = tqdm(
        #         initial=0,
        #         total=self.num_iters,
        #         desc="[Training Progress]",
        #         unit="epoch"
        #     )
        if self.resume_iters:
            start_epoch = self.resume_iters
        else:
            start_epoch = 0

        print("It's training")
        start_time = time.time()

        for epoch in range(start_epoch, self.num_iters+1):
            epoch_metrics = defaultdict(float)

            # Batch_wise训练
            for batch_idx, batch in enumerate(self.dataloader):
                # 数据预处理
                real_CT = batch['CT'].to(self.device).float()
                real_MR = batch['MR'].to(self.device).float()

                # ---------------------
                #  训练判别器
                # ---------------------
                d_metrics = self._train_discriminator_step(real_CT, real_MR)
                
                # ---------------------
                #  训练生成器 (每n_critic次迭代)
                # ---------------------
                if batch_idx % self.n_critic == 0:
                    g_metrics = self._train_generator_step(real_CT, real_MR)
                    epoch_metrics.update(g_metrics)

                epoch_metrics.update(d_metrics)
            
            # 日志记录
            if epoch % self.log_step == 0:
                self._log_training(self, epoch)

            # 保留中间结果
            if epoch % self.sample_step == 0:
                # 改为利用epoch来表示总step数字, 当epoch达到步数后, 就会自动利用该epoch最后的那组real_CT,real_MR,fake_MR作为采样结果
                fake_MR = self.G(real_CT)
                self._save_generated_samples(real_CT, fake_MR, epoch)


            # 保存模型
            if epoch % self.model_save_step == 0:
                self._save_checkpoint(self, epoch)
            
            # 学习率衰减
            self._update_learning_rates(epoch)

            