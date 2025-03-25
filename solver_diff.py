# solver_diff.py
from models import *

from tqdm import tqdm
import numpy as np

import torch
from torch.optim import Adam


class DiffusionSolver:
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = DiffusionUNet3D().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=float(config['lr']))
        
        # 扩散参数
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 数据加载
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader):
            ct = batch['CT'].to(self.device)  # [B,1,D,H,W]
            mr = batch['MR'].to(self.device)  # [B,1,D,H,W]
            
            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (mr.size(0),)).to(self.device)
            
            # 正向加噪过程
            alpha_bar = self.alpha_bars[t].view(-1,1,1,1,1)
            noise = torch.randn_like(mr)
            noisy_mr = torch.sqrt(alpha_bar)*mr + torch.sqrt(1-alpha_bar)*noise
            
            # 模型预测
            pred_noise = self.model(noisy_mr, ct, t.float()/self.num_timesteps)
            
            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def sample(self, ct, mr_init=None, steps=100):
        self.model.eval()
        x_t = mr_init if mr_init is not None else torch.randn_like(ct)
        
        for t in tqdm(reversed(range(steps)), desc="Sampling"):
            ts = torch.full((ct.size(0),), t, device=self.device)
            
            # 预测噪声
            pred_noise = self.model(x_t, ct, ts/steps)
            
            # 反向过程更新
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = (x_t - beta * pred_noise / torch.sqrt(1 - alpha_bar)) / torch.sqrt(self.alphas[t])
            x_t += torch.sqrt(beta) * noise
        
        return torch.clamp(x_t, -1, 1)
    
    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            
            if (epoch+1) % self.config['eval_interval'] == 0:
                val_loss = self.evaluate()
                print(f"Validation Loss: {val_loss:.4f}")
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            ct = batch['CT'].float().to(self.device)
            mr = batch['MR'].float().to(self.device)
            
            t = torch.randint(0, self.num_timesteps, (ct.size(0),)).to(self.device)
            
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(mr)
            noisy_mr = torch.sqrt(alpha_bar) * mr + torch.sqrt(1 - alpha_bar) * noise
            
            pred_noise = self.model(noisy_mr, ct, t/self.num_timesteps)
            loss = F.mse_loss(pred_noise, noise)
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)