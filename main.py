import os
from utils import *
from models import *
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def main():
    print(f'\033[1;34m[info]\033[0m main.py \033[1;32mstart\033[0m')
    
    # load config
    parser = argparse.ArgumentParser(
        description='main.py'
    )
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = load_yaml(opts.config)
    
    # print config
    print(f'\033[1;34m[info]\033[0m config: {config}')
    
    # make log directory
    log_root = f"{config['log_root']}/{config['name']}"
    for dirtype in config['log_dir_names'].keys():
        os.makedirs(f"{log_root}/{config['log_dir_names'][dirtype]}", exist_ok=True)

    # traindataset = ImagesDataset2D(config['dataset'], train=True)
    # testdataset = ImagesDataset2D(config['dataset'], train=False)
    trainloader = DataLoader(ImagesDataset2D(config['dataset'], train=True))
    testloader = DataLoader(ImagesDataset2D(config['dataset'], train=False))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = GMain(latent_size=512, resolution=256).to(device)
    discriminator = DStyleGAN2(resolution=256).to(device)
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.0, 0.99))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.0, 0.99))
    
    # 对抗损失函数
    def gan_loss(logits, target_is_real):
        labels = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return F.binary_cross_entropy_with_logits(logits, labels)

    # 训练函数
    def train_epoch(loader, epoch):
        generator.train()
        discriminator.train()
        
        for batch_idx, batch in enumerate(loader):
            real_B = batch['MR'].to(device)  # 目标风格图像
            
            # 训练判别器
            d_optimizer.zero_grad()
            
            # 生成假图像
            # z = torch.randn(real_B.size(0), 512).to(device)
            z = batch['CT'].to(device)
            fake_B = generator(z)
            
            # 判别器损失
            d_real = discriminator(real_B)
            d_fake = discriminator(fake_B.detach())
            d_loss = gan_loss(d_real, True) + gan_loss(d_fake, False)
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            d_fake = discriminator(fake_B)
            g_loss = gan_loss(d_fake, True)
            g_loss.backward()
            g_optimizer.step()
            
            # 打印训练日志
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

    # 主训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        train_epoch(trainloader, epoch)
        
if __name__ == '__main__':
    main()
