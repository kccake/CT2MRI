import os
import yaml
import torch
import numpy as np
from torch.backends import cudnn
from torchvision import transforms

import utils
from utils import *
# from utils import CTMR3DDataset, MedicalVolumePreprocessor, Solver, load_config, create_data_loaders
import models


if __name__ == '__main__':
    # 加载配置
    config = load_config('config.yaml')
    
    # ===== bugfree 绝活(可注释掉) ===== #
    bugfree = utils.BugFree(config['bugfree'])
    bugfree()
    print('\n\n')
    # ===== bugfree 绝活(可注释掉) ===== #
    
    # # 初始化数据集 ( ??? create_data_loaders()中有使用到 )
    # dataset = CTMR3DDataset(config=config['data'])
    # print(dataset[0]['CT'].shape, dataset[0]['MR'].shape)
    
    
    cudnn.benchmark = True # 加速训练
    
    # 创建需要的目录
    print(f"\033[1;34m[Info]\033[0m Enable cudnn benchmark")
    utils.make_dirs([directory for directory in config['directories'].values()])
    
    transform = transforms.Compose([
        MedicalVolumePreprocessor(target_contrast=50.0), # 3D医学影像预处理工具类(做图像增强)
    ])

    # 创建数据加载器
    train_loader = create_data_loaders(config, mode='train', transform=transform)
    test_loader = create_data_loaders(config, mode='test', transform=transform)

    solver = Solver(train_loader, test_loader, config)
    
    # 运行训练或测试
    if config['misc']['mode'] == 'train':
        solver.train()
    elif config['misc']['mode'] == 'test':
        solver.test()