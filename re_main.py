import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from re2_solver import Solver
from torch.backends import cudnn
from data.re_CTandMR_dataset import CTMR3DDataset
from torchvision import transforms


def str2bool(v):
    return v.lower() in ('true')

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(default_config, file_config, cli_config):
    """递归合并三种配置源"""
    config = default_config.copy()
    
    # 合并YAML文件配置
    if file_config:
        for key, value in file_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # 合并命令行参数
    for key, value in vars(cli_config).items():
        if value is not None:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    return config

class MedicalVolumePreprocessor:
    """3D医学影像预处理工具类"""
    def __init__(self, target_contrast=50.0):
        self.target_contrast = target_contrast
        
    def __call__(self, volume):
        # 转换为numpy数组处理（假设输入是numpy格式）
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # 对比度统一处理
        volume = self._adjust_contrast(volume)
        
        # 数据归一化到[-1,1]
        volume = self._normalize(volume)
        
        # 维度填充到4的倍数
        volume = self._pad_to_multiple(volume, base=4)
        
        return torch.from_numpy(volume).float()

    def _adjust_contrast(self, volume):
        # 转换为0-255范围
        volume_255 = cv2.convertScaleAbs(
            volume, 
            alpha=self.target_contrast / (volume.mean() + 1e-8),
            beta=0
        )
        return volume_255

    def _normalize(self, volume):
        return (volume - 127.5) / 127.5

    def _pad_to_multiple(self, volume, base=4):
        """将D/H/W维度填充到base的整数倍，保持C和batch维度不变"""
        # 输入维度假设为 (batch, C, D, H, W) 或 (C, D, H, W)
        original_shape = volume.shape
        pad_dims = []
        
        # 仅处理最后三个维度（D/H/W）
        for dim in original_shape[-3:]:  # 遍历 D, H, W
            remainder = dim % base
            if remainder != 0:
                pad = base - remainder
                pad_before = pad // 2
                pad_after = pad - pad_before
                pad_dims.append((pad_before, pad_after))
            else:
                pad_dims.append((0, 0))
        
        # 前两个维度（batch/C）不填充，补充空填充参数
        pad_dims = [(0, 0)] * (volume.ndim - 3) + pad_dims
        
        # 对D/H/W使用反射填充
        padded_volume = np.pad(
            volume,
            pad_dims,
            mode='reflect'
        )
        return padded_volume

def create_data_loaders(config, mode='train', transform=None):
    """创建数据加载器"""
    dataset = CTMR3DDataset(config['data'], mode=mode, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(mode == 'train' and not config['training']['serial_batches']),
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    return loader

def main(config):
    # 快速训练设置
    cudnn.benchmark = True

    # 创建目录
    os.makedirs(config['directories']['log_dir'], exist_ok=True)
    os.makedirs(config['directories']['model_save_dir'], exist_ok=True)
    os.makedirs(config['directories']['sample_dir'], exist_ok=True)
    os.makedirs(config['directories']['result_dir'], exist_ok=True)

    transform = transforms.Compose([
        MedicalVolumePreprocessor(target_contrast=50.0),
    ])

    # 数据加载
    train_loader = create_data_loaders(config, mode='train', transform=transform) # 等待在本文件中实现, 也就是最重要的对比度函数
    test_loader = create_data_loaders(config, mode='test', transform=transform)

    # 初始化 Solver
    solver = Solver(train_loader, test_loader, config)

    # 运行训练或测试
    if config['misc']['mode'] == 'train':
        solver.train()
    elif config['misc']['mode'] == 'test':
        solver.test()


if __name__ == '__main__':
    # 默认配置
    default_config = {
        'model': {
            'g_conv_dim': 64,
            'd_conv_dim': 64,
            'g_repeat_num': 6,
            'd_repeat_num': 6,
            'lambda_cls': 1.0,
            'lambda_rec': 10.0,
            'lambda_gp': 10.0
        },
        'training': {
            'batch_size': 1,
            'num_iters': 200000,
            'num_iters_decay': 100000,
            'g_lr': 0.0001,
            'd_lr': 0.0001,
            'n_critic': 5,
            'beta1': 0.5,
            'beta2': 0.999,
            'resume_iters': None,
            'serial_batches': False,
            'num_workers': 4,
            'test_interval': 1000  # 每 1000 次迭代评估一次训练集与测试集
        },
        'testing': {
            'test_iters':200000,
        },
        'data': {
            'trainCT_root': ['test_data/siteA/train', 'test_data/siteB/train'],
            'trainMR_root': ['test_data/siteA/train', 'test_data/siteB/train'],
            'testCT_root': 'test_data/siteA/test',
            'testMR_root': 'test_data/siteA/test',
            'file_pattern': r"(patient\d+)_(ct|mr)\.npy",
            'preload': True # 将dataset直接加入内存
        },
        'directories': {
            'log_dir': 'CT2MRI_3DGAN_test/metric_loss',
            'model_save_dir': 'CT2MRI_3DGAN_test/models',
            'sample_dir': 'CT2MRI_3DGAN_test/samples',
            'result_dir': 'CT2MRI_3DGAN_test/results'
        },
        'misc': {
            'mode': 'train',
            'use_tensorboard': True,
            'log_step': 10,
            'sample_step': 1000,
            'model_save_step': 10000,
            'lr_update_step': 1000
        }
    }

    # 配置解析
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 配置文件路径
    parser.add_argument('--config', type=str, help='Path to config YAML file')

    # 模型配置
    parser.add_argument('--model.g_conv_dim', type=int, help='Generator第一层卷积通道数')
    parser.add_argument('--model.d_conv_dim', type=int, help='Discriminator第一层卷积通道数')
    parser.add_argument('--model.g_repeat_num', type=int, help='Generator残差块数量')
    parser.add_argument('--model.d_repeat_num', type=int, help='Discriminator步长卷积层数')
    parser.add_argument('--model.lambda_cls', type=float, help='分类损失权重')
    parser.add_argument('--model.lambda_rec', type=float, help='重建损失权重')
    parser.add_argument('--model.lambda_gp', type=float, help='梯度惩罚权重')

    # 训练配置
    parser.add_argument('--training.batch_size', type=int, help='批次大小')
    parser.add_argument('--training.num_iters', type=int, help='总训练迭代次数')
    parser.add_argument('--training.num_iters_decay', type=int, help='学习率衰减迭代次数')
    parser.add_argument('--training.g_lr', type=float, help='生成器学习率')
    parser.add_argument('--training.d_lr', type=float, help='判别器学习率')
    parser.add_argument('--training.n_critic', type=int, help='判别器更新次数/生成器更新次数')
    parser.add_argument('--training.beta1', type=float, help='Adam beta1参数')
    parser.add_argument('--training.beta2', type=float, help='Adam beta2参数')
    parser.add_argument('--training.resume_iters', type=int, help='恢复训练的迭代步数')
    parser.add_argument('--training.serial_batches', type=str2bool, help='是否按顺序取batch')
    parser.add_argument('--training.num_workers', type=int, help='数据加载线程数')
    parser.add_argument('--training.test_interval', type=int, help='测试集评估间隔')
    
    # 测试配置
    parser.add_argument('--testing.test_iters', type=int, default=200000, help='test model from this step')

    # 数据配置
    parser.add_argument('--data.trainCT_root', type=str, nargs='+', help='训练CT数据路径')
    parser.add_argument('--data.trainMR_root', type=str, nargs='+', help='训练MR数据路径')
    parser.add_argument('--data.testCT_root', type=str, nargs='+', help='测试CT路径')
    parser.add_argument('--data.testMR_root', type=str, nargs='+', help='测试MR路径')
    parser.add_argument('--data.file_pattern', type=str, help='用于从文件名中提取病例ID的正则表达式模式，确保正确匹配和加载数据文件')
    parser.add_argument('--data.preload', type=str2bool, help='是否将数据集预加载到内存，以加快数据访问速度')

    # 其他配置
    parser.add_argument('--misc.mode', type=str, choices=['train', 'test'], help='运行模式')
    parser.add_argument('--misc.use_tensorboard', type=str2bool, help='是否使用TensorBoard')
    parser.add_argument('--misc.log_step', type=int, help='日志记录间隔')
    parser.add_argument('--misc.sample_step', type=int, help='采样间隔')
    parser.add_argument('--misc.model_save_step', type=int, help='模型保存间隔')
    parser.add_argument('--misc.lr_update_step', type=int, help='学习率更新间隔')

    # 解析命令行参数
    cli_args = parser.parse_args()

    # 加载 YAML 配置文件
    file_config = load_yaml(cli_args.config) if cli_args.config else None

    # 合并配置
    final_config = merge_configs(default_config, file_config, cli_args)

     # 运行主程序
    main(final_config)