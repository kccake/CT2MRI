import cv2
import numpy as np
import torch

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
        # print(f'min(volume): {np.min(volume)};  max(volume): {np.max(volume)}')
        # 对D/H/W使用反射填充
        # print(np.min(volume), np.max(volume))
        padded_volume = np.pad(
            volume,
            pad_dims,
            mode='reflect'
            # mode='constant', constant_values=-1 # volume的值 范围 [-1,1]
        )
        return padded_volume

class NoneMedicalVolumePreprocessor(MedicalVolumePreprocessor):
    """3D医学影像预处理工具类(不做数据增强，只变形)"""
    def __init__(self, target_contrast=50.0):
        self.target_contrast = target_contrast
        
    def __call__(self, volume):
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
            
        # 对比度统一处理（不做处理）
        # volume = self._adjust_contrast(volume)
        
        # 数据归一化到[-1,1]
        volume = self._normalize(volume)
        
        # 维度填充到4的倍数
        volume = self._pad_to_multiple(volume, base=4)
        
        return torch.from_numpy(volume).float()
        

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