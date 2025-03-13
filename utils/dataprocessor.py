import cv2
import numpy as np
import torch

class ContrastAdjuster:
    """对比度调整类"""
    def __init__(self, target_contrast=50.0):
        self.target_contrast = target_contrast
        
    def __call__(self, volume):
        # 如果输入是 torch.Tensor，转换为 numpy 数组
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # 对比度调整
        volume_255 = cv2.convertScaleAbs(
            volume, 
            alpha=self.target_contrast / (volume.mean() + 1e-8),
            beta=0
        )
        return torch.from_numpy(volume_255).float()

class Normalizer:
    """数据归一化类"""
    def __call__(self, volume):
        # 如果输入是 torch.Tensor，转换为 numpy 数组
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # 数据归一化到 [-1, 1]
        normalized_volume = (volume - 127.5) / 127.5
        return torch.from_numpy(normalized_volume).float()

class PaddingToMultiple:
    """维度填充到指定倍数的类"""
    def __init__(self, base=4):
        self.base = base
        
    def __call__(self, volume):
        # 如果输入是 torch.Tensor，转换为 numpy 数组
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        """将 D/H/W 维度填充到 base 的整数倍，保持 C 和 batch 维度不变"""
        original_shape = volume.shape
        pad_dims = []
        
        # 仅处理最后三个维度（D/H/W）
        for dim in original_shape[-3:]:  # 遍历 D, H, W
            remainder = dim % self.base
            if remainder != 0:
                pad = self.base - remainder
                pad_before = pad // 2
                pad_after = pad - pad_before
                pad_dims.append((pad_before, pad_after))
            else:
                pad_dims.append((0, 0))
        
        # 前两个维度（batch/C）不填充，补充空填充参数
        pad_dims = [(0, 0)] * (volume.ndim - 3) + pad_dims
        
        # 对 D/H/W 使用反射填充
        padded_volume = np.pad(
            volume,
            pad_dims,
            mode='reflect'
        )
        return torch.from_numpy(padded_volume).float()