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

# ver.2.0      

# class ContrastAdjuster:
#     """对比度调整类"""
#     def __init__(self, target_contrast=50.0):
#         self.target_contrast = target_contrast
        
#     def __call__(self, volume):
#         # 转换为numpy数组处理（假设输入是numpy格式）
#         if isinstance(volume, torch.Tensor):
#             volume = volume.numpy()
        
#         # 对比度统一处理
#         volume_255 = cv2.convertScaleAbs(
#             volume, 
#             alpha=self.target_contrast / (volume.mean() + 1e-8),
#             beta=0
#         )
#         return volume_255

# class Normalizer:
#     """数据归一化类"""
#     def __call__(self, volume):
#         return (volume - 127.5) / 127.5

# class PaddingToMultiple:
#     """维度填充到指定倍数的类"""
#     def __init__(self, base=4):
#         self.base = base
        
#     def __call__(self, volume):
#         """将D/H/W维度填充到base的整数倍，保持C和batch维度不变"""
#         # 输入维度假设为 (batch, C, D, H, W) 或 (C, D, H, W)
#         original_shape = volume.shape
#         pad_dims = []
        
#         # 仅处理最后三个维度（D/H/W）
#         for dim in original_shape[-3:]:  # 遍历 D, H, W
#             remainder = dim % self.base
#             if remainder != 0:
#                 pad = self.base - remainder
#                 pad_before = pad // 2
#                 pad_after = pad - pad_before
#                 pad_dims.append((pad_before, pad_after))
#             else:
#                 pad_dims.append((0, 0))
        
#         # 前两个维度（batch/C）不填充，补充空填充参数
#         pad_dims = [(0, 0)] * (volume.ndim - 3) + pad_dims
        
#         # 对D/H/W使用反射填充
#         padded_volume = np.pad(
#             volume,
#             pad_dims,
#             mode='reflect'
#         )
#         return padded_volume

# class MedicalVolumePreprocessor:
#     """3D医学影像预处理工具类"""
#     def __init__(self, contrast_adjuster=None, normalizer=None, padding_to_multiple=None):
#         self.contrast_adjuster = contrast_adjuster
#         self.normalizer = normalizer
#         self.padding_to_multiple = padding_to_multiple
        
#     def __call__(self, volume):
#         # 转换为numpy数组处理（假设输入是numpy格式）
#         if isinstance(volume, torch.Tensor):
#             volume = volume.numpy()
        
#         # 对比度调整
#         if self.contrast_adjuster is not None:
#             volume = self.contrast_adjuster(volume)
        
#         # 数据归一化
#         if self.normalizer is not None:
#             volume = self.normalizer(volume)
        
#         # 维度填充
#         if self.padding_to_multiple is not None:
#             volume = self.padding_to_multiple(volume)
        
#         return torch.from_numpy(volume).float()

# 使用示例
# transform = transforms.Compose([
#     MedicalVolumePreprocessor(
#         contrast_adjuster=ContrastAdjuster(target_contrast=50.0),
#         normalizer=Normalizer(),
#         padding_to_multiple=PaddingToMultiple(base=4)
#     )
# ])


# ver.2.1
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