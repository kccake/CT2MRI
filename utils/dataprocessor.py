import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Union


class CTMR3DDataset(Dataset):
    def __init__(self, 
                 config: dict,
                 mode: str = 'train', 
                 transform: callable = None, # 数据增强函数
                 ):
        """
        支持多目录、强化配对的3D数据集类
        
        参数：
            config : 至少包含以下键的字典
                trainCT_root : Union[str, List[str]]  训练CT数据路径（单个或多个）
                trainMR_root : Union[str, List[str]]  训练MR数据路径
                testCT_root : Union[str, List[str]]   测试CT路径
                testMR_root : Union[str, List[str]]   测试MR路径
                file_pattern : str                   用于从文件名提取病例ID的正则表达式
                preload : bool                       是否预加载数据到内存
            mode : str                               数据集模式（'train' 或 'test'）
            transform : Callable                     数据增强函数，应处理 (C,D,H,W) 格式张量
        """
        print(f"\033[1;34m[Info]\033[0m Initializing dataset in \033[34m{mode}\033[0m mode")
        # 从 config 中提取参数
        self.file_pattern = config.get('file_pattern', r".*patient[A-Za-z]*\s*\d+_(ct|mr)\.npy")
        self.preload = config.get('preload', True)
        self.id_regex = re.compile(self.file_pattern)
        self.transfrom = transform
        self.max_depth = config.get('max_depth', 29)
        
        self.ct_file_pattern = config.get('ct_file_pattern', r".*patient[A-Za-z]*\s*\d+_ct\.npy") # NEW
        self.mr_file_pattern = config.get('mr_file_pattern', r".*patient[A-Za-z]*\s*\d+_mr\.npy") # NEW
        self.ct_id_regex = re.compile(self.ct_file_pattern) # NEW
        self.mr_id_regex = re.compile(self.mr_file_pattern) # NEW
        
        # 根据 mode 选择数据路径
        if mode == 'train':
            ct_dirs = config['trainCT_root']
            mr_dirs = config['trainMR_root']
        else:
            ct_dirs = config['testCT_root']
            mr_dirs = config['testMR_root']

        # 统一转化为列表
        self.ct_dirs = [ct_dirs] if isinstance(ct_dirs, str) else ct_dirs
        self.mr_dirs = [mr_dirs] if isinstance(mr_dirs, str) else mr_dirs
        # 构建病例配对
        self.pairs = self._build_pairs() # 只是文件路径的配对, 而不是实体的配对
        # 预加载数据到内存
        if self.preload:
            print(f"\033[1;34m[Info]\033[0m Preloading data...")
            self._preload_data()
        
    def _build_pairs(self) -> List[tuple]:
        """建立CT-MR配对, 确保数据一致性"""
        # ct_files = self._find_files(self.ct_dirs)
        # mr_files = self._find_files(self.mr_dirs)
        ct_files = self._find_files(self.ct_dirs, self.ct_id_regex) # NEW
        mr_files = self._find_files(self.mr_dirs, self.mr_id_regex) # NEW
        
        common_ids = set(ct_files.keys()) & set(mr_files.keys())
        if not common_ids:
            raise ValueError("No valid CT-MR pairs found!")
        return [(ct_files[_id], mr_files[_id]) for _id in sorted(common_ids)]

    def _find_files(self, directories: List[str]) -> dict:
        """扫描多个目录，构建{病例ID: 文件路径}映射"""
        file_map = {}
        for d in directories:
            dir_tag = self._get_dir_tag(d)
            for fname in os.listdir(d):
                match = self.id_regex.match(fname)
                if match and fname.endswith('.npy'):
                    case_id = f"{dir_tag}_{match.group(1)}_{fname}"
                    file_map[case_id] = os.path.join(d, fname)
        return file_map # 这个map已经有了dir_tag作为保证, 哪怕病人的id号相同, 也可以进行区分, 从而不会因为病人的id号相同, 而导致被覆盖
    
    def _find_files(self, directories: List[str], regex)-> dict:  # NEW
        """扫描多个目录，构建{病例ID: 文件路径}映射"""
        file_map = {}
        for d in directories:
            dir_tag = self._get_dir_tag(d)
            for fname in os.listdir(d):
                match = regex.match(fname)
                if match and fname.endswith('.npy'):
                    filename = match.group(0)
                    # 分割，去掉最后一个下划线后的内容 去掉内容如'_ct.npy'
                    filename_untyped = filename.rsplit('_', 1)[0]
                    case_id = f"{dir_tag}_{filename_untyped}"
                    file_map[case_id] = os.path.join(d, fname)
        return file_map
    
    def _get_dir_tag(self, path: str) -> str:
        """生成目录标识(示例: siteA_train)"""
        parts = os.path.normpath(path).split(os.sep)[-2:]
        return "_".join(parts)

    def _preload_data(self):
        """将全部数据加载到内存"""
        self.ct_tensors = []
        self.mr_tensors = []
            

        for ct_path, mr_path in self.pairs:
            # 加载并转换为张量
            ct = np.load(ct_path)
            mr = np.load(mr_path)

            # 转换为 (C,D,H,W) 张量
            ct_tensor = torch.as_tensor(ct, dtype=torch.float32).unsqueeze(0)
            mr_tensor = torch.as_tensor(mr, dtype=torch.float32).unsqueeze(0)
            
            # 将D维度padding从上下两方填充到max_depth，用0填充
            if ct_tensor.shape[1] < self.max_depth:
                pad_total = self.max_depth - ct_tensor.shape[1]
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                ct_tensor = torch.nn.functional.pad(ct_tensor, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=0)
                mr_tensor = torch.nn.functional.pad(mr_tensor, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=0)
            
            self.ct_tensors.append(ct_tensor)
            self.mr_tensors.append(mr_tensor)
            
        # print(f"预加载完成！共加载 {len(self.ct_tensors)} 个样本")
        print(f"\033[1;32m[Success]\033[0m Preloading completed! \033[34m{len(self.ct_tensors)}\033[0m samples loaded\n")

    def __getitem__(self, idx):
        if self.preload:
            ct_tensor = self.ct_tensors[idx]
            mr_tensor = self.mr_tensors[idx]
        else:
            # 后备方案：动态加载
            ct_path, mr_path = self.pairs[idx]
            ct = np.load(ct_path)
            mr = np.load(mr_path)
            ct_tensor = torch.as_tensor(ct, dtype=torch.float32).unsqueeze(0)
            mr_tensor = torch.as_tensor(mr, dtype=torch.float32).unsqueeze(0)
        
        # 应用增强
        if self.transfrom:
            # print(f"\033[1;34m[Info]\033[0m Applying \033[34mtransform\033[0m for data augmentation")
            ct_tensor = self.transfrom(ct_tensor)
            mr_tensor = self.transfrom(mr_tensor)
        
        return {'CT': ct_tensor, 'MR': mr_tensor}
        
    def __len__(self):
        return len(self.pairs)

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