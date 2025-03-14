import os
import re
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
            config : 包含以下键的字典
                trainCT_root : Union[str, List[str]]  训练CT数据路径（单个或多个）
                trainMR_root : Union[str, List[str]]  训练MR数据路径
                testCT_root : Union[str, List[str]]   测试CT路径
                testMR_root : Union[str, List[str]]   测试MR路径
                file_pattern : str                   用于从文件名提取病例ID的正则表达式
                preload : bool                       是否预加载数据到内存
            mode : str                               数据集模式（'train' 或 'test'）
            transform : Callable                     数据增强函数，应处理 (C,D,H,W) 格式张量
        """

        # 从 config 中提取参数
        self.file_pattern = config.get('file_pattern', r"(patient\d+)_(ct|mr)\.npy")
        self.preload = config.get('preload', True)
        self.id_regex = re.compile(self.file_pattern)
        self.transfrom = transform
        
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
            self._preload_data()
        
    def _build_pairs(self) -> List[tuple]:
        """建立CT-MR配对, 确保数据一致性"""
        ct_files = self._find_files(self.ct_dirs, self.ct_id_regex)
        mr_files = self._find_files(self.mr_dirs, self.mr_id_regex)

        common_ids = set(ct_files.keys()) & set(mr_files.keys())
        if not common_ids:
            raise ValueError("No valid CT-MR pairs found!")
        
        return [(ct_files[_id], mr_files[_id]) for _id in sorted(common_ids)]

    def _find_files(self, directories: List[str],regex) -> dict:
        """扫描多个目录，构建{病例ID: 文件路径}映射"""
        file_map = {}
        for d in directories:
            dir_tag = self._get_dir_tag(d)
            for fname in os.listdir(d):
                match = regex.match(fname)
                if match and fname.endswith('.npy'):
                    # case_id = f"{dir_tag}_{match.group(1)}_{fname}"
                    # file_map[case_id] = os.path.join(d, fname)

                    filename = match.group(0)
                    # 分割，去掉最后一个下划线后的内容 去掉内容如'_ct.npy'
                    filename_untyped = filename.rsplit('_', 1)[0]
                    case_id = f"{dir_tag}_{filename_untyped}"
                    file_map[case_id] = os.path.join(d, fname)
        return file_map # 这个map已经有了dir_tag作为保证, 哪怕病人的id号相同, 也可以进行区分, 从而不会因为病人的id号相同, 而导致被覆盖
    
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
                
            self.ct_tensors.append(ct_tensor)
            self.mr_tensors.append(mr_tensor)
            
        print(f"预加载完成！共加载 {len(self.ct_tensors)} 个样本")

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
            ct_tensor = self.transfrom(ct_tensor)
            mr_tensor = self.transfrom(mr_tensor)
        
        return {'CT': ct_tensor, 'MR': mr_tensor}
        
    def __len__(self):
        return len(self.pairs)
    
    