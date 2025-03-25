import os
from utils import *
from models import *
# from solver import Solver
from solver_diff import DiffusionSolver
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

    trainloader = DataLoader(ImagesDataset3D(config['dataset'], train=True))
    testloader = DataLoader(ImagesDataset3D(config['dataset'], train=False))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    solver = DiffusionSolver(config, trainloader, testloader)
    solver.train()
    
    # 采样测试
    sample_ct = next(iter(testloader))['CT'].to(device)
    generated_mr = solver.sample(sample_ct, steps=100)
    # 将generated_mr保存到 ./test/ 文件夹下
    torch.save(sample_ct, './test/sample_ct.pt')
    torch.save(generated_mr, './test/generated_mr.pt')
    
    
        
if __name__ == '__main__':
    main()
