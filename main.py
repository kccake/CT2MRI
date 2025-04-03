import os
from utils import *
from models import *
from reggan3Dsolver import RegGAN3DSolver
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
    # print(f'\033[1;34m[info]\033[0m config: {config}')
    
    bugfree = BugFree(config['bugfree'])
    bugfree()
    
    # make log directory
    log_root = f"{config['misc']['log_root']}/{config['name']}"
    for dirtype in config['misc']['log_dir_names'].keys():
        os.makedirs(f"{log_root}/{config['misc']['log_dir_names'][dirtype]}", exist_ok=True)

    # trainloader = DataLoader(ImagesDataset3D(config['dataset'], train=True))
    # testloader = DataLoader(ImagesDataset3D(config['dataset'], train=False))
    solver = RegGAN3DSolver(config)
    solver.train()
    
        
if __name__ == '__main__':
    main()
