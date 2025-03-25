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
    for dirname in config['log_dir_names']:
        os.makedirs(f"{log_root}/{dirname}", exist_ok=True)

    # traindataset = ImagesDataset2D(config['dataset'], train=True)
    # testdataset = ImagesDataset2D(config['dataset'], train=False)
    trainloader = DataLoader(ImagesDataset2D(config['dataset'], train=True))
    testloader = DataLoader(ImagesDataset2D(config['dataset'], train=False))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
        
if __name__ == '__main__':
    main()
