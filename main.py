import os
from utils import *
import argparse

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

    # print(traindataset[0]['CT'].shape) # 

if __name__ == '__main__':
    main()
