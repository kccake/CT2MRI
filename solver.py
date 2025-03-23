from models import *
from utils import *

import os
import time
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Solver(object):
    def __init__(self, config, train_loader, val_loader):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._build_model()
        # self._load_checkpoint()
        pass
    
    def _build_model(self):
        pass
    
    def train(self):
        pass
    
    def _save_checkpoint(self, epoch):
        pass
    
    def _load_checkpoint(self):
        pass
    
    def _save_log(self, log):
        pass
    
    def _save_sample(self, sample):
        pass
    
    def _save_result(self, result):
        pass
        