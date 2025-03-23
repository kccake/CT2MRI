import numpy as np
import torch

class Solver:
    def __init__(self, config):
        # 加载参数
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化model(加载or创建)
    
        # 其它
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def test(self):
        pass