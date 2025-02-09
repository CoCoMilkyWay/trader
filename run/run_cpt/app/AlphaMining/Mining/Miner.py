import os
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Miner:
    def __init__(self):
        self.meta = torch.load('./Data/meta.pt')
        self.num_timestamps = self.meta[0]
        self.feature_names = self.meta[1]
        self.label_names = self.meta[2]
        self.code_info = self.meta[3]
        
        self.tensor = torch.load('./Data/tensor.pt')

        print(f'Days:{self.num_timestamps/60/24}')
        print(f'Codes:{len(self.code_info.keys())}')
        print(f'Features:{len(self.feature_names)}')
        print(f'Labels:{len(self.label_names)}')
        
if __name__ == '__main__':
    M = Miner()