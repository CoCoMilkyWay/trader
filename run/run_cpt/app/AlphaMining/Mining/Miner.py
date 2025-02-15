import os, sys
import torch
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from Mining.Data.Data import Data

class Miner:
    def __init__(self):
        self.Data = Data(init=True)
        
if __name__ == '__main__':
    M = Miner()

