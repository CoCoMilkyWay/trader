import torch
import random
from typing import Dict, List, Tuple

class CrossSectionAnalysis:
    def __init__(self, code_info: Dict[str, Dict], shared_tensor:torch.Tensor,):
        self.num_timestamps:int = 0
        self.code_info = code_info
        self.shared_tensor = shared_tensor
        self.init = True

    def analyze(self, results: List[Tuple[int, float]]):
        if self.num_timestamps == 0:
            return 0, 0.0
        else:
            # TODO
            self.num_timestamps += 1
            return random.randint(1, 100), random.uniform(1.5, 5.5)
