from dataclasses import dataclass
from typing import List, Dict

@dataclass
class vertex:
    idx: int
    value: float

@dataclass
class barrier_zone:
    idx_start:int # starting time
    idx_end:int # ending time
    top:float
    bottom:float
    init_volume:int
    types:int
    # 0: demand
    # 1: supply
    strength_rating:int # 0~10
    
    def __init_(self):
        self.broken:bool = False
        
        # multiple zone can merge when tested but not broken
        self.test_num:int = 0
        self.test_incident_angle:List[float] = []
        self.test_strength:List[float] = []
        self.test_volume:List[int] = []

        # broken zone can be retested from the other side
        self.rev_test_num:int = 0
        self.rev_test_incident_angle:List[float] = []
        self.rev_test_strength:List[float] = []
        self.rev_test_volume:List[int] = []