from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from collections import deque

@dataclass
class vertex:
    idx: int
    value: float

@dataclass
class barrier_zone:
    index:int # e.g. bi index
    
    left:int # starting time
    right:int # ending time
    top:float
    bottom:float
    
    volume:int
    types:int
    # 0: demand
    # 1: supply
    
    strength_rating:int # 0~10
    
    enter_bi_VP: Optional[List[Union[List[int], List[float]]]] = None
    leaving_bi_VP: Optional[List[Union[List[int], List[float]]]] = None
    
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