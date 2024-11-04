from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from collections import deque

@dataclass
class vertex:
    idx: int
    value: float

@dataclass
class barrier_zone:
    # typically supply and demand
    # NOTE: if a leaving bi is strong enough to 
    #           1. create a BoS(Break of Structure)
    #           2. create a leaving bi with opposite-side-inefficiency
    #       then this barrier_zone creating the leaving bi is strong enough
    #       to be considered an order block
    
    # if for some reason an order block is broken by even stronger
    # opposite force in the next bi (preferably with inefficiency), then
    # this region is of opposite effect and are marked as breaker block
    
    # Choch(Change of Character):
    #   for a trailing BoS that established a trend, if a BoS is formed
    #   from the other side and break the last trend (entering bi) that is
    #   at least 1/2 of the second last trend bi, then this 1st BoS of opposite
    #   side is considered strong enough to be a ChoCh and indicates a change in
    #   trend
    index:int # e.g. bi index
    
    left:int # starting time
    top:float
    bottom:float
    
    right0:int  # 1st ending time
    right1:int  # 2nd ending time
    # ... with weaker effects 
    
    # barrier types:
    # 0: demand
    # 1: supply
    # 2: demand (1st break supply)
    # 3: supply (1st break demand)
    # 4: ... with weaker effects
    type:int|None = None
    
    volume:int = 0
    
    strength_rating:int = 0 # 0~10
    
    # if re-tested and not broken, new VP will be recorded in the newly formed zones
    # so no need to record here
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