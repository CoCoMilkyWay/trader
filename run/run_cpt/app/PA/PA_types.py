from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from collections import deque

@dataclass
class vertex:
    idx: int
    value: float
    ts: float

@dataclass
class barrier_zone: # supply and demand and their derivatives
    
    # BoS(Break of Structure):
    #   for 2 consecutive bi, a BoS is formed if the new bi break
    #   the previous bi's level
    
    # Choch(Change of Character):
    #   1.  for a trailing of BoSs that established a trend
    #   2.  a new BoS is formed from the other side
    #   3.  it also breaks the last trend (entering bi) that is
    #           at least 1/2 of the second last trend bi
    # then this 1st BoS of opposite side is considered strong enough to 
    # be a ChoCh and indicates a change in trend
    
    # OB(Order Block)(strong support/resistance): if a leaving bi is strong enough to 
    #   1.  create a BoS
    #   2.  create a leaving bi with opposite-side-inefficiency
    #           FVG(fair value gap)
    #           SiBi(sell side imbalance(strong) buy side inefficiency(weak))
    #           BiSi(buy side imbalance(strong) sell side inefficiency(weak))
    # then this barrier_zone creating the leaving bi is considered strong enough
    # to be an order block
    
    # BB(Breaker Blocks)(failed order block)(potential reversal):
    #   if for some reason an order block is broken by even stronger
    #   opposite force in the next bi (preferably with inefficiency), then
    #   this region is of opposite effect and are marked as breaker block
    
    # MB(Mitigation Blocks)(incomplete/under-performing order block):
    #   1.  failed to break previous high/low (normal support/resistance)
    #   2.  or order block that is touched (like chattered glass, better wait for it to break
    #       and trade breakthrough)
    # if MB is touched frequently, wait for it to shatter(breakthrough), however, if it is touched
    #   less frequently, it may gather enough strength to hold and trade reversal
    
    # Reclaimed Order Block:
    # broken BB/MB that revert back to original order block
    
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
    
    tolerance:float = 0.0
    num_touch:int = 0
    
    # BoS/ChoCh
    BoS:Optional[List[int|float]] = None
    ChoCh:bool = False
    
    # OB
    OB:bool = False
    BB:bool = False
    MB:bool = False
    
    # FVG
    # contrary to common method, we use VP to determine FVG here:
    #   1.  consider pnl trading opportunity of 1.5~3
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
        