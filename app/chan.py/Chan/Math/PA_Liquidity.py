# PA: Price Action (order flow)
# ref: 价格行为分析框架 https://www.youtube.com/watch?v=20RnT4ruOmk
# ref: 供给需求区 https://www.youtube.com/watch?v=eujWFeE3TyE
# ref: 流动性交易策略 https://www.youtube.com/watch?v=UtspswubWSQ
# ref: 波段交易 https://tmipartner.com/lessons/swing-trade/
# ref: liquidity https://www.youtube.com/watch?v=YUUefUXeZwI
# ref: ICT Concepts Explained in 12 Minutes https://www.youtube.com/watch?v=arJI_3HhgxA
# ref: My List of Top ICT Concepts for Successful Trading https://www.youtube.com/watch?v=x7g2JU1lc_4
# ref: Learn ICT Concepts in 30 Minutes! https://www.youtube.com/watch?v=dokgVf0YdGY

# 适合swing trade（波段）级别分析
# 1. liquidity pool
# 2. H->L: Market Shift(no longer HH) / Break of Market(MS then break last 2 low)
#       等效于缠论线段转折（1买反转）
# 3. Premium/Discount area(Fibonacci) (balanced pnl at 0.7)
# 4. Liquidity Void/Imbalance: SIBI/BISI (Buyside Liquidity / Sellside inefficiency)
# 5. Turtle Soup / Stop Hunt / Stop Liquidity(external Liquidity)
# 6. FVG: Fair Value Gap(internal liquidity)
# 7. Daily Bias
# 8. Supply/Demand zone, Order Block, Mitigation Block, Break Block, Rejection Block
# 9. CISD: Change in state of delivery
# 10. market maker model

import os, sys
import math
import numpy as np
from typing import List, Dict
from enum import Enum, auto
from dataclasses import dataclass

from Chan.Math.PA_types import vertex, zone
    
class PA_Liquidity:
    # for supply/demand zone, the strength the bar goes to FX is important, 
    # thus for i-th bar that is a bottom FX (as bottom of supply zone), 
    # take close of negative bar and open of positive bar as top of supply zone
    POT_SD_ZONE = 0 # potential supply/ demand zone (formed with FX)
    SD_ZONE = 1 # established supply/demand zone (formed with breakthrough)
    
    def __init__(self):
        self.vertices:List[vertex] = []
        
        # list[0]: zones_formed
        # list[1]: zones_forming
        self.supply_zones:List[List[zone]] = [[],[]]
        self.demand_zones:List[List[zone]] = [[],[]]
        self.order_blocks:List[List[zone]] = [[],[]]
        self.mitigation_zones:List[List[zone]] = [[],[]]
        self.break_zones:List[List[zone]] = [[],[]]
        self.rejection_zones:List[List[zone]] = [[],[]]
        
    def add_vertex(self, new_vertex:vertex, end_open:float, end_close:float):
        TOP = 1
        BOT = -1
        default_end = 1<<31
        if len(self.vertices) >= 2:
            last_vertex = self.vertices[-1]
            second_last_vertex = self.vertices[-2]
            delta_y = new_vertex.value - last_vertex.value
            delta_x = new_vertex.idx - last_vertex.idx
            FX_dir = TOP if delta_y > 0 else BOT
            
            # close all zones within last bi when conditions are met
            for zones in [
                self.supply_zones,
                self.demand_zones,
                ]:
                for zone_forming in zones[1]:
                    bi_top:float = last_vertex.value if FX_dir==BOT else new_vertex.value
                    bi_bottom:float = new_vertex.value if FX_dir==BOT else last_vertex.value
                        
                    if bi_top > zone_forming.top and bi_bottom < zone_forming.bottom:
                        zone_forming.idx_end = new_vertex.idx
                        zones[0].append(zone_forming)
                        print('zone formed: ', len(zones[0]), len(zones[1]))
                        zones[1].remove(zone_forming)
            
            # update all forming zones
            if FX_dir==BOT:
                self.supply_zones[1].append(zone(new_vertex.idx, default_end, end_open, new_vertex.value))
            else:
                self.demand_zones[1].append(zone(new_vertex.idx, default_end, new_vertex.value, end_open))
        self.vertices.append(new_vertex)