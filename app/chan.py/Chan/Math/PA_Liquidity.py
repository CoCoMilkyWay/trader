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

# ICT SMC concepts:
# Customizable Timeframe - Calculate ICT concepts on off-chart timeframes
# Unicorn Strategy Model
# 2022 Strategy Model
# Liquidity Raid Strategy Model
# OTE (Optimal Trade Entry) Strategy Model
# Silver Bullet Strategy Model
# Order blocks
# Breaker blocks
# Rejection blocks
# FVG
# Strong highs and lows
# Displacements
# Liquidity sweeps
# Power of 3
# ICT Macros
# HTF previous bar high and low
# Break of Structure indications
# Market Structure Shift indications
# Equal highs and lows
# Swings highs and swing lows
# Fibonacci TPs and SLs
# Swing level TPs and SLs
# Previous day high and low TPs and SLs

import os, sys
import math
import numpy as np
from typing import List, Dict
from enum import Enum, auto
from dataclasses import dataclass

from Chan.Math.PA_types import vertex, barrier_zone

class PA_Liquidity:
    # for supply/demand zone, the strength the bar goes to FX is important,
    # thus for i-th bar that is a bottom FX (as bottom of supply zone),
    # take close of negative bar and open of positive bar as top of supply zone
    POT_SD_ZONE = 0 # potential supply/ demand zone (formed with FX)
    SD_ZONE = 1 # established supply/demand zone (formed with breakthrough)
    
    def __init__(self):
        self.vertices:List[vertex] = []
        
        # xxx_zones = List[zones_formed, zones_forming]
        self.supply_zones:      List[List[barrier_zone]] = [[],[]]
        self.demand_zones:      List[List[barrier_zone]] = [[],[]]
        self.order_blocks:      List[List[barrier_zone]] = [[],[]]
        self.mitigation_zones:  List[List[barrier_zone]] = [[],[]]
        self.break_zones:       List[List[barrier_zone]] = [[],[]]
        self.rejection_zones:   List[List[barrier_zone]] = [[],[]]
        
        # average & percentile
        self.supply_volume_sum: float = 0
        self.supply_sample_num: int = 0
        self.demand_volume_sum: float = 0
        self.demand_sample_num: int = 0
        
        self.snapshot:List = [] # snapshot of all liquidity zones
        
    def add_vertex(self, new_vertex:vertex, end_open:float, end_close:float, end_volume:int):
        TOP = 1
        BOT = -1
        default_end = 1<<31
        if len(self.vertices) >= 2:
            last_vertex = self.vertices[-1]
            delta_y = new_vertex.value - last_vertex.value
            delta_x = new_vertex.idx - last_vertex.idx
            FX_type = TOP if delta_y > 0 else BOT
            bi_top:float = last_vertex.value if FX_type==BOT else new_vertex.value
            bi_bottom:float = new_vertex.value if FX_type==BOT else last_vertex.value

            # close all zones within last bi when conditions are met
            for zones in [
                self.supply_zones,
                self.demand_zones,
                ]:
                zones_forming = []
                for zone_forming in zones[1]:
                    zone_broken = (bi_top > zone_forming.top) and (bi_bottom < zone_forming.bottom)
                    if zone_broken:
                        zone_level:float = (zone_forming.top + zone_forming.bottom)/2
                        zone_ratio_in_bi:float = (zone_level - bi_bottom) / abs(delta_y)
                        if FX_type==TOP:
                            zone_idx:int = last_vertex.idx + int(zone_ratio_in_bi * delta_x)
                        else:
                            zone_idx:int = last_vertex.idx + int((1-zone_ratio_in_bi) * delta_x)
                        zone_formed = zone_forming
                        zone_formed.idx_end = zone_idx
                        zones[0].append(zone_formed) # zone_formed
                        # print('zone formed: ', len(zones[0]), len(zones[1]))
                    else:
                        zones_forming.append(zone_forming) # zone_forming
                zones[1] = zones_forming
                
            # update all forming zones
            if FX_type==BOT:
                self.demand_volume_sum, self.demand_sample_num, strength_rating = self.get_strength_rating(self.demand_volume_sum, self.demand_sample_num, end_volume)
                self.demand_zones[1].append(barrier_zone(new_vertex.idx, default_end, end_open, new_vertex.value, end_volume, 0, strength_rating))
            else:
                self.supply_volume_sum, self.supply_sample_num, strength_rating = self.get_strength_rating(self.supply_volume_sum, self.supply_sample_num, end_volume)
                self.supply_zones[1].append(barrier_zone(new_vertex.idx, default_end, new_vertex.value, end_open, end_volume, 1, strength_rating))
        self.vertices.append(new_vertex)
        
    @staticmethod
    def get_strength_rating(historical_sum:float, historical_sample_num:int, new_sample_value:float):
        new_sum = historical_sum + new_sample_value
        new_sample_num = historical_sample_num + 1
        new_average = new_sum / new_sample_num
        strength_rating = round(new_sample_value / new_average / 0.5) # 0:<25%, 1:<75%, 2:<125%, ...
        if strength_rating > 10:
            strength_rating = 10
        return new_sum, new_sample_num, strength_rating