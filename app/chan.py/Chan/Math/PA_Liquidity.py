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
    # TOP and BOT of zones:
    # for supply/demand zone, the strength the bar goes to FX is important,
    # thus for i-th bar that is a bottom FX (as bottom of supply zone),
    # take close of negative bar and open of positive bar as top of supply zone
    
    # reversion between supply and demand:
    # most trades positions are opened around critical zones, once the zone is broken
    # it marks the conclusion of most reversion strategies,
    # also marks the beginning of some breakthrough strategies,
    # thus when price come back to this region, it would act as zone of reversed polarity
    # note that the effect of 2nd/3rd/... rejection is probably weaker than 1st rejection
    
    # TODO: need more accurate description
    # NOTE: liquidity are off different quality at different time, for organization to gather liquidity,
    #       the most effective location to gather equity is:
    #       1. at breakthrough, we are sure that retail-trader comes in
    #       2. pull up, but not far, they would not sell
    #       3. pull down, consolidation around or below (not far) previous 
    #          breakthrough position (with large retail volume)
    #       4. liquidity hunt on single equity (need volume, purpose is to wash-people-out, need to be unexpected)
    #       5. pull up (sell-out) again
    # NOTE: on single equity, lots of retail-holding-volume = no potential future pull up
    def __init__(self):
        self.bi_index: int = 0
        self.vertices:List[vertex] = []
        
        # xxx_zones = List[zones_formed, zones_forming]
        self.barrier_zones:     List[List[barrier_zone]] = [[],[]] # [formed, forming]
        self.order_blocks:      List[List[barrier_zone]] = [[],[]] # [formed, forming]
        self.mitigation_zones:  List[List[barrier_zone]] = [[],[]] # [formed, forming]
        self.break_zones:       List[List[barrier_zone]] = [[],[]] # [formed, forming]
        self.rejection_zones:   List[List[barrier_zone]] = [[],[]] # [formed, forming]
        
        # average & percentile
        self.supply_volume_sum: float = 0
        self.supply_sample_num: int = 0
        self.demand_volume_sum: float = 0
        self.demand_sample_num: int = 0
        
        self.snapshot:List = [] # snapshot of all liquidity zones
        
    def add_vertex(self, new_vertex:vertex, end_open:float, end_close:float):
        TOP = 1
        BOT = -1
        default_end = 1<<31
        
        self.bi_index += 1
        
        if len(self.vertices) >= 2:
            last_vertex = self.vertices[-1]
            delta_y = new_vertex.value - last_vertex.value
            delta_x = new_vertex.idx - last_vertex.idx
            FX_type = TOP if delta_y > 0 else BOT
            if FX_type==BOT:
                bi_top:float = last_vertex.value
                bi_bottom:float = new_vertex.value
            else:
                bi_top:float = new_vertex.value
                bi_bottom:float = last_vertex.value
                
            # close all zones within last bi when conditions are met
            for zones in [
                self.barrier_zones,
                ]:
                zones_forming_buffer = []
                for zone_forming in zones[1]:
                    zone_broken = (bi_top > zone_forming.top) and (bi_bottom < zone_forming.bottom)
                    if zone_broken:
                        zone_type = zone_forming.type
                        
                        # calculating horizontal span
                        zone_level:float = (zone_forming.top + zone_forming.bottom)/2
                        zone_ratio_in_bi:float = (zone_level - bi_bottom) / abs(delta_y)
                        if FX_type==TOP:
                            zone_idx:int = last_vertex.idx + int(zone_ratio_in_bi * delta_x)
                        else:
                            zone_idx:int = last_vertex.idx + int((1-zone_ratio_in_bi) * delta_x)
                        
                        # delete VP if zone is already formed to save memory
                        # NOTE: after 1st breakthrough, the old volume already means nothing
                        #       (even though this position is still critical with reversed polarity),
                        #       the only thing we care is the new breakthrough volume, which will be
                        #       recorded in the volume of newly formed zone(not the old zone)
                        zone_forming.enter_bi_VP = None # refer to above
                        zone_forming.leaving_bi_VP = None # refer to above
                        if zone_type == 0: # demand
                            zone_forming.right0 = zone_idx
                            zone_forming.type = 3
                            zones_forming_buffer.append(zone_forming) # zone_forming
                        elif zone_type == 1: # supply
                            zone_forming.right0 = zone_idx
                            zone_forming.type = 2
                            zones_forming_buffer.append(zone_forming) # zone_forming
                        elif zone_type == 2: # demand (1st break supply)
                            zone_forming.right1 = zone_idx
                            zones[0].append(zone_forming) # zone_formed
                        elif zone_type == 3: # supply (1st break demand)
                            zone_forming.right1 = zone_idx
                            zones[0].append(zone_forming) # zone_formed
                    else:
                        zones_forming_buffer.append(zone_forming) # zone_forming
                zones[1] = zones_forming_buffer
                
            # add new forming zones
            if FX_type==BOT:
                zone_bot = new_vertex.value
                zone_top = min(end_open, zone_bot + 0.1 * abs(delta_y)) # avoid zone to be too thick (from rapid price change)
                self.barrier_zones[1].append(barrier_zone(self.bi_index, new_vertex.idx, zone_top, zone_bot, default_end, default_end, 0, 0, 0, None, None))
            else:
                zone_top = new_vertex.value
                zone_bot = max(end_open, zone_top - 0.1 * abs(delta_y)) # avoid zone to be too thick (from rapid price change)
                self.barrier_zones[1].append(barrier_zone(self.bi_index, new_vertex.idx, zone_top, zone_bot, default_end, default_end, 1, 0, 0, None, None))
        self.vertices.append(new_vertex)
        
        #　s = self.supply_zones[1]
        #　d = self.demand_zones[1]
        #　if len(s) > 1 and len(d) > 1:
        #　    print(s[-1].index, s[-1].left, s[-2].index, s[-2].left)
        #　    print(d[-1].index, d[-1].left, d[-2].index, d[-2].left)
        
    @staticmethod
    def get_strength_rating(historical_sum:float, historical_sample_num:int, new_sample_value:float):
        new_sum = historical_sum + new_sample_value
        new_sample_num = historical_sample_num + 1
        new_average = new_sum / new_sample_num
        strength_rating = round(new_sample_value / new_average / 0.5) # 0:<25%, 1:<75%, 2:<125%, ...
        if strength_rating > 10:
            strength_rating = 10
        return new_sum, new_sample_num, strength_rating