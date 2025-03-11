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

import os
import sys
import math
import copy
import numpy as np
from typing import Tuple, List, Dict, Union
from enum import Enum, auto
from dataclasses import dataclass

from app.PA.PA_types import vertex, barrier_zone


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
        self.ts_per_index: float = 0
        self.vertices: List[vertex] = []
        # self.support: List[float] = []
        # self.resistance: List[float] = []

        # xxx_zones = List[zones_formed, zones_forming]
        self.barrier_zones:     List[List[barrier_zone]] = [
            [], []]  # [formed, forming]
        self.order_blocks:      List[List[barrier_zone]] = [
            [], []]  # [formed, forming]
        self.mitigation_zones:  List[List[barrier_zone]] = [
            [], []]  # [formed, forming]
        self.break_zones:       List[List[barrier_zone]] = [
            [], []]  # [formed, forming]
        self.rejection_zones:   List[List[barrier_zone]] = [
            [], []]  # [formed, forming]

        # snapshots
        self.barrier_snapshot: List[float] = []
        self.premium_snapshot: List[Tuple[float, float]] = []
        self.snapshot_ready: bool = False
        # self.history_barrier: List[List[float | int]] = []
        # self.anchor_barrier: float = 0.0

        # corner cases
        self.last_active_zones: List[Tuple[int, bool, int]] = []

        # average & percentile
        self.volume_sum: float = 1  # avoid division by 0
        self.sample_num: int = 0

        self.snapshot: List = []  # snapshot of all liquidity zones

        self.last_BoS: bool = False  # last bi also creates a BoS
        self.BoS_type_history: List[bool] = []
        self.BoS_price_history: List[float] = []

    def add_vertex(self, new_vertex: vertex):
        # liquidity zone should be formed at breakthrough, but for ease of computation
        # only update at FX formation
        TOP = 1
        BOT = -1
        self.bi_index += 1

        # NOTE: 2 consecutive bi should either both do or do not break the same zone
        # because of flexibility in top/bot definition, this cannot be guaranteed
        active_zones: List[Tuple[int, bool, int]] = []

        if len(self.vertices) > 1:
            last_1_vertex = self.vertices[-1]
            last_2_vertex = self.vertices[-2]
            if len(self.vertices) == 2:
                self.ts_per_index = (
                    last_1_vertex.ts - last_2_vertex.ts)/(last_1_vertex.idx - last_2_vertex.idx)
            delta_y = new_vertex.value - last_1_vertex.value
            delta_x = new_vertex.idx - last_1_vertex.idx
            delta_y_abs = abs(delta_y)
            # if the entering bi respect the level, it is not broken
            tolerance = 0.1 * delta_y_abs
            FX_type = TOP if delta_y > 0 else BOT
            if FX_type == BOT:
                bi_top: float = last_1_vertex.value
                bi_bottom: float = new_vertex.value
            else:
                bi_top: float = new_vertex.value
                bi_bottom: float = last_1_vertex.value

            # check if new BoS is formed
            # use 1.3 to check significance of break
            if FX_type == BOT:
                new_BoS = (last_1_vertex.value-new_vertex.value) > 1.3 * \
                    (last_1_vertex.value-last_2_vertex.value)
            else:
                new_BoS = (new_vertex.value-last_1_vertex.value) > 1.3 * \
                    (last_2_vertex.value-last_1_vertex.value)

            if new_BoS and len(self.barrier_zones[1]) > 0:
                self.barrier_zones[1][-1].BoS = [last_2_vertex.ts,
                                                 new_vertex.ts, last_2_vertex.value]
                # 0: break up 1: break down
                self.BoS_type_history.append(FX_type == BOT)
                # not BoS level, but the sup/res level
                self.BoS_price_history.append(self.vertices[-2].value)
                if len(self.BoS_type_history) > 3:
                    # check if a potential reversal against a trend happened
                    if all([self.BoS_type_history[-2] == self.BoS_type_history[-3-i] for i in [0,]]) and self.BoS_type_history[-2] != self.BoS_type_history[-1]:
                        down = 1 if FX_type == BOT else -1
                        # check previous trend formation (only need 2 BoS of the same type to form a trend)
                        if all([(down*self.BoS_price_history[-2-i]) > (down*self.BoS_price_history[-3-i]) for i in [0,]]):
                            # check if the structure it breaks is significant enough:
                            if abs(self.vertices[-2].value - self.vertices[-3].value) > 0.5*abs(self.vertices[-4].value - self.vertices[-5].value):
                                if self.last_BoS:  # check if it is a new BoS formed from the other side
                                    self.barrier_zones[1][-1].ChoCh = True

                # check if is order block
                # not check FVG here, check later
                self.barrier_zones[1][-1].OB = True

                # update last BoS
                self.last_BoS = new_BoS

                # update zone top/bot for order block (exit bi is stronger than entering bi)
                last_zone = self.barrier_zones[1][-1]

                if FX_type == BOT:  # vertex is top
                    last_zone.bottom = min(
                        last_zone.bottom, last_zone.top - tolerance)
                else:  # vertex is bot
                    last_zone.top = max(
                        last_zone.top, last_zone.bottom + tolerance)

            # 1. change state of previous zones
            # 2. form new zone
            for zones in [
                self.barrier_zones,
            ]:
                zones_forming_buffer = []
                for zone_forming in zones[1]:
                    top = zone_forming.top + tolerance
                    bot = zone_forming.bottom - tolerance
                    zone_broken = (bi_top > top) and (bi_bottom < bot)
                    active_zones.append(
                        (zone_forming.index, zone_broken, FX_type))
                    discrepency = False
                    for z in self.last_active_zones:
                        if z[0] == zone_forming.index:
                            # last bi is up and both "last price begin" and "current price ends" lower than zone
                            if z[2] == 1 and (new_vertex.value < zone_forming.bottom) and (last_2_vertex.value < zone_forming.bottom):
                                discrepency = zone_broken != z[1]
                            if z[2] == -1 and (new_vertex.value > zone_forming.top) and (last_2_vertex.value > zone_forming.top):
                                discrepency = zone_broken != z[1]
                            # correct potential discrepency
                            if discrepency:
                                zone_broken = z[1]
                    if FX_type == TOP:
                        zone_touched = bot < bi_top < top
                    else:
                        zone_touched = bot < bi_bottom < top

                    if zone_touched:
                        # if there was a large bi respecting this zone, we respect too
                        zone_forming.tolerance = max(
                            zone_forming.tolerance, tolerance)
                        zone_forming.num_touch += 1

                    if not zone_forming.MB and zone_touched:
                        zone_forming.MB = True

                    if zone_broken:
                        zone_type = zone_forming.type

                        # calculating horizontal span
                        zone_level: float = (
                            zone_forming.top + zone_forming.bottom)/2
                        zone_ratio_in_bi: float = (
                            zone_level - bi_bottom) / delta_y_abs
                        if FX_type == TOP:
                            zone_ts: float = last_1_vertex.ts + \
                                (zone_ratio_in_bi * delta_x * self.ts_per_index)
                        else:
                            zone_ts: float = last_1_vertex.ts + \
                                ((1-zone_ratio_in_bi) * delta_x * self.ts_per_index)

                        # delete VP if zone is already formed to save memory
                        # NOTE: after 1st breakthrough, the old volume already means nothing
                        #       (even though this position is still critical with reversed polarity),
                        #       the only thing we care is the new breakthrough volume, which will be
                        #       recorded in the volume of newly formed zone(not the old zone)
                        zone_forming.enter_bi_VP = None  # refer to above
                        zone_forming.leaving_bi_VP = None  # refer to above
                        if zone_type == 0:  # demand
                            zone_forming.right0 = zone_ts
                            zone_forming.type = 3
                            if zone_forming.OB:
                                zone_forming.BB = True
                            zones_forming_buffer.append(
                                zone_forming)  # zone_forming
                        elif zone_type == 1:  # supply
                            zone_forming.right0 = zone_ts
                            zone_forming.type = 2
                            if zone_forming.OB:
                                zone_forming.BB = True
                            zones_forming_buffer.append(
                                zone_forming)  # zone_forming
                        elif zone_type == 2:  # demand (1st break supply)
                            zone_forming.right1 = zone_ts
                            # zone_formed (2nd break)
                            zones[0].append(zone_forming)
                        elif zone_type == 3:  # supply (1st break demand)
                            zone_forming.right1 = zone_ts
                            # zone_formed (2nd break)
                            zones[0].append(zone_forming)
                    else:
                        zones_forming_buffer.append(
                            zone_forming)  # zone_forming
                zones[1] = zones_forming_buffer

            # add new forming zones
            if FX_type == BOT:
                zone_bot = new_vertex.value
                # avoid zone to be too thick (from rapid price change)
                # min(end_open, zone_bot + 0.1 * delta_y_abs)
                zone_top = zone_bot + tolerance
                self.barrier_zones[1].append(barrier_zone(
                    self.bi_index, new_vertex.ts, zone_top, zone_bot, 0))
            else:
                zone_top = new_vertex.value
                # avoid zone to be too thick (from rapid price change)
                # max(end_open, zone_top - 0.1 * delta_y_abs)
                zone_bot = zone_top - tolerance
                self.barrier_zones[1].append(barrier_zone(
                    self.bi_index, new_vertex.ts, zone_top, zone_bot, 1))

        self.vertices.append(new_vertex)
        self.last_active_zones = active_zones

        # update barrier/premium region snapshot (ease calculations)
        self.barrier_snapshot = []
        self.premium_snapshot = []

        # zones = []
        # for zone in self.barrier_zones[1]:
        #    if zone.strength_rating < 2:
        #        # newly formed zones are naturally weak
        #        # include them to function properly
        #        if zone.leaving_bi_VP:
        #            continue
        #    zones.append(zone)
        # sorted(zones, key=lambda x: x.top)
        # i = 0
        # while i < len(zones):
        #    current_zone = zones[i]
        #    if i + 1 < len(zones) and current_zone.bottom >= zones[i + 1].top:
        #        # If overlapping, merge with next zone
        #        merged_top = min(current_zone.top, zones[i + 1].top)
        #        merged_bottom = max(current_zone.bottom, zones[i + 1].bottom)
        #        self.barrier_snapshot.append((merged_top + merged_bottom) * 0.5)
        #        i += 2
        #    else:
        #        # No overlap, use current zone
        #        self.barrier_snapshot.append((current_zone.top + current_zone.bottom) * 0.5)
        #    i += 1

        for zone in self.barrier_zones[1]:
            if zone.strength_rating < 2:
                # newly formed zones are naturally weak
                # include them to function properly
                if zone.leaving_bi_VP:
                    continue
            # if not zone.OB:
            #     continue
            self.barrier_snapshot.append((zone.top+zone.bottom)*0.5)

        # merging needs to be performed

        if len(self.barrier_snapshot) > 1:
            self.snapshot_ready = True
            self.barrier_snapshot.sort()
            for i, barrier in enumerate(self.barrier_snapshot):
                dist_to_prev = barrier - self.barrier_snapshot[i-1] \
                    if i > 0 else self.barrier_snapshot[i+1] - barrier
                dist_to_next = self.barrier_snapshot[i+1] - barrier \
                    if i < len(self.barrier_snapshot)-1 else barrier - self.barrier_snapshot[i-1]
                self.premium_snapshot.append((
                    barrier - dist_to_prev*0.1,  # < 0.5
                    barrier + dist_to_next*0.1,  # < 0.5
                ))

    def check_sup_res(self, price: float):
        # 0: demand
        # 1: supply
        # 2: demand (1st break supply)
        # 3: supply (1st break demand)

        # price is like a ball bouncing between levels
        # once it touch a level, it anchor to that level
        # then it either breakout or reverse
        # and when it goes to new levels, it may react to it

        # we cannot use bi to determine the previous anchor point,
        # because bi update is kinda random(maybe too late because of update rules)

        # if the anchor point and price is at the same level,
        # we go along the price direction
        # otherwise we seek reversal(because this is reversal strategy)

        NO_RECORD = 4
        BREAKOUT_COUNT = 10

        if not self.snapshot_ready:  # not enough levels
            return False, [], [], ''

        if price > self.barrier_snapshot[-1]:  # newest high, short
            return True, self.barrier_snapshot, [], '^'
        if price < self.barrier_snapshot[0]:  # newest low, long
            return True, [], self.barrier_snapshot, 'v'

        # Binary search to find the matching region
        snapshot_len = len(self.premium_snapshot)
        left, right = 0, snapshot_len - 1
        while left <= right:
            mid = (left + right) >> 1  # Faster than division
            reg = self.premium_snapshot[mid]
            if reg[0] < price < reg[1]:  # level match found
                # Found matching region
                lower_targets = self.barrier_snapshot[:mid]
                upper_targets = self.barrier_snapshot[mid+1:]
                return True, lower_targets, upper_targets, '*'
                #
                # if not (reg[0] < anchor_level < reg[1]):
                #     self.history_barrier.append((self.barrier_snapshot[mid], 0))
                # long_short = price > anchor_price
                # if not (reg[0] < anchor_price < reg[1]):
                #     symbol = '*'
                #     long_short = not long_short

            if price <= reg[0]:
                right = mid - 1
            else:
                left = mid + 1

        return False, [], [], ''

        # # update anchor to get long_short
        # if self.history_barrier == []:
        #     # barrier_level, entry_dir
        #     # (-1: from under, 0: at barrier level, 1: from up)
        #     # entry times (avoid false breakout)
        #     self.history_barrier.append([price, 0, 0])
        #
        # anchor_level = self.history_barrier[-1][0]
        # anchor_flag = self.history_barrier[-1][1]
        # if anchor_flag == 0:  # last anchor is in premium region
        #     if not match:  # leaving premium region
        #         new_anchor_flag = -1 if price > anchor_level else 1
        #         self.history_barrier.append([price, new_anchor_flag, 0])
        #     pass
        # else:
        #     if match:
        #         barrier_level = self.barrier_snapshot[mid]
        #         self.history_barrier.append([barrier_level, 0, 0])
        #     else:
        #         if anchor_flag == -1:  # price rise above barrier
        #             if price > anchor_level:
        #                 self.history_barrier[-1][0] = price
        #                 # confirm breakouts
        #                 self.history_barrier[-1][2] += 1
        #         if anchor_flag == 1:  # price drop below barrier
        #             if price < anchor_level:
        #                 self.history_barrier[-1][0] = price
        #                 # confirm breakouts
        #                 self.history_barrier[-1][2] += 1
        #
        # num_records = len(self.history_barrier)
        # if num_records > NO_RECORD:
        #     self.history_barrier.pop(0)
        #
        # if match:
        #     # Found matching region
        #     lower_targets = self.barrier_snapshot[:mid]
        #     upper_targets = self.barrier_snapshot[mid+1:]
        #     if not lower_targets:
        #         symbol = 'v'
        #     if not upper_targets:
        #         symbol = '^'
        #     symbol = '*'
        #
        #     #if num_records == NO_RECORD:
        #     ## check elasticity built up for quality reversal
        #     #dir_1 = price > self.history_barrier[-2][0]
        #     #dir_2 = price > self.history_barrier[-3][0]
        #     elasticity = abs(price - self.history_barrier[-2][0])
        #     # print(elasticity/price,cfg_cpt.FEE * cfg_cpt.NO_FEE)
        #     if elasticity/price > (cfg_cpt.FEE * cfg_cpt.NO_FEE):
        #         long_short = elasticity < 0
        #         return True, long_short, lower_targets, upper_targets, symbol
        #     else:
        #         return False, False, [], [], ''
        # else:
        #     return False, False, [], [], ''

    def update_volume_zone(self, price_mapped_volume: List[Union[List[float], List[int]]]):
        A = price_mapped_volume
        n = len(price_mapped_volume[0])
        # Use slicing to split upper and lower halves
        # why 3? because we want pnl close to > 2
        lower_half = [x[:n // 2] for x in A]
        upper_half = [x[n // 2:] for x in A]
        # if len(lower_half) == 0:
        #     print('lower ==================================')
        # if len(upper_half) == 0:
        #     print('upper ==================================')
        bi_index = self.bi_index
        for zone in self.barrier_zones[1]:
            if zone.index == bi_index:  # enter_bi_VP
                if zone.type == 0:  # demand
                    zone.enter_bi_VP = lower_half
                else:  # supply
                    zone.enter_bi_VP = upper_half
            elif zone.index == (bi_index-1):  # leaving_bi_VP
                if zone.type == 0:  # demand
                    zone.leaving_bi_VP = lower_half
                else:  # supply
                    zone.leaving_bi_VP = upper_half
                if zone.enter_bi_VP and zone.leaving_bi_VP:
                    volume = sum(
                        zone.enter_bi_VP[1]) + sum(zone.leaving_bi_VP[1])
                    self.volume_sum, \
                        self.sample_num, \
                        zone.strength_rating = \
                        self.get_strength_rating(
                            self.volume_sum,
                            self.sample_num,
                            volume)

    @staticmethod
    def get_strength_rating(historical_sum: float, historical_sample_num: int, new_sample_value: float):
        new_sum = historical_sum + new_sample_value
        new_sample_num = historical_sample_num + 1
        new_average = new_sum / new_sample_num
        # 0:<25%, 1:50%, 2:100%, ...
        strength_rating = round(new_sample_value / new_average / 0.5)
        if strength_rating > 10:
            strength_rating = 10
        return new_sum, new_sample_num, strength_rating
