from typing import Dict, Iterable, List, Optional, Union
from collections import deque

from Chan.Common.CTime import CTime

# this to consider in volume profile analysis:
# POC: point of control
# VA: value area (30/70)
# HVN: high value node
# LVN: low volume node (gap)
# VP Shape:
#   D-shape: balanced: grid trade
#   p-shaped: bullish-trend, bearish if is new high
#   b-shaped: bearish-trend, bullish if new low
#   multiple peak distribution: watchout for news
#   trend proflie: one sided movement(trend) leaves an imbalanced area
# VWAP to track market cost (find out percent of people curently winning)
# sub-interval session VP (day, segment etc.)
# volume clusters

class PA_Volume_Profile():
    def __init__(self):
        self.volume_inited:bool = False
        self.price_bin_width:float
        self.volume_idx_min :int
        self.volume_idx_max :int
        
        # DAY: because of T+1 for A-stocks, day VP maybe of more importance
        # SESSION: how do you define session?
        #   1. a support can become(under some conditions) resistance after breakout once
        #   lost most of its effects after breakout twice
        #   However, effective levels will span across multiple timeframe with ineffective levels in between,
        #   thus, it is better to do it in liquidity/order_flow analysis rather than Volume Profile
        #   2. thus, session here means a higher level trend (aka. segment from Chan theory)
        
        # (price_bin * buy/sellside): volume
        self.day_volume_profile     :deque[List[int]] = deque()
        self.bi_volume_profile      :deque[List[int]] = deque()
        self.session_volume_profile :deque[List[int]] = deque()
        self.history_volume_profile :deque[List[int]] = deque()
        
    # trigger step -> kline iteration over levels -> add klu -> add klc / update bi
    #   -> check if new bi formed -> update volume profiles accordingly
    def update_volume_profile(self, batch_volume_profile:List, type:str):
        # batch -> bi (merge all batches within bi after new bi is sure)
        # bi -> session (with active trendlines)
        # session -> history
        # if type == 'batch':
        batch_time:CTime                = batch_volume_profile[0]
        index_range_low:int             = batch_volume_profile[1]
        index_range_high:int            = batch_volume_profile[2]
        batch_volume_buyside:List[int]  = batch_volume_profile[3]
        batch_volume_sellside:List[int] = batch_volume_profile[4]
        price_bin_width                 = batch_volume_profile[5]
        if not self.volume_inited:
            self.volume_idx_min = index_range_low
            self.volume_idx_max = index_range_high
            new_max_idx = index_range_high - index_range_low + 1
            new_min_idx = 0
            self.price_bin_width = price_bin_width
            self.volume_inited = True
        else:
            new_max_idx = index_range_high - self.volume_idx_max
            new_min_idx = self.volume_idx_min - index_range_low
        if new_max_idx > 0: # update profile index
            for _ in range(new_max_idx):
                self.bi_volume_profile.append([0,0])
                self.session_volume_profile.append([0,0])
                self.history_volume_profile.append([0,0])
            self.volume_idx_max = index_range_high
        if new_min_idx > 0:
            for _ in range(new_min_idx):
                self.bi_volume_profile.appendleft([0,0])
                self.session_volume_profile.appendleft([0,0])
                self.history_volume_profile.appendleft([0,0])
            self.volume_idx_min = index_range_low
            
        for i in range(index_range_low, index_range_high+1):
            idx_batch = i - index_range_low
            idx = i - self.volume_idx_min
            self.bi_volume_profile[idx][0] += batch_volume_buyside[idx_batch]
            self.bi_volume_profile[idx][1] += batch_volume_sellside[idx_batch]
        # print(f'{self.volume_idx_min}[{index_range_low}, {index_range_high}]{self.volume_idx_max}: {len(self.bi_volume_profile)}')
            
    def get_adjusted_volume_profile(self, max_mapped:float, type:str):
        if type == 'bi':
            volume_profile = self.bi_volume_profile
        elif type == 'session':
            volume_profile = self.session_volume_profile
        elif type == 'history':
            volume_profile = self.history_volume_profile
        buyside:List[int|float] = [price_bin[0] for price_bin in volume_profile]
        sellside:List[int|float] = [price_bin[1] for price_bin in volume_profile]
        
        max_volume = max(max(buyside), max(sellside))
        buyside = [price_bin/max_volume*max_mapped for price_bin in buyside]
        sellside = [price_bin/max_volume*max_mapped for price_bin in sellside]
        
        buyside_curve = self.normalized_gaussian(buyside)
        sellside_curve = self.normalized_gaussian(sellside)
        return buyside, sellside, buyside_curve, sellside_curve
    
    @staticmethod
    def normalized_gaussian(data):
        # gaussian smoothed volume_profile curve
        from scipy.ndimage import gaussian_filter1d
        smoothed_data = gaussian_filter1d(data, sigma=1.5)
        bar_area = sum(data)
        smoothed_area = sum(smoothed_data)
        smoothed_data_normalized = smoothed_data * (bar_area / smoothed_area)
        return smoothed_data_normalized
    