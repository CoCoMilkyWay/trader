from typing import Dict, Iterable, List, Optional, Union
from collections import deque

from Chan.Common.CTime import CTime

class PA_Volume_Profile():
    def __init__(self):
        self.volume_inited:bool = False
        self.price_bin_width:float
        self.volume_idx_min :int
        self.volume_idx_max :int
        self.bi_volume_profile      :deque[List[int]] = deque() # (price_bin * buy/sellside): volume
        self.session_volume_profile :deque[List[int]] = deque()
        self.history_volume_profile :deque[List[int]] = deque()
        
    def update_volume_profile(self, batch_volume_profile, type:str): 
        # batch -> bi (merge all batches within bi after new bi is sure)
        # bi -> session (with active trendlines)
        # session -> history
        if type == 'batch':
            batch_time:CTime                    = batch_volume_profile[0]
            index_range_low:int                 = batch_volume_profile[1]
            index_range_high:int                = batch_volume_profile[2]
            batch_volume_buyside:List[int]      = batch_volume_profile[3]
            batch_volume_sellside:List[int]     = batch_volume_profile[4]
            price_bin_width                     = batch_volume_profile[5]
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
    