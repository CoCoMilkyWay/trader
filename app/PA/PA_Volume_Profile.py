from typing import List, Dict, Tuple, Iterable, Optional, Union, cast
from collections import deque
from pprint import pprint
import numpy as np
import pandas as pd
import math
import copy

from Chan.Bi.Bi import CBi
from Chan.Common.CTime import CTime

# https://www.sohu.com/a/385016680_569161

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
# VWAP to track market cost (e.g. percent of people winning in 1/5 day)
# sub-interval session VP (day, segment etc.)
# volume clusters

# NOTE: what is the most powerful way to identify institution behavior?
# 1. volume from institutions: price oscillates within a range,
#   while volume peak remains relatively stable
# 2. pull up from institutions: price rise > 10%,
#   while most volumes accumulated from bottom consolidation still not sold out
# 3. selling out from institutions: as price moving up,
#   volume profile peak moving up


class PA_Volume_Profile():
    def __init__(self):
        self.volume_inited: bool = False
        self.price_bin_width: float = 0
        self.volume_idx_min: int
        self.volume_idx_max: int

        # self.batch_day_last: int
        # self.day_idx: int
        # self.list_day_idx: int

        # 0(all sell) < ratio < inf(all buy)
        self.inner_outer_order_ratio: List[float]

        self.bi_idx: int = 0
        self.list_bi_idx: int

        # DAY: because of T+1 for A-stocks, day VP maybe of more importance
        # SESSION: how do you define session?
        #   1. a support can become(under some conditions) resistance after breakout once
        #   lost most of its effects after breakout twice
        #   However, effective levels will span across multiple timeframe with ineffective levels in between,
        #   thus, it is better to do it in liquidity/order_flow analysis rather than Volume Profile
        #   2. thus, session here means a higher level trend (aka. segment from Chan theory)

        # self.bi_volume_profile: deque[List[int]] = deque()
        # self.session_volume_profile: deque[List[int]] = deque()
        self.history_volume_profile: deque[List[int]] = deque()

    # -> None | List[Union[List[float], List[int]]]
    def update_volume_profile(self, bi: CBi):
        # a bi is split into: sell_premium / indifferent / buy_premium region
        if self.price_bin_width == 0:  # auto determine price bin width
            self.price_bin_width = 10**round(math.log10(bi.amp()/10))

        total_volume: int = 0
        index_range_low: int = int(bi._low()//self.price_bin_width)  # floor
        index_range_high: int = int(bi._high()//self.price_bin_width)  # floor
        bi_volume_buyside: List[int] = [0] * \
            (index_range_high-index_range_low+1)
        bi_volume_sellside: List[int] = [0] * \
            (index_range_high-index_range_low+1)
        for klc in bi.klc_lst:
            for klu in klc.lst:
                total_volume += klu.volume
                index: int = int(((klu.close+klu.open)/2) //
                                 self.price_bin_width)
                # error correction:
                if not (index_range_low < index < index_range_high):
                    index = index_range_low
                if klu.close > klu.open:
                    bi_volume_buyside[index-index_range_low] += klu.volume
                else:
                    bi_volume_sellside[index-index_range_low] += klu.volume

        if not self.volume_inited:
            # self.inner_outer_order_ratio = []
            self.volume_idx_min = index_range_low
            self.volume_idx_max = index_range_high
            new_max_idx = index_range_high - index_range_low + 1
            new_min_idx = 0
            self.volume_inited = True
        else:
            new_max_idx = index_range_high - self.volume_idx_max
            new_min_idx = self.volume_idx_min - index_range_low

        # buyside_sum = sum(bi_volume_buyside) + 1  # avoid 0
        # sellside_sum = sum(bi_volume_sellside) + 1  # avoid 0
        # ratio = self.arctan_map(x=buyside_sum/sellside_sum)
        # self.inner_outer_order_ratio.append(ratio)

        self.bi_idx += 1
        self.list_bi_idx = self.bi_idx
        if self.volume_inited:
            bi_price_bins = len(bi_volume_buyside)
            bi_volume_profile_total = [
                bi_volume_buyside[i] + bi_volume_sellside[i] for i in range(bi_price_bins)]

            # this is bi volume profile (not a lot of data)
            # convert to price map first to ease later calculation
            # array[price, volume]
            bi_price_mapped_volume: None | List[Union[List[float], List[int]]] = [
                [], []]
            # index of the first non-zero volume from the left/right side
            left_cnt = next((i for i, x in enumerate(
                bi_volume_profile_total) if x != 0), None)
            right_cnt = next((i for i, x in enumerate(
                reversed(bi_volume_profile_total)) if x != 0), None)
            if left_cnt is not None and right_cnt is not None:
                left_index = left_cnt
                right_index = (bi_price_bins - 1) - right_cnt
                # price
                bi_price_mapped_volume[0] = [(i+index_range_low) * self.price_bin_width
                                          for i in range(left_index, right_index+1)]
                # volume
                bi_price_mapped_volume[1] = bi_volume_profile_total[left_index:right_index+1]

            # update profile index
            if new_max_idx > 0:
                for _ in range(new_max_idx):
                    self.history_volume_profile.append([0, 0])
                self.volume_idx_max = index_range_high
            if new_min_idx > 0:
                for _ in range(new_min_idx):
                    self.history_volume_profile.appendleft([0, 0])
                self.volume_idx_min = index_range_low

            self.bi_volume_profile = [bi_volume_buyside, bi_volume_sellside]
            for i in range(index_range_low, index_range_high+1):
                idx_bi = i - index_range_low
                idx_history = i - self.volume_idx_min
                buy_side = int(bi_volume_buyside[idx_bi])
                sell_side = int(bi_volume_sellside[idx_bi])
                self.history_volume_profile[idx_history][0] += buy_side
                self.history_volume_profile[idx_history][1] += sell_side

            # print(f'{self.volume_idx_min}[{index_range_low}, {index_range_high}]{self.volume_idx_max}: {len(self.bi_volume_profile)}')
            # print(bi_volume_profile_total)
            # print(bi_price_mapped_volume)
        return bi_price_mapped_volume

    def get_adjusted_volume_profile(self, max_mapped: float, type: str, sigma: float = 1.5):
        # if type == 'bi':
        #     _len = len(self.bi_volume_profile[0])
        #     volume_profile = [[self.bi_volume_profile[0][i],
        #                        self.bi_volume_profile[1][i]]
        #                       for i in range(_len)]
        if type == 'history':
            _len = len(self.history_volume_profile)
            volume_profile = self.history_volume_profile
        buyside: List[int | float] = [price_bin[0]
                                      for price_bin in volume_profile]
        sellside: List[int | float] = [price_bin[1]
                                       for price_bin in volume_profile]
        total: List[int | float] = [buyside[i] + sellside[i]
                                    for i in range(_len)]

        # max_volume = max(max(buyside), max(sellside))
        max_volume = max(total) + 1  # avoid 0
        buyside = [price_bin/max_volume*max_mapped for price_bin in buyside]
        sellside = [price_bin/max_volume*max_mapped for price_bin in sellside]

        buyside_curve = self.normalized_gaussian(buyside, sigma)
        sellside_curve = self.normalized_gaussian(sellside, sigma)

        # Calculate average cost
        idx_min = self.volume_idx_min
        idx_max = idx_min + _len
        prices = np.array(
            [idx * self.price_bin_width for idx in range(idx_min, idx_max)])
        # avoid 0 (similar but not VWAP)
        volume_weighted_cost = np.dot(prices, total) / (sum(total)+1)
        # Calculate the 30th, 50th, and 70th percentiles (cumulative)
        cumulative_sum = np.cumsum(total)
        pct_20_value = 0.2 * cumulative_sum[-1]
        pct_50_value = 0.5 * cumulative_sum[-1]
        pct_80_value = 0.8 * cumulative_sum[-1]
        pct_20_index = np.searchsorted(
            cumulative_sum, pct_20_value, side='left')
        pct_50_index = np.searchsorted(
            cumulative_sum, pct_50_value, side='left')
        pct_80_index = np.searchsorted(
            cumulative_sum, pct_80_value, side='left')
        pct_20 = (self.volume_idx_min + pct_20_index) * self.price_bin_width
        pct_50 = (self.volume_idx_min + pct_50_index) * self.price_bin_width
        pct_80 = (self.volume_idx_min + pct_80_index) * self.price_bin_width

        # NOTE if volume_weighted_cost ~= pct_50, it is a strong sign of consolidation phase

        results = [
            buyside,
            sellside,
            buyside_curve,
            sellside_curve,
            volume_weighted_cost,
            pct_20,
            pct_50,
            pct_80,
        ]
        return results  # [buyside, sellside, buy_curve, sell_curve, ...]

    @staticmethod
    def normalized_gaussian(data, sigma: float = 1.5):
        # gaussian smoothed volume_profile curve
        from scipy.ndimage import gaussian_filter1d
        smoothed_data = gaussian_filter1d(data, sigma=sigma)
        bar_area = sum(data)
        smoothed_area = sum(smoothed_data)
        smoothed_data_normalized = smoothed_data * (bar_area / smoothed_area)
        return smoothed_data_normalized

    @staticmethod
    def arctan_map(x):  # map 0, 1, inf to 0, 0.5, 1
        return np.arctan(x) / (np.pi / 2)
