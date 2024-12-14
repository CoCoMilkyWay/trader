from config.cfg_cpt import cfg_cpt
from typing import Tuple, List, Dict


class ChandelierIndicator:
    def __init__(self, length=22, atr_period=22, mult=3.0):
        self.length = length
        self.atr_period = atr_period
        self.mult = mult

        # Price histories
        self.highs:List[float] = []
        self.lows:List[float] = []
        self.closes:List[float] = []
        self.tr:List[float] = []

        # State variables
        self.prev_shortcs = 1<<30
        self.prev_longcs = 0
        self.prev_direction = 0
        self.prev_close = 0

        # History
        if cfg_cpt.dump_ind:
            # self.long_idx: int = 0
            # self.short_idx: int = 0
            # self.his_longts: List[List[float]] = [[]]
            # self.his_shortts:List[List[float]] = [[]]
            # self.his_longcs: List[List[float]] = [[]]
            # self.his_shortcs:List[List[float]] = [[]]

            self.switch_idx: int = 0
            self.his_switch_ts: List[float] = []
            self.his_switch_vs: List[float] = []

            self.his_ts: List[float] = []
            self.his_longcs: List[float] = []
            self.his_shortcs: List[float] = []

    def update(self, high: float, low: float, close: float, ts: float) -> Tuple[bool, bool]:
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        # Keep only needed history
        max_period = max(self.length, self.atr_period)
        if len(self.highs) > max_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        if len(self.highs) < 2:
            return False, False

        # Calculate TR
        tr = max(
            high - low,
            abs(high - self.prev_close) if self.prev_close else 0,
            abs(low - self.prev_close) if self.prev_close else 0
        )
        self.tr.append(tr)
        if len(self.tr) > self.atr_period:
            self.tr.pop(0)

        if len(self.highs) < max_period:
            return False, False

        # Calculate ATR
        atr = sum(self.tr) / len(self.tr)

        # Calculate stops
        highest = max(self.highs[-self.length:])
        lowest = min(self.lows[-self.length:])

        short_stop = highest + self.mult * atr  # max(tr, atr) * self.mult
        long_stop = lowest - self.mult * atr

        # Update stops
        shortcs = short_stop if close > self.prev_shortcs else min(short_stop, self.prev_shortcs)
        longcs = long_stop if close < self.prev_longcs else max(long_stop, self.prev_longcs)
        
        # Calculate switches (close price break through stop level)
        long_switch = close >= self.prev_shortcs and self.prev_close < self.prev_shortcs
        short_switch = close <= self.prev_longcs and self.prev_close > self.prev_longcs

        # Calculate direction
        direction = (1 if self.prev_direction <= 0 and long_switch else
                     -1 if self.prev_direction >= 0 and short_switch else
                     self.prev_direction)

        switch = direction != self.prev_direction
        
        # Update state
        self.prev_shortcs = shortcs
        self.prev_longcs = longcs
        self.prev_direction = direction
        self.prev_close = close

        if cfg_cpt.dump_ind:
            # if direction == 1 and not long_switch: # long
            #     self.his_longts[self.long_idx].append(ts)
            #     self.his_longcs[self.long_idx].append(longcs)
            # elif direction == -1 and not short_switch: # short
            #     self.his_shortts[self.short_idx].append(ts)
            #     self.his_shortcs[self.short_idx].append(shortcs)
            # if long_switch:
            #     self.long_idx += 1
            #     self.his_longts.append([])
            #     self.his_longcs.append([])
            # if short_switch:
            #     self.short_idx += 1
            #     self.his_shortts.append([])
            #     self.his_shortcs.append([])
            self.his_ts.append(ts)
            self.his_longcs.append(longcs)
            self.his_shortcs.append(shortcs)
            if switch:
                self.switch_idx += 1
                if long_switch:
                    self.his_switch_ts.append(ts)
                    self.his_switch_vs.append(longcs)
                elif short_switch:
                    self.his_switch_ts.append(ts)
                    self.his_switch_vs.append(shortcs)

        return long_switch, short_switch

# # Example usage
# indicator = ChandelierIndicator()
#
# # Feed bar data one at a time
# bar = {'high': 100, 'low': 98, 'close': 99}
# result = indicator.update(bar)
# print(result)
