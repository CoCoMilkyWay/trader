from typing import List, Dict, Optional

from wtpy.WtDataDefs import WtNpKline

from Chan.Common.CEnum import KL_TYPE, DATA_FIELD
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit

# handler = KLineHandler(lv_dict)
# for np_bar in bars:
#     results = handler.process_bar(np_bar)

class KLineHandler:
    __slots__ = ('levels', 'multiplier', 'periods', 'counts', 'opens', 'highs', 'lows', 'closes', 'volumes', 'start_times')
    
    def __init__(self, lv_list: list):
        # Store levels in reverse order (1m -> 5m -> 15m -> ...)
        self.levels     = [lv[0] for lv in reversed(lv_list)]
        self.multiplier = [lv[1] for lv in reversed(lv_list)]
                
        # Calculate periods for each level (how many bars from previous level needed)
        self.periods = {}
        for i in range(len(self.levels)):
            self.periods[self.levels[i]] = self.multiplier[i]
        

        # Single arrays for all levels
        self.counts = {level: 0 for level in self.periods}
        self.opens = {level: 0.0 for level in self.periods}
        self.highs = {level: 0.0 for level in self.periods}
        self.lows = {level: 0.0 for level in self.periods}
        self.closes = {level: 0.0 for level in self.periods}
        self.volumes = {level: 0 for level in self.periods}
        self.start_times = {level: CTime(1990,1,1,0,0) for level in self.periods}
    
    def process_bar(self, np_bars: WtNpKline) -> dict:
        """Single method to handle all processing"""
        # Extract bar data once
        time_str = str(np_bars.bartimes[-1])
        open_ = np_bars.opens[-1]
        high = np_bars.highs[-1]
        low = np_bars.lows[-1]
        close = np_bars.closes[-1]
        volume = int(np_bars.volumes[-1])
        
        # Create 1-min bar
        time = CTime(
            int(time_str[:4]),
            int(time_str[4:6]),
            int(time_str[6:8]),
            int(time_str[8:10]),
            int(time_str[10:12]),
            auto=False
        )
        
        results = {
            KL_TYPE.K_1M: [CKLine_Unit({
                DATA_FIELD.FIELD_TIME: time,
                DATA_FIELD.FIELD_OPEN: open_,
                DATA_FIELD.FIELD_HIGH: high,
                DATA_FIELD.FIELD_LOW: low,
                DATA_FIELD.FIELD_CLOSE: close,
                DATA_FIELD.FIELD_VOLUME: volume
            }, autofix=True)]
        }
        
        # Current bar values to feed into next level
        curr_open, curr_high = open_, high
        curr_low, curr_close = low, close
        curr_vol, curr_time = volume, time
        
        # Process each level using output from previous level
        for level in self.levels:
            count = self.counts[level]
            
            if count == 0:
                self.opens[level] = curr_open
                self.highs[level] = curr_high
                self.lows[level] = curr_low
                self.start_times[level] = curr_time
            else:
                self.highs[level] = max(self.highs[level], curr_high)
                self.lows[level] = min(self.lows[level], curr_low)
            
            self.closes[level] = curr_close
            self.volumes[level] += curr_vol
            count += 1
            
            if count >= self.periods[level]:
                # Create bar for this level
                results[level] = [CKLine_Unit({
                    DATA_FIELD.FIELD_TIME: time,
                    DATA_FIELD.FIELD_OPEN: self.opens[level],
                    DATA_FIELD.FIELD_HIGH: self.highs[level],
                    DATA_FIELD.FIELD_LOW: self.lows[level],
                    DATA_FIELD.FIELD_CLOSE: self.closes[level],
                    DATA_FIELD.FIELD_VOLUME: self.volumes[level]
                }, autofix=True)]
                
                # Update values for next level
                curr_open = self.opens[level]
                curr_high = self.highs[level]
                curr_low = self.lows[level]
                curr_close = self.closes[level]
                curr_vol = self.volumes[level]
                curr_time = self.start_times[level]
                
                # Reset counters
                self.counts[level] = 0
                self.volumes[level] = 0
            else:
                self.counts[level] = count
                return results  # Early return if this level isn't complete
                
        return results