from typing import List, Dict, Optional
from wtpy.WtDataDefs import WtNpKline
from Chan.Common.CEnum import KL_TYPE, DATA_FIELD
from Chan.Common.CTime import CTime
from Chan.KLine.KLine_Unit import CKLine_Unit

class KLineHandler:
    __slots__ = ('levels', 'multiplier', 'periods', 'counts', 'opens', 'highs', 
                 'lows', 'closes', 'volumes', 'start_times')
    
    def __init__(self, lv_list: list):
        # Store levels in reverse order (1m -> 5m -> 15m -> ...)
        self.levels = [lv[0] for lv in reversed(lv_list)]
        self.multiplier = [lv[1] for lv in reversed(lv_list)]
        
        # Pre-calculate periods
        self.periods = dict(zip(self.levels, self.multiplier))
        
        # Use lists instead of dictionaries for better access speed
        n_levels = len(self.levels)
        self.counts = [0] * n_levels
        self.opens = [0.0] * n_levels
        self.highs = [0.0] * n_levels
        self.lows = [0.0] * n_levels
        self.closes = [0.0] * n_levels
        self.volumes = [0] * n_levels
        self.start_times = {level: CTime(1990,1,1,0,0) for level in self.periods}
    
    def process_bar(self, np_bars: WtNpKline) -> dict[KL_TYPE, List[CKLine_Unit]]:
        """Optimized method to handle all processing"""
        # Extract bar data once and use local variables
        time_str = str(np_bars.bartimes[-1])
        time = CTime(
            int(time_str[:4]),
            int(time_str[4:6]),
            int(time_str[6:8]),
            int(time_str[8:10]),
            int(time_str[10:12]),
            auto=False
        )
        
        curr_open = np_bars.opens[-1]
        curr_high = np_bars.highs[-1]
        curr_low = np_bars.lows[-1]
        curr_close = np_bars.closes[-1]
        curr_vol = int(np_bars.volumes[-1])
        
        # Create initial bar dict to avoid repeated key lookups
        bar_dict = {
            DATA_FIELD.FIELD_TIME: time,
            DATA_FIELD.FIELD_OPEN: curr_open,
            DATA_FIELD.FIELD_HIGH: curr_high,
            DATA_FIELD.FIELD_LOW: curr_low,
            DATA_FIELD.FIELD_CLOSE: curr_close,
            DATA_FIELD.FIELD_VOLUME: curr_vol
        }
        
        # Pre-allocate results dict with known first entry
        results = {
            KL_TYPE.K_1M: [CKLine_Unit(bar_dict, autofix=True)]
        }
        
        # Process each level
        for i, level in enumerate(self.levels):
            count = self.counts[i]
            
            if count == 0:
                self.opens[i] = curr_open
                self.highs[i] = curr_high
                self.lows[i] = curr_low
                self.start_times[level] = time
            else:
                # Use direct comparison instead of min/max functions
                if curr_high > self.highs[i]:
                    self.highs[i] = curr_high
                if curr_low < self.lows[i]:
                    self.lows[i] = curr_low
            
            self.closes[i] = curr_close
            self.volumes[i] += curr_vol
            count += 1
            
            if count >= self.multiplier[i]:
                # Reuse bar_dict pattern for better performance
                bar_dict = {
                    DATA_FIELD.FIELD_TIME: time,
                    DATA_FIELD.FIELD_OPEN: self.opens[i],
                    DATA_FIELD.FIELD_HIGH: self.highs[i],
                    DATA_FIELD.FIELD_LOW: self.lows[i],
                    DATA_FIELD.FIELD_CLOSE: self.closes[i],
                    DATA_FIELD.FIELD_VOLUME: self.volumes[i]
                }
                results[level] = [CKLine_Unit(bar_dict, autofix=True)]
                
                # Update values for next level
                curr_open = self.opens[i]
                curr_high = self.highs[i]
                curr_low = self.lows[i]
                curr_close = self.closes[i]
                curr_vol = self.volumes[i]
                
                # Reset counters
                self.counts[i] = 0
                self.volumes[i] = 0
            else:
                self.counts[i] = count
                return results
                
        return results