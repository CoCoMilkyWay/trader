from typing import List
from config.cfg_cpt import cfg_cpt

class ParabolicSARIndicator:
    def __init__(self, acceleration=0.02, max_acceleration=0.2, hl_searching_range:int = 100):
        self.acc = acceleration
        self.max_acc = max_acceleration
        self.range = hl_searching_range
        
        # Price histories
        self.highs: List[float] = []
        self.lows: List[float] = []
        
        # State variables
        self.current_sar: float = 0.0
        self.extreme_point: float = 0.0
        self.acceleration_factor: float = acceleration
        self.prev_direction: int = 0  # 1 for long, -1 for short
        
        # History for plotting/analysis
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_sar: List[float] = []
            self.his_ep: List[float] = []
    
    def update(self, high: float, low: float, ts: float) -> float:
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        
        # Initial setup for first bar
        if len(self.highs) == 1:
            self.current_sar = low
            self.extreme_point = high
            self.prev_direction = 1
            return 0
            
        # Need at least self.range bars for SAR calculation
        if len(self.highs) < self.range:
            return 0
            
        # Calculate new SAR
        prior_sar = self.current_sar
        
        if self.prev_direction == 1:  # In uptrend
            # Update extreme point if we have a new high
            if high > self.extreme_point:
                self.extreme_point = high
                self.acceleration_factor = min(self.acceleration_factor + self.acc, self.max_acc)
                
            # Calculate new SAR
            self.current_sar = prior_sar + \
                             self.acceleration_factor * (self.extreme_point - prior_sar)
                             
            # Ensure SAR is below the prior two lows
            self.current_sar = min(self.current_sar, min(self.lows[-self.range:]))
            
            # Check for trend reversal
            if self.current_sar > low:  # SAR penetrated price
                self.current_sar = self.extreme_point
                self.extreme_point = low
                self.acceleration_factor = self.acc
                self.prev_direction = -1
                
        else:  # In downtrend
            # Update extreme point if we have a new low
            if low < self.extreme_point:
                self.extreme_point = low
                self.acceleration_factor = min(self.acceleration_factor + self.acc, self.max_acc)
                
            # Calculate new SAR
            self.current_sar = prior_sar + \
                             self.acceleration_factor * (self.extreme_point - prior_sar)
                             
            # Ensure SAR is above the prior two highs
            self.current_sar = max(self.current_sar, max(self.highs[-self.range:]))
            
            # Check for trend reversal
            if self.current_sar < high:  # SAR penetrated price
                self.current_sar = self.extreme_point
                self.extreme_point = high
                self.acceleration_factor = self.acc
                self.prev_direction = 1
        
        # Store history if enabled
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_sar.append(self.current_sar)
            self.his_ep.append(self.extreme_point)
        
        # Keep only needed history
        if len(self.highs) > self.range:
            self.highs.pop(0)
            self.lows.pop(0)
            
        return self.current_sar