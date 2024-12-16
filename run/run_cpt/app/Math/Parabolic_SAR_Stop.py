from typing import Tuple, List
from config.cfg_cpt import cfg_cpt

class ParabolicSARIndicator:
    def __init__(self, acceleration=0.02, max_acceleration=0.2, initial_acceleration=None):
        self.af = float(acceleration) if acceleration and acceleration > 0 else 0.02
        self.af0 = float(initial_acceleration) if initial_acceleration and initial_acceleration > 0 else self.af
        self.max_af = float(max_acceleration) if max_acceleration and max_acceleration > 0 else 0.2
        
        # Price histories for two bars
        self.highs: List[float] = []
        self.lows: List[float] = []
        
        # State variables
        self.current_sar: float = 0.0
        self.extreme_point: float = 0.0
        self.acceleration_factor: float = self.af0
        self.falling: bool = True
        
        # History for plotting/analysis
        if cfg_cpt.dump_ind:
            self.his_ts: List[float] = []
            self.his_sar: List[float] = []
            self.his_ep: List[float] = []
    
    def update(self, high: float, low: float, ts: float) -> Tuple[bool, bool]:
        """Update the PSAR indicator with new prices"""
        # Update price histories
        self.highs.append(high)
        self.lows.append(low)
        
        # Need at least 2 bars
        if len(self.highs) < 2:
            if len(self.highs) == 1:
                # Initialize on first bar
                up = high - high  # Will be 0 on first bar
                down = low - low  # Will be 0 on first bar
                self.falling = (down > up) and (down > 0)
                
                if self.falling:
                    self.current_sar = high
                    self.extreme_point = low
                else:
                    self.current_sar = low
                    self.extreme_point = high
                
                if cfg_cpt.dump_ind:
                    self.his_ts.append(ts)
                    self.his_sar.append(self.current_sar)
                    self.his_ep.append(self.extreme_point)
            return False, False
        
        # Calculate new SAR
        prior_sar = self.current_sar
        long_switch = False
        short_switch = False
        
        if self.falling:
            # Calculate SAR for downtrend
            self.current_sar = prior_sar + \
                             self.acceleration_factor * (self.extreme_point - prior_sar)
            
            # SAR can't be below the prior two lows
            self.current_sar = max(max(self.lows[-1], self.lows[-2]), self.current_sar)
            
            # Check for reversal
            if low < self.extreme_point:
                self.extreme_point = low
                self.acceleration_factor = min(self.acceleration_factor + self.af0, self.max_af)
                
            if high > self.current_sar:
                long_switch = True
                self.current_sar = self.extreme_point
                self.extreme_point = high
                self.acceleration_factor = self.af0
                self.falling = False
                
        else:
            # Calculate SAR for uptrend
            self.current_sar = prior_sar + \
                             self.acceleration_factor * (self.extreme_point - prior_sar)
            
            # SAR can't be above the prior two highs
            self.current_sar = min(min(self.highs[-1], self.highs[-2]), self.current_sar)
            
            # Check for new extreme point
            if high > self.extreme_point:
                self.extreme_point = high
                self.acceleration_factor = min(self.acceleration_factor + self.af0, self.max_af)
                
            # Check for reversal
            if low < self.current_sar:
                short_switch = True
                self.current_sar = self.extreme_point
                self.extreme_point = low
                self.acceleration_factor = self.af0
                self.falling = True
        
        # Store history if enabled
        if cfg_cpt.dump_ind:
            self.his_ts.append(ts)
            self.his_sar.append(self.current_sar)
            self.his_ep.append(self.extreme_point)
        
        # Keep only last two bars
        if len(self.highs) > 2:
            self.highs.pop(0)
            self.lows.pop(0)
            
        return long_switch, short_switch