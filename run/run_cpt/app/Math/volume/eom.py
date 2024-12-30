import array
from typing import List
class eom:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 volumes: array.array,
                 period: int = 14           
                 ):
        self.emv = array.array('d', [])
        
        # Store inputs
        self.highs = highs
        self.lows = lows
        self.volumes = volumes
        self.period = int(period)
        
        # Rolling window for normalization
        self.window_volumes:List[float] = []
        self.window_ranges:List[float] = []
        
        # Previous values for midpoint calc
        self.previous_mid:float = (highs[-1] + lows[-1]) / 2
        
    def update(self):
        """
        Updates Ease of Movement (EMV) with self-scaling normalization
        Formula: 
        1. Distance = ((H + L)/2 - (Prior H + Prior L)/2)
        2. BoxRatio = Volume/TypicalVolume / (Range/TypicalRange)
        3. EMV = Distance / BoxRatio
                
        Normalization:
        - Volume scaled by moving median volume
        - Price range scaled by moving median range
        - Results in more consistent values across different instruments
        """
        # Current values
        current_high = float(self.highs[-1])
        current_low = float(self.lows[-1])
        current_mid = (current_high + current_low) / 2
        current_volume = float(self.volumes[-1]) + 1 # avoid 0
        current_range = current_high - current_low or 1e-8  # Avoid div by 0
        
        # Update normalization windows
        self.window_volumes.append(current_volume)
        self.window_ranges.append(current_range)
        if len(self.window_volumes) > self.period:
            self.window_volumes.pop(0)
            self.window_ranges.pop(0)
            
        # Get median values (middle element of sorted array)
        typical_volume = sorted(self.window_volumes)[len(self.window_volumes)//2]
        typical_range = sorted(self.window_ranges)[len(self.window_ranges)//2]
        
        # Calculate EMV
        distance = current_mid - self.previous_mid
        box_ratio = (current_volume/typical_volume) / (current_range/typical_range)
        raw_emv = distance / box_ratio
        
        # Simple moving average
        window = list(self.emv[-self.period+1:]) + [raw_emv] if len(self.emv) else [raw_emv]
        smoothed_emv = sum(window) / len(window)
        
        # Update values
        self.emv.append(float(smoothed_emv))
        self.previous_mid = current_mid
        
        # Maintain fixed length
        if len(self.emv) > 200:
            del self.emv[:-100]