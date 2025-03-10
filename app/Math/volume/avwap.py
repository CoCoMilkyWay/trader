import array

class avwap:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 volumes: array.array,
                 atr: array.array,          
                 period: int = 20,          # Rolling window size
                 smooth_period: int = 0     # 0: disable deviation smoothing
                 ):
        # Initialize arrays for final indicators
        self.avwap = array.array('d', [])
        self.deviation = array.array('d', [])
        
        # Store inputs
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes
        self.atr = atr
        self.period = int(period)
        self.smooth_period = int(smooth_period)
        
        # Rolling window queues
        self.window_pv = []    # Price * volume pairs for window
        self.window_vol = []   # Volume values for window
        
        self.previous_avwap = 0.0
        self.previous_deviation = 0.0
        self.data_points = 0
        
    def update(self):
        """
        Updates AVWAP and deviation multiplier:
        1. Maintains rolling window of price-volume data
        2. Calculates VWAP over current window
        3. Computes deviation from VWAP in ATR units
        """
        self.data_points += 1
        
        # Calculate typical price
        typical_price = (self.highs[-1] + self.lows[-1] + self.closes[-1]) / 3.0
        current_volume = float(self.volumes[-1])
        
        # Update rolling window
        self.window_pv.append(typical_price * current_volume)
        self.window_vol.append(current_volume)
        
        # Maintain window size
        if len(self.window_pv) > self.period:
            self.window_pv.pop(0)
            self.window_vol.pop(0)
        
        # Calculate VWAP over current window
        total_pv = sum(self.window_pv)
        total_vol = sum(self.window_vol)
        
        if total_vol > 0:
            vwap_value = float(total_pv / total_vol)
        else:
            vwap_value = typical_price
            
        self.avwap.append(vwap_value)
        self.previous_avwap = vwap_value
        
        # Calculate ATR deviation
        if self.atr[-1] != 0:
            dev_value = float((self.closes[-1] - vwap_value) / self.atr[-1])
        else:
            dev_value = 0.0
            
        # Optional smoothing using EMA
        if self.smooth_period > 0:
            alpha = 2.0 / (self.smooth_period + 1)
            if not self.deviation:  # First value
                smoothed_dev = dev_value
            else:
                smoothed_dev = float(dev_value * alpha + 
                                   self.deviation[-1] * (1 - alpha))
            dev_value = smoothed_dev
            
        self.deviation.append(dev_value)
        self.previous_deviation = dev_value
        
        # Maintain fixed length for both arrays
        LEN = 100
        if len(self.avwap) > 2*LEN:
            del self.avwap[:-LEN]
            del self.deviation[:-LEN]