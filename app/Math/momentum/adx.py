import array

class adx:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 period: int = 14,      # Default period is 14
                 adx_period: int = 14   # Period for ADX smoothing
                 ):
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.period = period
        self.adx_period = adx_period
        
        # Initialize arrays for final values
        self.pdi = array.array('d', [])  # Plus DI
        self.ndi = array.array('d', [])  # Negative DI
        self.adx = array.array('d', [])  # ADX
        
        # Keep track of previous values for TR and DM calculations
        self.tr_sum = 0.0      # True Range sum
        self.plus_dm_sum = 0.0  # Plus Directional Movement sum
        self.minus_dm_sum = 0.0 # Minus Directional Movement sum
        self.dx_sum = 0.0       # For ADX smoothing
        self.data_points = 0
        
    def wilder_smooth(self, previous_sum: float, current_value: float, period: int) -> float:
        """Apply Wilder's smoothing formula"""
        return previous_sum - (previous_sum / period) + current_value
        
    def update(self):
        """
        Calculate ADX:
        1. Calculate True Range (TR) and Directional Movement (DM)
        2. Smooth TR and DM using Wilder's smoothing
        3. Calculate DI+ and DI-
        4. Calculate DX and smooth it to get ADX
        """
        self.data_points += 1
        
        # Need at least 2 points to calculate price changes
        if self.data_points < 2:
            self.pdi.append(0.0)
            self.ndi.append(0.0)
            self.adx.append(0.0)
            return
            
        # Calculate True Range
        high = self.highs[-1]
        low = self.lows[-1]
        prev_close = self.closes[-2]
        
        tr = max(
            high - low,  # Current high - low
            abs(high - prev_close),  # Current high - previous close
            abs(low - prev_close)    # Current low - previous close
        )
        
        # Calculate Directional Movement
        prev_high = self.highs[-2]
        prev_low = self.lows[-2]
        
        plus_dm = max(0, high - prev_high)
        minus_dm = max(0, prev_low - low)
        
        # If plus_dm and minus_dm are equal, both become 0
        if plus_dm == minus_dm:
            plus_dm = minus_dm = 0
        elif plus_dm < minus_dm:
            plus_dm = 0
        else:
            minus_dm = 0
            
        # Apply Wilder's smoothing
        if self.data_points <= self.period:
            # Initial accumulation
            self.tr_sum += tr
            self.plus_dm_sum += plus_dm
            self.minus_dm_sum += minus_dm
            
            self.pdi.append(0.0)
            self.ndi.append(0.0)
            self.adx.append(0.0)
            return
        elif self.data_points == self.period + 1:
            # First smoothed values
            self.tr_sum = tr
            self.plus_dm_sum = plus_dm
            self.minus_dm_sum = minus_dm
        else:
            # Continue smoothing
            self.tr_sum = self.wilder_smooth(self.tr_sum, tr, self.period)
            self.plus_dm_sum = self.wilder_smooth(self.plus_dm_sum, plus_dm, self.period)
            self.minus_dm_sum = self.wilder_smooth(self.minus_dm_sum, minus_dm, self.period)
            
        # Calculate DI+ and DI-
        if self.tr_sum != 0:
            pdi = 100 * self.plus_dm_sum / self.tr_sum
            ndi = 100 * self.minus_dm_sum / self.tr_sum
        else:
            pdi = ndi = 0.0
            
        self.pdi.append(pdi)
        self.ndi.append(ndi)
        
        # Calculate DX
        if pdi + ndi != 0:
            dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        else:
            dx = 0.0
            
        # Calculate ADX using another smoothing period
        if self.data_points <= self.period + self.adx_period:
            self.dx_sum += dx
            adx = 0.0
        elif self.data_points == self.period + self.adx_period + 1:
            # First ADX value
            adx = self.dx_sum / self.adx_period
            self.dx_sum = adx
        else:
            # Continue smoothing ADX
            self.dx_sum = self.wilder_smooth(self.dx_sum, dx, self.adx_period)
            adx = self.dx_sum
            
        self.adx.append(adx)
        LEN = 100
        if len(self.adx) > 2*LEN:
            del self.pdi[:-LEN]
            del self.ndi[:-LEN]
            del self.adx[:-LEN]
        return