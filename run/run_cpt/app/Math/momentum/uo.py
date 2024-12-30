import array

class uo:
    def __init__(self,
                 closes: array.array,
                 highs: array.array,
                 lows: array.array,
                 short_period: int = 7,    # Shortest timeframe
                 mid_period: int = 14,     # Medium timeframe
                 long_period: int = 28,    # Longest timeframe
                 short_weight: float = 4.0,  # Weight for short BP average
                 mid_weight: float = 2.0,    # Weight for medium BP average
                 long_weight: float = 1.0    # Weight for long BP average
                 ):
        # Initialize indicator array
        self.uo = array.array('d', [])
        
        # Store inputs
        self.closes = closes
        self.highs = highs
        self.lows = lows
        self.short_period = int(short_period)
        self.mid_period = int(mid_period)
        self.long_period = int(long_period)
        
        # Store buying pressure and true range
        self.buying_pressure = []
        self.true_range = []
        
        # Normalize weights
        total_weight = short_weight + mid_weight + long_weight
        self.short_weight = short_weight / total_weight
        self.mid_weight = mid_weight / total_weight
        self.long_weight = long_weight / total_weight
        
        # Previous values needed for calculations
        self.previous_close = 0.0

    def _true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range"""
        return max(
            high - low,  # Current high-low range
            abs(high - prev_close),  # Current high to previous close
            abs(low - prev_close)    # Current low to previous close
        )
        
    def _buying_pressure(self, close: float, low: float, prev_close: float) -> float:
        """Calculate Buying Pressure"""
        return close - min(low, prev_close)
        
    def _average_ratio(self, period: int) -> float:
        """Calculate average ratio for a specific period"""
        if len(self.buying_pressure) < period:
            return 0.0
            
        bp_sum = sum(self.buying_pressure[-period:])
        tr_sum = sum(self.true_range[-period:])
        
        return bp_sum / tr_sum if tr_sum != 0 else 0.0

    def update(self):
        """
        Updates Ultimate Oscillator:
        UO = 100 * (4*Avg7 + 2*Avg14 + 1*Avg28)/(4+2+1)
        where AvgN = Sum(BP)/Sum(TR) over N periods
        BP = Close - min(Low, PrevClose)
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        """
        # Get current values
        close = self.closes[-1]
        high = self.highs[-1]
        low = self.lows[-1]
        
        if not self.buying_pressure:  # First update
            self.previous_close = close
            self.buying_pressure.append(0.0)
            self.true_range.append(high - low)
            self.uo.append(50.0)  # Start at neutral
            return
            
        # Calculate current period values
        bp = self._buying_pressure(close, low, self.previous_close)
        tr = self._true_range(high, low, self.previous_close)
        
        # Store values
        self.buying_pressure.append(bp)
        self.true_range.append(tr)
        
        # Keep only needed periods of history
        max_period = max(self.short_period, self.mid_period, self.long_period)
        if len(self.buying_pressure) > max_period:
            self.buying_pressure.pop(0)
            self.true_range.pop(0)
            
        # Calculate averages for each period
        avg1 = self._average_ratio(self.short_period)
        avg2 = self._average_ratio(self.mid_period)
        avg3 = self._average_ratio(self.long_period)
            
        # Calculate Ultimate Oscillator
        uo_value = 100.0 * (
            self.short_weight * avg1 +
            self.mid_weight * avg2 +
            self.long_weight * avg3
        )
        
        self.uo.append(uo_value)
        self.previous_close = close
        
        # Maintain fixed length
        LEN = 100
        if len(self.uo) > 2*LEN:
            del self.uo[:-LEN]