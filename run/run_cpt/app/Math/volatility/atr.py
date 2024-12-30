import array

class atr:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 period: int = 14
                 ):
        self.atr = array.array('d', [])
        self.tr = array.array('d', [])     # True Range values
        
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.period = period
        
        # Store only necessary data points
        self.previous_atr = None
        self.previous_tr = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate ATR:
        1. True Range = max(
            current_high - current_low,
            abs(current_high - previous_close),
            abs(current_low - previous_close)
        )
        2. ATR = EMA of True Range
        """
        self.data_points += 1
        
        if self.data_points < 2:  # Need at least 2 data points for TR
            tr_value = self.highs[-1] - self.lows[-1]  # Just high-low for first point
        else:
            # Calculate True Range
            tr_value = max(
                self.highs[-1] - self.lows[-1],                    # Current high-low
                abs(self.highs[-1] - self.closes[-2]),            # High-previous close
                abs(self.lows[-1] - self.closes[-2])              # Low-previous close
            )
        
        self.tr.append(tr_value)
        self.previous_tr = tr_value
        
        # Get ATR value (from pre-calculated EMA of TR)
        # If no EMA provided, use simple moving average for the period
        if self.data_points < self.period:
            atr_value = sum(self.tr) / self.data_points
        else:
            atr_value = sum(self.tr[-self.period:]) / self.period
                
        self.atr.append(atr_value)
        self.previous_atr = atr_value
        
        return