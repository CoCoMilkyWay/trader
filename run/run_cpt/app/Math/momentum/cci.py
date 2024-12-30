import array

class cci:
    def __init__(self,
                 highs: array.array,
                 lows: array.array,
                 closes: array.array,
                 ma: array.array,     # Pre-calculated MA of typical price
                 period: int = 20,
                 constant: float = 0.015  # Traditional CCI constant
                 ):
        self.cci = array.array('d', [])
        self.typical_price = array.array('d', [])
        
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.ma = ma
        self.period = period
        self.constant = constant
        
        # Store only necessary data points
        self.previous_cci = None
        self.data_points = 0
        
    def update(self):
        """
        Calculate CCI:
        1. Typical Price = (High + Low + Close) / 3
        2. Mean Deviation = sum(abs(typical_price - MA)) / period
        3. CCI = (Typical Price - MA) / (constant * Mean Deviation)
        """
        self.data_points += 1
        
        # Calculate typical price
        typical = (self.highs[-1] + self.lows[-1] + self.closes[-1]) / 3
        self.typical_price.append(typical)
        
        # Need enough data points for calculation
        if self.data_points < self.period:
            self.cci.append(0)
            return
            
        # Get MA of typical price (pre-calculated)
        ma_typical = self.ma[-1]
        
        # Calculate mean deviation
        period_typical = self.typical_price[-self.period:]
        mean_deviation = sum(abs(price - ma_typical) for price in period_typical) / self.period
        
        # Handle zero mean deviation
        if mean_deviation == 0:
            cci_value = 0  # or could be set to a large number depending on preference
        else:
            # Calculate CCI
            cci_value = (typical - ma_typical) / (self.constant * mean_deviation)
            
        self.cci.append(cci_value)
        self.previous_cci = cci_value
        LEN = 100
        if len(self.cci) > 2*LEN:
            del self.cci[:-LEN]
            del self.typical_price[:-LEN]
        return